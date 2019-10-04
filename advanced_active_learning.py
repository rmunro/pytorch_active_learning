#!/usr/bin/env python

"""Diversity Sampling 


"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import math
import datetime
import csv
import re
import os
import getopt, sys

from random import shuffle
from collections import defaultdict	

from diversity_sampling import DiversitySampling
from uncertainty_sampling import UncertaintySampling
from pytorch_clusters import CosineClusters 
from pytorch_clusters import Cluster

if sys.argv[0] == "advanced_active_learning.py":
    import active_learning

__author__ = "Robert Munro"
__license__ = "MIT"
__version__ = "1.0.1"


class AdvancedActiveLearning():


    def __init__(self, verbose=False):
        self.verbose = verbose
        self.uncertainty_sampling = UncertaintySampling(self.verbose)
        self.diversity_sampling = DiversitySampling(self.verbose)


    def get_clustered_uncertainty_samples(self, model, unlabeled_data, method, feature_method,
                                perc_uncertain = 0.1, num_clusters=20, max_epochs=10, limit=10000):
        """Gets the most uncertain items and then clusters the, sampling from each cluster
        
        Keyword arguments:
            model -- machine learning model to get predictions from to determine uncertainty
            unlabeled_data -- data that does not yet have a label
            method -- method for uncertainty sampling (eg: least_confidence())
            feature_method -- the method for extracting features from your data
            perc_uncertain -- percentage of items through uncertainty sampling to cluster
            num_clusters -- the number of clusters to create
            max_epochs -- maximum number of epochs to create clusters
            limit -- sample from only this many predictions for faster sampling (-1 = no limit)      
        """ 
                
        if limit > 0:
            shuffle(unlabeled_data)
            unlabeled_data = unlabeled_data[:limit]            
        uncertain_count = math.ceil(len(unlabeled_data) * perc_uncertain)
        
        uncertain_samples = self.uncertainty_sampling.get_samples(model, unlabeled_data, method, 
                                                                feature_method, uncertain_count, limit=limit)
                    
        samples = self.diversity_sampling.get_cluster_samples(uncertain_samples, 
                                                                num_clusters=num_clusters)
        
        for item in samples:
            item[3] = method.__name__+"_"+item[3]
            
        return samples



    def get_uncertain_model_outlier_samples(self, model, outlier_model, unlabeled_data, training_data, validation_data, method, feature_method,
                                                        perc_uncertain = 0.1, number=10, limit=10000): 
        """Gets the most uncertain items and samples the biggest model outliers among them
        
        Keyword arguments:
            model -- machine learning model to get predictions from to determine uncertainty
            outlier_model -- machine learning model for outlier prediction
            validation_data -- data not used for the outlier_model but from the same distribution
            unlabeled_data -- data that does not yet have a label
            method -- method for uncertainty sampling (eg: least_confidence())
            feature_method -- the method for extracting features from your data
            perc_uncertain -- percentage of items through uncertainty sampling to cluster            
            number -- the final number of items to sample
            limit -- sample from only this many predictions for faster sampling (-1 = no limit)      
        """ 
        
        if limit > 0:
            shuffle(unlabeled_data)
            unlabeled_data = unlabeled_data[:limit]            
        uncertain_count = math.ceil(len(unlabeled_data) * perc_uncertain)

        uncertain_samples = self.uncertainty_sampling.get_samples(model, unlabeled_data, method, 
                                                                feature_method, uncertain_count, limit=limit)
        
        samples = self.diversity_sampling.get_model_outliers(outlier_model, uncertain_samples, validation_data, 
                                                            feature_method, number=number, limit=limit)

        for item in samples:
            item[3] = method.__name__+"_"+item[3]
            
        return samples
        


    def get_representative_cluster_samples(self, training_data, unlabeled_data, number=10, num_clusters=20, max_epochs=10, limit=10000):
        """Gets the most representative unlabeled items, compared to training data, across multiple clusters
        
        Keyword arguments:
            training_data -- data with a label, that the current model is trained on
            unlabeled_data -- data that does not yet have a label
            number -- number of items to sample
            limit -- sample from only this many items for faster sampling (-1 = no limit)
            num_clusters -- the number of clusters to create
            max_epochs -- maximum number of epochs to create clusters
       
        """ 
            
        if limit > 0:
            shuffle(training_data)
            training_data = training_data[:limit]
            shuffle(unlabeled_data)
            unlabeled_data = unlabeled_data[:limit]
            
        # Create clusters for training data
    
        training_clusters = CosineClusters(num_clusters)
        training_clusters.add_random_training_items(training_data)
        
        for i in range(0, max_epochs):
            print("Epoch "+str(i))
            added = training_clusters.add_items_to_best_cluster(training_data)
            if added == 0:
                break
    
        # Create clusters for unlabeled data
    
        unlabeled_clusters = CosineClusters(num_clusters)    
        unlabeled_clusters.add_random_training_items(unlabeled_data)
        
        for i in range(0, max_epochs):
            print("Epoch "+str(i))
            added = unlabeled_clusters.add_items_to_best_cluster(unlabeled_data)
            if added == 0:
                break
    
        # get scores
        
        most_representative_items = []
        
        # for each cluster of unlabeled data
        for cluster in unlabeled_clusters.clusters:
            most_representative = None
            representativeness = float("-inf")
            
            # find the item in that cluster most like the unlabeled data 
            item_keys = list(cluster.members.keys())
             
            for key in item_keys:
                item = cluster.members[key]
                
                _r, unlabeled_score = unlabeled_clusters.get_best_cluster(item)
                _, training_score = training_clusters.get_best_cluster(item)
    
                cluster_representativeness = unlabeled_score - training_score
    
                if cluster_representativeness > representativeness:
                    representativeness = cluster_representativeness 
                    most_representative = item
                    
            most_representative[3] = "representative_clusters"            
            most_representative[4] = representativeness
            most_representative_items.append(most_representative)
                     
        most_representative_items.sort(reverse=True, key=lambda x: x[4])       
        return most_representative_items[:number:]    
    
    
    
    def get_high_uncertainty_cluster(self, model, unlabeled_data, method, feature_method,
                                                number=10, num_clusters=20, max_epochs=10, limit=10000):
        """Gets items from the cluster with the highest average uncertainty
        
        Keyword arguments:
            model -- machine learning model to get predictions from to determine uncertainty
            unlabeled_data -- data that does not yet have a label
            method -- method for uncertainty sampling (eg: least_confidence())
            feature_method -- the method for extracting features from your data
            number -- number of items to sample
            num_clusters -- the number of clusters to create
            max_epochs -- maximum number of epochs to create clusters
            limit -- sample from only this many items for faster sampling (-1 = no limit)       
        """ 
                
        if limit > 0:
            shuffle(unlabeled_data)
            unlabeled_data = unlabeled_data[:limit]            

        unlabeled_clusters = CosineClusters(num_clusters)    
        unlabeled_clusters.add_random_training_items(unlabeled_data)
        
        for i in range(0, max_epochs):
            print("Epoch "+str(i))
            added = unlabeled_clusters.add_items_to_best_cluster(unlabeled_data)
            if added == 0:
                break
    
        # get scores
        
        most_uncertain_cluster = None
        highest_average_uncertainty = 0.0
        
        # for each cluster of unlabeled data
        for cluster in unlabeled_clusters.clusters:
            total_uncertainty = 0.0
            count = 0

            item_keys = list(cluster.members.keys())
             
            for key in item_keys:
                item = cluster.members[key]
                text = item[1]
                
                feature_vector = feature_method(text)
                hidden, logits, log_probs = model(feature_vector, return_all_layers=True)  
    
                prob_dist = torch.exp(log_probs) # the probability distribution of our prediction
                
                score = method(prob_dist.data[0]) # get the specific type of uncertainty sampling
                
                total_uncertainty += 1.0
                count += 1
                
            average_uncertainty = total_uncertainty / count
            if average_uncertainty > highest_average_uncertainty:
                highest_average_uncertainty = average_uncertainty
                most_uncertain_cluster = cluster
            
        samples = most_uncertain_cluster.get_random_members(number)
            
        return samples

    
    
    
    
    
    
    
     

