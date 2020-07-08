#!/usr/bin/env python

"""DIVERSITY SAMPLING
 
Diversity Sampling examples for Active Learning in PyTorch 

This is an open source example to accompany Chapter 4 from the book:
"Human-in-the-Loop Machine Learning"

It contains four Active Learning strategies:
1. Model-based outlier sampling
2. Cluster-based sampling
3. Representative sampling
4. Adaptive Representative sampling


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
# from numpy import rank

from uncertainty_sampling import UncertaintySampling
from pytorch_clusters import CosineClusters 
from pytorch_clusters import Cluster

if sys.argv[0] == "diversity_sampling.py":
    import active_learning


__author__ = "Robert Munro"
__license__ = "MIT"
__version__ = "1.0.1"

   
class DiversitySampling():


    def __init__(self, verbose=False):
        self.verbose = verbose


    
    def get_cluster_samples(self, data, num_clusters=5, max_epochs=5, limit=5000):
        """Create clusters using cosine similarity
        
        Keyword arguments:
            data -- data to be clustered
            num_clusters -- the number of clusters to create
            max_epochs -- maximum number of epochs to create clusters
            limit -- sample only this many items for faster clustering (-1 = no limit)
        
        Creates clusters by the K-Means clustering algorithm,
        using cosine similarity instead of more common euclidean distance
        
        Creates clusters until converged or max_epochs passes over the data 
            
        """ 
        
        if limit > 0:
            shuffle(data)
            data = data[:limit]
        
        cosine_clusters = CosineClusters(num_clusters)
        
        cosine_clusters.add_random_training_items(data)
        
        for i in range(0, max_epochs):
            print("Epoch "+str(i))
            added = cosine_clusters.add_items_to_best_cluster(data)
            if added == 0:
                break
    
        centroids = cosine_clusters.get_centroids()
        outliers = cosine_clusters.get_outliers()
        randoms = cosine_clusters.get_randoms(3, self.verbose)
        
        return centroids + outliers + randoms
             
    
    def get_representative_samples(self, training_data, unlabeled_data, number=20, limit=10000):
        """Gets the most representative unlabeled items, compared to training data
    
        Keyword arguments:
            training_data -- data with a label, that the current model is trained on
            unlabeled_data -- data that does not yet have a label
            number -- number of items to sample
            limit -- sample from only this many items for faster sampling (-1 = no limit)
    
        Creates one cluster for each data set: training and unlabeled
        
        
        """ 
            
        if limit > 0:
            shuffle(training_data)
            training_data = training_data[:limit]
            shuffle(unlabeled_data)
            unlabeled_data = unlabeled_data[:limit]
            
        training_cluster = Cluster()
        for item in training_data:
            training_cluster.add_to_cluster(item)
        
        unlabeled_cluster = Cluster()    
        for item in unlabeled_data:
            unlabeled_cluster.add_to_cluster(item)
    
        
        for item in unlabeled_data:
            training_score = training_cluster.cosine_similary(item)
            unlabeled_score = unlabeled_cluster.cosine_similary(item)
                        
            representativeness = unlabeled_score - training_score
            
            item[3] = "representative"            
            item[4] = representativeness
                
                     
        unlabeled_data.sort(reverse=True, key=lambda x: x[4])       
        return unlabeled_data[:number:]       
    
    
    def get_adaptive_representative_samples(self, training_data, unlabeled_data, number=20, limit=5000):
        """Adaptively gets the most representative unlabeled items, compared to training data
        
        Keyword arguments:
            training_data -- data with a label, that the current model is trained on
            unlabeled_data -- data that does not yet have a label
            number -- number of items to sample
            limit -- sample from only this many items for faster sampling (-1 = no limit)
            
        Adaptive variant of get_representative_samples() where the training_data is updated
        after each individual selection in order to increase diversity of samples
        
        """
        
        samples = []
        
        for i in range(0, number):
            print("Epoch "+str(i))
            representative_item = self.get_representative_samples(training_data, unlabeled_data, 1, limit)[0]
            samples.append(representative_item)
            unlabeled_data.remove(representative_item)
            
        return samples
    
    
    
    def get_validation_rankings(self, model, validation_data, feature_method):
        """Get model outliers from unlabeled data 
    
        Keyword arguments:
            model -- current Machine Learning model for this task
            unlabeled_data -- data that does not yet have a label
            validation_data -- held out data drawn from the same distribution as the training data
            feature_method -- the method to create features from the raw text
            number -- number of items to sample
            limit -- sample from only this many items for faster sampling (-1 = no limit)
    
        An outlier is defined as 
        unlabeled_data with the lowest average from rank order of logits
        where rank order is defined by validation data inference 
    
        """
                
        validation_rankings = [] # 2D array, every neuron by ordered list of output on validation data per neuron    
    
        # Get per-neuron scores from validation data
        if self.verbose:
            print("Getting neuron activation scores from validation data")
    
        with torch.no_grad():
            v=0
            for item in validation_data:
                textid = item[0]
                text = item[1]
                
                feature_vector = feature_method(text)
                hidden, logits, log_probs = model(feature_vector, return_all_layers=True)  
        
                neuron_outputs = logits.data.tolist()[0] #logits
                
                # initialize array if we haven't yet
                if len(validation_rankings) == 0:
                    for output in neuron_outputs:
                        validation_rankings.append([0.0] * len(validation_data))
                            
                n=0
                for output in neuron_outputs:
                    validation_rankings[n][v] = output
                    n += 1
                            
                v += 1
        
        # Rank-order the validation scores 
        v=0
        for validation in validation_rankings:
            validation.sort() 
            validation_rankings[v] = validation
            v += 1
          
        return validation_rankings 
    
    
    
    def get_rank(self, value, rankings):
        """ get the rank of the value in an ordered array as a percentage 
    
        Keyword arguments:
            value -- the value for which we want to return the ranked value
            rankings -- the ordered array in which to determine the value's ranking
        
        returns linear distance between the indexes where value occurs, in the
        case that there is not an exact match with the ranked values    
        """
        
        index = 0 # default: ranking = 0
        
        for ranked_number in rankings:
            if value < ranked_number:
                break #NB: this O(N) loop could be optimized to O(log(N))
            index += 1        
        
        if(index >= len(rankings)):
            index = len(rankings) # maximum: ranking = 1
            
        elif(index > 0):
            # get linear interpolation between the two closest indexes 
            
            diff = rankings[index] - rankings[index - 1]
            perc = value - rankings[index - 1]
            linear = perc / diff
            index = float(index - 1) + linear
        
        absolute_ranking = index / len(rankings)
    
        return(absolute_ranking)
    
                
    
    def get_model_outliers(self, model, unlabeled_data, validation_data, feature_method, number=5, limit=10000):
        """Get model outliers from unlabeled data 
    
        Keyword arguments:
            model -- current Machine Learning model for this task
            unlabeled_data -- data that does not yet have a label
            validation_data -- held out data drawn from the same distribution as the training data
            feature_method -- the method to create features from the raw text
            number -- number of items to sample
            limit -- sample from only this many items for faster sampling (-1 = no limit)
    
        An outlier is defined as 
        unlabeled_data with the lowest average from rank order of logits
        where rank order is defined by validation data inference 
    
        """
    
        # Get per-neuron scores from validation data
        validation_rankings = self.get_validation_rankings(model, validation_data, feature_method)
        
        # Iterate over unlabeled items
        if self.verbose:
            print("Getting rankings for unlabeled data")
    
        outliers = []
        if limit == -1 and len(unlabeled_data) > 10000 and self.verbose: # we're drawing from *a lot* of data this will take a while                                               
            print("Get rankings for a large amount of unlabeled data: this might take a while")
        else:
            # only apply the model to a limited number of items                                                                            
            shuffle(unlabeled_data)
            unlabeled_data = unlabeled_data[:limit]
    
        with torch.no_grad():
            for item in unlabeled_data:
                text = item[1]
    
                feature_vector = feature_method(text)
                hidden, logits, log_probs = model(feature_vector, return_all_layers=True)            
                
                neuron_outputs = logits.data.tolist()[0] #logits
                   
                n=0
                ranks = []
                for output in neuron_outputs:
                    rank = self.get_rank(output, validation_rankings[n])
                    ranks.append(rank)
                    n += 1 
                
                item[3] = "logit_rank_outlier"
                
                item[4] = 1 - (sum(ranks) / len(neuron_outputs)) # average rank
                
                outliers.append(item)
                
        outliers.sort(reverse=True, key=lambda x: x[4])       
        return outliers[:number:]       
      
                

    
