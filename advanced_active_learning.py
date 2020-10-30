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
import copy

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



    def get_uncertain_model_outlier_samples(self, model, outlier_model, unlabeled_data, validation_data, method, feature_method,
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
                
                total_uncertainty += score
                count += 1
                
            average_uncertainty = total_uncertainty / count
            if average_uncertainty > highest_average_uncertainty:
                highest_average_uncertainty = average_uncertainty
                most_uncertain_cluster = cluster
            
        samples = most_uncertain_cluster.get_random_members(number)
            
        return samples



        
    
    def get_deep_active_transfer_learning_uncertainty_samples(self, model, unlabeled_data, validation_data, feature_method,
                                                number=100, limit=10000, epochs=10, select_per_epoch=100):
        """Uses transfer learning to predict uncertainty within the model
        
        Keyword arguments:
            model -- machine learning model to get predictions from to determine uncertainty
            unlabeled_data -- data that does not yet have a label
            validation_data -- data with a label that is not in the training set, to be used for transfer learning
            feature_method -- the method for extracting features from your data
            number -- number of items to sample
            epochs -- number of epochs to train transfer-learning model
            select_per_epoch -- number of items to train on per epoch of training
            limit -- sample from only this many items for faster sampling (-1 = no limit)       
        """ 

        correct_predictions = [] # validation items predicted correctly
        incorrect_predictions = [] # validation items predicted incorrectly
        item_hidden_layers = {} # hidden layer of each item, by id

        # 1 GET PREDICTIONS ON VALIDATION DATA FROM MODEL

        for item in validation_data:
                    
            id = item[0]
            text = item[1]
            label = item[2]
                
            feature_vector = feature_method(text)
            hidden, logits, log_probs = model(feature_vector, return_all_layers=True)  

            item_hidden_layers[id] = hidden
        
            # get confidence that item is disaster-related
            prob_related = math.exp(log_probs.data.tolist()[0][1]) 

            if item[3] == "seen":
                correct_predictions.append(item)
                
            elif (label == "1" and prob_related > 0.5) or (label == "0" and prob_related <= 0.5):
                correct_predictions.append(item)
            else:
                incorrect_predictions.append(item)
                        
            # item.append(hidden) # the hidden layer will be the input to our new model
            

        # 2 BUILD A NEW MODEL TO PREDICT WHETHER VALIDATION ITEMS WERE CORRECT OR INCORRECT                
        correct_model = SimpleUncertaintyPredictor(128)
        loss_function = nn.NLLLoss()
        optimizer = optim.SGD(correct_model.parameters(), lr=0.01)                    

        # print(correct_predictions)
       
        for epoch in range(epochs):
            if self.verbose:
                print("Epoch: "+str(epoch))
            current = 0
    
            # make a subset of data to use in this epoch
            # with an equal number of items from each label
    
            shuffle(correct_predictions) #randomize the order of the validation data       
            shuffle(incorrect_predictions) #randomize the order of the validation data       
 
            correct_ids = {}
            for item in correct_predictions:
                correct_ids[item[0]] = True         
            epoch_data = correct_predictions[:select_per_epoch]
            epoch_data += incorrect_predictions[:select_per_epoch]
            shuffle(epoch_data) 
                    
            # train the final layers model
            for item in epoch_data:                
                id = item[0]
                label = 0
                if id in correct_ids:
                    label = 1
    
                correct_model.zero_grad() 
    
                # print(item)
                feature_vec = item_hidden_layers[id]
                target = torch.LongTensor([label])
    
                log_probs = correct_model(feature_vec)
    
                # compute loss function, do backward pass, and update the gradient
                loss = loss_function(log_probs, target)
                loss.backward(retain_graph=True)
                optimizer.step()    
            
         
           
        # 3 PREDICT WHETHER UNLABELED ITEMS ARE CORRECT

        if limit > 0:
            shuffle(unlabeled_data)
            unlabeled_data = unlabeled_data[:limit]            
        

        deep_active_transfer_preds = []

        with torch.no_grad():
            v=0
            for item in unlabeled_data:
                text = item[1]
                
                # get prediction from main model
                feature_vector = feature_method(text)
                hidden, logits, log_probs = model(feature_vector, return_all_layers=True)  

                # use hidden layer from main model as input to model predicting correct/errors
                logits, log_probs = correct_model(hidden, return_all_layers=True)  
                
                # get confidence that item is correctly labeled
                prob_correct = 1 - math.exp(log_probs.data.tolist()[0][1]) 

                if(label == "0"):
                    prob_correct = 1 - prob_correct
                    
                item[3] = "predicted_error"            
                item[4] = 1 - prob_correct
                deep_active_transfer_preds.append(item)

       
        deep_active_transfer_preds.sort(reverse=True, key=lambda x: x[4])
          
        return deep_active_transfer_preds[:number:]    
   
  
  
    def get_atlas_samples(self, model, unlabeled_data, validation_data, feature_method,
                                    number=100, limit=10000, number_per_iteration=10, epochs=10, select_per_epoch=100):
        """Uses transfer learning to predict uncertainty within the model
        
        Keyword arguments:
            model -- machine learning model to get predictions from to determine uncertainty
            unlabeled_data -- data that does not yet have a label
            validation_data -- data with a label that is not in the training set, to be used for transfer learning
            feature_method -- the method for extracting features from your data
            number -- number of items to sample
            number_per_iteration -- number of items to sample per iteration
            limit -- sample from only this many items for faster sampling (-1 = no limit)       
        """ 
        
        if(len(unlabeled_data) < number):
            raise Exception('More samples requested than the number of unlabeled items')
                
        atlas_samples = [] # all items sampled by atlas
                
        print(number)
        while(len(atlas_samples) < number):
            samples = self.get_deep_active_transfer_learning_uncertainty_samples(model, unlabeled_data, validation_data, 
                                                        feature_method, number_per_iteration, limit, epochs, select_per_epoch)
            for item in samples:
                atlas_samples.append(item)
                unlabeled_data.remove(item)

                item = copy.deepcopy(item)
                item[3] = "seen" # mark this item as already seen
                
                validation_data.append(item) # append so that it is in the next iteration
            
        print("DONE!")    
        return atlas_samples        
                  
  
    
class SimpleUncertaintyPredictor(nn.Module):  # inherit pytorch's nn.Module
    """Simple model to predict whether an item will be classified correctly    

    """
    
    def __init__(self, vocab_size):
        super(SimpleUncertaintyPredictor, self).__init__() # call parent init
    
        # Define single layer model predicting 2 classes
        self.linear = nn.Linear(vocab_size, 2)
    
    def forward(self, feature_vec, return_all_layers=False):
        # Define how data is passed through the model and what gets returned
    
        output = self.linear(feature_vec).clamp(min=-1) # ReLU
        log_softmax = F.log_softmax(output, dim=1)
    
        if return_all_layers:
            return [output, log_softmax]
        else:
            return log_softmax
            

class AdvancedUncertaintyPredictor(nn.Module):  # inherit pytorch's nn.Module
    """Simple model to predict whether an item will be classified correctly    

    """
    
    def __init__(self, vocab_size):
        super(AdvancedUncertaintyPredictor, self).__init__() # call parent init
    
        # Define model with one hidden layer with 128 neurons
        self.linear1 = nn.Linear(vocab_size, 128)
        self.linear2 = nn.Linear(128, num_labels)

    def forward(self, feature_vec, return_all_layers=False):
        # Define how data is passed through the model and what gets returned

        hidden1 = self.linear1(feature_vec).clamp(min=0) # ReLU
        output = self.linear2(hidden1)
        log_softmax = F.log_softmax(output, dim=1)

        if return_all_layers:
            return [hidden1, output, log_softmax]
        else:
            return log_softmax
            

