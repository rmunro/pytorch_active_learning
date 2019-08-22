#!/usr/bin/env python

"""Cosine distance kernal for KMeans-type clustering

This is an open source example to accompany Chapter 4 from the book:
"Human-in-the-Loop Machine Learning"

It is a general clustering library.

In this code-base, it supports three Active Learning strategies:
1. Cluster-based sampling
2. Representative sampling
3. Adaptive Representative sampling


"""
import torch
import torch.nn.functional as F
from random import shuffle


class CosineClusters():
    """Represents a set of clusters over a dataset
    
    
    """
    
    
    def __init__(self, num_clusters=100):
        
        self.clusters = [] # clusters for unsupervised and lightly supervised sampling
        self.item_cluster = {} # each item's cluster by the id of the item


        # Create initial clusters
        for i in range(0, num_clusters):
            self.clusters.append(Cluster())
        
        
    def add_random_training_items(self, items):
        """ Adds items randomly to clusters    
        """ 
        
        cur_index = 0
        for item in items:
            self.clusters[cur_index].add_to_cluster(item)
            textid = item[0]
            self.item_cluster[textid] = self.clusters[cur_index]
            
            cur_index += 1
            if cur_index >= len(self.clusters):
                cur_index = 0 


    def add_items_to_best_cluster(self, items):
        """ Adds multiple items to best clusters
        
        """
        added = 0
        for item in items:
            new = self.add_item_to_best_cluster(item)
            if new:
                added += 1
                
        return added



    def get_best_cluster(self, item):
        """ Finds the best cluster for this item
            
            returns the cluster and the score
        """
        best_cluster = None 
        best_fit = float("-inf")        
             
        for cluster in self.clusters:
            fit = cluster.cosine_similary(item)
            if fit > best_fit:
                best_fit = fit
                best_cluster = cluster 
        
        return [best_cluster, best_fit]
    
       

    def add_item_to_best_cluster(self, item):
        """ Adds items to best fit cluster    
            
            Removes from previous cluster if it existed in one
            Returns True if item is new or moved cluster
            Returns Fales if the item remains in the same cluster
        """ 
        
        best_cluster = None 
        best_fit = float("-inf")        
        previous_cluster = None
        
        # Remove from current cluster so it isn't contributing to own match
        textid = item[0]
        if textid in self.item_cluster:
            previous_cluster = self.item_cluster[textid]
            previous_cluster.remove_from_cluster(item)
            
        for cluster in self.clusters:
            fit = cluster.cosine_similary(item)
            if fit > best_fit:
                best_fit = fit
                best_cluster = cluster 
        
        best_cluster.add_to_cluster(item)
        self.item_cluster[textid] = best_cluster
        
        if best_cluster == previous_cluster:
            return False
        else:
            return True
 
 
    def get_items_cluster(self, item):  
        textid = item[0]
        
        if textid in self.item_cluster:
            return self.item_cluster[textid]
        else:
            return None      
        
        
    def get_centroids(self):  
        centroids = []
        for cluster in self.clusters:
            centroids.append(cluster.get_centroid())
        
        return centroids
    
        
    def get_outliers(self):  
        outliers = []
        for cluster in self.clusters:
            outliers.append(cluster.get_outlier())
        
        return outliers
 
         
    def get_randoms(self, number_per_cluster=1, verbose=False):  
        randoms = []
        for cluster in self.clusters:
            randoms += cluster.get_random_members(number_per_cluster, verbose)
        
        return randoms
   
      
    def shape(self):  
        lengths = []
        for cluster in self.clusters:
            lengths.append(cluster.size())
        
        return str(lengths)



class Cluster():
    """Represents on cluster for unsupervised or lightly supervised clustering
            
    """
    
    feature_idx = {} # the index of each feature as class variable to be constant 


    def __init__(self):
        self.members = {} # dict of items by item ids in this cluster
        self.feature_vector = [] # feature vector for this cluster
    
    def add_to_cluster(self, item):
        textid = item[0]
        text = item[1]
        
        self.members[textid] = item
                
        words = text.split()   
        for word in words:
            if word in self.feature_idx:
                while len(self.feature_vector) <= self.feature_idx[word]:
                    self.feature_vector.append(0)
                    
                self.feature_vector[self.feature_idx[word]] += 1
            else:
                # new feature that is not yet in any cluster                
                self.feature_idx[word] = len(self.feature_vector)
                self.feature_vector.append(1)
                
        
            
    def remove_from_cluster(self, item):
        """ Removes if exists in the cluster        
            
        """
        textid = item[0]
        text = item[1]
        
        exists = self.members.pop(textid, False)
        
        if exists:
            words = text.split()   
            for word in words:
                index = self.feature_idx[word]
                if index < len(self.feature_vector):
                    self.feature_vector[index] -= 1
        
        
    def cosine_similary(self, item):
        text = item[1]
        words = text.split()  
        
        vector = [0] * len(self.feature_vector)
        for word in words:
            if word not in self.feature_idx:
                self.feature_idx[word] = len(self.feature_vector)
                self.feature_vector.append(0)
                vector.append(1)
            else:
                while len(vector) <= self.feature_idx[word]:
                    vector.append(0)
                    self.feature_vector.append(0)
                              
                vector[self.feature_idx[word]] += 1
        
        item_tensor = torch.FloatTensor(vector)
        cluster_tensor = torch.FloatTensor(self.feature_vector)
        
        similarity = F.cosine_similarity(item_tensor, cluster_tensor, 0)
        
        # Alternatively using `F.pairwise_distance()` but normalize the cluster first
        
        return similarity.item() # item() converts tensor value to float
    
    
    def size(self):
        return len(self.members.keys())
 
 
  
    def get_centroid(self):
        if len(self.members) == 0:
            return []
        
        best_item = None
        best_fit = float("-inf")
        
        for textid in self.members.keys():
            item = self.members[textid]
            similarity = self.cosine_similary(item)
            if similarity > best_fit:
                best_fit = similarity
                best_item = item
                
        best_item[3] = "cluster_centroid"
        best_item[4] = best_fit 
                
        return best_item
     
         

    def get_outlier(self):
        if len(self.members) == 0:
            return []
        
        best_item = None
        biggest_outlier = float("inf")
        
        for textid in self.members.keys():
            item = self.members[textid]
            similarity = self.cosine_similary(item)
            if similarity < biggest_outlier:
                biggest_outlier = similarity
                best_item = item

        best_item[3] = "cluster_outlier"
        best_item[4] = 1 - biggest_outlier
                
        return best_item



    def get_random_members(self, number=1, verbose=False):
        if len(self.members) == 0:
            return []        
        
        keys = list(self.members.keys())
        shuffle(keys)

        randoms = []
        for i in range(0, number):
            if i < len(keys):
                textid = keys[i] 
                item = self.members[textid]
                item[3] = "cluster_member"
                item[4] = self.cosine_similary(item)

                randoms.append(item)
         
        if verbose:
            print("\nRandomly items selected from cluster:")
            for item in randoms:
                print("\t"+item[1])         
                
        return randoms
    




         
