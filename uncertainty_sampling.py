#!/usr/bin/env python

"""UNCERTAINTY SAMPLING
 
Uncertainty Sampling examples for Active Learning in PyTorch 

This is an open source example to accompany Chapter 3 from the book:
"Human-in-the-Loop Machine Learning"

It contains four Active Learning strategies:
1. Least Confidence Sampling
2. Margin of Confidence Sampling
3. Ratio of Confidence Sampling
4. Entropy-based Sampling


"""

import torch 
import math
from random import shuffle


__author__ = "Robert Munro"
__license__ = "MIT"
__version__ = "1.0.1"

   

class UncertaintySampling():
    """Active Learning methods to sample for uncertainty
    
    
    """
    
    def __init__(self, verbose=False):
        self.verbose = verbose
    

    def least_confidence(self, prob_dist, sorted=False):
        """ 
        Returns the uncertainty score of an array using
        least confidence sampling in a 0-1 range where 1 is the most uncertain
        
        Assumes probability distribution is a pytorch tensor, like: 
            tensor([0.0321, 0.6439, 0.0871, 0.2369])
                    
        Keyword arguments:
            prob_dist -- a pytorch tensor of real numbers between 0 and 1 that total to 1.0
            sorted -- if the probability distribution is pre-sorted from largest to smallest
        """
        if sorted:
            simple_least_conf = prob_dist.data[0] # most confident prediction
        else:
            simple_least_conf = torch.max(prob_dist) # most confident prediction
                    
        num_labels = prob_dist.numel() # number of labels
         
        normalized_least_conf = (1 - simple_least_conf) * (num_labels / (num_labels -1))
        
        return normalized_least_conf.item()
    
    
    def margin_confidence(self, prob_dist, sorted=False):
        """ 
        Returns the uncertainty score of a probability distribution using
        margin of confidence sampling in a 0-1 range where 1 is the most uncertain
        
        Assumes probability distribution is a pytorch tensor, like: 
            tensor([0.0321, 0.6439, 0.0871, 0.2369])
            
        Keyword arguments:
            prob_dist -- a pytorch tensor of real numbers between 0 and 1 that total to 1.0
            sorted -- if the probability distribution is pre-sorted from largest to smallest
        """
        if not sorted:
            prob_dist, _ = torch.sort(prob_dist, descending=True) # sort probs so largest is first
        
        difference = (prob_dist.data[0] - prob_dist.data[1]) # difference between top two props
        margin_conf = 1 - difference 
        
        return margin_conf.item()
        
    
    def ratio_confidence(self, prob_dist, sorted=False):
        """ 
        Returns the uncertainty score of a probability distribution using
        ratio of confidence sampling in a 0-1 range where 1 is the most uncertain
        
        Assumes probability distribution is a pytorch tensor, like: 
            tensor([0.0321, 0.6439, 0.0871, 0.2369])
                    
        Keyword arguments:
            prob_dist --  pytorch tensor of real numbers between 0 and 1 that total to 1.0
            sorted -- if the probability distribution is pre-sorted from largest to smallest
        """
        if not sorted:
            prob_dist, _ = torch.sort(prob_dist, descending=True) # sort probs so largest is first        
            
        ratio_conf = prob_dist.data[1] / prob_dist.data[0] # ratio between top two props
        
        return ratio_conf.item()
    
    
    def entropy_based(self, prob_dist):
        """ 
        Returns the uncertainty score of a probability distribution using
        entropy 
        
        Assumes probability distribution is a pytorch tensor, like: 
            tensor([0.0321, 0.6439, 0.0871, 0.2369])
                    
        Keyword arguments:
            prob_dist -- a pytorch tensor of real numbers between 0 and 1 that total to 1.0
            sorted -- if the probability distribution is pre-sorted from largest to smallest
        """
        log_probs = prob_dist * torch.log2(prob_dist) # multiply each probability by its base 2 log
        raw_entropy = 0 - torch.sum(log_probs)
    
        normalized_entropy = raw_entropy / math.log2(prob_dist.numel())
        
        return normalized_entropy.item()
        
 
   
    def softmax(self, scores, base=math.e):
        """Returns softmax array for array of scores
        
        Converts a set of raw scores from a model (logits) into a 
        probability distribution via softmax.
            
        The probability distribution will be a set of real numbers
        such that each is in the range 0-1.0 and the sum is 1.0.
    
        Assumes input is a pytorch tensor: tensor([1.0, 4.0, 2.0, 3.0])
            
        Keyword arguments:
            prediction -- a pytorch tensor of any positive/negative real numbers.
            base -- the base for the exponential (default e)
        """
        exps = (base**scores.to(dtype=torch.float)) # exponential for each value in array
        sum_exps = torch.sum(exps) # sum of all exponentials

        prob_dist = exps / sum_exps # normalize exponentials 
        return prob_dist
        
   
        
        
    def get_samples(self, model, unlabeled_data, method, feature_method, number=5, limit=10000):
        """Get samples via the given uncertainty sampling method from unlabeled data 
    
        Keyword arguments:
            model -- current Machine Learning model for this task
            unlabeled_data -- data that does not yet have a label
            method -- method for uncertainty sampling (eg: least_confidence())
            feature_method -- the method for extracting features from your data
            number -- number of items to sample
            limit -- sample from only this many predictions for faster sampling (-1 = no limit)
    
        Returns the number most uncertain items according to least confidence sampling
    
        """
    
        samples = []
    
        if limit == -1 and len(unlabeled_data) > 10000 and self.verbose: # we're drawing from *a lot* of data this will take a while                                               
            print("Get predictions for a large amount of unlabeled data: this might take a while")
        else:
            # only apply the model to a limited number of items                                                                            
            shuffle(unlabeled_data)
            unlabeled_data = unlabeled_data[:limit]
    
        with torch.no_grad():
            v=0
            for item in unlabeled_data:
                text = item[1]
                
                feature_vector = feature_method(text)
                hidden, logits, log_probs = model(feature_vector, return_all_layers=True)  
    
                prob_dist = torch.exp(log_probs) # the probability distribution of our prediction
                
                score = method(prob_dist.data[0]) # get the specific type of uncertainty sampling
                
                item[3] = method.__name__ # the type of uncertainty sampling used 
                item[4] = score
                
                samples.append(item)
                
                
        samples.sort(reverse=True, key=lambda x: x[4])       
        return samples[:number:]        
        
    

        
