#!/usr/bin/env python

"""ACTIVE LEARNING
 
This is an open source example to accompany Chapters 3 and 4 from the book:
"Human-in-the-Loop Machine Learning"

This example tries to classify news headlines into one of two categories:
  disaster-related
  not disaster-related

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

from diversity_sampling import DiversitySampling
from uncertainty_sampling import UncertaintySampling
from advanced_active_learning import AdvancedActiveLearning
from pytorch_clusters import CosineClusters 
from pytorch_clusters import Cluster


__author__ = "Robert Munro"
__license__ = "MIT"
__version__ = "1.0.1"

   
# settings
minimum_evaluation_items = 1200 # annotate this many randomly sampled items first for evaluation data before creating training data
minimum_validation_items = 200 # annotate this many randomly sampled items first for validation data before creating training data
minimum_training_items = 100 # minimum number of training items before we first train a model

epochs = 20 # default number of epochs per training session
select_per_epoch = 200  # number to select per epoch per label


data = []
test_data = []

# directories with data
unlabeled_data = "unlabeled_data/unlabeled_data.csv"

evaluation_related_data = "evaluation_data/related.csv"
evaluation_not_related_data = "evaluation_data/not_related.csv"

validation_related_data  = "validation_data/related.csv" 
validation_not_related_data = "validation_data/not_related.csv" 

training_related_data = "training_data/related.csv"
training_not_related_data = "training_data/not_related.csv"

# default number to sample for each method
number_random = 5

number_least_confidence = 0
number_margin_confidence = 0
number_ratio_confidence = 0
number_entropy_based = 0

number_model_outliers = 0
number_cluster_based = 0
number_representative = 0
number_adaptive_representative = 0

number_representative_clusters = 0
number_clustered_uncertainty = 0
number_uncertain_model_outliers = 0
number_high_uncertainty_cluster = 0
number_transfer_learned_uncertainty = 0
number_atlas = 0

verbose = False

cli_args = sys.argv
arg_list = cli_args[1:]

# default option, random:
gnu_options = ["random_remaining="]
# uncertainty sampling
gnu_options += ["least_confidence=", "margin_confidence=", "ratio_confidence=","entropy_based="]
# diversity sampling
gnu_options += ["model_outliers=", "cluster_based=","representative=","adaptive_representative="]
# advanced active learning
gnu_options += ["representative_clusters=", "clustered_uncertainty=", "uncertain_model_outliers="]
gnu_options += ["high_uncertainty_cluster=", "transfer_learned_uncertainty="]
gnu_options += ["atlas="]
# options
gnu_options += ["help", "verbose"]

try:
    arguments, values = getopt.getopt(arg_list, "", gnu_options)
except getopt.error as err:
    # output error, and return with an error code
    print (str(err))
    sys.exit(2)

for arg, value in arguments:
    if arg == "--random_remaining":
        number_random = int(value)
    if arg == "--model_outliers":
        number_model_outliers = int(value)
    if arg == "--cluster_based":
        number_cluster_based = int(value)
    if arg == "--representative":
        number_representative = int(value)
    if arg == "--adaptive_representative":
        number_adaptive_representative = int(value)
    if arg == "--least_confidence":
        number_least_confidence = int(value)
    if arg == "--margin_confidence":
        number_margin_confidence = int(value)
    if arg == "--ratio_confidence":
        number_ratio_confidence = int(value)
    if arg == "--entropy_based":
        number_entropy_based = int(value)
    if arg == "--representative_clusters":
        number_representative_clusters = int(value)
    if arg == "--clustered_uncertainty":
        number_clustered_uncertainty = int(value)
    if arg == "--uncertain_model_outliers":
        number_uncertain_model_outliers = int(value)
    if arg == "--high_uncertainty_cluster":
        number_high_uncertainty_cluster = int(value)
    if arg == "--transfer_learned_uncertainty":
        number_transfer_learned_uncertainty = int(value)
    if arg == "--atlas":
        number_atlas = int(value)
    if arg == "--verbose":
        verbose = True
    if arg == "--help":
        print("\nValid options for Active Learning sampling: ")
        for option in gnu_options:
            print("\t"+option)
        print("\nFor example `model_outliers=100` will sample 100 unlabeled items through model-based outlier sampling.\n")

        exit()
    


already_labeled = {} # tracking what is already labeled
feature_index = {} # feature mapping for one-hot encoding


def load_data(filepath, skip_already_labeled=False):
    # csv format: [ID, TEXT, LABEL, SAMPLING_STRATEGY, CONFIDENCE]
    with open(filepath, 'r') as csvfile:
        data = []
        reader = csv.reader(csvfile)
        for row in reader:
            if skip_already_labeled and row[0] in already_labeled:
        	    continue
        		
            if len(row) < 3:
                row.append("") # add empty col for LABEL to add later
            if len(row) < 4:
                row.append("") # add empty col for SAMPLING_STRATEGY to add later        
            if len(row) < 5:
                row.append(0) # add empty col for CONFIDENCE to add later         
            data.append(row)

            label = str(row[2])
            if row[2] != "":
                textid = row[0]
                already_labeled[textid] = label

    csvfile.close()
    return data

def append_data(filepath, data):
    with open(filepath, 'a', errors='replace') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    csvfile.close()

def write_data(filepath, data):
    with open(filepath, 'w', errors='replace') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    csvfile.close()


# LOAD ALL UNLABELED, TRAINING, VALIDATION, AND EVALUATION DATA
training_data = load_data(training_related_data) + load_data(training_not_related_data)
training_count = len(training_data)

validation_data = load_data(validation_related_data) + load_data(validation_not_related_data)
validation_count = len(validation_data)
    
evaluation_data = load_data(evaluation_related_data) + load_data(evaluation_not_related_data)
evaluation_count = len(evaluation_data)

data = load_data(unlabeled_data, skip_already_labeled=True)

annotation_instructions = "Please type 1 if this message is disaster-related, "
annotation_instructions += "or hit Enter if not.\n"
annotation_instructions += "Type 2 to go back to the last message, "
annotation_instructions += "type d to see detailed definitions, "
annotation_instructions += "or type s to save your annotations.\n"

last_instruction = "All done!\n"
last_instruction += "Type 2 to go back to change any labels,\n"
last_instruction += "or Enter to save your annotations."

detailed_instructions = "A 'disaster-related' headline is any story about a disaster.\n"
detailed_instructions += "It includes:\n"
detailed_instructions += "  - human, animal and plant disasters.\n"
detailed_instructions += "  - the response to disasters (aid).\n"
detailed_instructions += "  - natural disasters and man-made ones like wars.\n"
detailed_instructions += "It does not include:\n"
detailed_instructions += "  - criminal acts and non-disaster-related police work\n"
detailed_instructions += "  - post-response activity like disaster-related memorials.\n\n"


def get_annotations(data, default_sampling_strategy="random"):
    """Prompts annotator for label from command line and adds annotations to data 
    
    Keyword arguments:
        data -- an list of unlabeled items where each item is 
                [ID, TEXT, LABEL, SAMPLING_STRATEGY, CONFIDENCE]
        default_sampling_strategy -- strategy to use for each item if not already specified
    """

    ind = 0
    while ind <= len(data):
        if ind < 0:
            ind = 0 # in case you've gone back before the first
        if ind < len(data):
            textid = data[ind][0]
            text = data[ind][1]
            label = data[ind][2]
            strategy =  data[ind][3]
            score = data[ind][4]
            
            if strategy == "":
                strategy = "random"

            if textid in already_labeled:
                if verbose:
                    print("Skipping seen "+str(textid)+" with label "+label)
                    print(data[ind])
                ind+=1
            else:
                print(annotation_instructions)
                if verbose:
                    print("Sampled with strategy `"+str(strategy)+"` and score "+str(round(score,3)))
                label = str(input(text+"\n\n> ")) 

                if label == "2":                   
                    ind-=1  # go back
                elif label == "d":                    
                    print(detailed_instructions) # print detailed instructions
                elif label == "s":
                    break  # save and exit
                else:
                    if not label == "1":
                        label = "0" # treat everything other than 1 as 0
                        
                    data[ind][2] = label # add label to our data

                    if data[ind][3] is None or data[ind][3] == "":
                        data[ind][3] = default_sampling_strategy # add default if none given
                    ind+=1        

        else:
            #last one - give annotator a chance to go back
            print(last_instruction)
            label = str(input("\n\n> ")) 
            if label == "2":
                ind-=1
            else:
                ind+=1

    return data


def create_features(minword = 3):
    """Create indexes for one-hot encoding of words in files
    
    """

    total_training_words = {}
    for item in data + training_data:
        text = item[1]
        for word in text.split():
            if word not in total_training_words:
                total_training_words[word] = 1
            else:
                total_training_words[word] += 1

    for item in data + training_data:
        text = item[1]
        for word in text.split():
            if word not in feature_index and total_training_words[word] >= minword:
                feature_index[word] = len(feature_index)

    return len(feature_index)


class SimpleTextClassifier(nn.Module):  # inherit pytorch's nn.Module
    """Text Classifier with 1 hidden layer 

    """
    
    def __init__(self, num_labels, vocab_size):
        super(SimpleTextClassifier, self).__init__() # call parent init

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
                                

def make_feature_vector(text):
    features = text.split()
    vec = torch.zeros(len(feature_index))
    for feature in features:
        if feature in feature_index:
            vec[feature_index[feature]] += 1
    return vec.view(1, -1)


def train_model(training_data, validation_data = "", evaluation_data = "", num_labels=2, vocab_size=0):
    """Train model on the given training_data

    Tune with the validation_data
    Evaluate accuracy with the evaluation_data
    """

    model = SimpleTextClassifier(num_labels, vocab_size)
    # let's hard-code our labels for this example code 
    # and map to the same meaningful booleans in our data, 
    # so we don't mix anything up when inspecting our data
    label_to_ix = {"not_disaster_related": 0, "disaster_related": 1} 

    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # epochs training
    for epoch in range(epochs):
        if verbose:
            print("Epoch: "+str(epoch))
        current = 0

        # make a subset of data to use in this epoch
        # with an equal number of items from each label

        shuffle(training_data) #randomize the order of the training data        
        related = [row for row in training_data if '1' in row[2]]
        not_related = [row for row in training_data if '0' in row[2]]
        
        epoch_data = related[:select_per_epoch]
        epoch_data += not_related[:select_per_epoch]
        shuffle(epoch_data) 
                
        # train our model
        for item in epoch_data:
            text = item[1]
            label = int(item[2])

            model.zero_grad() 

            feature_vec = make_feature_vector(text)
            target = torch.LongTensor([int(label)])

            log_probs = model(feature_vec)

			# compute loss function, do backward pass, and update the gradient
            loss = loss_function(log_probs, target)
            loss.backward()
            optimizer.step()	

    fscore, auc = evaluate_model(model, evaluation_data)
    fscore = round(fscore,3)
    auc = round(auc,3)

    # save model to path that is alphanumeric and includes number of items and accuracies in filename
    timestamp = re.sub('\.[0-9]*','_',str(datetime.datetime.now())).replace(" ", "_").replace("-", "").replace(":","")
    training_size = "_"+str(len(training_data))
    accuracies = str(fscore)+"_"+str(auc)
                     
    model_path = "models/"+timestamp+accuracies+training_size+".params"

    torch.save(model.state_dict(), model_path)
    return model_path



def get_random_items(unlabeled_data, number = 10):
    shuffle(unlabeled_data)

    random_items = []
    for item in unlabeled_data:
        textid = item[0]
        if textid in already_labeled:
            continue
        item[3] = "random_remaining"
        random_items.append(item)
        if len(random_items) >= number:
            break

    return random_items

        

def evaluate_model(model, evaluation_data):
    """Evaluate the model on the held-out evaluation data

    Return the f-value for disaster-related and the AUC
    """

    related_confs = [] # related items and their confidence of being related
    not_related_confs = [] # not related items and their confidence of being _related_

    true_pos = 0.0 # true positives, etc 
    false_pos = 0.0
    false_neg = 0.0

    with torch.no_grad():
        for item in evaluation_data:
            _, text, label, _, _, = item

            feature_vector = make_feature_vector(text)
            log_probs = model(feature_vector)

            # get confidence that item is disaster-related
            prob_related = math.exp(log_probs.data.tolist()[0][1]) 

            if(label == "1"):
                # true label is disaster related
                related_confs.append(prob_related)
                if prob_related > 0.5:
                    true_pos += 1.0
                else:
                    false_neg += 1.0
            else:
                # not disaster-related
                not_related_confs.append(prob_related)
                if prob_related > 0.5:
                    false_pos += 1.0

    # Get FScore
    if true_pos == 0.0:
        fscore = 0.0
    else:
        precision = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)
        fscore = (2 * precision * recall) / (precision + recall)

    # GET AUC
    not_related_confs.sort()
    total_greater = 0 # count of how many total have higher confidence
    for conf in related_confs:
        for conf2 in not_related_confs:
            if conf < conf2:
                break
            else:                  
                total_greater += 1


    denom = len(not_related_confs) * len(related_confs) 
    auc = total_greater / denom

    return[fscore, auc]



if evaluation_count <  minimum_evaluation_items:
    #Keep adding to evaluation data first
    print("Creating evaluation data:\n")

    shuffle(data)
    needed = minimum_evaluation_items - evaluation_count
    data = data[:needed]
    print(str(needed)+" more annotations needed")

    data = get_annotations(data) 
	
    related = []
    not_related = []

    for item in data:
        label = item[2]    
        if label == "1":
            related.append(item)
        elif label == "0":
            not_related.append(item)

    # append evaluation data
    append_data(evaluation_related_data, related)
    append_data(evaluation_not_related_data, not_related)

if validation_count <  minimum_validation_items:
    #Keep adding to evaluation data first
    print("Creating validation data:\n")

    shuffle(data)
    needed = minimum_validation_items - validation_count
    data = data[:needed]
    print(str(needed)+" more annotations needed")

    data = get_annotations(data) 
    
    related = []
    not_related = []

    for item in data:
        label = item[2]    
        if label == "1":
            related.append(item)
        elif label == "0":
            not_related.append(item)

    # append validation data
    append_data(validation_related_data, related)
    append_data(validation_not_related_data, not_related)



elif training_count < minimum_training_items:
    # lets create our first training data! 
    print("Creating initial training data:\n")

    shuffle(data)
    needed = minimum_training_items - training_count
    data = data[:needed]
    print(str(needed)+" more annotations needed")

    data = get_annotations(data)

    related = []
    not_related = []

    for item in data:
        label = item[2]
        if label == "1":
            related.append(item)
        elif label == "0":
            not_related.append(item)

    # append training data
    append_data(training_related_data, related)
    append_data(training_not_related_data, not_related)
else:
    # lets start Active Learning!! 
    sampled_data = []

        
    # GET RANDOM SAMPLES
    if number_random > 0:
        print("Sampling "+str(number_random)+" Random Remaining Items\n")
        sampled_data += get_random_items(data, number=number_random)


    # RETRAIN WHOLE MODEL IF WE NEED IT FOR ANY METHOD:
    if (number_least_confidence + number_margin_confidence + number_ratio_confidence +                                 
                number_entropy_based + number_clustered_uncertainty + number_uncertain_model_outliers +
                number_high_uncertainty_cluster > 0):
        print("Retraining model for Uncertainty Sampling \n")    

        vocab_size = create_features()
        model_path = train_model(training_data, evaluation_data=evaluation_data, vocab_size=vocab_size)
        model = SimpleTextClassifier(2, vocab_size)
        model.load_state_dict(torch.load(model_path)) 

    # RETRAIN MODEL WITH TRAIN/VALIDATION SPLIT IF WE NEED IT FOR ANY METHOD:
    if number_model_outliers + number_uncertain_model_outliers + number_transfer_learned_uncertainty + number_atlas > 0:
        print("Retraining model for Model-based Outliers or Deep Active Transfer Learning \n")    

        # Need to split our training data to make a leave-out validation set:            
        new_training_data = training_data[:int(len(training_data)*0.9)] 
        new_validation_data = training_data[len(new_training_data):] 

        vocab_size = create_features()
        model_path = train_model(new_training_data, evaluation_data=evaluation_data, vocab_size=vocab_size)
        validation_model = SimpleTextClassifier(2, vocab_size)
        validation_model.load_state_dict(torch.load(model_path))


    uncert_sampling = UncertaintySampling(verbose)    
    diversity_samp = DiversitySampling(verbose)      
    adv_samping = AdvancedActiveLearning(verbose)   


    if number_cluster_based + number_representative + number_adaptive_representative + number_model_outliers > 0:
        print("Sampling for Diversity")
        
        # MODEL-BASED OUTLIER SAMPLES
        if number_model_outliers > 0:
            print("Sampling "+str(number_model_outliers)+" Model Outliers\n")
        
            sampled_data += diversity_samp.get_model_outliers(validation_model, data, new_validation_data, 
                                                              make_feature_vector, number=number_model_outliers)


        # CLUSTER-BASED SAMPLES
        if number_cluster_based > 0:
            print("Sampling "+str(number_cluster_based)+" via Clustering")
            num_clusters = math.ceil(number_cluster_based / 5) # sampling 5 items per cluster by default
            print("Creating "+str(num_clusters)+" Clusters")
    
            if num_clusters * 5 > number_cluster_based:
                print("Adjusting sample to "+str(num_clusters * 5)+" to get an equal number per sample\n")
            
            sampled_data += diversity_samp.get_cluster_samples(data, num_clusters=num_clusters)
        
    
        # REPRESENTATIVE SAMPLES
        if number_representative > 0:   
            print("Sampling "+str(number_representative)+" via Representative Sampling\n")
            sampled_data += diversity_samp.get_representative_samples(training_data, data, number=number_representative)
    
    
        # REPRESENTATIVE SAMPLES USING ADAPTIVE SAMPLING
        if number_adaptive_representative > 0:
            print("Sampling "+str(number_adaptive_representative)+" via Adaptive Representative Sampling\n")    
            sampled_data += diversity_samp.get_adaptive_representative_samples(training_data, data, 
                                                                               number=number_adaptive_representative)
  
        
    if number_least_confidence + number_margin_confidence + number_ratio_confidence + number_entropy_based > 0:   
  
        # LEAST CONFIDENCE SAMPLES
        if number_least_confidence > 0:
            print("Sampling "+str(number_least_confidence)+" via Least Confidence Sampling\n")    
    
            sampled_data += uncert_sampling.get_samples(model, data, uncert_sampling.least_confidence, 
                                                        make_feature_vector, number=number_least_confidence)

        # MARGIN OF CONFIDENCE SAMPLES
        if number_margin_confidence > 0:
            print("Sampling "+str(number_margin_confidence)+" via Margin of Confidence Sampling\n")    
    
            # margin_confidence_samples = get_margin_confidence_samples(model, data, number=number_margin_confidence)
            sampled_data += uncert_sampling.get_samples(model, data, uncert_sampling.margin_confidence, 
                                                        make_feature_vector, number=number_margin_confidence)

        # RATIO OF CONFIDENCE SAMPLES
        if number_ratio_confidence > 0:
            print("Sampling "+str(number_ratio_confidence)+" via Ratio of Confidence Sampling\n")    
    
            # ratio_confidence_samples = get_ratio_confidence_samples(model, data, number=number_ratio_confidence)
            sampled_data += uncert_sampling.get_samples(model, data, uncert_sampling.ratio_confidence, 
                                                        make_feature_vector, number=number_ratio_confidence)
            
        # ENTROPY-BASED SAMPLES
        if number_entropy_based > 0:
            print("Sampling "+str(number_entropy_based)+" via Entropy-based Sampling\n")    
    
            # entropy_based_samples = get_entropy_based_samples(model, data, number=number_entropy_based)
            sampled_data += uncert_sampling.get_samples(model, data, uncert_sampling.entropy_based, 
                                                        make_feature_vector, number=number_entropy_based)



    # ADVANCED TECHNIQUES
    
    # REPRESENTATIVE CLUSTERS
    if number_representative_clusters > 0:
        print("Sampling "+str(number_representative_clusters)+" via Representative Clusters\n")    
        
        sampled_data += adv_samping.get_representative_cluster_samples(training_data, data, 
                                                            number=number_representative_clusters)
                                                            
    # CLUSTERED UNCERTAINTY
    if number_clustered_uncertainty > 0:
        print("Sampling "+str(number_clustered_uncertainty)+" via Clustered Least Confidence\n")
        uncert_sampling = UncertaintySampling(verbose)
        
        sampled_data += adv_samping.get_clustered_uncertainty_samples(model, data, 
                                        uncert_sampling.least_confidence, make_feature_vector, 
                                        num_clusters=math.ceil(number_clustered_uncertainty/5))


    # UNCERTAIN MODEL OUTLIERS
    if number_uncertain_model_outliers > 0:
        print("Sampling "+str(number_uncertain_model_outliers)+" via Model-Outlier Least Confidence\n")
        
        
        sampled_data += adv_samping.get_uncertain_model_outlier_samples(model, validation_model, data,  
                        new_validation_data, uncert_sampling.least_confidence, make_feature_vector, 
                        number=number_uncertain_model_outliers)


    # HIGH UNCERTAINY CLUSTERS
    if number_high_uncertainty_cluster > 0:
        print("Sampling "+str(number_high_uncertainty_cluster)+" via highest entropy clusters\n")
        sampled_data += adv_samping.get_high_uncertainty_cluster(model, data, uncert_sampling.entropy_based, 
                                                make_feature_vector, number=number_high_uncertainty_cluster)

    # ACTIVE TRANSFER LEARNING FOR UNCERTAINTY
    if number_transfer_learned_uncertainty > 0:
        print("Sampling "+str(number_transfer_learned_uncertainty)+" via deep active transfer learning for uncertainty\n")       
        sampled_data += adv_samping.get_deep_active_transfer_learning_uncertainty_samples(validation_model, 
                                                data, new_validation_data, 
                                                make_feature_vector, number=number_transfer_learned_uncertainty)

    # ACTIVE TRANSFER LEARNING FOR ADAPTIVE SAMPLING
    if number_atlas > 0:
        print("Sampling "+str(number_atlas)+" via adaptive transfer learning for active samplng (ATLAS)\n")       
        sampled_data += adv_samping.get_atlas_samples(validation_model, 
                                                data, new_validation_data, 
                                                make_feature_vector, number=number_atlas)
                                                

   
    # GET ANNOTATIONS FROM OUR SAMPLES
    shuffle(sampled_data)
    
    sampled_data = get_annotations(sampled_data)
    
    related = []
    not_related = []
    for item in sampled_data:
        label = item[2]
        if label == "1":
            related.append(item)
        elif label == "0":
            not_related.append(item)
        
    # append training data files
    append_data(training_related_data, related)
    append_data(training_not_related_data, not_related)
    

if training_count > minimum_training_items:
    print("\nRetraining model with new data")
    
	# UPDATE OUR DATA AND (RE)TRAIN MODEL WITH NEWLY ANNOTATED DATA
    training_data = load_data(training_related_data) + load_data(training_not_related_data)
    training_count = len(training_data)

    evaluation_data = load_data(evaluation_related_data) + load_data(evaluation_not_related_data)
    evaluation_count = len(evaluation_data)

    vocab_size = create_features()
    model_path = train_model(training_data, evaluation_data=evaluation_data, vocab_size=vocab_size)
    model = SimpleTextClassifier(2, vocab_size)
    model.load_state_dict(torch.load(model_path))

    accuracies = evaluate_model(model, evaluation_data)
    print("[fscore, auc] =")
    print(accuracies)
    print("Model saved to: "+model_path)
    

