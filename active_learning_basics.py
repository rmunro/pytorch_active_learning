#!/usr/bin/env python

"""INTRODUCTION TO ACTIVE LEARNING

A simple text classification algorithm in PyTorch 

This is an open source example to accompany the book:
"Human-in-the-Loop Machine Learning"

This example is trys to classify news headlines into one of two categories:
  disaster-related
  not disaster-related

It looks for low confidence items that it humans should review

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
from random import shuffle

__author__ = "Robert Munro"
__license__ = "MIT"
__version__ = "1.0.1"

# settings

minimum_evaluation_items = 600 # annotate this many randomly selected items first for evaluation data before creating training data
minimum_training_items = 400 # minimum number of training items before we first train a model

epochs = 20 # number of epochs per training session
samples = 1000  # number to sample per epoch per label


data = []
test_data = []

# directories with data
unlabeled_data = "unlabeled_data/unlabeled_data.csv"

evaluation_related_data = "evaluation_data/related.csv"
evaluation_not_related_data = "evaluation_data/not_related.csv"

#validation_related_data # not used in this example
#validation_not_related_data # not used in this example

training_related_data = "training_data/related.csv"
training_not_related_data = "training_data/not_related.csv"


already_labeled = {} # tracking what is already labeled
feature_index = {} # feature mapping for one-hot encoding


def load_data(filepath):
    # csv format: ID, TEXT, LABEL (if any), SAMPLING_STRATEGY (if any) 
    with open(filepath, 'r') as csvfile:
        data = []
        reader = csv.reader(csvfile)
        for row in reader:
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
data = load_data(unlabeled_data)

training_data = load_data(training_related_data) + load_data(training_not_related_data)
training_count = len(training_data)
    
evaluation_data = load_data(evaluation_related_data) + load_data(evaluation_not_related_data)
evaluation_count = len(evaluation_data)


def get_annotations(data, default_sampling_strategy="random"):
    """Prompts annotator for label from command line and add annotations to data 
    
    Keyword arguments:
        data -- an list of unlabeled items where each item is [ID, TEXT, LABEL, SAMPLING_STRATEGY, CONFIDENCE]
        default_sampling_strategy -- the default sampling strategy to use for each item if not already specified
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

            if textid in already_labeled:
                print("Skipping seen "+label)
                ind+=1
            else:
                print("Please type 1 if this message is disaster-related, or hit Enter if not.")
                print("Type 2 to go back to the last message, or type s to save annotations\n")
                label = str(input(text+"\n\n> "))  # NB: python 2.* will not accept an empty response but with otherwise work

                if label == "2":
                    # go back
                    ind-=1
                elif label == "s":
                    # save and exit
                    break
                else:
                    if not label == "1":
                        label = "0" # treat everything other than 1 as 0

                    data[ind][2] = label # add label to our data

                    if data[ind][3] is None or data[ind][3] == "":
                        data[ind][3] = default_sampling_strategy # add default sampling strategy if none given

                    ind+=1        

        else:
            #last one - give annotator a chance to go back
            print("All done! Type 2 to go back to the last message if you need to change the label,\nor Enter to save your annotations.")
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

    def __init__(self, num_labels, vocab_size):
        super(SimpleTextClassifier, self).__init__() # call parent init

        # Define model with one hidden layer with 128 neurons
        self.linear1 = nn.Linear(vocab_size, 128)
        self.linear2 = nn.Linear(128, num_labels)

    def forward(self, bow_vec):
        # Define how data is passed through the model

        hidden1 = self.linear1(bow_vec).clamp(min=0) # ReLU
        output = self.linear2(hidden1)
        return F.log_softmax(output, dim=1)
                                

def make_feature_vector(features, feature_index):
    vec = torch.zeros(len(feature_index))
    for feature in features:
        if feature in feature_index:
            vec[feature_index[feature]] += 1
    return vec.view(1, -1)


def train_model(training_data, validation_data = "", evaluation_data = "", vocab_size=0):
    """Train model on the given training_data

    Tune with the validation_data
    Evaluate accuracy with the evaluation_data
    """

    model = SimpleTextClassifier(2, vocab_size)
    # let's hard-code our labels for this example code and map to meaningful booleans, so we don't mix them up
    label_to_ix = {"not_disaster_related": 0, "disaster_related": 1} 

    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # epochs training
    for epoch in range(epochs):
        print("Epoch: "+str(epoch))
        current = 0

        # current subset of data to use in this epoch
        epoch_data = []

        not_related = []
        related = []
        shuffle(training_data) #randomize the order of the training data
        for item in training_data:
            label = item[2]
            if label == "0" and len(not_related) < samples:
                epoch_data.append(item)
            elif label == "1" and len(related) < samples:
                epoch_data.append(item)
            elif len(not_related) >= samples and len(related) >= samples:
                break        

        for item in epoch_data:
            training_idx = random.randint(0,len(data)-1)
            words = item[1].split()
            label = int(item[2])

            model.zero_grad() 

            feature_vec = make_feature_vector(words, feature_index)
            target = torch.LongTensor([int(label)])

            # torch.LongTensor([label_to_ix[label]]) 

            #make_target(label, label_to_ix)

            log_probs = model(feature_vec)

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


def get_low_conf_unlabeled(model, unlabeled_data, number=80):
    confidences = []
    print("Get confidences for unlabeled data (this might take a while)")
    with torch.no_grad():
        c=0
        for item in unlabeled_data:
            textid = item[0]
            if textid in already_labeled:
                continue

            text = item[1]

            feature_vector = make_feature_vector(text.split(), feature_index)
            log_probs = model(feature_vector)

            prob_related = math.exp(log_probs.data.tolist()[0][1]) # get confidence that it is related
            if prob_related < 0.5:
                confidence = 1 - prob_related
            else:
                confidence = prob_related 

            item[3] = "low confidence"
            item[4] = confidence
            confidences.append(item)
            c+=1
            if c > 10000:
                break

    confidences.sort(key=lambda x: x[4])
    return confidences[:number:]


def get_random_items(unlabeled_data, number = 10):
    shuffle(unlabeled_data)

    random_items = []
    for item in unlabeled_data:
        textid = item[0]
        if textid in already_labeled:
            continue
        random_items.append(item)
        if len(random_items) >= number:
            break

    return random_items
        

def get_outliers(training_data, unlabeled_data, number=10):
    """Get outliers from unlabeled data in training data
    return number outliers
    """
    outliers = []

    total_feature_counts = {}
    for item in training_data:
        text = item[1]
        features = text.split()

        for feature in features:
            if feature in total_feature_counts:
                total_feature_counts[feature] += 1
            else:
                total_feature_counts[feature] = 1
                
    while(len(outliers) < number):
        top_outlier = []
        top_match = float("inf")

        for item in unlabeled_data:
            textid = item[0]
            if textid in already_labeled:
                continue

            text = item[1]
            features = text.split()
            total_matches = 1 # start at 1 for slight smoothing to avoid only short sentences
            for feature in features:
                if feature in total_feature_counts:
                    total_matches += total_feature_counts[feature]

            ave_matches = total_matches / len(features)
            if ave_matches < top_match:
                top_match = ave_matches
                top_outlier = item

        # add this outlier to list and update what is 'labeled', assuming this new outlier will get a label
        top_outlier[3] = "outlier"
        outliers.append(top_outlier)
        text = top_outlier[1]
        features = text.split()
        for feature in features:
            if feature in total_feature_counts:
                total_feature_counts[feature] += 1
            else:
                total_feature_counts[feature] = 1

    return outliers
    


def evaluate_model(model, evaluation_data):
    """Evalute the model on the held-out evaluation data

    Return the f-value for disaster-related and the AUC
    """

    related_confs = [] # related items and their confidence of being related
    not_related_confs = [] # not related items and their confidence of being _related_

    tp = 0.0 # true positives, etc 
    fp = 0.0
    fn = 0.0

    with torch.no_grad():
        for item in evaluation_data:
            _, text, label, _, _, = item

            feature_vector = make_feature_vector(text.split(), feature_index)
            log_probs = model(feature_vector)

            prob_related = math.exp(log_probs.data.tolist()[0][1]) # get confidence that it is related

            if(label == "1"):
                # true label is disaster related
                related_confs.append(prob_related)
                if prob_related > 0.5:
                    tp+=1.0
                else:
                    fn += 1
            else:
                # not disaster-related
                not_related_confs.append(prob_related)
                if prob_related > 0.5:
                    fn+=1.0

    # Get FScore
    if tp == 0.0:
        fscore = 0.0
    else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        fscore = (2 * precision * recall) / (precision + recall)

    # GET AUC
    not_related_confs.sort()
    total_greater = 0 # count of how many total are greater
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

elif training_count < minimum_training_items:
    # lets create our first training data! 
    print("Creating initial training data:\n")

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
    print("Sampling via Active Learning:\n")

    # load most recent model
    models = os.listdir('./models')
    models.sort()
    model_path = "models/"+ models[-1]

    vocab_size = create_features()
    model = SimpleTextClassifier(2, vocab_size)
    model.load_state_dict(torch.load(model_path))

    random_items = get_random_items(data, number=10)
    low_confidences = get_low_conf_unlabeled(model, data, number=80)
    outliers = get_outliers(training_data+random_items+low_confidences, data, number=10)

    sampled_data = random_items + low_confidences + outliers
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

    # append training data
    append_data(training_related_data, related)
    append_data(training_not_related_data, not_related)


# UPDATE OUR DATA AND (RE)TRAIN MODEL WITH NEWLY ANNOTATED DATA
training_data = load_data(training_related_data) + load_data(training_not_related_data)
training_count = len(training_data)

evaluation_data = load_data(evaluation_related_data) + load_data(evaluation_not_related_data)
evaluation_count = len(evaluation_data)

if training_count > minimum_training_items:
    print("\nRetraining model with new data")

    vocab_size = create_features()
    model_path = train_model(training_data, evaluation_data=evaluation_data, vocab_size=vocab_size)
    model = SimpleTextClassifier(2, vocab_size)
    model.load_state_dict(torch.load(model_path))

    accuracies = evaluate_model(model, evaluation_data)
    print("[fscore, auc] =")
    print(accuracies)
    print("Model saved to: "+model_path)
    
