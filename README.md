# PyTorch Active Learning

Library for common Active Learning methods to accompany:  
Human-in-the-Loop Machine Learning  
Robert Munro  
Manning Publications
https://www.manning.com/books/human-in-the-loop-machine-learning  

The code is stand-alone and can be used with the book.

# Active Learning methods in the library

The code currently contains methods for:

*Least Confidence sampling* 

*Margin of Confidence sampling*

*Ratio of Confidence sampling*

*Entropy (classification entropy)*

*Model-based outlier sampling*

*Cluster-based sampling*

*Representative sampling* 

*Adaptive Representative sampling*

*Active Transfer Learning for Uncertainty Sampling*

*Active Transfer Learning for Representative Sampling*

*Active Transfer Learning for Adaptive Sampling (ATLAS)*


The book covers how to apply them indepedently, in combination, and for different use cases in Computer Vision and Natural Language Processing. It also covers strategies for sampling for real-world diversity to avoid bias.

## Installation: 

If you clone this repo and already have PyTorch installed, you should be able to get going immediately:

`git clone https://github.com/rmunro/pytorch_active_learning`

`cd pytorch_active_learning`  

### Running Chapter 2, Getting Started with Human-in-the-Loop Machine Learning

`python active_learning_basics.py`

When you run the software, you will be prompted to classify news headlines as being disaster-related or not. The prompt will also tell you give you the option to see a precise definitions for what constitutes "disaster-related". You can also read those definitions in the code in the `detailed_instructions` variable: https://github.com/rmunro/pytorch_active_learning/blob/master/active_learning_basics.py

After you have classified (annotated) enough data for evaluation and to begin training, you will see that machine learning models now train after each iteration of annotation, reporting the accuracy on your held-out evaluation data as F-Scores and AUC. 

After the initial iteration of training, which will just be on randomly-chosen data, you will start to see Active Learning kick-in to find unlabeled items that the model is confused about or are outliers with novel features. The Active Learning will be evident in the annotations, too, as the disaster-related headlines will be very rare initially, but should become around 40% of the data that you are annotating after a few iterations.


### Running Chapter 4, Diversity Sampling

`python diversity_sampling.py`

This builds on the earlier dataset. See the chapter for the details of the feature flags that allow you to sample using different types of Diversity Sampling, like Model-based Outliers, Clustering, and Representative Sampling.


## Requirements: 
The code assumes that you are using python3.6 or later. 

If you really need to get this working on python2.\*, please let me know: the PyTorch and Active Learning algorithms _should_ all be 2.\* compliant and it is only python's methods for getting command-line inputs that will need to be changed (python2.\* expects integrer inputs only). If enough people request it, then I'll try to update the code to be compatible for earlier versions of python! 

## Installing PyTorch:

### AWS
I recommend using the Deep Learning AMI on AWS, because PyTorch is already installed and can be activated with:  
`source activate pytorch_p36`  
That should be all you need to run the program immediately.

For more details on using PyTorch on AWS, see:  
https://docs.aws.amazon.com/dlami/latest/devguide/tutorial-pytorch.html

### Google Cloud
I recommend using a PyTorch image for a Deep Learning virtual machine on Google Cloud, because PyTorch is already installed. Both the CPU and GPU should work:
`pytorch-latest-cpu`

`pytorch-latest-gpu`

For more details on using PyTorch on Google Cloud, see:  
https://cloud.google.com/deep-learning-vm/docs/images

### Microsoft Azure
I recommend using a Data Science pre-configured virtual machine on Microsoft Azure:  
https://azure.microsoft.com/en-us/develop/pytorch/
The Azure Notebook option might also be a good option, but I haven't tested it out: please let me know if you do! 

### Linux / Mac / Windows
If you're installing locally or on a cloud server without PyTorch pre-installed, you can use these options:  

Mac:  
`conda install pytorch torchvision -c pytorch`

Linux/Windows:  
 `conda install pytorch torchvision cudatoolkit=9.0 -c pytorch`
 
These local instructions are current as of June 2019. PyTorch are great about maintaining quickstart instructions, so I recommend going there if these commands don't work for you for some reason. See "QUICK START LOCALLY" at:  
https://pytorch.org/

Mac users should also make sure they are using python3.6 or later, as Mac's still ship with python2.7 by default. See above re support for 2.7 if you really require it.

For pip users, it is possible that you can install pytorch with the following commands:
 `pip3 install torch`
or
 `pip3 install torch`
However, this sometimes works and sometimes doesn't depending on the versions of various libraries and your exact operating system. That's why `conda` is recommended over `pip` on the pytorch website.



## Data Sources

Currently, the data used is from the "Million News Headlines" dataset posted on Kaggle:  
 https://www.kaggle.com/therohk/million-headlines
The data is taken from headlines from Australia's "ABC" news organization. They are in Austalian English, which will be closer to UK English than US English, but a complete lexical subset of UK & US English, differing only in that some words in Australian English have meanings that do not occur in UK or US English.
 
However, I intend to replace it soonish. The headlines are all lower-case and stripped of all characters other than a-z and 0-9: no punctuation, accented characters, etc. Many of the headlines seem to be truncated for some reason, too. So, I will update it with a dataset that is closer to true headlines. 

This dataset is perfectly fine for everything that you need to learn in this code - it is just that the resulting annotations/models will be less useful in real-world situations.

