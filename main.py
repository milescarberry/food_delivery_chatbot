# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 17:38:33 2021

@author: LENOVO

"""


import nltk

from nltk.stem.lancaster import LancasterStemmer


# Creating an instance of the LancasterStemmer class.

stemmer = LancasterStemmer()


import numpy as np
import tflearn
import tensorflow as tf
import json
import random
import pdb


# Opening the "intents.json" file and then assigning it to the variable "dataset_file".

# Then we are gonna load the contents of that .json file and print it(printing is optional).



with open("intents.json") as dataset_file:
    
    dataset = json.load(dataset_file)
    
    
    
    
words = []

labels = []

docs_x = []

docs_y = []





for intent in dataset["intents"]:
    
    
    for pattern in intent["patterns"]:
        
        
        # Perform "Stemming"!
        
        # Firstly, grab all the words in our pattern :-
        
        wrds = nltk.word_tokenize(pattern)
        
        # wrds is a "list" of words. (Split at "space")
        
        # "wrds" is a list here. So, we just "extend" our existing "words" list with "wrds".
        
        words.extend(wrds)
        
        
        docs_x.append(wrds)        # append the tokenized words to the "docs_x" list.
        
    
        docs_y.append(intent["tag"])
        
        
        # The corresponding "tag" for each "pattern" goes in "docs_y" list.
    
    
    # We are gonna store each and every pattern of the intents in docs_x & its corresponding tag(intent_name) in docs_y.
    
    # The above step is important for training our ML model.
    
    # We are actually classifying all the patterns inside our intents.(Important for training our model)
        
        
        
    if intent["tag"] not in labels:
        
        labels.append(intent["tag"])
        
        
        
        
        # The "labels" list stores all the "tags"(all the "intent" names). (To classify each "pattern" in the docs_x list.)
        
        


# Now, we are gonna "stem" each word and measure how many "unique" words are there in the "words" list.

# In short, we are gonna "stem" each word( Example:- Extract the "root" word from "Is anyone there?, Extract the root word from What's up?") and also remove all the duplicate words from our "words" list.

# Don't forget to convert all words to lowercase before "stemming". [For Uniformity!]

# We are also going to use "list comprehension" here.



words = [stemmer.stem(w.lower()) for w in words if w not in ["?" , "!" , ":)" , ":("]]


# Now, let's remove all the duplicate elements from our "words" list :-
 
# (We are also gonna "sort" the list in the end.)(Using the sorted() in-built function)
 
 
words = sorted(list(set(words)))




labels = sorted(labels)


# The "labels" list contains all the "tags" of each "intent".



# Neural Networks only understands "numbers".




training = []


output = []


out_empty = [0 for _ in range(len(labels))]



for x, doc in enumerate(docs_x):
    
    bag = []
    
    # "bag" list is our "bag of words". The "Bag Of Words" is a list containing 0s and 1s.
    
    # "output" is a list containing 0s and 1s.
    
    
    docs_x_wrds = [stemmer.stem(w.lower()) for w in doc]
    
    
    
    for w in words:
        
        if w in docs_x_wrds:
            
            bag.append(1)
            
            
            
        else:
            
            
            bag.append(0)
            
            
            
            
    # output_row is a copy of "out_empty" list.(Initially)    
    
        
    output_row =  out_empty[::1]


    output_row[labels.index(docs_y[x])] = 1
    
    
    training.append(bag)
    
    output.append(output_row)
    
    
    
    
    
    
    
    
 # Converting "training" and "output" lists to numpy arrays :-

   

training = np.array(training)


output = np.array(output)



# "Resetting" the underlying data graph. ("Reset the previous settings and stuff.")

tf.reset_default_graph()



# Let's specify the input data for our model. Each training input array should be of the same length. (shape = [len(training[0])]  ==> equal to the total number of words (i.e. the length of the bag of words))



net = tflearn.input_data(shape = [len(training[0])])               # Specify the shape of the input data array.




















    
    
    
    
























        
        
        
        
        
        
        
        
        
        
        
        



    
    



