# -*- coding: utf-8 -*-


import nltk



from nltk.stem.lancaster import LancasterStemmer

# Creating an instance of the LancasterStemmer class.

stemmer = LancasterStemmer()



import numpy as np

import tflearn

import tensorflow as tf

import random

import json

import pdb

import pickle


with open("swiggy_intents.json") as file:
    
    
    data = json.load(file)
    
    
    
try:
    
    
    
    
    asdasd
    
    with open("data.pickle", "rb") as f:
        
        words, labels, docs_x, docs_y, training, output = pickle.load(f)
    
    
except:
    
        
    
        words = []
        
        labels = []
        
        docs_x = []
        
        
        # Every entry in docs_x corresponds to an entry in docs_y.
        
        # For every "list of words" in docs_x, there is a corresponding intent (tag) in docs_y.
        
        docs_y = []
        
        
        
            
            
            
        
        
        
        for intent in data["intents"]:
            
            
            for pattern in intent["patterns"]:
                
                
                # patterns is a list containing "string phrases".
                
                
                
                wrds = nltk.word_tokenize(pattern)          # This returns a list with all of the different words in our "string pattern".
                
                
                words.extend(wrds)
                
                docs_x.append(wrds)
                
                docs_y.append(intent["tag"])
                
                
                
                
            if intent["tag"] not in labels:
                
                labels.append(intent["tag"])
                
                
                
                
        # stemmer is an instance (child) of the LancasterStemmer class.
        
        
        words = [stemmer.stem(w.lower()) for w in words if w not in ["!","?","!?","?!", ":"]]
        
        
        # Now, we remove all the duplicate elements from our words list :-
        
        
        words = sorted(list(set(words)))
        
        
        
        
        # Now, let's remove any duplicate elements from our "words" list.
        
        
        # Sorting our "labels" list.
        
        labels = sorted(labels)
        
        
        
        
        training = []
        
        
        output = []
        
        
        out_empty = [0 for _ in range(len(labels))]
        
        
        
        for (x, doc) in enumerate(docs_x):
            
            
            bag = []
            
            
            
            
            wrds = [stemmer.stem(w.lower()) for w in doc if w not in ["!","?","!?","?!", ":"]]
            
            
            
            
            
            
            for w in words:
                
                if w in wrds:
                    
                    bag.append(1)
                    
                else:
                    
                    bag.append(0)
                    
                    
            
            
            output_row = out_empty[::1]
        
        
            output_row[labels.index(docs_y[x])] = 1           
            
            
            
            
            training.append(bag)
            
            
            output.append(output_row)
            
            
            
        
        
        
        training = np.array(training)
        
        
        output = np.array(output)
        
        
        
        
        
        with open("data.pickle", "wb") as f:
        
            pickle.dump((words, labels, docs_x, docs_y, training, output), f)              # Dump the mentioned variables into .pickle file "f".
        
    
   
    
    



# Let's provide input data (numpy array of a particular shape) for our model :-
    
    
    
network = tflearn.input_data(shape = [None, len(training[0])])         # The number of rows is "not specified" for our input array.  
    
    
# The length of each "bag of words" list is equal to our "word vocabulary" size (len(words)).
    
    
# Now, our input layer is connected to "two hidden fully_connected neural layers" containing eight neurons each.
    
    
network = tflearn.fully_connected(network, 8)
    
network = tflearn.fully_connected(network, 8)


    

    
    
# Our "hidden" fully_connected neural layers is connected to an output fully-connected layer containing neurons equal to the total number of labels.
    
    
network = tflearn.fully_connected(network, len(output[0]), activation = "softmax")      # fully connected output layer containing a neuron for each label (tag).
    
    
# activation = "softmax" kw_argument allows us to get "probabilities" on each of our outputs.
    
# "activation = softmax" is going to go through and provide probabilities for each of our outputs.
    
    
network = tflearn.regression(network)
    
    
    
    # Now finally, we are going to "train" our model using the neural network we have defined above:-
    
model = tflearn.DNN(network)
    
    # "DNN" is a type of a neural network.
    

    
    
try:
    hjghghg
    
    
    model.load("chatbot_model.tflearn")
    
    
    
except:
    
    
    
    # Now, let's train and also provide the "training parameters" for our newly created model :-
    
    
    model.fit(training, output, n_epoch = 1000, batch_size = 8, show_metric = True)
    
    
    
    # n_epoch parameter specifies the total number of times our model is gonna see through the training data.
    
    
    # Now, let's save our model :-
    
    
    model.save("chatbot_model.tflearn")
    
    
    
    
    
    
    
    
    
    



    
    
    

    
    

    
    

    
    
    
    
def bag_of_words(sentence, words):
    
    
    
    
        
        
    bag = [0 for _ in range(len(words))]
        
        
    
    my_sentence = nltk.word_tokenize(sentence)
    
    
    my_sentence = [stemmer.stem(word.lower()) for word in my_sentence if word not in ["?","!", ":", "?!", "!?"]]
            
            
        
   
    for w in my_sentence:
        
        
        for i, word in enumerate(words):
            
            if word == w:
                
                
                bag[i] = 1
                
                
            else:
                
                
                bag[i] = 0
                
                
                
                
                
    
    
              
    return np.array(bag)




def chat():
    
    
    print("Start chatting with Swiggy Bot! (Type in 'quit' to stop chatting)")
    
    
    while True:
        
        inp = input("You: ")
        
        
        if(inp.lower()=="quit"):
            
            break           # break out of the "while" loop. (stop chatting with the bot)
            
            
            
        else:
            
            
            results = model.predict([bag_of_words(inp, words)])[0]         # The prediction for our entry (provided inside of a list) is stored inside the variable "result".
            
            
            
            
              # Actually, the model.predict(list) method makes predictions for multiple entries at once (entries are provided in the form of a list). 
            
              # But, we have only one entry. 
            
              # Still, we need to pass in the argument (that one entry) in the form of a list.
              
              
            
            
            results_index = np.argmax(results)           # numpy.argmax(results) will return the "index" position of the greatest number in the "results" array.
            
            
            if results[results_index] >= 0.7 :
                
                
                
            
            
            
                tag = labels[results_index]
         
                
                for intent in data["intents"]:
                    
                    if tag in intent["tag"]:
                        
                        
                        responses = intent["responses"]
                        
                        
                        
                        print("Swiggy Bot:", random.choice(responses))
                        
                        
                        
                        
                        
                    
                        
                        
                        
                        
                
                
                
            
            
            else:
                
                
                print("Swiggy Bot: Sorry, I didn't get that. Please try again.")
            
            
            
            
            
            
            
                
                
                
            
            
            
            
                    
                    
                    
                    
                    
                    
            
    
    
if __name__ == "__main__" :

    
    
    

    # Call and execute the chat() function :-



    chat()
            
            
            
            
            
            


            
            
        
        
        
        
        
        
            
            
            
            
            
        
        
        
        
        
        
    
    
    
    
    



















            











            
    
    









        
        
        
        
    
    
    
    


