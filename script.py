# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 00:13:58 2021

@author: LENOVO
"""

import json


import nltk


from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()



import tensorflow as tf


import tflearn


import numpy as np


import pickle


import random







with open("swiggy_intents.json") as file:

    data = json.load(file)





try:

    with open("swiggy_data.pickle", "rb") as f:


        training, output, words, labels, docs_x, docs_y = pickle.load(f)





except:











    words = []

    labels = []


    docs_x = []


    docs_y = []



    for intent in data["intents"]:





        for pattern in intent["patterns"]:


            words_list = nltk.word_tokenize(pattern)                       # nltk.word_tokenize(pattern) function itself returns a list.

            words.extend(words_list)

            docs_x.append(words_list)

            docs_y.append(intent["tag"])




        if intent["tag"] not in labels:


            labels.append(intent["tag"])














    words = [stemmer.stem(w.lower()) for w in words if w not in ["?", "!"]]


    words = sorted(list(set(words)))


    labels = sorted(labels)







    training = []

    output = []


    out_empty = [0 for _ in range(len(labels))]
















    for i, words_list in enumerate(docs_x):


        words_list = [stemmer.stem(word.lower()) for word in words_list if word not in ["?", "!"]]


        bag = []













        for word in words:


            if word in words_list:


                bag.append(1)


            else:

                bag.append(0)




        output_row = out_empty[::1]


        output_row[labels.index(docs_y[i])] = 1


        output.append(output_row)


        training.append(bag)





    output = np.array(output)


    training = np.array(training)



    with open("swiggy_data.pickle", "wb") as f:


        pickle.dump((training, output, words, labels, docs_x, docs_y), f)





#tf.reset_default_graph()             # reset the previously applied settings and the underlying "data graph".







network = tflearn.input_data(shape = [None, len(training[0])])


network = tflearn.fully_connected(network, 8)

network = tflearn.fully_connected(network, 8)

network = tflearn.fully_connected(network, len(output[0]), activation = "softmax")



network = tflearn.regression(network)

model = tflearn.DNN(network)





try:

    # If the model is already trained and saved, try to execute this block of code :-



    model.load("swiggy_chatbot.tflearn")



except:



    model.fit(training, output, n_epoch = 1000, batch_size = 8, show_metric = True)



    model.save("swiggy_chatbot.tflearn")




def bag_of_words(sentence, words):



    bag = [0 for _ in range(len(words))]




    list_of_words = nltk.word_tokenize(sentence)


    stemmed_list_of_words = [stemmer.stem(w.lower()) for w in list_of_words if w not in ["?", "!"]]



    for word in stemmed_list_of_words:


        for i,w in enumerate(words):


            if word in words:


                bag[i] = 1


            else:

                bag[i] = 0





    return np.array(bag)





def chat():



    print("Start chatting with the Swiggy Bot! (Type in 'quit' to end the chat session.)")


    while True:

        inp = input("You: ")


        if inp.lower() == "quit":

            break


        else:


            result_array = model.predict([bag_of_words(inp, words)])[0]


            result_index = np.argmax(result_array)



            if result_array[result_index] >= 0.7:


                tag = labels[result_index]


                for intent in data["intents"]:


                    if intent["tag"] == tag:


                        responses = intent["responses"]


                        print("Swiggy Bot:", random.choice(responses))


            else:


                print("Swiggy Bot: I couldn't understand your query. Please try again.")








if __name__ == "__main__":


    # Call and execute the chat() function.


    chat()
