import tensorflow as tf
import tflearn
import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import random
import json
import pickle

pickle.dump({"words":words, "classes":classes, "train_x":train_x, "train_y":train_y}, open("training_data", "wb"))

data = pickle.load(open("training_data","rb"))
words = data['words']
classes = data['classes']
train_x = data['train_x']
train_y = data['train_y']

with open("intents.json") as json_data:
    intents = json.load(json_data)

model.load("./model.tflearn")


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words= [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=False):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("Found in bag: %s"% w)
    return(np.array(bag))

context = {}
ERROR_THRESHOLD = 0.25
def classify(sentence):
    results = model.predict([bow(sentence, words)])[0]
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    return return_list

def response(sentence, userID='123', show_details=False):
    results = classify(sentence)
    if results:
        while results:
            for i in intents['intents']:
                if i['tag'] == results[0][0]:
                    if 'context_set' in i:
                        if show_details: print ('context:', i['context_set'])
                        context[userID] = i['context_set']
                    if not 'context_filter' in i or \
                            (userID in context and 'context_filter' in i and i['context_filter'] == context[userID]):
                        if show_details: print ('tag:', i['tag'])
                        return print(random.choice(i['responses']))
            results.pop(0)

response("define TLS")
r="statue of equality"
response(r)