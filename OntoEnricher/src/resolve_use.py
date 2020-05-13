import time, sys, pickle
from multiprocessing import Value
import numpy as np
from math import ceil
import concurrent.futures
import tensorflow as tf
import tensorflow_hub as hub

USE_folder = "/home/vlead/USE"

f = open("../junk/failed_words", "rb") 
failed, words = pickle.load(f)

def extractUSEEmbeddings(words):

    embed = hub.KerasLayer(USE_folder)






    word_embeddings = embed(words)
    return word_embeddings.numpy()

def compare_sim(words, word_to_compare, max_sim, closest_word):
    print ("Extracting for", len(words))
    word_embeddings = extractUSEEmbeddings(words)
    print ("emb extracted for ", word_to_compare)
    for i,w in enumerate(word_embeddings):
        sim = np.dot(word_to_compare, w)
        if sim > max_sim:
            max_sim = sim
            closest_word = words[i]
    print ("Original word: ", word, "Closest Word: ", closest_word, "Sim: ", max_sim)
    return (closest_word, max_sim)

def closest_word_USE(argument):
    
    word, embed = argument
    len_part = 10000
    max_sim = -1000
    n_parts = ceil(len(words)/len_part)
    closest_word = ""
    for i in range(n_parts):
        words_part = words[i*len_part:(i+1)*len_part]
        closest_word, max_sim = compare_sim(words_part, embed, max_sim, closest_word)
    with counter.get_lock():
        counter.value += 1
    print ("RESOLVED: Original word: ", word, "Closest Word: ", closest_word, "Sim: ", max_sim)
    print ("Percentage done: ", float(counter.value*100/len(failed)))
    return word1, closest_word, max_sim


resolved = dict()
print ("Working on it...")
counter = Value('i', 0)
a = time.time()
failed_embeddings = extractUSEEmbeddings(failed)
print ("Took me {} seconds to extract USE embeddings...".format(time.time()-a))
with concurrent.futures.ProcessPoolExecutor() as executor:
    for res in executor.map(closest_word_USE, zip(failed, failed_embeddings)):
        resolved[res[0]] = (res[1], res[2])

f = open("resolved", "wb")
pickle.dump(resolved, f)
    
