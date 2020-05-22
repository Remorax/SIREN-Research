import time, sys, pickle
from multiprocessing import Value
import numpy as np
from math import ceil
import concurrent.futures
import tensorflow as tf
import tensorflow_hub as hub

USE_folder = "/home/vlead/USE"
embeds, words = [], []
def extractUSEEmbeddings(words):

    embed = hub.KerasLayer(USE_folder)
    word_embeddings = embed(words)
    return word_embeddings.numpy()

def compare_sim(word_to_compare):
    global embeds, words
    max_sim = -1000
    closest_word = ""
    for i,w in enumerate(embeds):
        sim = np.dot(word_to_compare, w)
        if sim > max_sim:
            max_sim = sim
            closest_word = words[i]
    print ("Original word: ", word, "Closest Word: ", closest_word, "Sim: ", max_sim)
    return (closest_word, max_sim)

def run():
    global embeds, words
    f = open("../junk/failed_words", "rb") 
    failed, words = pickle.load(f)

    len_part = 10000
    n_parts = ceil(len(words)/len_part)
    closest_word = ""
    embeds = []
    for i in range(n_parts):
        print (float(i*100/n_parts))
        words_part = words[i*len_part:(i+1)*len_part]
        embeds.append(extractUSEEmbeddings(words_part))

    f = open("../junk/use_embeddings", "wb")
    pickle.dump(embeds, f)

    resolved = dict()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for res in executor.map(compare_sim, failed):
            resolved[res[0]] = (res[1], res[2])

    f = open("../junk/use_resolved", "wb")
    pickle.dump(resolved, f)
    return

if __name__ == "__main__": 
    run()
