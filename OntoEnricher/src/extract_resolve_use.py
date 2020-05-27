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

    counter = 0
    len_part = 10000
    n_parts = ceil(len(words)/len_part)
    closest_word = ""
    embeds = []
    a = 0
    for i in range(n_parts):
        counter = float(i*100/n_parts)
        print (counter, "%done")
        if i>0 and float(i*100/n_parts)%5 < float((i-1)*100/n_parts)%5:
           
            if counter<95:
                a+=1
            else:
                f = open("../junk/use_embeddings_" + str(a), "wb")
                pickle.dump(embeds, f)
                f.close()
                a += 1
                del embeds
                embeds = []
        if counter<90:
            continue
        words_part = words[i*len_part:(i+1)*len_part]
        embeds.extend(list(zip(words_part, extractUSEEmbeddings(words_part))))

    f = open("../junk/use_embeddings_" + str(a), "wb")
    pickle.dump(embeds, f)

    return

if __name__ == "__main__": 
    run()

