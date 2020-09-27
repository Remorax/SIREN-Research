import time, sys, pickle
from multiprocessing import Value
import numpy as np
from math import ceil
import concurrent.futures
import tensorflow as tf
import tensorflow_hub as hub

f = open("../junk/failed_words", "rb") 
failed, words = pickle.load(f)
print (len(words))
USE = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5?tf-hub-format=compressed")
def extractUSEEmbeddings(words):
    word_embeddings = USE(words)
    return word_embeddings.numpy()

def compare_sim(words, word_to_compare, max_sim, closest_word):
    print ("Extracting for", len(words))
    sys.stdout.flush()
    word_embeddings = USE(words).numpy()
    print ("emb extracted for ", word_to_compare)
    sys.stdout.flush()
    for i,w in enumerate(word_embeddings):
        sim = np.dot(word_to_compare, w)
        if sim > max_sim:
            max_sim = sim
            closest_word = words[i]
    print ("Original word: ", word, "Closest Word: ", closest_word, "Sim: ", max_sim)
    sys.stdout.flush()
    return (closest_word, max_sim)

def closest_word_USE(argument):
    
    word, embed = argument
    len_part = 10000
    max_sim = -1000
    n_parts = ceil(len(words)/len_part)
    closest_word = ""
    print ("Word len", len(words))
    print ("Num_parts", n_parts)
    sys.stdout.flush()
    for i in range(n_parts):
        print (i)
        sys.stdout.flush()
        words_part = words[i*len_part:(i+1)*len_part]
        closest_word, max_sim = compare_sim(words_part, embed, max_sim, closest_word)
    with counter.get_lock():
        counter.value += 1
    print ("RESOLVED: Original word: ", word, "Closest Word: ", closest_word, "Sim: ", max_sim)
    sys.stdout.flush()
    print ("Percentage done: ", float(counter.value*100/len(failed)))
    sys.stdout.flush()
    return word1, closest_word, max_sim

def run():
    resolved = dict()
    print ("Working on it...", len(failed))
    sys.stdout.flush()
    counter = Value('i', 0)
    a = time.time()
    failed_embeddings = extractUSEEmbeddings([elem.split("(")[0].strip() for elem in failed])
    print ("Took me {} seconds to extract USE embeddings...".format(time.time()-a))
    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
        for res in executor.map(closest_word_USE, zip(failed, failed_embeddings), chunksize=int(len(failed)/100)):
            resolved[res[0]] = (res[1], res[2])

    f = open("../junk/resolved_unbracketed.pkl", "wb")
    pickle.dump(resolved, f)

if __name__ == '__main__':
    run()
