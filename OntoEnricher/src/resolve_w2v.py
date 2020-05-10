from copy import deepcopy
from gensim.models import KeyedVectors
import pickle, os, sys, time
import concurrent.futures
import numpy as np
from scipy import spatial
from sparse_dot_topn import awesome_cossim_topn
from scipy.sparse import coo_matrix

f = open("w2v_data", "rb")
words, failed = pickle.load(f)

og_dict = deepcopy(wiki2vec.wv.vocab)
for k in og_dict:
    if "/" in k:
        wiki2vec.wv.vocab[k.split("/")[1].lower()] = wiki2vec.wv.vocab[k]
        del wiki2vec.wv.vocab[k]

del og_dict
def calculate_sim(words, word1, max_sim, closest_word):
    t = time.time()
    i = 0
    for word2 in words:
        try:
            sim = wiki2vec.similarity("_".join(word1.lower().split()), "_".join(word2.split()))
            if sim > max_sim:
                max_sim = sim
                closest_word = word2
            i += 1
        except Exception as e:
            continue
    print ("Original word: ", word1, "Closest Word: ", closest_word)
    print ("Took me {} seconds to iteration of sim compare...".format(time.time()-a))
    sys.stdout.flush()
    return (closest_word, max_sim)

def closest_word_w2v(word1):
    len_part = 100000
    max_sim = -1000
    n_parts = ceil(len(words)/len_part)
    closest_word = ""
    if word1 not in wiki2vec.wv.vocab:
        print ("Original word not in vocab", word1)
        return (closest_word, max_sim)
    for i in range(n_parts):
        words_part = words[i*len_part:(i+1)*len_part]
        closest_word, max_sim = calculate_sim(words_part, word1, max_sim, closest_word)
    return word1, closest_word          

a = time.time()

# closest_word = closest_word_w2v("margherita pizza")

# closest_word_w2v("nelson mandela")

resolved = dict()
with concurrent.futures.ProcessPoolExecutor() as executor:
    for res in executor.map(closest_word_w2v, failed):
        resolved[res[0]] = res[1]

f = open("resolved", "wb")
pickle.dump(resolved, f)