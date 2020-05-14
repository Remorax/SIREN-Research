from gensim.models.keyedvectors import KeyedVectors
import bcolz, pickle, os, sys, pickledb, time
import concurrent.futures
import numpy as np
from math import ceil
from itertools import count
from collections import defaultdict
from difflib import SequenceMatcher
import tensorflow as tf
import tensorflow_hub as hub
from scipy import spatial
from gensim.models.keyedvectors import KeyedVectors
from copy import deepcopy

wiki2vec = KeyedVectors.load_word2vec_format("/home/vlead/enwiki_20180420_win10_300d.txt")
og_dict = deepcopy(wiki2vec.wv.vocab)
for k in og_dict:
    if "/" in k:
        wiki2vec.wv.vocab[k.split("/")[1].lower()] = wiki2vec.wv.vocab[k]
        del wiki2vec.wv.vocab[k]

del og_dict
