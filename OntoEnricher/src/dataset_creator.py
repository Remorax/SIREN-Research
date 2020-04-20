from bsddb3 import btopen
import bcolz, pickle, os, sys, shelve
import numpy as np
from math import ceil
from itertools import count
from collections import defaultdict
from difflib import SequenceMatcher
import tensorflow as tf
import tensorflow_hub as hub
from scipy import spatial

prefix = "/data/Vivek/Final/SIREN-Research/OntoEnricher/junk/Files/security_threshold_10_10/security"
output_folder = "../junk/Output/"
embeddings_folder = "../junk/Glove.dat"
USE_folder = "/home/vlead/USE"
embeddings_file = "/data/Vivek/glove.6B.300d.txt"
use_embeddings = "../files/embeddings.pt"

POS_DIM = 4
DEP_DIM = 5
DIR_DIM = 1
EMBEDDING_DIM = 300
NULL_PATH = ((0, 0, 0, 0),)
relations = ["hypernym", "hyponym", "synonym", "none"]
# relations = ["True", "False"]
NUM_RELATIONS = len(relations)

inp_file = sys.argv[1]

failed = []
success = []



def id_to_entity(db, entity_id):
    entity = db[str(entity_id)]
    if db == id2path_db:
        entity = "/".join(["*##*".join(e.split("_", 1)) for e in entity.split("/")])
    return entity


def entity_to_id(db, entity):
    if entity in db:
        success.append(entity)
        return int(db[entity])
    failed.append(entity)
    return -1

def extract_paths(db, x, y):
    key = (str(x) + '###' + str(y))
    try:
        relation = db[key]
        return {int(path_count.split(":")[0]): int(path_count.split(":")[1]) for path_count in relation.split(",")}
    except Exception as e:
        return {}

def load_embeddings_from_disk():
    try:
        vectors = bcolz.open(embeddings_folder)[:]
        words = pickle.load(open(embeddings_folder + 'words.pkl', 'rb'))
        word2idx = pickle.load(open(embeddings_folder + 'words_index.pkl', 'rb'))

        embeddings = vectors
    except:
        embeddings, word2idx = create_embeddings()
    return embeddings, word2idx
        

def create_embeddings():
    words = ['_unk_']
    idx = 1
    word2idx = {"_unk_": 0}
    vectors = bcolz.carray(np.random.random(300), rootdir=embeddings_folder, mode='w')
    with open(embeddings_file, 'r') as f:
        for l in f:
            line = l.split()
            word, vector = line[0], line[1:]
            words.append(word)
            vectors.append(np.array(vector).astype(np.float))
            word2idx[word] = idx
            idx += 1
    vectors = vectors.reshape((-1, EMBEDDING_DIM))
    row_norm = np.sum(np.abs(vectors)**2, axis=-1)**(1./2)
    vectors /= row_norm[:, np.newaxis]
    vectors = bcolz.carray(vectors, rootdir=embeddings_folder, mode='w')
    vectors.flush()

    pickle.dump(words, open(embeddings_folder + 'words.pkl', 'wb'))
    pickle.dump(word2idx, open(embeddings_folder + 'words_index.pkl', 'wb'))
    
    return vectors, word2idx

word2id_db = shelve.open(prefix + "_word_to_id_dict.db", 'r')
id2word_db = shelve.open(prefix + "_id_to_word_dict.db", "r")
path2id_db = shelve.open(prefix + "_path_to_id_dict.db", "r")
id2path_db = shelve.open(prefix + "_id_to_path_dict.db", "r")
relations_db = shelve.open(prefix + "_relations_map.db", "r")

embeddings, emb_indexer = load_embeddings_from_disk()

train_dataset = {}
for l in open(inp_file).read().split("\n"):
    try:
        if l:
            train_dataset[tuple(l.split("\t")[:2])] = l.split("\t")[2] 
    except:
        print (l)
        raise

arrow_heads = {">": "up", "<":"down"}

def extract_direction(edge):

    if edge[0] == ">" or edge[0] == "<":
        direction = "start_" + arrow_heads[edge[0]]
        edge = edge[1:]
    elif edge[-1] == ">" or edge[-1] == "<":
        direction = "end_" + arrow_heads[edge[-1]]
        edge = edge[:-1]
    else:
        direction = ' '
    return direction, edge

def parse_path(path):
    parsed_path = []
    for edge in path.split("*##*"):
        direction, edge = extract_direction(edge)
        if edge.split("/"):
            try:
                embedding, pos, dependency = edge.split("/")
            except:
                print (edge, path)
                raise
            emb_idx, pos_idx, dep_idx, dir_idx = emb_indexer.get(embedding, 0), pos_indexer[pos], dep_indexer[dependency], dir_indexer[direction]
            parsed_path.append(tuple([emb_idx, pos_idx, dep_idx, dir_idx]))
        else:
            return None
    return tuple(parsed_path)

def extract_all_paths(x,y):

    paths = list(extract_paths(relations_db,x,y).items()) + list(extract_paths(relations_db,y,x).items())
    print ("extracted paths xy and yx...")
    x_word = id_to_entity(id2word_db, x) if x!=-1 else "X"
    y_word = id_to_entity(id2word_db, y) if y!=-1 else "Y"
    path_count_dict = { id_to_entity(id2path_db, path).replace("X/", x_word+"/").replace("Y/", y_word+"/") : freq for (path, freq) in paths }
    print ("bug fixing for xy and yx")
    path_count_dict = { parse_path(path) : path_count_dict[path] for path in path_count_dict }
    print ("counted paths")

    return { path : path_count_dict[path] for path in path_count_dict if path}

def parse_dataset(dataset):
    print ("Entering parse dataset")
    keys = [(entity_to_id(word2id_db, x), entity_to_id(word2id_db, y)) for (x, y) in dataset]
    print ("Extracting all paths...")
    paths = [extract_all_paths(x,y) for (x,y) in keys]
    empty = [list(dataset)[i] for i, path_list in enumerate(paths) if len(list(path_list.keys())) == 0]
    print('Pairs without paths:', len(empty), ', all dataset:', len(dataset))
    embed_indices = [(emb_indexer.get(x,0), emb_indexer.get(y,0)) for (x,y) in keys]
    print ("emb indices done")
    return embed_indices, paths

pos_indexer, dep_indexer, dir_indexer = defaultdict(count(0).__next__), defaultdict(count(0).__next__), defaultdict(count(0).__next__)
unk_pos, unk_dep, unk_dir = pos_indexer["#UNKNOWN#"], dep_indexer["#UNKNOWN#"], dir_indexer["#UNKNOWN#"]

dataset_keys = list(train_dataset.keys())
dataset_vals = list(train_dataset.values())

embed_indices, x = parse_dataset(dataset_keys)
y = [i for (i,relation) in enumerate(dataset_vals)]

it = inp_file.split("_")[-1]
op_file = "../junk/Files/parsed_dataset_parts/parsed_dataset_" + str(it)

f = open(op_file, "wb+")
pickle.dump([success, failed, embed_indices, x, y], f)
f.close()

print ("Parsed",inp_file) 