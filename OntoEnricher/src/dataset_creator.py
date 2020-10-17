import bcolz, pickle, os, sys, pickledb
import concurrent.futures
import numpy as np
from math import ceil
from itertools import count
from collections import defaultdict
from difflib import SequenceMatcher
import tensorflow as tf
import tensorflow_hub as hub
from scipy import spatial

train_file = "../files/dataset/train_final.tsv"
test_file = "../files/dataset/test_final.tsv"
train_file = "/data/Vivek/original/HypeNET/dataset/custom_train_0.0_0.2.tsv"
test_file =  "/data/Vivek/original/HypeNET/dataset/custom_test_0.0_0.2.tsv"
instances_file = '../files/dataset/test_instances.tsv'
knocked_file = '../files/dataset/test_knocked.tsv'
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
relations = ["hypernym", "hyponym", "concept", "instance", "none"]
# relations = ["True", "False"]
NUM_RELATIONS = len(relations)

def run(args):
    prefix, op_file = args
    failed = []
    success = []

    def id_to_entity(db, entity_id):
        try:
            entity = db[str(entity_id)]    
        except:
            entity = db[str(entity_id).decode("utf-8")]
        return entity

    def id_to_path(db, entity_id):
        try:
            entity = db[str(entity_id)]
        except:
            entity = db[str(entity_id).decode("utf-8")]
        entity = "/".join(["*##*".join(e.split("_", 1)) for e in entity.split("/")])
        return entity

    def entity_to_id(db, entity):
        if entity in db:
            success.append(entity)
            try:
                return int(db[entity])
            except:
                return int(db[entity.decode("utf-8")])
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

    try:
        word2id_db = shelve.open(prefix + "_word_to_id_dict.db", 'r')
    except:
        print (prefix)
        raise
    id2word_db = shelve.open(prefix + "_id_to_word_dict.db", "r")
    path2id_db = shelve.open(prefix + "_path_to_id_dict.db", "r")
    id2path_db = shelve.open(prefix + "_id_to_path_dict.db", "r")
    relations_db = shelve.open(prefix + "_relations_map.db", "r")

    embeddings, emb_indexer = load_embeddings_from_disk()

    train_dataset = {tuple(l.split("\t")[:2]): l.split("\t")[2] for l in open(train_file).read().split("\n")}
    test_dataset = {tuple(l.split("\t")[:2]): l.split("\t")[2] for l in open(test_file).read().split("\n")}
    test_instances = {tuple(l.split("\t")[:2]): l.split("\t")[2] for l in open(instances_file).read().split("\n")}
    test_knocked = {tuple(l.split("\t")[:2]): l.split("\t")[2] for l in open(knocked_file).read().split("\n")}

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
                    embedding, pos, dependency = tuple([a[::-1] for a in edge[::-1].split("/",2)][::-1])
                except:
                    print (edge, path)
                    raise
                emb_idx, pos_idx, dep_idx, dir_idx = emb_indexer.get(embedding, 0), pos_indexer[pos], dep_indexer[dependency], dir_indexer[direction]
                parsed_path.append(tuple([emb_idx, pos_idx, dep_idx, dir_idx]))
            else:
                return None
        return tuple(parsed_path)

    def parse_tuple(tup):
        x, y = entity_to_id(word2id_db, tup[0]), entity_to_id(word2id_db, tup[1])
        # paths = list(extract_paths(relations_db,x,y).items()) + list(extract_paths(relations_db,y,x).items())
        # x_word = id_to_entity(id2word_db, x) if x!=-1 else "X"
        # y_word = id_to_entity(id2word_db, y) if y!=-1 else "Y"
        # path_count_dict = { id_to_path(id2path_db, path).replace("X/", x_word+"/").replace("Y/", y_word+"/") : freq for (path, freq) in paths }
        paths_xy = list(extract_paths(relations_db,x,y).items())
        paths_yx = list(extract_paths(relations_db,y,x).items())
        path_count_dict = { id_to_path(id2path_db, path) : freq for (path, freq) in paths_xy }
        path_count_dict.update({ id_to_path(id2path_db, path).replace("X/", '@@@').replace('Y/', 'X/').replace('@@@', 'Y/') : freq for (path, freq) in paths_yx })
        return path_count_dict

    def parse_dataset(dataset):
        print ("Parsing dataset for ", prefix)
        
        parsed_dicts = [parse_tuple(tup) for tup in dataset]
        parsed_dicts = [{ parse_path(path) : path_count_dict[path] for path in path_count_dict } for path_count_dict in parsed_dicts]
        paths = [{ path : path_count_dict[path] for path in path_count_dict if path} for path_count_dict in parsed_dicts]
        empty = [list(dataset)[i] for i, path_list in enumerate(paths) if len(list(path_list.keys())) == 0]
        print('Pairs without paths:', len(empty), ', all dataset:', len(dataset))
        embed_indices = [(emb_indexer.get(x,0), emb_indexer.get(y,0)) for (x,y) in dataset]
        
        return embed_indices, paths

    pos_indexer, dep_indexer, dir_indexer = defaultdict(count(0).__next__), defaultdict(count(0).__next__), defaultdict(count(0).__next__)
    unk_pos, unk_dep, unk_dir = pos_indexer["#UNKNOWN#"], dep_indexer["#UNKNOWN#"], dir_indexer["#UNKNOWN#"]

    dataset_keys = list(train_dataset.keys()) + list(test_dataset.keys()) + list(test_instances.keys()) + list(test_knocked.keys())
    dataset_vals = list(train_dataset.values()) + list(test_dataset.values()) + list(test_instances.values()) + list(test_knocked.values())

    mappingDict = {key: idx for (idx,key) in enumerate(relations)}

    embed_indices, x = parse_dataset(dataset_keys)
    y = [mappingDict[relation] for relation in dataset_vals]

    f = open(op_file, "wb+")

    s1 = len(train_dataset)
    s2 = len(train_dataset) + len(test_dataset)
    s3 = len(train_dataset)+len(test_dataset)+len(test_instances)

    parsed_train = (embed_indices[:s1], x[:s1], y[:s1], dataset_keys[:s1], dataset_vals[:s1])
    parsed_test = (embed_indices[s1:s2], x[s1:s2], y[s1:s2], dataset_keys[s1:s2], dataset_vals[s1:s2])
    parsed_instances = (embed_indices[s2:s3], x[s2:s3], y[s2:s3], dataset_keys[s2:s3], dataset_vals[s2:s3])
    parsed_knocked = (embed_indices[s3:], x[s3:], y[s3:], dataset_keys[s3:], dataset_vals[s3:])
    pickle.dump([parsed_train, parsed_test, parsed_instances, parsed_knocked, pos_indexer, dep_indexer, dir_indexer], f)
    print ("Successful hits: ", len(success), "Failed hits: ", len(failed))
    f.close()

    print ("Parsed",prefix) 

thresholds_path = "/data/Vivek/Final/SIREN-Research/OntoEnricher/junk/Files/"
folders = [l for l in os.listdir(thresholds_path) if l.startswith("security_threshold")]
args = [(thresholds_path + l + "/security", "../junk/Files/parsed_dataset_parts/parsed_dataset_" + "_".join(l.split("_")[-2:])) for l in folders]
#with concurrent.futures.ProcessPoolExecutor() as executor:
#    for res in executor.map(run, args):
#        pass
run(("/data/Vivek/Final/SIREN-Research/OntoEnricher/junk/Files/security_threshold_7_10/security", "dataset_optimized_alt0.2.pkl"))
