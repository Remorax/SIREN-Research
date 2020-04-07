from bsddb3 import btopen
import bcolz, pickle, torch
import numpy as np
from math import ceil
from itertools import count
from collections import defaultdict
import torch.optim as optim
import torch.nn as nn

# prefix = "../junk/Files/temp_threshold_3_4/temp"
# train_file = "../junk/train.tsv"
# test_file = "../junk/test.tsv"
# output_folder = "../junk/Output/"
# embeddings_folder = "../junk/Glove.dat"
# embeddings_file = "/Users/vivek/SIREN-Research/Archive-LSTM/glove.6B/glove.6B.300d.txt"

prefix = "/home/hduser_/security"
train_file = "../files/dataset/train.tsv"
test_file = "../files/dataset/test.tsv"
output_folder = "../junk/Output/"
embeddings_folder = "../junk/Glove.dat"
embeddings_file = "/home/hduser_/glove.6B.300d.txt"

POS_DIM = 4
DEP_DIM = 5
DIR_DIM = 1
EMBEDDING_DIM = 300
NULL_PATH = ((0, 0, 0, 0),)
relations = ["hypernym", "hyponym", "synonym", "none"]
# relations = ["True", "False"]
NUM_RELATIONS = len(relations)


def id_to_entity(db, entity_id):
    entity_id = str(entity_id).encode("utf-8")
    return db[entity_id].decode("utf-8")

def entity_to_id(db, entity):
    entity = entity.encode("utf-8")
    if entity in db:
        return int(db[entity])
    return -1

def extract_paths(db, x, y):
    key = (str(x) + '_' + str(y)).encode("utf-8")
    try:
        relation = db[key].decode("utf-8")
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
        embeddings = create_embeddings()
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


word2id_db = btopen(prefix + "_term_to_id.db", "r")
id2word_db = btopen(prefix + "_id_to_term.db", "r")
path2id_db = btopen(prefix + "_path_to_id.db", "r")
id2path_db = btopen(prefix + "_id_to_path.db", "r")
relations_db = btopen(prefix + "_l2r.db", "r")

embeddings, emb_indexer = load_embeddings_from_disk()

train_dataset = {tuple(l.split("\t")[:2]): l.split("\t")[2] for l in open(train_file).read().split("\n")}
test_dataset = {tuple(l.split("\t")[:2]): l.split("\t")[2] for l in open(test_file).read().split("\n")}

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
    for edge in path.split("_"):
        direction, edge = extract_direction(edge)
        if edge.split("/"):
            embedding, pos, dependency = edge.split("/")
            emb_idx, pos_idx, dep_idx, dir_idx = emb_indexer.get(embedding, 0), pos_indexer[pos], dep_indexer[dependency], dir_indexer[direction]
            parsed_path.append(tuple([emb_idx, pos_idx, dep_idx, dir_idx]))
        else:
            return None
    return tuple(parsedPath)


def extract_all_paths(x,y):
    
    paths = list(extract_paths(relations_db,x,y).items()) + list(extract_paths(relations_db,y,x).items())
    x_word = id_to_entity(id2word_db, x) if x!=-1 else "X"
    y_word = id_to_entity(id2word_db, y) if y!=-1 else "Y"
    path_count_dict = { id_to_entity(id2path_db, path).replace("X/", x_word+"/").replace("Y/", y_word+"/") : freq for (path, freq) in paths }
    path_count_dict = { parse_path(path) : path_count_dict[path] for path in path_count_dict }

    return { path : path_count_dict[path] for path in path_count_dict if path}
    
def parse_dataset(dataset):
    keys = [(entity_to_id(word2id_db, x), entity_to_id(word2id_db, y)) for (x, y) in dataset]
    paths = [extract_all_paths(x,y) for (x,y) in keys]
    empty = [list(dataset)[i] for i, path_list in enumerate(paths) if len(list(path_list.keys())) == 0]
    print('Pairs without paths:', len(empty), ', all dataset:', len(dataset))
    embed_indices = [(emb_indexer.get(x,0), emb_indexer.get(y,0)) for (x,y) in keys]
    return embed_indices, paths
  
torch.set_default_dtype(torch.float64)

pos_indexer, dep_indexer, dir_indexer = defaultdict(count(0).__next__), defaultdict(count(0).__next__), defaultdict(count(0).__next__)
unk_pos, unk_dep, unk_dir = pos_indexer["#UNKNOWN#"], dep_indexer["#UNKNOWN#"], dir_indexer["#UNKNOWN#"]

dataset_keys = list(train_dataset.keys()) + list(test_dataset.keys())
dataset_vals = list(train_dataset.values()) + list(test_dataset.values())

embed_indices, x = parse_dataset(dataset_keys)
y = [i for (i,relation) in enumerate(dataset_vals)]

embed_indices_train, embed_indices_test = np.array(embed_indices[:len(train_dataset)]), np.array(embed_indices[len(train_dataset):len(train_dataset)+len(test_dataset)])
x_train, x_test = np.array(x[:len(train_dataset)]), np.array(x[len(train_dataset):len(train_dataset)+len(test_dataset)])
y_train, y_test = np.array(y[:len(train_dataset)]), np.array(y[len(train_dataset):len(train_dataset)+len(test_dataset)])

class LSTM(nn.Module):

    def __init__(self):
        
        super(LSTM, self).__init__()
        self.cache = {}
        
        self.hidden_dim = HIDDEN_DIM + 2 * EMBEDDING_DIM
        self.input_dim = POS_DIM + DEP_DIM + EMBEDDING_DIM + DIR_DIM
        self.W = nn.Linear(NUM_RELATIONS, self.input_dim)
        self.dropout_layer = nn.Dropout(p=dropout)
        self.softmax = nn.LogSoftmax()
        
        self.word_embeddings = nn.Embedding(len(embeddings), EMBEDDING_DIM)
        self.word_embeddings.load_state_dict({'weight': torch.from_numpy(np.array(embeddings))})
        self.word_embeddings.require_grad = False
        
        self.pos_embeddings = nn.Embedding(len(pos_indexer), POS_DIM)
        self.dep_embeddings = nn.Embedding(len(dep_indexer), DEP_DIM)
        self.dir_embeddings = nn.Embedding(len(dir_indexer), DIR_DIM)
        
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, NUM_LAYERS)
    
    def normalize_embeddings(self, embeds):
        row_norm = torch.sum(torch.abs(embeds)**2, axis=-1)**(1./2)
        embeds /= row_norm.view(-1,1)
        embed = torch.flatten(self.dropout_layer(embeds))
        return embed

    def embed_path(self, elem):
        path, count = elem
        if path in self.cache:
            return self.cache[path] * count
        lstm_inp = torch.Tensor([])
        for edge in path:
            inputs = [torch.Tensor([[el]]).long() for el in edge]
            word_embed = self.normalize_embeddings(self.word_embeddings(inputs[0]))
            pos_embed = self.normalize_embeddings(self.pos_embeddings(inputs[1]))
            dep_embed = self.normalize_embeddings(self.dep_embeddings(inputs[2]))
            dir_embed = self.normalize_embeddings(self.dir_embeddings(inputs[3]))
            embeds = torch.cat((word_embed, pos_embed, dep_embed, dir_embed)).view(1, -1)
            lstm_inp = torch.cat((lstm_inp, embeds), 0)

        lstm_inp = lstm_inp.view(-1, 1, self.input_dim)
        print (lstm_inp.shape)
        output, _ = self.lstm(lstm_inp)
        self.cache[path] = output
        print (output.shape)
        return output * count
    
    def forward(self, data, emb_indexer):
        for el in data:
            if not el:
                el[NULL_PATH] = 1
        print ("Data: ", data.shape, emb_indexer.shape)
        num_paths = [sum(list(paths.values())) for paths in data]
        print ("Number of paths: ", num_paths)
        for paths in data:
            for path in paths.items():
                toself.embed_path(path)
        path_embeddings = np.array([np.sum([self.embed_path(path) for path in paths.items()]) for paths in data])
        #print ("Path Embeddings: ", path_embeddings)
        
        h = np.divide(path_embeddings, num_paths)
        print (h.shape)
        h = [np.concatenate((self.word_embeddings(emb[0]), h[i], self.word_embeddings(emb[1]))) for i,emb in enumerate(emb_indexer)]
        return self.softmax(self.W(h))

HIDDEN_DIM = 60
NUM_LAYERS = 2
num_epochs = 3
batch_size = 10

dataset_size = len(y_train)
batch_size = min(batch_size, dataset_size)
num_batches = int(ceil(dataset_size/batch_size))

lr = 0.001
dropout = 0.3
lstm = LSTM()
criterion = nn.NLLLoss()
optimizer = optim.Adam(lstm.parameters(), lr=lr)

for epoch in range(num_epochs):
    
    total_loss, epoch_idx = 0, np.random.permutation(dataset_size)
    
    for batch_idx in range(num_batches):
        batch_end = (batch_idx+1) * batch_size
        batch_start = batch_idx * batch_size
        batch = epoch_idx[batch_start:batch_end]
        
        data, labels, embeddings_idx = x_train[batch], y_train[batch], embed_indices_train[batch]
        
        # Run the forward pass
        outputs = lstm(data, embeddings_idx)
        loss = criterion(outputs, labels)

        # Backprop and perform Adam optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    total_loss /= dataset_size
    print('Epoch [{}/{}] Loss: {:.4f}'.format(epoch + 1, num_epochs, total_loss))
    loss_list.append(loss.item())

lstm.eval()
with torch.no_grad():
    predictedLabels = []
    for batch_idx in range(num_batches):
        outputs = lstm(data)
        print (outputs)
        _, predicted = torch.max(outputs.data, 1)
        predictedLabels.extend(predicted)
