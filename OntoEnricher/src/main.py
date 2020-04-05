from bsddb3 import btopen
import bcolz, pickle
import numpy as np
from math import ceil
from itertools import count
from collections import defaultdict

prefix = "../junk/Files/temp_threshold_3_4/temp"
train_file = "../junk/train.tsv"
test_file = "../junk/test.tsv"
output_folder = "../junk/Output/"
embeddings_folder = "../junk/Glove.dat"
embeddings_file = "/Users/vivek/SIREN-Research/Archive-LSTM/glove.6B/glove.6B.300d.txt"

POS_DIM = 4
DEP_DIM = 5
DIR_DIM = 1
EMBEDDING_DIM = 300
NULL_PATH = 
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

        embeddings = {w: vectors[word2idx[w]] for w in words}
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
    
    embeddings = {w: vectors[word2idx[w]] for w in words}
    return embeddings, word2idx

word2id_db = btopen(prefix + "_word_to_id.db", "r")
id2word_db = btopen(prefix + "_id_to_word.db", "r")
path2id_db = btopen(prefix + "_path_to_id.db", "r")
id2path_db = btopen(prefix + "_id_to_path.db", "r")
relations_db = btopen(prefix + "_word_occurence_map.db", "r")

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
    x_word, y_word = id_to_entity(id2word_db, x), id_to_entity(id2word_db, y)
    path_count_dict = { id_to_entity(id2path_db, path).replace("X/", x_word+"/").replace("Y/", y_word+"/") : freq for (path, freq) in paths }
    path_count_dict = { parse_path(path) : path_count_dict[path] for path in path_count_dict }

    return { path : path_count_dict[path] for path in path_count_dict if path}
    
def parse_dataset(dataset):
    keys = [(entity2id(word2id_db, x), entity2id(word2id_db, y)) for (x, y) in dataset]
    paths = [extract_all_paths(x,y) for (x,y) in keys]
    empty = [list(dataset)[i] for i, path_list in enumerate(paths) if len(list(path_list.keys())) == 0]
    print('Pairs without paths:', len(empty), ', all dataset:', len(dataset))
    embed_indices = [(embeddings.get(x,0), embeddings.get(y,0)) for (x,y) in keys]
    return embed_indices, paths
    
pos_indexer, dep_indexer, dir_indexer = defaultdict(count(0).__next__), defaultdict(count(0).__next__), defaultdict(count(0).__next__)
unk_pos, unk_dep, unk_dir = pos_indexer["#UNKNOWN#"], dep_indexer["#UNKNOWN#"], dir_indexer["#UNKNOWN#"]

dataset_keys = train_dataset.keys() + test_dataset.keys()
dataset_vals = train_dataset.values() + test_dataset.values()

embed_indices, x = parse_dataset(dataset_keys)
y = [i for (i,relation) in enumerate(dataset_vals)]

embed_indices_train, embed_indices_test = embed_indices[:len(train_dataset)], embed_indices[len(train_dataset):len(train_dataset)+len(test_dataset)]
x_train, x_test = x[:len(train_dataset)], x[len(train_dataset):len(train_dataset)+len(test_dataset)]
y_train, y_test = y[:len(train_dataset)], y[len(train_dataset):len(train_dataset)+len(test_dataset)]

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
        self.word_embeddings.load_state_dict({'weight': embeddings})
        self.word_embeddings.require_grad = False
        
        self.pos_embeddings = nn.Embedding(len(pos_indexer), POS_DIM)
        self.dep_embeddings = nn.Embedding(len(dep_indexer), DEP_DIM)
        self.dir_embeddings = nn.Embedding(len(dir_indexer), DIR_DIM)
        
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, NUM_LAYERS)
    
    def embed_path(self, elem):
        path, count = elem
        if path in self.cache:
            return cache[path] * count
        
        word_embed = self.dropout_layer(self.word_embeddings(elem[0]))
        pos_embed = self.dropout_layer(self.pos_embeddings(elem[1]))
        dep_embed = self.dropout_layer(self.dep_embeddings(elem[2]))
        dir_embed = self.dropout_layer(self.dir_embeddings(elem[2]))
        
        embeds = np.concatenate((word_embed, pos_embed, dep_embed, dir_embed))
        output, _ = self.lstm(embeds)
        cache[path] = output

        return output * count
    
    def forward(self, data, emb_indexer):
        if not data:
            data[NULL_PATH] = 1
        
        num_paths = [sum(list(paths.values())) for paths in data]
        path_embeddings = [np.sum([self.embed_path(path) for path in paths.items()]) for paths in data]
        
        h = np.divide(path_embeddings, num_paths)
        h = [np.concatenate((self.word_embeddings(elem[0]), h[i], self.word_embeddings(elem[1]))) for i,emb in enumerate(emb_indexer)]
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