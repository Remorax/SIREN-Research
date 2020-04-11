from bsddb3 import btopen
import bcolz, pickle, torch, os
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

prefix = "/home/vivek.iyer/security"
train_file = "../files/dataset/train.tsv"
test_file = "../files/dataset/test.tsv"
output_folder = "../junk/Output/"
embeddings_folder = "../junk/Glove.dat"
embeddings_file = "/home/vivek.iyer/glove.6B.300d.txt"
model_filename = "/home/vivek.iyer/SIREN-Research/OntoEnricher/src/model.pt"

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
        write("Emb: " + str(type(embeddings)))
    except:
        embeddings, word2idx  = create_embeddings()
    return embeddings, word2idx
        
def write(statement):
    op_file = open("Logs", "a+")
    op_file.write("\n" + statement + "\n")
    op_file.close()

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
    write("vecsize " + str(vectors.size))
    row_norm = np.sum(np.abs(vectors)**2, axis=-1)**(1./2)
    vectors /= row_norm[:, np.newaxis]
    vectors = bcolz.carray(vectors, rootdir=embeddings_folder, mode='w')
    vectors.flush()

    pickle.dump(words, open(embeddings_folder + 'words.pkl', 'wb'))
    pickle.dump(word2idx, open(embeddings_folder + 'words_index.pkl', 'wb'))
    
    return vectors, word2idx

def load_checkpoint(model, optimizer, filename='model.pt'):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(filename):
        write("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        write("=> loaded checkpoint '{}' (epoch {})".format(filename, checkpoint['epoch']))
    else:
        write("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, start_epoch


word2id_db = btopen(prefix + "_term_to_id.db", "r")
id2word_db = btopen(prefix + "_id_to_term.db", "r")
path2id_db = btopen(prefix + "_path_to_id.db", "r")
id2path_db = btopen(prefix + "_id_to_path.db", "r")
relations_db = btopen(prefix + "_l2r.db", "r")

embeddings, emb_indexer = load_embeddings_from_disk()

write("Embeddings shape: " + str(embeddings.shape))
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
    write('Pairs without paths: ' + str(len(empty)) + str(' , all dataset: ') + str(len(dataset)))
    embed_indices = [(emb_indexer.get(x,0), emb_indexer.get(y,0)) for (x,y) in keys]
    return embed_indices, paths
  
torch.set_default_dtype(torch.float64)

pos_indexer, dep_indexer, dir_indexer = defaultdict(count(0).__next__), defaultdict(count(0).__next__), defaultdict(count(0).__next__)
unk_pos, unk_dep, unk_dir = pos_indexer["#UNKNOWN#"], dep_indexer["#UNKNOWN#"], dir_indexer["#UNKNOWN#"]

dataset_keys = list(train_dataset.keys()) + list(test_dataset.keys())
dataset_vals = list(train_dataset.values()) + list(test_dataset.values())

embed_indices, x = parse_dataset(dataset_keys)
mappingDict = {key: idx for (idx,key) in enumerate(list(set(dataset_vals)))}
# print (mappingDict)
y = [mappingDict[relation] for relation in dataset_vals]

embed_indices_train, embed_indices_test = np.array(embed_indices[:len(train_dataset)]), np.array(embed_indices[len(train_dataset):len(train_dataset)+len(test_dataset)])
x_train, x_test = np.array(x[:len(train_dataset)]), np.array(x[len(train_dataset):len(train_dataset)+len(test_dataset)])
y_train, y_test = np.array(y[:len(train_dataset)]), np.array(y[len(train_dataset):len(train_dataset)+len(test_dataset)])

class LSTM(nn.Module):

    def __init__(self):
        
        super(LSTM, self).__init__()
        self.cache = {}
        
        self.hidden_dim = HIDDEN_DIM + 2 * EMBEDDING_DIM
        self.input_dim = POS_DIM + DEP_DIM + EMBEDDING_DIM + DIR_DIM
        self.W = nn.Linear(self.hidden_dim, NUM_RELATIONS)
        self.dropout_layer = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax()
        
        self.word_embeddings = nn.Embedding(len(emb_indexer), EMBEDDING_DIM)
        self.word_embeddings.load_state_dict({'weight': torch.from_numpy(np.array(embeddings))})
        self.word_embeddings.require_grad = False
        
        self.pos_embeddings = nn.Embedding(len(pos_indexer), POS_DIM)
        self.dep_embeddings = nn.Embedding(len(dep_indexer), DEP_DIM)
        self.dir_embeddings = nn.Embedding(len(dir_indexer), DIR_DIM)
        
        nn.init.xavier_uniform_(self.pos_embeddings.weight)
        nn.init.xavier_uniform_(self.dep_embeddings.weight)
        nn.init.xavier_uniform_(self.dir_embeddings.weight)

        self.lstm = nn.LSTM(self.input_dim, HIDDEN_DIM, NUM_LAYERS)
    
    def normalize_embeddings(self, embeds):
        
        embed = torch.flatten(self.dropout_layer(embeds))
        return embed

    def embed_path(self, elem):
        path, count = elem
        if path in self.cache:
            return self.cache[path] * count
        lstm_inp = torch.Tensor([]).to(device)
        for edge in path:
            word_embed = self.normalize_embeddings(self.word_embeddings(edge[0]))
            pos_embed = self.normalize_embeddings(self.pos_embeddings(edge[1]))
            dep_embed = self.normalize_embeddings(self.dep_embeddings(edge[2]))
            dir_embed = self.normalize_embeddings(self.dir_embeddings(edge[3]))
            # print (word_embed.shape, pos_embed.shape, dep_embed.shape, dir_embed.shape)
            embeds = torch.cat((word_embed, pos_embed, dep_embed, dir_embed)).view(1, -1)
            lstm_inp = torch.cat((lstm_inp, embeds), 0)

        lstm_inp = lstm_inp.view(-1, 1, self.input_dim)
        # print ("LSTM inp:", lstm_inp.shape)
        output, _ = self.lstm(lstm_inp)
        self.cache[path] = output
        # print ("LSTM op:", output.shape)
        return output * count
    
    def forward(self, data, emb_indexer):
        
        # print ("Data: ", data.shape, emb_indexer.shape)
        h = torch.Tensor([]).to(device)
        idx = 0
        for paths in data:
            paths_embeds = torch.Tensor([]).to(device)
            # print (paths)
            for path in paths.items():
                paths_embeds = torch.cat((paths_embeds, self.embed_path(path).view(1,-1)), 0)
                # print ("paths_embeds:", paths_embeds.shape)
            path_embedding = torch.div(torch.sum(paths_embeds, 0), sum(list(paths.values())))
            # print (emb_indexer[idx][0].shape, emb_indexer[idx][1].shape, emb_indexer[idx])
            x = self.word_embeddings(torch.LongTensor([[emb_indexer[idx][0]]]).to(device)).view(EMBEDDING_DIM)
            y = self.word_embeddings(torch.LongTensor([[emb_indexer[idx][1]]]).to(device)).view(EMBEDDING_DIM)
            # print (x.shape, path_embedding.shape, y.shape)
            path_embedding_cat = torch.cat((x, path_embedding, y))
            # print ("Path embedding after cat with embeddings: ", path_embedding.shape)
            probabilities = self.softmax(self.W(path_embedding_cat))
            # print ("Probabilities: ", probabilities)
            h = torch.cat((h, probabilities.view(1,-1)), 0)
            idx += 1
         
        # print ("h shape: ", h.shape)
        return h

def log_loss(output, target):
    prob_losses = torch.Tensor([]).double().to(device)
    for i,batch in enumerate(output):
        # print (i, batch)
        log_prob = -1 * torch.log(batch[target[i]])
        # print (log_prob, log_prob.shape)
        prob_losses = torch.cat((prob_losses, torch.unsqueeze(log_prob, 0)), 0)
    # print (prob_losses, prob_losses.shape)
    return torch.sum(prob_losses)

def tensorifyTuple(tup):
    return tuple([tuple([torch.LongTensor([[e]]).to(device) for e in edge]) for edge in tup])
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

NUM_RELATIONS = len(mappingDict)
# print ("num_relations:", NUM_RELATIONS)
HIDDEN_DIM = 60
NUM_LAYERS = 2
num_epochs = 10
batch_size = 32

dataset_size = len(y_train)
batch_size = min(batch_size, dataset_size)
num_batches = int(ceil(dataset_size/batch_size))

lr = 0.001
dropout = 0.3
lstm = nn.DataParallel(LSTM()).to(device)
criterion = nn.NLLLoss()
optimizer = optim.Adam(lstm.parameters(), lr=lr)

loss_list = []
for epoch in range(num_epochs):
    
    total_loss, epoch_idx = 0, np.random.permutation(dataset_size)
    
    if False:
        lstm, optimizer, curr_epoch = load_checkpoint(lstm, optimizer)
        lstm = lstm.to(device)
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
                    
    for batch_idx in range(num_batches):
        if batch_idx % 100 == 0:
            write("Batch_idx " + str(batch_idx))
        batch_end = (batch_idx+1) * batch_size
        batch_start = batch_idx * batch_size
        batch = epoch_idx[batch_start:batch_end]
        
        # print ("x_train", x_train[batch], "emb", embed_indices_train[batch])
        
        data = [{NULL_PATH: 1} if not el else el for el in x_train[batch]]
        data = [{tensorifyTuple(e): dictElem[e] for e in dictElem} for dictElem in data]
        labels, embeddings_idx = y_train[batch], embed_indices_train[batch]
        
        # Run the forward pass
        outputs = lstm(data, embeddings_idx)
        # print (outputs, labels)
        loss = log_loss(outputs, torch.LongTensor(labels).to(device))
        # loss = criterion(outputs, torch.LongTensor(labels).to(device))
        if batch_idx % 100 == 0:
            write("Loss: " + str(loss.item()))
        # Backprop and perform Adam optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    state = {'epoch': epoch + 1, 'state_dict': lstm.state_dict(),
             'optimizer': optimizer.state_dict()}
    torch.save(state, model_filename)
    
    total_loss /= dataset_size
    write('Epoch [{}/{}] Loss: {:.4f}'.format(epoch + 1, num_epochs, total_loss))
    loss_list.append(loss.item())
lstm.eval()
with torch.no_grad():
    predictedLabels = []
    for batch_idx in range(num_batches):
        outputs = lstm(data)
        print (outputs)
        _, predicted = torch.max(outputs.data, 1)
        predictedLabels.extend(predicted)
