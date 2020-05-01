import bcolz, pickle, torch, os, shelve
import concurrent.futures
import numpy as np
from math import ceil
from itertools import count
from collections import defaultdict
from difflib import SequenceMatcher
from scipy import spatial
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import accuracy_score
from skorch import NeuralNetClassifier

relations = ["hypernym", "hyponym", "concept", "instance", "none"]
NUM_RELATIONS = len(relations)

mappingDict = {key: idx for (idx,key) in enumerate(relations)}
mappingDict_inv = {idx: key for (idx,key) in enumerate(relations)}

prefix = "/home/vivek.iyer/"
output_folder = "../junk/Output/"
embeddings_folder = "../junk/Glove.dat"
embeddings_file = "/home/vivek.iyer/glove.6B.300d.txt"
model_filename = "/home/vivek.iyer/SIREN-Research/OntoEnricher/src/baseline_debugged.pt"


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
    op_file.write("\n" + str(statement) + "\n")
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

def load_checkpoint(model, optimizer, filename=model_filename):
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


POS_DIM = 4
DEP_DIM = 5
DIR_DIM = 1
EMBEDDING_DIM = 300
NULL_PATH = ((0, 0, 0, 0),)

file = open("dataset_parsed.pkl", 'rb')
parsed_train, parsed_test, parsed_instances, parsed_knocked, pos_indexer, dep_indexer, dir_indexer  = pickle.load(file)
relations = ["hypernym", "hyponym", "concept", "instance", "none"]
NUM_RELATIONS = len(relations)

embeddings, emb_indexer = load_embeddings_from_disk()

torch.set_default_dtype(torch.float64)

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
        output, _ = self.lstm(lstm_inp)
        output = output[-1]
        self.cache[path] = output
        return output * count
    
    def forward(self, data, emb_indexer):
        
        # print ("Data: ", data.shape, emb_indexer.shape)
        h = torch.Tensor([]).to(device)
        idx = 0
        for paths in data:
            paths_embeds = torch.Tensor([]).to(device)
            for path in paths.items():
                emb = self.embed_path(path).view(1,-1)
                paths_embeds = torch.cat((paths_embeds, emb), 0)
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
         
        print ("h shape: ", h.shape)
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

parsed_train = tuple(el[:1000] for el in parsed_train)

data = [{NULL_PATH: 1} if not el else el for el in np.array(parsed_train[1])]
data = [{tensorifyTuple(e): dictElem[e] for e in dictElem} for dictElem in data]
labels, embeddings_idx = np.array(parsed_train[2]), np.array(parsed_train[0])

lstm = nn.DataParallel(LSTM()).to(device)
net = NeuralNetClassifier(
    lstm,
    criterion=nn.NLLLoss,
    optimizer=Adam,
    max_epochs=5,
    lr=0.001,
    device='cuda',  # uncomment this to train with CUDA
)

# X_dict = {'X': data, 'length': X_len}
net.fit((data, embeddings_idx), labels)

data_test = [{NULL_PATH: 1} if not el else el for el in np.array(parsed_test[1])]
data_test = [{tensorifyTuple(e): dictElem[e] for e in dictElem} for dictElem in data_test]
labels_test, embeddings_idx_test = np.array(parsed_test[2]), np.array(parsed_test[0])

y_test = net.predict((data_test, embeddings_idx_test))


def calculate_precision(true, pred):
    true_f, pred_f = [], []
    for l in true:
        if l!=4:
            true_f.append(l)
            pred_f.append(l)
    return accuracy_score(true_f, pred_f)

print ("Accuracy score: ", accuracy_score(labels_test, y_test))
print ("Precision score: ", calculate_precision(labels_test, y_test))