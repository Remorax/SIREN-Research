import bcolz, pickle, torch, os, shelve, sys
import concurrent.futures
import numpy as np
from math import ceil
from itertools import count
from collections import defaultdict
from difflib import SequenceMatcher
import tensorflow as tf
import tensorflow_hub as hub
from scipy import spatial
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import accuracy_score

relations = ["hypernym", "hyponym", "concept", "instance", "none"]
NUM_RELATIONS = len(relations)

mappingDict = {key: idx for (idx,key) in enumerate(relations)}
mappingDict_inv = {idx: key for (idx,key) in enumerate(relations)}

output_folder = "../junk/Output/"
dataset_file = sys.argv[1]
prev_epoch = float(sys.argv[2])
print (prev_epoch)
prefix = "/home/vivek.iyer/"
output_folder = "../junk/Output/glove_vanilla_output/"
model_filename = "/home/vivek.iyer/SIREN-Research/OntoEnricher/src/glove-vanilla.pt"
embeddings_folder = "../junk/Glove.dat/"


if not os.path.isdir(output_folder):  
    os.mkdir(output_folder)
#if os.path.exists("Logs"):
#    os.remove("Logs")
        
def write(statement):
    op_file = open("Logs", "a+")
    op_file.write("\n" + str(statement) + "\n")
    op_file.close()


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

    pickle.dump(word2idx, open(embeddings_folder + 'words_index.pkl', 'wb'))
    
    return vectors, word2idx

def load_embeddings_from_disk():
    try:
        embeddings = bcolz.open(embeddings_folder)[:]
        word2idx = pickle.load(open(embeddings_folder + 'words_index.pkl', 'rb'))
        write("Emb: " + str(type(embeddings)))
    except:
        embeddings, word2idx  = create_embeddings()
    return embeddings, word2idx

POS_DIM = 4
DEP_DIM = 5
DIR_DIM = 1
EMBEDDING_DIM = 300
NULL_PATH = ((0, 0, 0, 0),)


file = open(dataset_file, 'rb')
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
            path_embedding_cat = torch.cat((x, path_embedding, y))
            # print ("Path embedding after cat with embeddings: ", path_embedding.shape)
            probabilities = self.softmax(self.W(path_embedding_cat))
            # print ("Probabilities: ", probabilities)
            h = torch.cat((h, probabilities.view(1,-1)), 0)
            idx += 1
         
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


# print ("num_relations:", NUM_RELATIONS)
HIDDEN_DIM = 60
NUM_LAYERS = 2
num_epochs = 10
batch_size = 5000

dataset_size = len(parsed_train[2])
batch_size = min(batch_size, dataset_size)
num_batches = int(ceil(dataset_size/batch_size))

lr = 0.01
dropout = 0.3
lstm = nn.DataParallel(LSTM()).to(device)
criterion = nn.NLLLoss()
optimizer = optim.AdamW(lstm.parameters(), lr=lr)

                   
loss_list = []

epochs_range = range(prev_epoch, num_epochs) if prev_epoch!=-1 else range(num_epochs)

try:
    for epoch in epochs_range:
        
        total_loss, epoch_idx = 0, np.random.permutation(dataset_size)
        
        if prev_epoch!=-1 and epoch==prev_epoch:
            lstm, optimizer, curr_epoch = load_checkpoint(lstm, optimizer)
            lstm = lstm.to(device)
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
                        
        for batch_idx in range(num_batches):
            
            write("Batch_idx " + str(batch_idx))
            batch_end = (batch_idx+1) * batch_size
            batch_start = batch_idx * batch_size
            batch = epoch_idx[batch_start:batch_end]
            
            # print ("x_train", x_train[batch], "emb", embed_indices_train[batch])
            data = [{NULL_PATH: 1} if not el else el for el in np.array(parsed_train[1])[batch]]
            data = [{tensorifyTuple(e): dictElem[e] for e in dictElem} for dictElem in data]
            labels, embeddings_idx = np.array(parsed_train[2])[batch], np.array(parsed_train[0])[batch]
            
            # Run the forward pass
            outputs = lstm(data, embeddings_idx)
            # print (outputs, labels)
            loss = log_loss(outputs, torch.LongTensor(labels).to(device))
            #loss = criterion(outputs, torch.LongTensor(labels).to(device))
            
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
except Exception as e:
    print (e)
    sys.exit(epoch)

def calculate_recall(true, pred):
    true_f, pred_f = [], []
    for i,l in enumerate(true):
        if l!=4:
            true_f.append(l)
            pred_f.append(pred[i])
    return accuracy_score(true_f, pred_f)

def calculate_precision(true, pred):
    true_f, pred_f = [], []
    for i,l in enumerate(pred):
        if l!=4:
            pred_f.append(l)
            true_f.append(true[i])
    return accuracy_score(true_f, pred_f)

def test(test_dataset, message, output_file):
    predictedLabels, trueLabels = [], []
    results = []
    global mappingDict, mappingDict_inv
    dataset_size = len(test_dataset[2])
    batch_size = min(5000, dataset_size)
    num_batches = int(ceil(dataset_size/batch_size))

    test_perm = np.random.permutation(dataset_size)
    for batch_idx in range(num_batches):
        
        batch_end = (batch_idx+1) * batch_size
        batch_start = batch_idx * batch_size
        batch = test_perm[batch_start:batch_end]

        data = [{NULL_PATH: 1} if not el else el for el in np.array(test_dataset[1])[batch]]
        data = [{tensorifyTuple(e): dictElem[e] for e in dictElem} for dictElem in data]
        labels, embeddings_idx = np.array(test_dataset[2])[batch], np.array(test_dataset[0])[batch]

        outputs = lstm(data, embeddings_idx)
        _, predicted = torch.max(outputs, 1)
        predicted = [el.item() for el in predicted]
        labels = [el.item() for el in labels]
        predictedLabels.extend(predicted)
        trueLabels.extend(labels)
        results.extend(["\t".join(tup) for tup in zip(["\t".join(l) for l in np.array(test_dataset[3])[batch]], [mappingDict_inv[l] for l in predicted], [mappingDict_inv[l] for l in labels])])
    open(output_file, "w+").write("\n".join(results))
    print ("\n\n{}\n\n".format(message))
    
    if output_file!="test_knocked.tsv":
        accuracy = accuracy_score(trueLabels, predictedLabels)
        recall = calculate_recall(trueLabels, predictedLabels)
        precision = calculate_precision(trueLabels, predictedLabels)
        write ("Accuracy:" + str(accuracy) + str( " Precision:") + str(precision) + str(" Recall: ") + str(recall) + str("F1-score: ") + str(2 * (precision * recall/(precision + recall))))
    else:
        write ("Knocked:" + len([l for l in predictedLabels if l!=4]))

lstm.eval()
with torch.no_grad():
    test(parsed_test, "Test Set:", output_folder + "test.tsv")
    test(parsed_instances, "Instance Set:", output_folder + "test_instance.tsv")
    test(parsed_knocked, "Knocked out Set:", output_folder + "test_knocked.tsv")

sys.exit(0)
