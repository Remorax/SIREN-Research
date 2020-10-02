import pickle, torch, os, sys, random
import numpy as np
from math import ceil
import torch.optim as optim
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from sklearn.metrics import accuracy_score

dataset_file = os.path.abspath("../Input/" + sys.argv[1])
results_file = os.path.abspath("../Results/" + sys.argv[2])
model_file =  os.path.abspath("../Models/" + sys.argv[3])

f = open(dataset_file, "rb")
(nodes_train, paths_train, counts_train, targets_train, 
 nodes_test, paths_test, counts_test, targets_test,
 nodes_instances, paths_instances, counts_instances, targets_instances,
 nodes_knocked, paths_knocked, counts_knocked, targets_knocked,
 emb_indexer, emb_indexer_inv, emb_vals, 
 pos_indexer, dep_indexer, dir_indexer, rel_indexer) = pickle.load(f)

op_file = open(results_file, "w+")

def distributed_main(batch_size, gpu_ids, use_flatten=True):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '5555'
    mp.spawn(
        distributed_warning,
        nprocs=len(gpu_ids),
        args=(batch_size, len(gpu_ids), use_flatten))


def write(statement):
    global op_file
    op_file.write("\n" + str(statement) + "\n")

POS_DIM = 4
DEP_DIM = 6
DIR_DIM = 3
NUM_RELATIONS = len(rel_indexer)
NULL_EDGE = [0, 0, 0, 0]

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

flatten = lambda l: [item for sublist in l for item in sublist]

class RelationPredictor(nn.Module):

    def __init__(self, emb_vals):
        
        super(RelationPredictor, self).__init__()

        self.EMBEDDING_DIM = np.array(emb_vals).shape[1]
        self.n_directions = 2 if bidirectional else 1
        
        self.input_dim = POS_DIM + DEP_DIM + self.EMBEDDING_DIM + DIR_DIM
        self.output_dim = self.n_directions * HIDDEN_DIM + 2 * self.EMBEDDING_DIM
        # self.layer1_dim = LAYER1_DIM
        # self.W1 = nn.Linear(self.hidden_dim, self.layer1_dim)
        # self.W2 = nn.Linear(self.layer1_dim, NUM_RELATIONS)

        self.dropout_layer = nn.Dropout(p=dropout)
        self.log_softmax = nn.LogSoftmax()
        
        self.name_embeddings = nn.Embedding(len(emb_vals), self.EMBEDDING_DIM)
        self.name_embeddings.load_state_dict({'weight': torch.from_numpy(np.array(emb_vals))})
        self.name_embeddings.weight.requires_grad = False

        self.pos_embeddings = nn.Embedding(len(pos_indexer), POS_DIM)
        self.dep_embeddings = nn.Embedding(len(dep_indexer), DEP_DIM)
        self.dir_embeddings = nn.Embedding(len(dir_indexer), DIR_DIM)

        nn.init.xavier_uniform_(self.pos_embeddings.weight)
        nn.init.xavier_uniform_(self.dep_embeddings.weight)
        nn.init.xavier_uniform_(self.dir_embeddings.weight)
        
        self.lstm = nn.LSTM(self.input_dim, HIDDEN_DIM, NUM_LAYERS, bidirectional=bidirectional, batch_first=True)

        self.W = nn.Linear(self.output_dim, NUM_RELATIONS)

    def forward(self, nodes, paths, counts, edgecounts, max_paths, max_edges):
        '''
            nodes: batch_size * 2
            paths: batch_size * max_paths * max_edges * 4
            counts: batch_size * max_paths
            edgecounts: batch_size * max_paths
        '''
        word_embed = self.dropout_layer(self.name_embeddings(paths[:,:,:,0]))
        pos_embed = self.dropout_layer(self.pos_embeddings(paths[:,:,:,1]))
        dep_embed = self.dropout_layer(self.dep_embeddings(paths[:,:,:,2]))
        dir_embed = self.dropout_layer(self.dir_embeddings(paths[:,:,:,3]))
        paths_embed = torch.cat((word_embed, pos_embed, dep_embed, dir_embed), dim=-1)
        nodes_embed = self.dropout_layer(self.name_embeddings(nodes)).reshape(-1, 2*self.EMBEDDING_DIM)

        paths_embed = paths_embed.reshape((-1, max_edges, self.input_dim))

        paths_packed = pack_padded_sequence(paths_embed, torch.flatten(edgecounts), batch_first=True, enforce_sorted=False)
        _, (hidden_state, _) = self.lstm(paths_packed)
        paths_output = hidden_state.permute(1,2,0)
        paths_output_reshaped = paths_output.reshape(-1, max_paths, HIDDEN_DIM*NUM_LAYERS*self.n_directions)
        # paths_output has dim (batch_size, max_paths, HIDDEN_DIM, NUM_LAYERS*self.n_directions)

        paths_weighted = torch.bmm(paths_output_reshaped.permute(0,2,1), counts.unsqueeze(-1)).squeeze(-1)
        representation = torch.cat((nodes_embed, paths_weighted), dim=-1)
        probabilities = self.log_softmax(self.W(representation))
        return probabilities

def to_list(seq):
    for item in seq:
        if isinstance(item, tuple):
            yield list(to_list(item))
        elif isinstance(item, list):
            yield [list(to_list(elem)) for elem in item]
        else:
            yield item

def pad_paths(paths, max_paths, max_edges):
    paths_edgepadded = [[path + [NULL_EDGE for i in range(max_edges-len(path))]
        for path in element]
    for element in paths]
    NULL_PATH = [NULL_EDGE for i in range(max_edges)]
    paths_padded = [element + [NULL_PATH for i in range(max_paths-len(element))] 
        for element in paths_edgepadded]
    return np.array(paths_padded)
        
def pad_counts(counts, max_paths):
    return np.array([elem + [0 for i in range(max_paths - len(elem))] for elem in counts])

def pad_edgecounts(edgecounts, max_paths):
    return np.array([elem + [1 for i in range(max_paths - len(elem))] for elem in edgecounts])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

HIDDEN_DIM = 250
# LAYER1_DIM = 120
NUM_LAYERS = 1
num_epochs = 50
batch_size = 32
bidirectional = True

lr = 0.001
dropout = 0.3
weight_decay = 0.001

model = nn.DataParallel(RelationPredictor(emb_vals)).to(device)
criterion = nn.NLLLoss()
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

for epoch in range(num_epochs):
    
    all_inp = list(zip(nodes_train, paths_train, counts_train, targets_train))
    all_inp_shuffled = random.sample(all_inp, len(all_inp))
    nodes_train, paths_train, counts_train, targets_train = list(zip(*all_inp_shuffled))

    num_edges_all = [[len(path) for path in element] for element in paths_train]
    max_edges = max(flatten(num_edges_all))
    max_paths = max([len(elem) for elem in counts_train])

    dataset_size = len(nodes_train)
    batch_size = min(batch_size, dataset_size)
    num_batches = int(ceil(dataset_size/batch_size))

    for batch_idx in range(num_batches):
        
        batch_start = batch_idx * batch_size
        batch_end = (batch_idx+1) * batch_size
        
        nodes = torch.LongTensor(nodes_train[batch_start:batch_end]).to(device)
        paths = torch.LongTensor(pad_paths(paths_train[batch_start:batch_end], max_paths, max_edges)).to(device)
        counts = torch.DoubleTensor(pad_counts(counts_train[batch_start:batch_end], max_paths)).to(device)
        edgecounts = torch.LongTensor(pad_edgecounts(num_edges_all[batch_start:batch_end], max_paths)).to(device)
        targets = torch.LongTensor(targets_train[batch_start:batch_end]).to(device)
        
        # Backprop and perform Adam optimisation
        optimizer.zero_grad()

        # Run the forward pass
        outputs = model(nodes, paths, counts, edgecounts, max_paths, max_edges)

        #loss = log_loss(outputs, torch.LongTensor(labels).to(device))
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        write("Epoch: {} Idx: {} Loss: {}".format(epoch, batch_idx, loss.item()))  

write("Training Complete!")

def calculate_recall(true, pred):
    true_f, pred_f = [], []
    for i,elem in enumerate(true):
        if elem!=4:
            true_f.append(elem)
            pred_f.append(pred[i])
    return accuracy_score(true_f, pred_f)

def calculate_precision(true, pred):
    true_f, pred_f = [], []
    for i,elem in enumerate(pred):
        if elem!=4:
            pred_f.append(elem)
            true_f.append(true[i])
    return accuracy_score(true_f, pred_f)

def test(nodes_test, paths_test, counts_test, targets_test, message):
    predictedLabels, trueLabels = [], []

    num_edges_all = [[len(path) for path in element] for element in paths_test]
    max_edges = max(flatten(num_edges_all))
    max_paths = max([len(elem) for elem in counts_test])

    dataset_size = len(nodes_test)
    batch_size = min(32, dataset_size)
    num_batches = int(ceil(dataset_size/batch_size))

    for batch_idx in range(num_batches):
        
        batch_start = batch_idx * batch_size
        batch_end = (batch_idx+1) * batch_size

        nodes = torch.LongTensor(nodes_test[batch_start:batch_end])
        paths = torch.LongTensor(pad_paths(paths_test[batch_start:batch_end], max_paths, max_edges))
        counts = torch.DoubleTensor(pad_counts(counts_test[batch_start:batch_end], max_paths))
        edgecounts = torch.LongTensor(pad_edgecounts(num_edges_all[batch_start:batch_end], max_paths))
        targets = torch.LongTensor(targets_test[batch_start:batch_end])
        
        outputs = model(nodes, paths, counts, edgecounts, max_paths, max_edges)
        _, predicted = torch.max(outputs, 1)
        predicted = [el.item() for el in predicted]
        targets = [el.item() for el in targets]
        predictedLabels.extend(predicted)
        trueLabels.extend(targets)
    
    accuracy = accuracy_score(trueLabels, predictedLabels)
    recall = calculate_recall(trueLabels, predictedLabels)
    precision = calculate_precision(trueLabels, predictedLabels)
    final_metrics = [accuracy, precision, recall, 2 * (precision * recall/(precision + recall))]
    write("Final Results ({}): [{}]".format(message, ", ".join([str(el) for el in final_metrics])))

model.eval()
with torch.no_grad():
    test(nodes_test, paths_test, counts_test, targets_test, "Test")
    test(nodes_instances, paths_instances, counts_instances, targets_instances, "Instances")
    test(nodes_knocked, paths_knocked, counts_knocked, targets_knocked, "Knocked out")

op_file.close()