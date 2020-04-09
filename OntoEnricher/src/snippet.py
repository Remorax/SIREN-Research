import torch.nn as nn

data = [{}, {}, {}, {}]
labels = [0, 0, 0, 0]

class LSTM(nn.Module):

    def __init__(self):
        
        super(LSTM, self).__init__()
        self.W = nn.Linear(60, 5)
        self.dropout_layer = nn.Dropout(p=0.3)
        self.softmax = nn.LogSoftmax()
        self.embeddings = nn.Embedding(50, 10)
        nn.init.xavier_uniform_(self.embeddings.weight)
        self.lstm = nn.LSTM(310, 60, 2)

    def embed_path(self, path):
        edge, count = path
        inputs = torch.Tensor([[edge]]).long().to(device)
        embed = torch.flatten(self.dropout_layer(self.embeddings(inputs)))
        output, _ = self.lstm(lstm_inp.view(-1, 1, 310))
        return output * count
    
    def forward(self, data):
        for el in data:
            if not el:
                el[0] = 1
        h = torch.Tensor([]).to(device)
        for path in data:
            lstm_output = self.embed_path(path).view(1,-1)
            probabilities = self.softmax(self.W(lstm_output))
            h = torch.cat((h, probabilities.view(1,-1)), 0)
        return h

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

lstm = nn.DataParallel(LSTM()).to(device)
criterion = nn.NLLLoss()
optimizer = optim.Adam(lstm.parameters(), lr=0.001)

for epoch in range(3):

    outputs = lstm(data)
    loss = criterion(outputs, torch.LongTensor(labels))
    
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()

    print ("Loss: ", loss.item())
