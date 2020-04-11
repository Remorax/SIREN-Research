import torch.nn as nn
import torch
import torch.optim as optim

data = [(), (), (), (4,5)]
labels = [0, 0, 0, 0]

class LSTM(nn.Module):

    def __init__(self):
        
        super(LSTM, self).__init__()
        self.W = nn.Linear(60, 5)
        self.dropout_layer = nn.Dropout(p=0.3)
        self.softmax = nn.LogSoftmax()
        self.embeddings = nn.Embedding(10, 50)
        nn.init.xavier_uniform_(self.embeddings.weight)
        self.lstm = nn.LSTM(50, 60, 2)

    def embed_path(self, path):
        edge, count = path
        embed = torch.flatten(self.dropout_layer(self.embeddings(edge)))
        output, _ = self.lstm(embed.view(-1, 1, 50))
        return output * count
    
    def forward(self, data, h):
        print (data)
        for path in data:
            lstm_output = self.embed_path(path).view(1,-1)
            probabilities = self.softmax(self.W(lstm_output))
            h = torch.cat((h, probabilities.view(1,-1)), 0)
        return h

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

lstm = nn.DataParallel(LSTM()).to(device)
criterion = nn.NLLLoss()
optimizer = optim.Adam(lstm.parameters(), lr=0.001)

data = [(0,1) if not el else el for el in data]
data = [(torch.Tensor([[edge]]).long().to(device), count) for edge, count in data]

for epoch in range(3):

    h = torch.Tensor([]).to(device)
    outputs = lstm(data, h)
    loss = criterion(outputs, torch.LongTensor(labels).to(device))
    
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()

    print ("Loss: ", loss.item())
