import torch
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self,input_size = (28 * 28), output_size = 10, hidden_layers = [512,512]):
        super(Network, self).__init__()
        
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        self.hidden_layers.extend([nn.Linear(h1,h2) for h1,h2 in zip(hidden_layers[:-1], hidden_layers[1:])])
        self.output = nn.Linear(hidden_layers[-1], output_size)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # adding hidden layer with relu activation function and dropout
        for each in self.hidden_layers:
            x = F.relu(each(x))
            x = self.dropout(x)
        x = self.output(x)
        return x
        
        
        
        
