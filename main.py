from config import maze
import torch.nn as nn
from torch.autograd import Function

class FNN(nn.Module):
    def __init__(self, num_hidden_layers=3, num_neurons=64):
        super().__init__()
        self.hidden_layers = []
        
        self.inp_layer = nn.Linear(2,num_neurons) # take in the two coords
        
        for _ in range(num_hidden_layers):
            self.hidden_layers.append(nn.Linear(num_neurons,num_neurons))
        
        self.out_layer = nn.Linear(num_neurons, 4)
    
    def forward(self, x):
        x = nn.ReLU(self.inp_layer)
        
        for hd in self.hidden_layers:
            x = nn.ReLU(hd(x))
        
        x = nn.Softmax(self.out_layer)
        
        return x
    

class PolicyGradient:
    def __init__(self):
        pass
    