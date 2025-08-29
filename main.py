from config import maze
from environment import MyEnv
from enum import Enum

import torch.nn as nn


class MazeEnv(MyEnv):
    class Actions(Enum):
        LEFT: 0
        UP: 1
        RIGHT: 2
        DOWN: 3
        
    def __init__(self, board=None):
        super().__init__()
        self.reset(board)
    
    def reset(self, board):
        # TODO: doesn't account for if board isn't lists inside list
        if not board:
            self.board = maze
        else:
            self.board = board
        
        self.num_rows = len(board)
        self.num_cols = len(board[0])
        
        self.player_location = (0,0)
    
    def step(self, action):
        if action not in self.Actions:
            return ValueError("Invalid action performed.")        
    
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
        
        x = nn.Softmax(self.out_layer(x))
        
        return x
    

class PolicyGradient:
    def __init__(self):
        self.policy = FNN()
        self.trajectories = []
    
    def run_episode():
        pass
    