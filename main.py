from config import maze
from environment import MyEnv
from enum import Enum

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import matplotlib.pyplot as plt
from tqdm import tqdm

class FNN(nn.Module):
    def __init__(self, num_hidden_layers=3, num_neurons=64):
        super().__init__()
        self.hidden_layers = []
        
        self.inp_layer = nn.Linear(2,num_neurons) # take in the two coords
        
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(num_neurons, num_neurons) for _ in range(num_hidden_layers)]
        )
        
        self.out_layer = nn.Linear(num_neurons, 4)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.inp_layer(x))
        
        for hd in self.hidden_layers:
            x = self.relu(hd(x))
        
        x = self.out_layer(x)
        
        return x
    
class MazeEnv(MyEnv):
    class Actions(Enum):
        LEFT = 0
        UP = 1
        RIGHT = 2
        DOWN = 3
    
    def get_actions(self):
        return list(self.Actions)
        
    def __init__(self, board=None):
        super().__init__()
        if not board:
            self.board = maze
            
        self.max_steps = 200
        self.reset()
    
    def reset(self):
        self.num_rows = len(self.board)
        self.num_cols = len(self.board[0])
        self.num_steps = 0
        self.player_location = (0,0)
    
    def get_initial_state(self):
        return self.player_location
    
    def is_valid_square(self, y, x):
        # bounds
        if y < 0 or y > self.num_rows - 1:
            return False
        
        # bounds
        if x < 0 or x > self.num_cols - 1:
            return False

        # wall
        if self.board[y][x] == 1:
            return False
        
        return True
        
    def step(self, action):
        self.num_steps += 1
        
        if action not in self.Actions:
            return ValueError("Action not found in list of actions performed.")        

        y,x = self.player_location
        
        if action == self.Actions.LEFT:
            next_pos = (y, x-1)
        elif action == self.Actions.RIGHT:
            next_pos = (y, x+1)
        elif action == self.Actions.UP:
            next_pos = (y-1, x)
        elif action == self.Actions.DOWN:
            next_pos = (y+1, x)
        
        is_valid = self.is_valid_square(*next_pos)
        if is_valid:
            self.player_location = next_pos
        
        new_y, new_x = self.player_location
        
        if self.board[new_y][new_x] == 2:
            reward = 10
            done = True
        else:
            reward = -1
            done = False
        
        if self.num_steps >= self.max_steps:
            done = True
            
        return self.player_location, reward, done  
    
class PolicyGradient:
    def __init__(self):
        self.policy = FNN()
        self.trajectories = []
        self.env = MazeEnv()
        self.gamma = 0.99
        
    def get_total_return(self, trj):
        rewards = [d['reward'] for d in trj]
        G = []
        R = 0.0
        for r in reversed(rewards):
            R = r + self.gamma * R
            G.insert(0, R)
        G = torch.tensor(G, dtype=torch.float32)
        
        # normalize
        G = (G - G.mean()) / (G.std() + 1e-8)

        return G
        
    def run_episode(self):
        trj = []
        log_probs = []
        
        done = False
        state = self.env.get_initial_state()
        
        while not done:
            # follow policy
            logits = self.policy(torch.tensor(state, dtype=torch.float32))
            probs = torch.softmax(logits, dim=-1)
            action_idx = torch.argmax(probs).item()
            
            # get the action
            dist = torch.distributions.Categorical(probs)
            action_idx = dist.sample()                    # tensor
            log_probs.append(dist.log_prob(action_idx))   # OK
            action = self.env.get_actions()[action_idx.item()]  # use .item() here
                        
            # get reward / update state
            state, reward, done = self.env.step(action)
            
            trj.append({'state': state, 'reward': reward, 'done': done})

        self.env.reset()
        
        return trj, log_probs
    
    def get_loss(self, trj, log_probs):
        G = self.get_total_return(trj)
    
        log_probs = torch.stack(log_probs)
        loss = -(log_probs * G).mean()
        
        return loss
        
    def train(self, num_episodes=100000):
        optimizer = optim.Adam(self.policy.parameters(), lr=1e-5)
        steps_taken = []
        plotting_steps = []
        
        plt.ion()
        fig, ax = plt.subplots()
        line, = ax.plot([], [])
        
        for i in tqdm(range(num_episodes)):
            trj, log_probs = self.run_episode()
            num_steps = len(trj)
            steps_taken.append(num_steps)
            
            loss = self.get_loss(trj, log_probs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i + 1) % 100 == 0 and i != 0:
                avg = np.array(steps_taken[:-10]).mean()
                
                plotting_steps.append(avg)
                # inside your loop
                line.set_data(range(len(plotting_steps)), plotting_steps)
                ax.relim()
                ax.autoscale_view()
                plt.pause(0.001)

        plt.ioff()
        plt.show()
    
    def test(self):
        for i in range(5):
            trj, _ = self.run_episode()
            
            for r in trj:
                print(r)
            
            print('total reward:', self.get_total_return(trj))
        

pg = PolicyGradient()
pg.train()
pg.test()     
            
    