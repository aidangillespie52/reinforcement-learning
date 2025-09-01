from .environment import MyEnv
from enum import Enum

maze = \
[
    [0,0,0,1,0,0],
    [0,0,0,1,0,2],
    [0,0,0,0,0,0],
]

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
            reward = 2
            done = True
        else:
            reward = -1
            done = False
        
        if self.num_steps >= self.max_steps:
            done = True
            
        return self.player_location, reward, done  