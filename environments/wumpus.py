from environment import MyEnv
from enum import Enum
from dataclasses import dataclass
import os
import random

from colorama import Fore, Style, init
init()

false_tup = (False,)*6
map_size = 4
wumpus_map = [[0 for _ in range(map_size)] for _ in range(map_size)]

# TODO: implement movement and only partial vision

class WumpusEnv(MyEnv):
    class Actions(Enum):
        LEFT = 0
        UP = 1
        RIGHT = 2
        DOWN = 3
        SHOOT = 4

    @dataclass
    class Cell:
        stench: bool
        breeze: bool
        pit: bool
        wumpus: bool
        player: bool
        gold: bool

        def default(self):
            self.stench = False
            self.breeze = False
            self.pit = False
            self.wumpus = False
            self.player = False
            self.gold = False

            return self
        
        def plot_str(self):
            parts = []

            if self.stench:
                parts.append(f"{Fore.GREEN}#{Style.RESET_ALL}")
            else:
                parts.append(".")

            if self.breeze:
                parts.append(f"{Fore.CYAN}~{Style.RESET_ALL}")
            else:
                parts.append(".")

            if self.pit:
                parts.append(f"{Fore.RED}P{Style.RESET_ALL}")
            else:
                parts.append(".")

            if self.wumpus:
                parts.append(f"{Fore.MAGENTA}W{Style.RESET_ALL}")
            else:
                parts.append(".")

            if self.player:
                parts.append(f"{Fore.WHITE}@{Style.RESET_ALL}")
            else:
                parts.append(".")

            if self.gold:
                parts.append(f"{Fore.YELLOW}G{Style.RESET_ALL}")
            else:
                parts.append(".")

            return "".join(parts) + '|'
    
    def get_adjacent_cells(self, y, x):
        adjs = []

        # down
        if y+1 <= self.map_size - 1:
            adjs.append(self.map[y+1][x])

        # up
        if y-1 >= 0:
            adjs.append(self.map[y-1][x])

        # right
        if x+1 <= self.map_size - 1:
            adjs.append(self.map[y][x+1])

        # left
        if x-1 >= 0:
            adjs.append(self.map[y][x-1])
        
        return adjs

    def place_wumpus(self):
        available_cells = []
        for row in self.map:
            for cell in row:
                if not cell.pit and not cell.player:
                    available_cells.append(cell)
        
        
        wumpus_cell = random.choice(available_cells)
        wumpus_cell.wumpus = True
    
    def place_gold(self):
        available_cells = []
        for row in self.map:
            for cell in row:
                if not cell.pit and not cell.wumpus and not cell.player:
                    available_cells.append(cell)
                    
        gold_cell = random.choice(available_cells)
        gold_cell.gold = True

    def create_map(self, map_size):
        self.map = [[self.Cell(*false_tup) for _ in range(map_size)] for _ in range(map_size)]
        self.map_size = map_size
        player_y, player_x = map_size-1, 0

        c = self.Cell(*false_tup)
        c = c.default()
        c.player = True
        self.map[player_y][player_x] = c

        # place pits
        for y,row in enumerate(self.map):
            for x,cell in enumerate(row):
                if cell.player:
                    continue
                
                is_pit = random.random() <= self.pit_chance
                if is_pit:
                    cell.pit = True
                    adj_cells = self.get_adjacent_cells(y, x)
                    for c in adj_cells:
                        c.breeze = True
        
        self.place_wumpus()
        self.place_gold()

    def __init__(self, map_size=4, map=None, pit_chance = 0.10):
        super().__init__()
        self.pit_chance = pit_chance

        if not map:
            self.create_map(map_size)
    
    def plot_map(self):
        os.system("clear")

        for row in self.map:
            for cell in row:
                s = cell.plot_str()
                print(s, end='')
                
            print('\n' + '-------' * self.map_size)


    def get_actions(self):
        pass

    def get_initial_state(self):
        pass

    def reset(self):
        pass

    def step(self):
        pass

if __name__ == '__main__':
    print(map)
    wenv = WumpusEnv()
    wenv.plot_map()
