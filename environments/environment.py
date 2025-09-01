from abc import ABC, abstractmethod

class MyEnv(ABC):
    @property
    @abstractmethod
    def Actions(self):
        pass
    
    @abstractmethod
    def get_actions(self):
        pass
    
    @abstractmethod
    def get_initial_state(self):
        pass
    
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, action):
        pass