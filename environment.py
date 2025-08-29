from abc import ABC, abstractmethod

class MyEnv(ABC):
    @property
    @abstractmethod
    def Actions(self):
        """Return the Enum of valid actions"""
        pass
    
    @abstractmethod
    def get_actions(self):
        """Return iterable of valid actions"""
        pass
    
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, action):
        pass