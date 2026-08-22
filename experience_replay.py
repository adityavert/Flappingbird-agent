from collections import deque
import random

class ReplayMemory():
    #Create FIFO queue - Experience Replay
    def __init__(self,maxlen,seed=None):
        self.memory=deque([],maxlen=maxlen)

    def append(self,new_exp):
        self.memory.append(new_exp)

    def sample(self,sample_size):
        return random.sample(self.memory,sample_size)

    #Current buffer size
    def __len__(self):
        return len(self.memory)
