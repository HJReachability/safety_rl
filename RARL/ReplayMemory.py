"""
Please contact the author(s) of this library if you have any questions.
Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )
"""

import numpy as np


class ReplayMemory(object):

    def __init__(self, capacity, seed=0):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.isfull = False
        self.seed = seed
        np.random.seed(self.seed)

    def reset(self):
        self.memory = []
        self.position = 0
        self.isfull = False

    def update(self, transition):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition
        self.position = int((self.position + 1) % self.capacity)
        if len(self.memory) == self.capacity:
            self.isfull = True

    def sample(self, batch_size):
        length = len(self.memory)
        indices = np.random.randint(low=0, high=length, size=(batch_size,))
        return [self.memory[i] for i in indices]

    def __len__(self):
        return len(self.memory)
