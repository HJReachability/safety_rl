"""
Please contact the author(s) of this library if you have any questions.
Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )

This module implements the replay memory (buffer) for off-policy reinforcement
learning. In this codespace, we use it for double deep Q-network.

This file is based on Adam Paszke's implementation of Replay Memory,
available at:

https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

"""

import numpy as np


class ReplayMemory(object):
  """Contains a replay memory (or a memory buffer).

  Attribures:
      capacity (int): the maximum number of transitions can be stored.
      memory (list): the transitions stored.
      position (list): the index where the new transition will be stored.
      isfull (bool): whether the memory is fully occupied.
      seed (int): the random seed for this memory, which influences the
          sampling method.
  """

  def __init__(self, capacity, seed=0):
    """Initializes the memory with the maximum capacity and a random seed.
    """
    self.capacity = capacity
    self.memory = []
    self.position = 0
    self.isfull = False
    self.seed = seed
    np.random.seed(self.seed)

  def reset(self):
    """Clears the memory and reset the position to be zero.
    """
    self.memory = []
    self.position = 0
    self.isfull = False

  def update(self, transition):
    """Updates the memory given the newcoming transition.
    """
    if len(self.memory) < self.capacity:
      self.memory.append(None)
    self.memory[self.position] = transition
    self.position = int((self.position + 1) % self.capacity)
    if len(self.memory) == self.capacity:
      self.isfull = True

  def sample(self, batch_size):
    """Samples batch_size transitions from the memory uniformly at random.
    """
    length = len(self.memory)
    indices = np.random.randint(low=0, high=length, size=(batch_size,))
    return [self.memory[i] for i in indices]

  def __len__(self):
    """Returns the number of transitions in the memory.
    """
    return len(self.memory)
