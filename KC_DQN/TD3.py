# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )

import torch
import torch.nn as nn
from torch.nn.functional import mse_loss, smooth_l1_loss

from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
import os
import time

from .model import GaussianPolicy, DeterministicPolicy, TwinnedQNetwork
from .ActorCritic import ActorCritic, Transition

class TD3(ActorCritic):
    def __init__(self, CONFIG, actionSpace, dimList, actType=['Tanh', 'Tanh'],
        verbose=True):
        """
        __init__: initialization.

        Args:
            CONFIG (Class object): hyper-parameter configuration.
            actionSpace (Class object): consists of `high` and `low` attributes.
            dimList (list): consists of dimension lists
            actType (list, optional): consists of activation types.
                Defaults to ['Tanh', 'Tanh'].
            verbose (bool, optional): print info or not. Defaults to True.
        """        
        super(TD3, self).__init__(actorType='TD3', CONFIG)

        #== ENV PARAM ==
        self.actionSpace = actionSpace

        #== Build NN for (D)DQN ==
        assert dimList is not None, "Define the architectures"
        self.dimList = dimList
        self.actType = actType
        self.build_network(dimList, actType, verbose)


    def build_actor(self, dimList, actType='Tanh'):
        self.actor = DeterministicPolicy(dimList, actType)
        self.actorTarget = DeterministicPolicy(dimList, actType)
        pass


    def initBuffer(self, env, ratio=1.):
        pass


    def initQ(self):
        pass


    def update_critic(self, addBias=False):
        pass


    def update_actor(self):
        pass