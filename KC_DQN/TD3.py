# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )

import torch
import torch.nn as nn
from torch.nn.functional import mse_loss, smooth_l1_loss
from torch.nn.utils import clip_grad_norm_

from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
import os
import time

from .model import DeterministicPolicy
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
        super(TD3, self).__init__(actorType='TD3', CONFIG, actionSpace)

        #== Build NN for (D)DQN ==
        assert dimList is not None, "Define the architectures"
        self.dimList = dimList
        self.actType = actType
        self.build_network(dimList, actType, verbose)


    def build_actor(self, dimList, actType='Tanh'):
        self.actor = DeterministicPolicy(dimList, actType)
        self.actorTarget = DeterministicPolicy(dimList, actType)


    def initBuffer(self, env, ratio=1.):
        cnt = 0
        while len(self.memory) < self.memory.capacity * ratio:
            cnt += 1
            print('\rWarmup Buffer [{:d}]'.format(cnt), end='')
            s = env.reset()
            a, _, _ = self.select_action(s, explore=True)
            s_, r, done, info = env.step(a)
            if done:
                s_ = None
            self.store_transition(s, a, r, s_, info)
        print(" --- Warmup Buffer Ends")


    def initQ(self, env, warmupIter, outFolder, num_warmup_samples=200,
                vmin=-1, vmax=1, plotFigure=True, storeFigure=True):
        for ep_tmp in range(warmupIter):
            print('\rWarmup Q [{:d}]'.format(ep_tmp+1), end='')
            states, value = env.get_warmup_examples(num_warmup_samples)
            actions = self.genRandomActions(num_warmup_samples)

            self.critic.train()
            value = torch.from_numpy(value[:, 0]).float().to(self.device)
            stateTensor = torch.from_numpy(states).float().to(self.device)
            actionTensor = torch.from_numpy(actions).float().to(self.device)
            q1, q2 = self.critic(stateTensor, actionTensor)
            q1Loss = smooth_l1_loss(input=q1, target=value)
            q2Loss = smooth_l1_loss(input=q2, target=value)
            loss = q1Loss + q2Loss

            self.criticOptimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.criticOptimizer.step()

        print(" --- Warmup Q Ends")
        if plotFigure or storeFigure:
            self.critic.eval()
            env.visualize(self.critic.q1, vmin=vmin, vmax=vmax, cmap='seismic')
            if storeFigure:
                figureFolder = '{:s}/figure/'.format(outFolder)
                os.makedirs(figureFolder, exist_ok=True)
                plt.savefig('{:s}initQ.png'.format(figureFolder))
            if plotFigure:
                plt.show()
                plt.pause(0.001)
                plt.close()

        # hard replace
        self.criticTarget.load_state_dict(self.critic.state_dict())
        del self.criticOptimizer
        self.build_optimizer()


    def update_critic(self, addBias=False):
        pass


    def update_actor(self):
        pass