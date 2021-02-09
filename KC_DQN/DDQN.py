# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )


import torch
import torch.nn as nn
from torch.nn.functional import mse_loss, smooth_l1_loss
from torch.autograd import Variable
import torch.optim as optim

from collections import namedtuple
import numpy as np
import os
import glob

from .model import StepLR, StepLRMargin
from .ReplayMemory import ReplayMemory


Transition = namedtuple('Transition', ['s', 'a', 'r', 's_', 'info'])
class DDQN():
    def __init__(self, CONFIG):
        self.memory = ReplayMemory(CONFIG.MEMORY_CAPACITY, CONFIG.SEED)

        #== PARAM ==
        # Exploration
        self.EpsilonScheduler = StepLR( initValue=CONFIG.EPSILON, period=CONFIG.EPS_PERIOD,
                                        decay=CONFIG.EPS_DECAY, endValue=CONFIG.EPS_END)
        self.EPSILON = self.EpsilonScheduler.get_variable()
        # Learning Rate
        self.LR_C = CONFIG.LR_C
        self.LR_C_PERIOD = CONFIG.LR_C_PERIOD
        self.LR_C_DECAY = CONFIG.LR_C_DECAY
        self.LR_C_END = CONFIG.LR_C_END
        # NN: batch size, maximal number of NNs stored
        self.BATCH_SIZE = CONFIG.BATCH_SIZE
        self.MAX_MODEL = CONFIG.MAX_MODEL
        self.device = CONFIG.DEVICE
        # Discount Factor
        self.GammaScheduler = StepLRMargin( initValue=CONFIG.GAMMA, period=CONFIG.GAMMA_PERIOD,
                                            decay=CONFIG.GAMMA_DECAY, endValue=CONFIG.GAMMA_END,
                                            goalValue=1.)
        self.GAMMA = self.GammaScheduler.get_variable()
        # Target Network Update
        self.double = CONFIG.DOUBLE
        self.TAU = CONFIG.TAU
        self.HARD_UPDATE = CONFIG.HARD_UPDATE # int, update period
        self.SOFT_UPDATE = CONFIG.SOFT_UPDATE # bool


    def build_network(self):
        raise NotImplementedError


    def build_optimizer(self):
        # self.optimizer = optim.Adam(self.Q_network.parameters(), lr=self.LR_C)
        self.optimizer = torch.optim.AdamW(self.Q_network.parameters(), lr=self.LR_C, weight_decay=1e-3)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer,
            step_size=self.LR_C_PERIOD, gamma=self.LR_C_DECAY)
        self.max_grad_norm = 1
        self.cntUpdate = 0


    def update(self):
        raise NotImplementedError


    def initBuffer(self, env):
        raise NotImplementedError


    def initQ(self):
        raise NotImplementedError


    def learn(self):
        raise NotImplementedError


    def update_target_network(self):
        if self.SOFT_UPDATE:
            # Soft Replace
            for module_tar, module_pol in zip(self.target_network.modules(), self.Q_network.modules()):
                if isinstance(module_tar, nn.Linear):
                    module_tar.weight.data = (1-self.TAU)*module_tar.weight.data + self.TAU*module_pol.weight.data
                    module_tar.bias.data   = (1-self.TAU)*module_tar.bias.data   + self.TAU*module_pol.bias.data
        elif self.cntUpdate % self.HARD_UPDATE == 0:
            # Hard Replace
            self.target_network.load_state_dict(self.Q_network.state_dict())


    def updateHyperParam(self):
        if self.optimizer.state_dict()['param_groups'][0]['lr'] <= self.LR_C_END:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.LR_C_END
        else:
            self.scheduler.step()

        self.EpsilonScheduler.step()
        self.EPSILON = self.EpsilonScheduler.get_variable()
        self.GammaScheduler.step()
        self.GAMMA = self.GammaScheduler.get_variable()


    def select_action(self):
        raise NotImplementedError


    def store_transition(self, *args):
        self.memory.update(Transition(*args))


    def save(self, step, logs_path):
        os.makedirs(logs_path, exist_ok=True)
        model_list =  glob.glob(os.path.join(logs_path, '*.pth'))
        #print(model_list)
        if len(model_list) > self.MAX_MODEL - 1 :
            min_step = min([int(li.split('/')[-1][6:-4]) for li in model_list])
            os.remove(os.path.join(logs_path, 'model-{}.pth' .format(min_step)))
        logs_path = os.path.join(logs_path, 'model-{}.pth' .format(step))
        torch.save(self.Q_network.state_dict(), logs_path)
        print('  => Save {} after [{}] updates' .format(logs_path, step))


    def restore(self, logs_path):
        self.Q_network.load_state_dict(
            torch.load(logs_path, map_location=torch.device('cpu')))
        self.target_network.load_state_dict(
            torch.load(logs_path, map_location=torch.device('cpu')))
        print('  => Restore {}' .format(logs_path))


    # ! Deprecated method, do not use
    def updateEpsilon(self):
        if self.cntUpdate % self.EPS_PERIOD == 0 and self.cntUpdate != 0:
            self.EPSILON = max(self.EPSILON*self.EPS_DECAY, self.EPS_END)


    # ! Deprecated method, do not use
    def updateGamma(self):
        if self.cntUpdate % self.GAMMA_PERIOD == 0 and self.cntUpdate != 0:
            self.GAMMA = min(1 - (1-self.GAMMA) * self.GAMMA_DECAY, self.GAMMA_END)