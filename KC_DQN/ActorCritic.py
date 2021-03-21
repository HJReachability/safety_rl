# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )

# ? how to deal with the discrete action space since SAC and TD3 are proposed
# ? for continuous action.
# Jaime: Just let the agent learn in the continuous action space. We expect
# that it will automatically catch the fact that it should always adopt the
# maximal speed.

# TODO
# * Objects
#   - Twinned Q-Network: note that here we take an extra input, i.e., action
#       + Q1, Q2
#       + target networks
#   - Actor Model
#       + SAC: GaussianPolicy,      without target, care entropy loss
#       + TD3: DeterministicPolicy, with target
#       + select_action() calls self.actor.sample()
#   - Replay Buffer
#       + (extension) prioritized experience replay
#   - Hyper-Parameter Scheduler
#       + learning rate: Q1, Q2, Actor
#       + contraction factor: gamma
#       + exploration-exploitation trade-off: epsilon
#   - Optimizer: Q1, Q2, Actor

# * Functions
#   - build_network:
#       + build_actor -> in child class
#       + build_critic (o)
#       + build_optimizer (o)
#   - learn
#       + initBuffer -> in child class
#       + initQ -> in child class
#       + update_critic -> in child class
#       + update_actor -> in child class
#       + update_target_network (o)
#       + update_critic_hyperParam (o)
#       + update_actor_hyperParam (o)
#   - Others
#       + __init__
#       + store_transition (o)
#       + select_action (o)
#       + save (o)
#       + restore (o)


import torch
import torch.nn as nn
from torch.nn.functional import mse_loss, smooth_l1_loss
from torch.optim import AdamW
from torch.optim import lr_scheduler

from collections import namedtuple
import numpy as np
import os
import glob

from .model import StepLR, StepLRMargin, GaussianPolicy, DeterministicPolicy, TwinnedQNetwork
from .ReplayMemory import ReplayMemory
from .utils import soft_update, save_model

Transition = namedtuple('Transition', ['s', 'a', 'r', 's_', 'info'])


class ActorCritic(object):
    def __init__(self, actorType, CONFIG):
        """
        __init__ : initializes actor-critic model.

        Args:
            actorType (str): The type of actor model, Currently supports SAC, TD3.
            CONFIG (dict): configurations.
        """
        self.actorType = actorType
        self.memory = ReplayMemory(CONFIG.MEMORY_CAPACITY)

        #== PARAM ==
        # Exploration
        self.EpsilonScheduler = StepResetLR( initValue=CONFIG.EPSILON, 
            period=CONFIG.EPS_PERIOD, decay=CONFIG.EPS_DECAY,
            endValue=CONFIG.EPS_END, resetPeriod=CONFIG.EPS_RESET_PERIOD)
        self.EPSILON = self.EpsilonScheduler.get_variable()

        # Learning Rate
        self.LR_C = CONFIG.LR_C
        self.LR_C_PERIOD = CONFIG.LR_C_PERIOD
        self.LR_C_DECAY = CONFIG.LR_C_DECAY
        self.LR_C_END = CONFIG.LR_C_END
        self.LR_A = CONFIG.LR_A
        self.LR_A_PERIOD = CONFIG.LR_A_PERIOD
        self.LR_A_DECAY = CONFIG.LR_A_DECAY
        self.LR_A_END = CONFIG.LR_A_END

        # NN: batch size, maximal number of NNs stored
        self.BATCH_SIZE = CONFIG.BATCH_SIZE
        self.MAX_MODEL = CONFIG.MAX_MODEL
        self.device = CONFIG.DEVICE

        # Discount Factor
        self.GammaScheduler = StepLRMargin( initValue=CONFIG.GAMMA,
            period=CONFIG.GAMMA_PERIOD, decay=CONFIG.GAMMA_DECAY,
            endValue=CONFIG.GAMMA_END, goalValue=1.)
        self.GAMMA = self.GammaScheduler.get_variable()

        # Target Network Update
        self.TAU = CONFIG.TAU


    # * BUILD NETWORK BEGINS
    def build_network(self, dimList, actType=['Tanh', 'Tanh'], device='cpu'):
        self.build_critic(dimList[0], actType[0], device)
        self.build_actor(dimList[1], actType[1], device)
        self.build_optimizer()


    def build_actor(self, dimList, actType='Tanh', device='cpu'): # in child class
        # self.actor = DeterministicPolicy(dimList, actType, device)
        # self.actorTarget = DeterministicPolicy(dimList, actType, device)
        raise NotImplementedError


    def build_critic(self, dimList, actType='Tanh', device='cpu'):
        self.critic = TwinnedQNetwork(dimList, actType, device)
        self.criticTarget = TwinnedQNetwork(dimList, actType, device)


    def build_optimizer(self):
        self.Q1Optimizer    = AdamW(self.critic.Q1.parameters(), lr=self.LR_C, weight_decay=1e-3)
        self.Q2Optimizer    = AdamW(self.critic.Q2.parameters(), lr=self.LR_C, weight_decay=1e-3)
        self.ActorOptimizer = AdamW(self.actor.parameters(),     lr=self.LR_A, weight_decay=1e-3)
        self.Q1Scheduler    = lr_scheduler.StepLR(self.Q1Optimizer,    step_size=self.LR_C_PERIOD, gamma=self.LR_C_DECAY)
        self.Q2Scheduler    = lr_scheduler.StepLR(self.Q2Optimizer,    step_size=self.LR_C_PERIOD, gamma=self.LR_C_DECAY)
        self.ActorScheduler = lr_scheduler.StepLR(self.ActorOptimizer, step_size=self.LR_A_PERIOD, gamma=self.LR_A_DECAY)
        self.max_grad_norm = 1
        self.cntUpdate = 0
    # * BUILD NETWORK ENDS


    # * LEARN STARTS
    def initBuffer(self, env, ratio=1.): # in child class
        raise NotImplementedError


    def initQ(self): # in child class
        raise NotImplementedError


    def update_critic_hyperParam(self):
        if self.Q1Optimizer.state_dict()['param_groups'][0]['lr'] <= self.LR_C_END:
            for param_group in self.Q1Optimizer.param_groups:
                param_group['lr'] = self.LR_C_END
        else:
            self.Q1Scheduler.step()

        if self.Q2Optimizer.state_dict()['param_groups'][0]['lr'] <= self.LR_C_END:
            for param_group in self.Q2Optimizer.param_groups:
                param_group['lr'] = self.LR_C_END
        else:
            self.Q2Scheduler.step()

        self.EpsilonScheduler.step()
        self.EPSILON = self.EpsilonScheduler.get_variable()
        self.GammaScheduler.step()
        self.GAMMA = self.GammaScheduler.get_variable()


    def update_actor_hyperParam(self):
        if self.ActorOptimizer.state_dict()['param_groups'][0]['lr'] <= self.LR_A_END:
            for param_group in self.ActorOptimizer.param_groups:
                param_group['lr'] = self.LR_A_END
        else:
            self.ActorScheduler.step()


    def update_target_network(self):
        soft_update(self.criticTarget.Q1, self.critic.Q1, self.TAU)
        soft_update(self.criticTarget.Q2, self.critic.Q2, self.TAU)
        if actorType == 'TD3':
            soft_update(self.actorTarget, self.actor, self.TAU)


    def update_critic(self, addBias=False): # in child class
        raise NotImplementedError


    def update_actor(self): # in child class
        raise NotImplementedError


    def learn(self): # TODO: Not yet implemented
        raise NotImplemented
    # * LEARN ENDS


    # * OTHERS STARTS
    def store_transition(self, *args):
        self.memory.update(Transition(*args))


    def save(self, step, logs_path):
        logs_path_critic = os.path.join(logs_path, 'critic/')
        logs_path_actor = os.path.join(logs_path, 'actor/')
        save_model(self.critic, step, logs_path_critic, 'critic', self.MAX_MODEL)
        save_model(self.actor,  step, logs_path_actor, 'actor',  self.MAX_MODEL)


    def restore(self, step, logs_path):
        logs_path_critic = os.path.join(logs_path, 'critic/critic-{}.pth'.format(step))
        logs_path_actor  = os.path.join(logs_path, 'actor/actor-{}.pth'.format(step))
        self.critic.load_state_dict(
            torch.load(logs_path_critic, map_location=self.device))
        self.critic.to(self.device)
        self.criticTarget.load_state_dict(
            torch.load(logs_path_critic, map_location=self.device))
        self.criticTarget.to(self.device)
        self.actor.load_state_dict(
            torch.load(logs_path_actor, map_location=self.device))
        self.actor.to(self.device)    
        if self.actorType == 'TD3':
            self.actorTarget.load_state_dict(
                torch.load(logs_path_actor, map_location=self.device))
            self.actorTarget.to(self.device)   
        print('  => Restore {}' .format(logs_path))


    def select_action(self, state, explore=False):
        stateTensor = torch.from_numpy(state).float().to(self.device).unsqueeze(0)
        if explore:
            action, _, _ = self.actor.sample(stateTensor)
        else:
            _, _, action = self.actor.sample(stateTensor)
        return action.detach().cpu().numpy()[0]
    # * OTHERS ENDS