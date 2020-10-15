# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )

# Here we aim to minimize the cost. We make the following two modifications:
#  - a' = argmin_a' Q_policy(s', a'), y = c(s,a) + gamma * Q_tar(s', a')
#  - loss = E[ ( y - Q_policy(s,a) )^2 ] 

#import sys
#sys.path.append('..')
#print(sys.path)

import torch
import torch.nn as nn
from torch.nn.functional import mse_loss, smooth_l1_loss
from torch.autograd import Variable
import torch.optim as optim

from collections import namedtuple
import random
import numpy as np

from .model import model
from .ReplayMemory import ReplayMemory

Transition = namedtuple('Transition', ['s', 'a', 'r', 's_'])

class DDQN():

    def __init__(self, state_num, action_num, CONFIG, action_list):
        self.action_list = action_list
        self.memory = ReplayMemory(CONFIG.MEMORY_CAPACITY)
        
        #== ENV PARAM ==
        self.state_num = state_num
        self.action_num = action_num
        
        #== PARAM ==
        # Exploration
        self.EPSILON = CONFIG.EPSILON
        self.EPS_END = CONFIG.EPS_END
        self.EPS_PERIOD = CONFIG.EPS_PERIOD
        self.EPS_DECAY = CONFIG.EPS_DECAY
        # Learning Rate
        self.LR_C = CONFIG.LR_C
        self.LR_C_PERIOD = CONFIG.LR_C_PERIOD
        self.LR_C_DECAY = CONFIG.LR_C_DECAY
        # NN: batch size, maximal number of NNs stored
        self.BATCH_SIZE = CONFIG.BATCH_SIZE
        self.MAX_MODEL = CONFIG.MAX_MODEL
        self.device = CONFIG.DEVICE
        # Contraction Mapping
        self.GAMMA = CONFIG.GAMMA
        self.GAMMA_PERIOD = CONFIG.GAMMA_PERIOD
        self.GAMMA_DECAY = CONFIG.GAMMA_DECAY
        # Target Network Update
        self.double = CONFIG.DOUBLE
        self.TAU = CONFIG.TAU
        self.HARD_UPDATE = CONFIG.HARD_UPDATE # int, update period
        self.SOFT_UPDATE = CONFIG.SOFT_UPDATE # bool
        # Build NN(s) for DQN 
        self.build_network()


    def build_network(self):
        self.Q_network = model(self.state_num, self.action_num)
        self.target_network = model(self.state_num, self.action_num)
        if self.device == torch.device('cuda'):
            self.Q_network.cuda()
            self.target_network.cuda()
        self.optimizer = optim.Adam(self.Q_network.parameters(), lr=self.LR_C)
        self.scheduler =  optim.lr_scheduler.StepLR(self.optimizer, step_size=self.LR_C_PERIOD, gamma=self.LR_C_DECAY)
        self.max_grad_norm = 0.5
        self.training_epoch = 0


    def update_target_network(self):
        if self.SOFT_UPDATE:
            # Soft Replace
            for module_tar, module_pol in zip(self.target_network.modules(), self.Q_network.modules()):
                if isinstance(module_tar, nn.Linear):
                    module_tar.weight.data = (1-self.TAU)*module_tar.weight.data + self.TAU*module_pol.weight.data
                    module_tar.bias.data   = (1-self.TAU)*module_tar.bias.data   + self.TAU*module_pol.bias.data
        elif self.training_epoch % self.HARD_UPDATE == 0:
            # Hard Replace
            self.target_network.load_state_dict(self.Q_network.state_dict())
        
         
    def update(self):
        if len(self.memory) < self.BATCH_SIZE*20:
        #if not self.memory.isfull:
            return
        
        #== EXPERIENCE REPLAY ==
        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))
        
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.s_)), 
                                      device=self.device, dtype=torch.bool)
        non_final_state_nxt = torch.FloatTensor([s for s in batch.s_ if s is not None], 
                                                 device=self.device)
        state = torch.FloatTensor(batch.s, device=self.device)
        action = torch.LongTensor(batch.a, device=self.device).view(-1,1)
        reward = torch.FloatTensor(batch.r, device=self.device)
        
        #== get Q(s,a) ==
        # gather reguires idx to be Long, i/p and idx should have the same shape with only diff at the dim we want to extract value
        # o/p = Q [ i ][ action[i] ], which has the same dim as idx, 
        state_action_values = self.Q_network(state).gather(1, action).view(-1)
        
        #== get a' by Q_policy: a' = argmin_a' Q_policy(s', a') ==
        with torch.no_grad():
            action_nxt = self.Q_network(non_final_state_nxt).min(1, keepdim=True)[1]
        
        #== get expected value: y = r + gamma * Q_tar(s', a') ==
        state_value_nxt = torch.zeros(self.BATCH_SIZE, device=self.device)
        with torch.no_grad():
            if self.double:
                Q_expect = self.target_network(non_final_state_nxt)
            else:
                Q_expect = self.Q_network(non_final_state_nxt)
        state_value_nxt[non_final_mask] = Q_expect.gather(1, action_nxt).view(-1)
        expected_state_action_values = (state_value_nxt * self.GAMMA) + reward
        
        #== regression Q(s, a) -> y ==
        self.Q_network.train()
        loss = smooth_l1_loss(input=state_action_values, target=expected_state_action_values.detach())
        
        #== backward optimize ==
        self.optimizer.zero_grad()
        loss.backward()
        #nn.utils.clip_grad_norm_(self.Q_network.parameters(), self.max_grad_norm)
        self.optimizer.step()

        self.update_target_network()

        return loss.item()


    def updateEpsilon(self):
        if self.training_epoch % self.EPS_PERIOD == 0 and self.training_epoch != 0:
            self.EPSILON = max(self.EPSILON*self.EPS_DECAY, self.EPS_END)


    def updateGamma(self):
        if self.training_epoch % self.GAMMA_PERIOD == 0 and self.training_epoch != 0:
            self.GAMMA = 1 - (1-self.GAMMA) * self.GAMMA_DECAY

    #== Hyper-Parameter Update ==
    def updateHyperParam(self):
        self.scheduler.step()
        self.updateEpsilon()
        self.updateGamma()
        self.training_epoch += 1


    def select_action(self, state, explore=True):
        # tensor.min() returns (value, indices), which are in tensor form
        state = torch.from_numpy(state).float().unsqueeze(0)
        if (random.random() < self.EPSILON) and explore:
            action_index = random.randint(0, self.action_num-1)
        else:
            action_index = self.Q_network(state).min(dim=1)[1].item()
        return self.action_list[action_index], action_index


    def store_transition(self, *args):
        self.memory.update(Transition(*args))

        
    def save(self, step, logs_path):
        os.makedirs(logs_path, exist_ok=True)
        model_list =  glob.glob(os.path.join(logs_path, '*.pth'))
        if len(model_list) > self.MAX_MODEL - 1 :
            min_step = min([int(li.split('/')[-1][6:-4]) for li in model_list]) 
            os.remove(os.path.join(logs_path, 'model-{}.pth' .format(min_step)))
        logs_path = os.path.join(logs_path, 'model-{}.pth' .format(step))
        self.Q_network.save(logs_path, step=step)
        print('=> Save {}' .format(logs_path)) 


    def restore(self, logs_path):
        self.Q_network.load(logs_path)
        self.target_network.load(logs_path)
        print('=> Restore {}' .format(logs_path))