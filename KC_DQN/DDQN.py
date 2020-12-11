# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )

# Here we aim to minimize the cost. We make the following two modifications:
#  - a' = argmin_a' Q_policy(s', a'), y = c(s,a) + gamma * Q_tar(s', a')
#  - loss = E[ ( y - Q_policy(s,a) )^2 ]

import torch
import torch.nn as nn
from torch.nn.functional import mse_loss, smooth_l1_loss
from torch.autograd import Variable
import torch.optim as optim

from collections import namedtuple
import random
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

from .model import *
from .ReplayMemory import ReplayMemory

Transition = namedtuple('Transition', ['s', 'a', 'r', 's_', 'info'])

class DDQN():

    def __init__(self, state_num, action_num, CONFIG, actionList, 
                    mode='normal', dimList=None, actType='Tanh'):
        self.actionList = actionList
        self.memory = ReplayMemory(CONFIG.MEMORY_CAPACITY)
        self.mode = mode # 'normal' or 'RA'

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
        self.LR_C_END = CONFIG.LR_C_END
        # NN: batch size, maximal number of NNs stored
        self.BATCH_SIZE = CONFIG.BATCH_SIZE
        self.MAX_MODEL = CONFIG.MAX_MODEL
        self.device = CONFIG.DEVICE
        # Discount Factor
        self.GAMMA = CONFIG.GAMMA
        self.GAMMA_END = CONFIG.GAMMA_END
        self.GAMMA_PERIOD = CONFIG.GAMMA_PERIOD
        self.GAMMA_DECAY = CONFIG.GAMMA_DECAY
        # Target Network Update
        self.double = CONFIG.DOUBLE
        self.TAU = CONFIG.TAU
        self.HARD_UPDATE = CONFIG.HARD_UPDATE # int, update period
        self.SOFT_UPDATE = CONFIG.SOFT_UPDATE # bool
        # Build NN for (D)DQN
        self.dimList = dimList
        self.actType = actType
        self.build_network(dimList, actType)
        self.build_optimizer()


    def build_network(self, dimList=None, actType='Tanh'):
        assert self.dimList is not None, "Define the architecture"
        self.Q_network = model(dimList, actType, verbose=True)
        self.target_network = model(dimList, actType)

        if self.device == torch.device('cuda'):
            self.Q_network.cuda()
            self.target_network.cuda()


    def build_optimizer(self):
        self.optimizer = optim.Adam(self.Q_network.parameters(), lr=self.LR_C)
        self.scheduler =  optim.lr_scheduler.StepLR(self.optimizer, step_size=self.LR_C_PERIOD, gamma=self.LR_C_DECAY)
        self.max_grad_norm = 1
        self.cntUpdate = 0


    def update(self, addBias=False):
        if len(self.memory) < self.BATCH_SIZE*20:
            return

        #== EXPERIENCE REPLAY ==
        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for detailed explanation).
        # This converts batch-array of Transitions to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # `non_final_mask` is used for environments that have next state to be None
        non_final_mask = torch.tensor(  tuple(map(lambda s: s is not None, batch.s_)),
                                        device=self.device, dtype=torch.bool)
        non_final_state_nxt = torch.FloatTensor([s for s in batch.s_ if s is not None],
                                                device=self.device)
        state  = torch.FloatTensor(batch.s, device=self.device)
        action = torch.LongTensor(batch.a,  device=self.device).view(-1,1)
        reward = torch.FloatTensor(batch.r, device=self.device)
        if self.mode == 'RA':
            g_x = torch.FloatTensor([info['g_x'] for info in batch.info],
                                    device=self.device).view(-1)
            l_x = torch.FloatTensor([info['l_x'] for info in batch.info],
                                    device=self.device).view(-1)
            g_x_nxt = torch.FloatTensor([info['g_x_nxt'] for info in batch.info],
                                    device=self.device).view(-1)
            l_x_nxt = torch.FloatTensor([info['l_x_nxt'] for info in batch.info],
                                    device=self.device).view(-1)

        #== get Q(s,a) ==
        # `gather` reguires idx to be Long, input and index should have the same shape
        # with only difference at the dimension we want to extract value
        # out[i][j][k] = input[i][j][ index[i][j][k] ], which has the same dim as index
        # -> state_action_values = Q [ i ][ action[i] ]
        # view(-1): from mtx to vector
        state_action_values = self.Q_network(state).gather(1, action).view(-1)

        #== get a' by Q_policy: a' = argmin_a' Q_policy(s', a') ==
        with torch.no_grad():
            action_nxt = self.Q_network(non_final_state_nxt).min(1, keepdim=True)[1]

        #== get expected value ==
        state_value_nxt = torch.zeros(self.BATCH_SIZE, device=self.device)

        with torch.no_grad(): # V(s') = Q_tar(s', a'), a' is from Q_policy
            if self.double:
                Q_expect = self.target_network(non_final_state_nxt)
            else:
                Q_expect = self.Q_network(non_final_state_nxt)
        state_value_nxt[non_final_mask] = Q_expect.gather(1, action_nxt).view(-1)

        #== Discounted Reach-Avoid Bellman Equation (DRABE) ==
        if self.mode == 'RA':
            expected_state_action_values = torch.zeros(self.BATCH_SIZE).float().to(self.device)
            if addBias: # Bias version: V(s) = gamma ( max{ g(s), min{ l(s), V_diff(s') } } - max{ g(s), l(s) } ),
                        # where V_diff(s') = V(s') + max{ g(s'), l(s') }
                min_term = torch.min(l_x, state_value_nxt+torch.max(l_x_nxt, g_x_nxt))
                terminal = torch.max(l_x, g_x)
                non_terminal = torch.max(min_term, g_x) - terminal
                expected_state_action_values[non_final_mask] = self.GAMMA * non_terminal[non_final_mask]
                expected_state_action_values[torch.logical_not(non_final_mask)] = terminal[torch.logical_not(non_final_mask)]
            else:   # Better version instead of DRABE on the paper (discussed on Nov. 18, 2020)
                    # V(s) = gamma ( max{ g(s), min{ l(s), V_better(s') } } + (1-gamma) max{ g(s), l(s) },
                    # where V_better(s') = max{ g(s'), min{ l(s'), V(s') } }
                #success_mask = torch.logical_and(torch.logical_not(non_final_mask), l_x<=0)
                #failure_mask = torch.logical_and(torch.logical_not(non_final_mask), g_x>0)
                V_better = torch.max( g_x_nxt, torch.min(l_x_nxt, state_value_nxt))
                #V_better = state_value_nxt
                min_term = torch.min(l_x, V_better)
                non_terminal = torch.max(min_term, g_x)
                terminal = torch.max(l_x, g_x)

                expected_state_action_values[non_final_mask] = non_terminal[non_final_mask] * self.GAMMA + \
                                                               terminal[non_final_mask] * (1-self.GAMMA)
                # if next state is None, we will use g(x) as the expected V(s)
                expected_state_action_values[torch.logical_not(non_final_mask)] = g_x[torch.logical_not(non_final_mask)]
        else: # V(s) = c(s, a) + gamma * V(s')
            expected_state_action_values = state_value_nxt * self.GAMMA + reward

        #== regression: Q(s, a) <- V(s) ==
        self.Q_network.train()
        loss = smooth_l1_loss(input=state_action_values, target=expected_state_action_values.detach())

        #== backpropagation ==
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.Q_network.parameters(), self.max_grad_norm)
        self.optimizer.step()

        self.update_target_network()

        return loss.item()


    def learn(  self, env, MAX_UPDATES=2000000, MAX_EP_STEPS=100,
                warmupBuffer=True, warmupQ=False, warmupIter=10000,
                addBias=False, doneTerminate=True, running_cost_th=None,
                curUpdates=None, checkPeriod=50000, 
                plotFigure=True, storeFigure=False,
                showBool=False, vmin=-1, vmax=1, num_rnd_traj=200,
                storeModel=True, storeBest=False, 
                outFolder='RA', verbose=True):
        #== TRAINING RECORD ==
        TrainingRecord = namedtuple('TrainingRecord', ['ep', 'avg_cost', 'cost', 'loss_c'])
        trainingRecords = []
        running_cost = 0.

        # == Warmup Buffer ==
        if warmupBuffer:
            cnt = 0
            while len(self.memory) < self.memory.capacity:
                cnt += 1
                print('\rWarmup Buffer [{:d}]'.format(cnt), end='')
                s = env.reset()
                a, a_idx = self.select_action(s)
                s_, r, done, info = env.step(a_idx)
                if done:
                    s_ = None
                self.store_transition(s, a_idx, r, s_, info)
            print(" --- Warmup Buffer Ends")

        # == Warmup Q ==
        if warmupQ:
            num_warmup_samples = 200
            for ep_tmp in range(warmupIter):
                print('\rWarmup Q [{:d}]'.format(ep_tmp+1), end='')
                states, heuristic_v = env.get_warmup_examples(num_warmup_samples=num_warmup_samples)

                self.Q_network.train()
                heuristic_v = torch.from_numpy(heuristic_v).float().to(self.device)
                states = torch.from_numpy(states).float().to(self.device)
                v = self.Q_network(states)
                loss = smooth_l1_loss(input=v, target=heuristic_v)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.Q_network.parameters(), self.max_grad_norm)
                self.optimizer.step()

            print(" --- Warmup Q Ends")
            env.visualize(self.Q_network, True, vmin=vmin, vmax=vmax, cmap='seismic', addBias=addBias)
            plt.pause(0.001)
            self.target_network.load_state_dict(self.Q_network.state_dict()) # hard replace
            self.build_optimizer()

        # == Main Training ==
        trainProgress = []
        checkPointSucc = 0.
        ep = 0
        if curUpdates is not None:
            self.cntUpdate = curUpdates
            print("starting from {:d} updates".format(self.cntUpdate))
        while self.cntUpdate <= MAX_UPDATES:
            s = env.reset()
            ep_cost = 0.
            ep += 1
            # Rollout
            for step_num in range(MAX_EP_STEPS):
                # Select action
                a, a_idx = self.select_action(s)

                # Interact with env
                s_, r, done, info = env.step(a_idx)
                ep_cost += r

                # Store the transition in memory
                self.store_transition(s, a_idx, r, s_, info)
                s = s_

                # Check after fixed number of gradient updates
                if self.cntUpdate != 0 and self.cntUpdate % checkPeriod == 0:
                    _, results = env.simulate_trajectories( self.Q_network, T=MAX_EP_STEPS, 
                                                            num_rnd_traj=num_rnd_traj,
                                                            keepOutOf=False, toEnd=False)
                    success  = np.sum(results==1) / num_rnd_traj
                    failure  = np.sum(results==-1)/ num_rnd_traj
                    unfinish = np.sum(results==0) / num_rnd_traj
                    trainProgress.append([success, failure, unfinish])
                    if verbose:
                        lr = self.optimizer.state_dict()['param_groups'][0]['lr']
                        print('\nAfter [{:d}] updates:'.format(self.cntUpdate))
                        print('  - eps={:.2f}, gamma={:.6f}, lr={:.1e}.'.format(
                            self.EPSILON, self.GAMMA, lr))
                        print('  - success/failure/unfinished ratio: {:.3f}, {:.3f}, {:.3f}'.format(
                            success, failure, unfinish))

                    if storeModel:
                        if storeBest:
                            if success > checkPointSucc:
                                checkPointSucc = success
                                self.save(self.cntUpdate, 'models/{:s}/'.format(outFolder))
                        else:
                            self.save(self.cntUpdate, 'models/{:s}/'.format(outFolder))

                    if plotFigure or storeFigure:
                        if showBool:
                            env.visualize(self.Q_network, True, vmin=0, boolPlot=True, addBias=addBias)
                        else:
                            env.visualize(self.Q_network, True, vmin=vmin, vmax=vmax, cmap='seismic', addBias=addBias)
                        if storeFigure:
                            figureFolder = 'figure/{:s}/'.format(outFolder)
                            os.makedirs(figureFolder, exist_ok=True)
                            plt.savefig('{:s}/{:d}.png'.format(figureFolder, self.cntUpdate))
                        if plotFigure:
                            plt.pause(0.001)

                # Perform one step of the optimization (on the target network)
                loss_c = self.update(addBias=addBias)
                self.cntUpdate += 1
                self.updateHyperParam()

                # Terminate early
                if done and doneTerminate:
                    break

            # Rollout report
            running_cost = running_cost * 0.9 + ep_cost * 0.1
            trainingRecords.append(TrainingRecord(ep, running_cost, ep_cost, loss_c))
            if verbose:
                print('\r{:3.0f}: This episode gets running/episode cost = ({:3.2f}/{:.2f}) after {:d} steps.'.format(\
                    ep, running_cost, ep_cost, step_num+1), end=' ')
                print('The agent currently updates {:d} times'.format(self.cntUpdate), end='\t\t')

            # Check stopping criteria
            if running_cost_th != None:
                if running_cost <= running_cost_th:
                    print("\n At Updates[{:3.0f}] Solved! Running cost is now {:3.2f}!".format(self.cntUpdate, running_cost))
                    env.close()
                    break
        print()
        self.save(self.cntUpdate, 'models/{:s}/'.format(outFolder))
        return trainingRecords, trainProgress


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


    def updateEpsilon(self):
        if self.cntUpdate % self.EPS_PERIOD == 0 and self.cntUpdate != 0:
            self.EPSILON = max(self.EPSILON*self.EPS_DECAY, self.EPS_END)


    def updateGamma(self):
        if self.cntUpdate % self.GAMMA_PERIOD == 0 and self.cntUpdate != 0:
            self.GAMMA = min(1 - (1-self.GAMMA) * self.GAMMA_DECAY, self.GAMMA_END)


    def updateHyperParam(self):
        if self.optimizer.state_dict()['param_groups'][0]['lr'] <= self.LR_C_END:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.LR_C_END
        else:
            self.scheduler.step()
        self.updateEpsilon()
        self.updateGamma()


    def select_action(self, state, explore=True):
        # tensor.min() returns (value, indices), which are in tensor form
        state = torch.from_numpy(state).float().unsqueeze(0)
        if (random.random() < self.EPSILON) and explore:
            action_index = random.randint(0, self.action_num-1)
        else:
            action_index = self.Q_network(state).min(dim=1)[1].item()
        return self.actionList[action_index], action_index


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
        print('=> Save {} after [{}] updates' .format(logs_path, step))


    def restore(self, logs_path):
        self.Q_network.load_state_dict(torch.load(logs_path))
        self.target_network.load_state_dict(torch.load(logs_path))
        print('=> Restore {}' .format(logs_path))