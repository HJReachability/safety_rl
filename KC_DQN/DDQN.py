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

from .model import model
from .ReplayMemory import ReplayMemory

Transition = namedtuple('Transition', ['s', 'a', 'r', 's_', 'info'])

class DDQN():

    def __init__(self, state_num, action_num, CONFIG, action_list, mode='normal', RA_scaling=1.):
        self.action_list = action_list
        self.memory = ReplayMemory(CONFIG.MEMORY_CAPACITY)
        self.mode = mode # 'normal' or 'RA'
        
        #== ENV PARAM ==
        self.state_num = state_num
        self.action_num = action_num
        self.RA_scaling = RA_scaling
        
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
        self.max_grad_norm = 1
        self.training_epoch = 0
        
         
    def update(self, verbose=False, addBias=False):
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

        #== Discounted Reach-Avoid Bellman Equation ==
        if self.mode == 'RA':
            expected_state_action_values = torch.zeros(self.BATCH_SIZE).float().to(self.device)
            
            l_x [l_x < 0] *=  self.RA_scaling
            l_x_nxt [l_x_nxt < 0] *=  self.RA_scaling
            #g_x [g_x > 0] *=  self.RA_scaling
            # Bias version
            if addBias:
                min_term = torch.min(l_x, state_value_nxt+torch.max(l_x_nxt, g_x_nxt))
                terminal = torch.max(l_x, g_x)
                non_terminal = torch.max(min_term, g_x) - terminal
                expected_state_action_values[non_final_mask] = self.GAMMA * non_terminal[non_final_mask]
                expected_state_action_values[torch.logical_not(non_final_mask)] = terminal[torch.logical_not(non_final_mask)]
            else:
                success_mask = torch.logical_and(torch.logical_not(non_final_mask), l_x<=0)
                failure_mask = torch.logical_and(torch.logical_not(non_final_mask), g_x>0)

                min_term = torch.min(l_x, state_value_nxt)
                non_terminal = torch.max(min_term, g_x)
                terminal = torch.max(l_x, g_x)

                expected_state_action_values[non_final_mask] = non_terminal[non_final_mask] * self.GAMMA + \
                                                               terminal[non_final_mask] * (1-self.GAMMA)
                #expected_state_action_values[success_mask] = -10.
                #expected_state_action_values[failure_mask] = 10.
                #expected_state_action_values[success_mask] = l_x[success_mask]
                expected_state_action_values[torch.logical_not(non_final_mask)] = terminal[torch.logical_not(non_final_mask)]
            if verbose:
                np.set_printoptions(precision=3)
                print(non_final_mask[:10])
                print('V_target:', state_value_nxt[:10].numpy())
                print('ell     :', l_x[:10].numpy())
                print('g       :', g_x[:10].numpy())
                print('V_expect:', expected_state_action_values[:10].numpy())
                print('V_policy:', state_action_values[:10].detach().numpy(), end='\n\n')
        else:
            expected_state_action_values = state_value_nxt * self.GAMMA + reward
        
        #== regression Q(s, a) -> y ==
        self.Q_network.train()
        loss = smooth_l1_loss(input=state_action_values, target=expected_state_action_values.detach())
        
        #== backward optimize ==
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.Q_network.parameters(), self.max_grad_norm)
        self.optimizer.step()

        self.update_target_network()

        return loss.item()


    def learn(self, env, MAX_EPISODES=20000, MAX_EP_STEPS=100, running_cost_th=None, addBias=False, 
              report_period=5000, plotFigure=True,
              vmin=-100, vmax=100, randomPlot=False, num_rnd_traj=10, toEnd=False,
              warmupBuffer=True, warmupQ=False,
              check_period=50000, storeModel=True, outFolder='RA'):

        #== TRAINING RECORD ==
        TrainingRecord = namedtuple('TrainingRecord', ['ep', 'avg_cost', 'cost', 'loss_c'])
        training_records = []
        running_cost = 0.
        running_cost_th = running_cost_th
        report_period = report_period
        vmin = vmin
        vmax = vmax

        # == Warmup Buffer ==
        if warmupBuffer:
            cnt = 0
            while len(self.memory) < self.BATCH_SIZE*20:
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
            ep_warmup = 1000
            num_warmup_samples = 100
            for ep_tmp in range(ep_warmup):
                print('Warmup Q [{:d}]'.format(ep_tmp), end='\r')
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
                '''
                if ep_tmp % 500 == 0:
                    env.visualize_analytic_comparison(self.Q_network, True, vmin=vmin, vmax=vmax)
                    plt.pause(0.001)
                '''
            print("\nWarmup Q Ends")
            env.visualize_analytic_comparison(self.Q_network, True, vmin=vmin, vmax=vmax)
            env.visualize_analytic_comparison(self.target_network, True, vmin=vmin, vmax=vmax)
            plt.pause(0.001)
            self.target_network.load_state_dict(self.Q_network.state_dict()) # hard replace
 
        # == Main Training ==
        cnt_access = 0
        trainProgress = []
        checkPointSucc = 0.
        for ep in range(MAX_EPISODES):
            s = env.reset()
            ep_cost = 0.
            for step_num in range(MAX_EP_STEPS):
                cnt_access += 1
                # Select action
                a, a_idx = self.select_action(s)
                # Interact with env
                s_, r, done, info = env.step(a_idx)
                ep_cost += r
                if done:
                    s_ = None
                # Store the transition in memory
                self.store_transition(s, a_idx, r, s_, info)
                s = s_
                # Perform one step of the optimization (on the target network)
                if cnt_access % check_period == 0:
                    loss_c = self.update(verbose=False, addBias=addBias)
                else:
                    loss_c = self.update(addBias=addBias)
                # Report after fixed number of gradient updates / accesses
                if cnt_access % check_period == 0:
                    num_rnd_traj_test=2000
                    _, results = env.simulate_trajectories(self.Q_network, T=200, num_rnd_traj=num_rnd_traj_test, 
                                                           keepOutOf=False, toEnd=False)
                    success  = np.sum(results==1) / num_rnd_traj_test 
                    failure  = np.sum(results==-1)/ num_rnd_traj_test
                    unfinish = np.sum(results==0) / num_rnd_traj_test
                    trainProgress.append([success, failure, unfinish])
                    print('After [{:d}] accesses, success/failure/unfinished ratio: {:.3f}, {:.3f}, {:.3f}'.format(\
                        cnt_access, success, failure, unfinish))
                    if success > checkPointSucc and storeModel:
                        checkPointSucc = success
                        self.save(cnt_access, 'models/{:s}/'.format(outFolder))
                # Terminate early
                if done:
                    break
                    
            self.updateHyperParam()

            running_cost = running_cost * 0.9 + ep_cost * 0.1
            training_records.append(TrainingRecord(ep, running_cost, ep_cost, loss_c))
            print('{:d}: {:.1f} after {:d} steps, currently access {:d} times'.format(\
                ep, ep_cost, step_num+1, cnt_access), end='\r')
            
            # Report after fixed number of epochs
            if ep % report_period == 0:
                lr = self.optimizer.state_dict()['param_groups'][0]['lr']
                print('Ep[{:3.0f} - ({:.2f},{:.6f},{:.1e})]: Running/Real cost: {:3.2f}/{:.2f}; '.format(
                    ep, self.EPSILON, self.GAMMA, lr, running_cost, ep_cost))
                if plotFigure:
                    env.visualize_analytic_comparison(self.Q_network, True, vmin=vmin, vmax=vmax, labels=None)
                    env.plot_reach_avoid_set()
                    if randomPlot:
                        _ = env.plot_trajectories(self.Q_network, T=500, num_rnd_traj=num_rnd_traj, 
                                                keepOutOf=True, toEnd=toEnd)
                    else:
                        _ = env.plot_trajectories(self.Q_network, T=500, 
                                                states=env.visual_initial_states, toEnd=toEnd)
                    plt.pause(0.001)
            
            # Check stopping criteria  
            if running_cost_th != None:
                if running_cost <= running_cost_th:
                    print("\r At Ep[{:3.0f}] Solved! Running cost is now {:3.2f}!".format(ep, running_cost))
                    env.close()
                    break
        
        self.save(cnt_access, 'models/{:s}/'.format(outFolder))
        return training_records, trainProgress


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


    def updateEpsilon(self):
        if self.training_epoch % self.EPS_PERIOD == 0 and self.training_epoch != 0:
            self.EPSILON = max(self.EPSILON*self.EPS_DECAY, self.EPS_END)


    def updateGamma(self):
        if self.training_epoch % self.GAMMA_PERIOD == 0 and self.training_epoch != 0:
            self.GAMMA = min(1 - (1-self.GAMMA) * self.GAMMA_DECAY, 0.99999999)


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
        #print(model_list)
        if len(model_list) > self.MAX_MODEL - 1 :
            min_step = min([int(li.split('/')[-1][6:-4]) for li in model_list]) 
            os.remove(os.path.join(logs_path, 'model-{}.pth' .format(min_step)))
        logs_path = os.path.join(logs_path, 'model-{}.pth' .format(step))
        torch.save(self.Q_network.state_dict(), logs_path)
        print('=> Save {} at [{}] accesses' .format(logs_path, step)) 


    def restore(self, logs_path):
        self.Q_network.load_state_dict(torch.load(logs_path))
        self.target_network.load_state_dict(torch.load(logs_path))
        print('=> Restore {}' .format(logs_path))