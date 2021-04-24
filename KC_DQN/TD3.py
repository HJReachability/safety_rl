# Please contact the author(s) of this library if you have any questions.
# Authors: Vicenc Rubies-Royo (vrubies@berkeley.edu)
#          Kai-Chieh Hsu ( kaichieh@princeton.edu )

import torch
import torch.nn as nn
from torch.nn.functional import mse_loss, smooth_l1_loss
from torch.nn.utils import clip_grad_norm_

from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from copy import deepcopy

from .model import DeterministicPolicy
from .ActorCritic import ActorCritic, Transition

class TD3(ActorCritic):
    def __init__(self, CONFIG, actionSpace, dimLists, actType={'critic':'Tanh', 'actor':'Tanh'},
        verbose=True):
        """
        __init__: initialization.

        Args:
            CONFIG (Class object): hyper-parameter configuration.
            actionSpace (Class object): consists of `high` and `low` attributes.
            dimList (list): consists of dimension lists
            actType (dict, optional): consists of activation types.
                Defaults to ['Tanh', 'Tanh'].
            verbose (bool, optional): print info or not. Defaults to True.
        """
        super(TD3, self).__init__('TD3', CONFIG, actionSpace)

        #== Build NN for (D)DQN ==
        assert dimLists is not None, "Define the architectures"
        self.dimListCritic = dimLists[0]
        self.dimListActor = dimLists[1]
        self.actType = actType
        self.build_network(dimLists, actType)


    def build_actor(self, dimListActor, actType='Tanh', noiseStd=0.2,
        noiseClamp=0.5):
        self.actor = DeterministicPolicy(dimListActor, self.actionSpace,
            actType=actType, noiseStd=noiseStd, noiseClamp=noiseClamp)
        self.actorTarget = deepcopy(self.actor)
        for p in self.actorTarget.parameters():
            p.requires_grad = False


    def initBuffer(self, env, ratio=1.):
        cnt = 0
        s = env.reset()
        while len(self.memory) < self.memory.capacity * ratio:
            cnt += 1
            print('\rWarmup Buffer [{:d}]'.format(cnt), end='')
            a = env.action_space.sample()
            s_, r, done, info = env.step(a)
            s_ = None if done else s_
            self.store_transition(s, a, r, s_, info)
            if done:
                s = env.reset()
            else:
                s = s_
        print(" --- Warmup Buffer Ends")


    def initQ(self, env, warmupIter, outFolder, num_warmup_samples=200,
                vmin=-1, vmax=1, plotFigure=True, storeFigure=True):
        loss = 0.0
        lossList = np.empty(warmupIter, dtype=float)
        for ep_tmp in range(warmupIter):
            states, value = env.get_warmup_examples(num_warmup_samples)
            actions = self.genRandomActions(num_warmup_samples)

            self.critic.train()
            value = torch.from_numpy(value).float().to(self.device)
            stateTensor = torch.from_numpy(states).float().to(self.device)
            actionTensor = torch.from_numpy(actions).float().to(self.device)
            q1, q2 = self.critic(stateTensor, actionTensor)
            q1Loss = mse_loss(input=q1, target=value, reduction='sum')
            q2Loss = mse_loss(input=q2, target=value, reduction='sum')
            loss = q1Loss + q2Loss

            self.criticOptimizer.zero_grad()
            loss.backward()
            # clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.criticOptimizer.step()

            lossList[ep_tmp] = loss.detach().cpu().numpy()
            print('\rWarmup Q [{:d}]. MSE = {:f}'.format(
                ep_tmp+1, loss.detach()), end='')

        print(" --- Warmup Q Ends")
        if plotFigure or storeFigure:
            env.visualize(self.critic.Q1, self.actor, vmin=vmin, vmax=vmax)
            if storeFigure:
                figureFolder = os.path.join(outFolder, 'figure')
                os.makedirs(figureFolder, exist_ok=True)
                figurePath = os.path.join(figureFolder, 'initQ.png')
                plt.savefig(figurePath)
            if plotFigure:
                plt.show()
                plt.pause(0.001)
                plt.close()

        # hard replace
        self.criticTarget.load_state_dict(self.critic.state_dict())
        del self.criticOptimizer
        self.build_optimizer()

        return lossList


    def update_critic(self, batch, addBias=False):

        (non_final_mask, non_final_state_nxt, state, action, reward,
            g_x, l_x, g_x_nxt, l_x_nxt) = self.unpack_batch(batch)

        #== get Q(s,a) ==
        q1, q2 = self.critic(state, action)  # Used to compute loss (non-target part).

        #== placeholder for target ==
        target_q = torch.zeros(self.BATCH_SIZE).float().to(self.device)

        #== compute actorTarget next_actions and feed to criticTarget ==
        with torch.no_grad():
            _, next_actions = self.actorTarget.sample(non_final_state_nxt)  # clip(pi_targ(s')+clip(eps,-c,c),a_low, a_high)
            next_q1, next_q2 = self.criticTarget(non_final_state_nxt, next_actions)
            q_max = torch.max(next_q1, next_q2).view(-1)  # max because we are doing reach-avoid.

        target_q[non_final_mask] =  (
            (1.0 - self.GAMMA) * torch.max(l_x_nxt[non_final_mask], g_x_nxt[non_final_mask]) +
            self.GAMMA * torch.max( g_x_nxt[non_final_mask], torch.min(l_x_nxt[non_final_mask], q_max)))
        target_q[torch.logical_not(non_final_mask)] = torch.max(
            l_x_nxt[torch.logical_not(non_final_mask)], g_x_nxt[torch.logical_not(non_final_mask)])

        #== MSE update for both Q1 and Q2 ==
        loss_q1 = mse_loss(input=q1.view(-1), target=target_q)
        loss_q2 = mse_loss(input=q2.view(-1), target=target_q)
        loss_q = loss_q1 + loss_q2

        #== backpropagation ==
        self.criticOptimizer.zero_grad()
        loss_q.backward()
        # nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.criticOptimizer.step()

        return loss_q.item()


    def update_actor(self, batch):

        (non_final_mask, non_final_state_nxt, state,
         action, reward, g_x, l_x, g_x_nxt, l_x_nxt) = self.unpack_batch(batch)

        for p in self.critic.parameters():
            p.requires_grad = False

        q_pi_1, q_pi_2 = self.critic(state, self.actor(state))
        q_pi = torch.max(q_pi_1, q_pi_2)#q_pi_1 if np.random.randint(2) == 0 else q_pi_2  # Kai-Chieh: why randomly decide instead of pick 1.

        loss_pi = q_pi.mean()
        self.actorOptimizer.zero_grad()
        loss_pi.backward()
        # nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actorOptimizer.step()

        for p in self.critic.parameters():
            p.requires_grad = True

        return loss_pi.item()


    def unpack_batch(self, batch):
        # `non_final_mask` is used for environments that have next state to be None
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.s_)),
            dtype=torch.bool).to(self.device)
        non_final_state_nxt = torch.FloatTensor([s for s in batch.s_ if s is not None]).to(self.device)
        state  = torch.FloatTensor(batch.s).to(self.device)
        action = torch.LongTensor(batch.a).to(self.device).view(-1,1)
        reward = torch.FloatTensor(batch.r).to(self.device)

        g_x = torch.FloatTensor([info['g_x'] for info in batch.info]).to(self.device).view(-1)
        l_x = torch.FloatTensor([info['l_x'] for info in batch.info]).to(self.device).view(-1)
        g_x_nxt = torch.FloatTensor([info['g_x_nxt'] for info in batch.info]).to(self.device).view(-1)
        l_x_nxt = torch.FloatTensor([info['l_x_nxt'] for info in batch.info]).to(self.device).view(-1)

        return non_final_mask, non_final_state_nxt, state, action, reward, g_x, l_x, g_x_nxt, l_x_nxt