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

from .model import GaussianPolicy
from .ActorCritic import ActorCritic, Transition

class SAC(ActorCritic):
    def __init__(self, CONFIG, actionSpace, dimLists, terminalType='g', 
        actType={'critic':'Tanh', 'actor':'Tanh'}, verbose=True):
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
        super(SAC, self).__init__('SAC', CONFIG, actionSpace)
        self.alpha = CONFIG.ALPHA
        self.terminalType = terminalType

        #== Build NN for (D)DQN ==
        assert dimLists is not None, "Define the architectures"
        self.dimListCritic = dimLists[0]
        self.dimListActor = dimLists[1]
        self.actType = actType
        self.build_network(dimLists, actType)


    def build_actor(self, dimListActor, actType='Tanh'):
        self.actor = GaussianPolicy(dimListActor, self.actionSpace, actType=actType)


    def initBuffer(self, env, ratio=1.):
        cnt = 0
        s = env.reset()
        while len(self.memory) < self.memory.capacity * ratio:
            cnt += 1
            print('\rWarmup Buffer [{:d}]'.format(cnt), end='')
            a = env.action_space.sample()
            # a = self.genRandomActions(1)[0]
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

        non_final_mask, non_final_state_nxt, state, action, _, g_x, l_x = \
            self.unpack_batch(batch)
        self.critic.train()
        self.criticTarget.eval()

        #== get Q(s,a) ==
        q1, q2 = self.critic(state, action)  # Used to compute loss (non-target part).

        #== placeholder for target ==
        target_q = torch.zeros(self.BATCH_SIZE).float().to(self.device)

        #== compute actor next_actions and feed to criticTarget ==
        with torch.no_grad():
            next_actions, _ = self.actor.sample(non_final_state_nxt)
            next_q1, next_q2 = self.criticTarget(non_final_state_nxt, next_actions)
            q_max = torch.max(next_q1, next_q2).view(-1)  # max because we are doing reach-avoid.

        target_q[non_final_mask] =  (
            (1.0 - self.GAMMA) * torch.max(l_x[non_final_mask], g_x[non_final_mask]) +
            self.GAMMA * torch.max( g_x[non_final_mask], torch.min(l_x[non_final_mask], q_max)))
        if self.terminalType == 'g':
            target_q[torch.logical_not(non_final_mask)] = g_x[torch.logical_not(non_final_mask)]
        elif self.terminalType == 'max':
            target_q[torch.logical_not(non_final_mask)] = torch.max(
                l_x[torch.logical_not(non_final_mask)], g_x[torch.logical_not(non_final_mask)])
        else:
            raise ValueError("invalid terminalType")

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

        _, _, state, _, _, _, _ = self.unpack_batch(batch)

        self.critic.eval()
        self.actor.train()
        for p in self.critic.parameters():
            p.requires_grad = False

        action_sample, log_prob = self.actor.sample(state)
        q_pi_1, q_pi_2 = self.critic(state, action_sample)
        q_pi = torch.max(q_pi_1, q_pi_2)

        # Obj: min_theta E [ Q(s, pi_theta(s, \xi)) + alpha * log(pi_theta(s, \xi)) ]
        # loss_pi = (q_pi - self.alpha * log_prob.view(-1)).mean()
        loss_entropy = log_prob.view(-1).mean()
        loss_q_eval = q_pi.mean()
        loss_pi = loss_q_eval + self.alpha * loss_entropy
        self.actorOptimizer.zero_grad()
        loss_pi.backward()
        self.actorOptimizer.step()

        for p in self.critic.parameters(): 
            p.requires_grad = True

        return loss_pi.item(), loss_entropy.item()


    def update(self, timer, update_period=2):
        if len(self.memory) < self.start_updates:
            return 0.0, 0.0

        #== EXPERIENCE REPLAY ==
        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        loss_q = self.update_critic(batch)
        loss_pi = 0.
        loss_entropy = 0.
        if timer % update_period == 0:
            loss_pi, loss_entropy = self.update_actor(batch)
            print('\r{:d}: (q, pi, ent) = ({:3.5f}/{:3.5f}/{:3.5f}).'.format(
                self.cntUpdate, loss_q, loss_pi, loss_entropy), end=' ')

        self.update_target_networks()

        return loss_q, loss_pi, loss_entropy


    def learn(  self, env, MAX_UPDATES=2000000, MAX_EP_STEPS=100,
                warmupBuffer=True, warmupQ=False, warmupIter=10000,
                addBias=False, doneTerminate=True, runningCostThr=None,
                curUpdates=None, checkPeriod=50000,
                plotFigure=True, storeFigure=False,
                showBool=False, vmin=-1, vmax=1, numRndTraj=200,
                storeModel=True, saveBest=True, outFolder='RA', verbose=True):

        # == Warmup Buffer ==
        startInitBuffer = time.time()
        if warmupBuffer:
            self.initBuffer(env)
        endInitBuffer = time.time()

        # == Warmup Q ==
        startInitQ = time.time()
        if warmupQ:
            self.initQ(env, warmupIter=warmupIter, outFolder=outFolder,
                vmin=vmin, vmax=vmax, plotFigure=plotFigure,
                storeFigure=storeFigure)
        endInitQ = time.time()

        # == Main Training ==
        startLearning = time.time()
        TrainingRecord = namedtuple('TrainingRecord',
            ['ep', 'runningCost', 'cost', 'loss_q', 'loss_pi'])
        trainingRecords = []
        runningCost = 0.
        trainProgress = []
        checkPointSucc = 0.
        ep = 0

        if storeModel:
            modelFolder = os.path.join(outFolder, 'model')
            os.makedirs(modelFolder, exist_ok=True)
        if storeFigure:
            figureFolder = os.path.join(outFolder, 'figure')
            os.makedirs(figureFolder, exist_ok=True)

        if curUpdates is not None:
            self.cntUpdate = curUpdates
            print("starting from {:d} updates".format(self.cntUpdate))

        while self.cntUpdate <= MAX_UPDATES:
            s = env.reset()
            epCost = np.inf
            ep += 1

            # Rollout
            for step_num in range(MAX_EP_STEPS):
                # Select action
                if warmupBuffer or self.cntUpdate > max(warmupIter, self.start_updates):
                    with torch.no_grad():
                        a, _ = self.actor.sample(
                            torch.from_numpy(s).float().to(self.device))
                        a = a.cpu().numpy()
                else:
                    a = env.action_space.sample()

                # Interact with env
                s_, r, done, info = env.step(a)
                s_ = None if done else s_
                epCost = max(info["g_x"], min(epCost, info["l_x"]))

                # Store the transition in memory
                self.store_transition(s, a, r, s_, info)
                s = s_

                # Check after fixed number of gradient updates
                if self.cntUpdate != 0 and self.cntUpdate % checkPeriod == 0:
                    actor_sim = self.actor
                    results= env.simulate_trajectories(actor_sim,
                        T=MAX_EP_STEPS, num_rnd_traj=numRndTraj,
                        keepOutOf=False, toEnd=False)[1]
                    success  = np.sum(results==1) / numRndTraj
                    failure  = np.sum(results==-1)/ numRndTraj
                    unfinish = np.sum(results==0) / numRndTraj
                    trainProgress.append([success, failure, unfinish])
                    if verbose:
                        lr = self.actorOptimizer.state_dict()['param_groups'][0]['lr']
                        print('\nAfter [{:d}] updates:'.format(self.cntUpdate))
                        print('  - gamma={:.6f}, lr={:.1e}.'.format(
                            self.GAMMA, lr))
                        print('  - success/failure/unfinished ratio: {:.3f}, {:.3f}, {:.3f}'.format(
                            success, failure, unfinish))

                    if storeModel:
                        if saveBest:
                            if success > checkPointSucc:
                                checkPointSucc = success
                                self.save(self.cntUpdate, modelFolder)
                        else:
                            self.save(self.cntUpdate, modelFolder)

                    if plotFigure or storeFigure:
                        if showBool:
                            env.visualize(self.critic.Q1, actor_sim, vmin=0, boolPlot=True, addBias=addBias)
                        else:
                            env.visualize(self.critic.Q1, actor_sim, vmin=vmin, vmax=vmax, cmap='seismic', addBias=addBias)

                        if storeFigure:
                            figurePath = os.path.join(figureFolder,
                                '{:d}.png'.format(self.cntUpdate))
                            plt.savefig(figurePath)
                        if plotFigure:
                            plt.show()
                            plt.pause(0.001)
                            plt.close()

                # Perform one step of the optimization (on the target network)
                loss_q, loss_pi, loss_entropy = 0, 0, 0
                update_every = 100
                if self.cntUpdate % update_every == 0:
                    for timer in range(update_every):
                        loss_q, loss_pi, loss_entropy = self.update(timer)
                        trainingRecords.append([loss_q, loss_pi, loss_entropy])
                self.cntUpdate += 1
                # Update gamma, lr etc.
                self.updateHyperParam()

                # Terminate early
                if done:
                    break

        endLearning = time.time()
        timeInitBuffer = endInitBuffer - startInitBuffer
        timeInitQ = endInitQ - startInitQ
        timeLearning = endLearning - startLearning
        self.save(self.cntUpdate, '{:s}/model/'.format(outFolder))
        print('\nInitBuffer: {:.1f}, InitQ: {:.1f}, Learning: {:.1f}'.format(
            timeInitBuffer, timeInitQ, timeLearning))

        trainingRecords = np.array(trainingRecords)
        trainProgress = np.array(trainProgress)
        return trainingRecords, trainProgress


    def unpack_batch(self, batch):
        # `non_final_mask` is used for environments that have next state to be None
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.s_)),
            dtype=torch.bool).to(self.device)
        non_final_state_nxt = torch.FloatTensor([s for s in batch.s_ if s is not None]).to(self.device)
        state  = torch.FloatTensor(batch.s).to(self.device)
        action = torch.FloatTensor(batch.a).to(self.device).view(-1, self.actionSpace.shape[0])
        reward = torch.FloatTensor(batch.r).to(self.device)

        g_x = torch.FloatTensor([info['g_x'] for info in batch.info]).to(self.device).view(-1)
        l_x = torch.FloatTensor([info['l_x'] for info in batch.info]).to(self.device).view(-1)

        return non_final_mask, non_final_state_nxt, state, action, reward, g_x, l_x