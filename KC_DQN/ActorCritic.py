# Please contact the author(s) of this library if you have any questions.
# Authors: Vicenc Rubies-Royo (vrubies@berkeley.edu)
#          Kai-Chieh Hsu ( kaichieh@princeton.edu )

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
#           when update_actor, use samples action
#       + TD3: DeterministicPolicy, with target
#           when update_actor, use the deterministic action
#       + select_action() calls self.actor.sample()
#   - Replay Buffer
#       + (extension) prioritized experience replay
#   - Hyper-Parameter Scheduler
#       + learning rate: Q1, Q2, Actor
#       + contraction factor: gamma
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
#       + __init__ (o)
#       + store_transition (o)
#       + select_action (o)
#       + save (o)
#       + restore (o)


import torch
import torch.nn as nn
from torch.nn.functional import mse_loss, smooth_l1_loss
from torch.optim import AdamW, Adam
from torch.optim import lr_scheduler

from collections import namedtuple
import numpy as np
import os
import time
from copy import deepcopy
import matplotlib.pyplot as plt
import pickle

from .model import StepLR, StepLRMargin, TwinnedQNetwork, StepResetLR
from .ReplayMemory import ReplayMemory
from .utils import soft_update, save_model


Transition = namedtuple('Transition', ['s', 'a', 'r', 's_', 'info'])


class ActorCritic(object):
    def __init__(self, actorType, CONFIG, actionSpace):
        """
        __init__ : initializes actor-critic model.

        Args:
            actorType (str): The type of actor model, Currently supports SAC, TD3.
            CONFIG (dict): configurations.
        """
        self.CONFIG = CONFIG
        self.saved = False
        self.actorType = actorType
        self.memory = ReplayMemory(CONFIG.MEMORY_CAPACITY)

        #== ENV PARAM ==
        self.actionSpace = actionSpace

        #== PARAM ==

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
        self.start_updates = 1000
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
    def build_network(self, dimLists, actType={'critic':'Tanh', 'actor':'Tanh'},
        verbose=True):
        self.build_critic(dimLists[0], actType['critic'], verbose=verbose)
        self.build_actor(dimLists[1], actType['actor'], verbose=verbose)
        self.build_optimizer()


    def build_actor(self):
        raise NotImplementedError


    def build_critic(self, dimList, actType='Tanh', verbose=True):
        self.critic = TwinnedQNetwork(dimList, actType, self.device, verbose=verbose)
        self.criticTarget = deepcopy(self.critic)
        for p in self.criticTarget.parameters():
            p.requires_grad = False


    def build_optimizer(self):
        self.criticOptimizer = Adam(self.critic.parameters(), lr=self.LR_C)
        self.actorOptimizer = Adam(self.actor.parameters(), lr=self.LR_A)
        self.criticScheduler = lr_scheduler.StepLR(self.criticOptimizer,
            step_size=self.LR_C_PERIOD, gamma=self.LR_C_DECAY)
        self.actorScheduler = lr_scheduler.StepLR(self.actorOptimizer,
            step_size=self.LR_A_PERIOD, gamma=self.LR_A_DECAY)
        self.max_grad_norm = 1
        self.cntUpdate = 0
    # * BUILD NETWORK ENDS


    # * LEARN STARTS
    def initBuffer(self, env, ratio=1.): # in child class
        raise NotImplementedError


    def initQ(self): # in child class
        raise NotImplementedError


    def update_critic_hyperParam(self):
        if self.criticOptimizer.state_dict()['param_groups'][0]['lr'] <= self.LR_C_END:
            for param_group in self.criticOptimizer.param_groups:
                param_group['lr'] = self.LR_C_END
        else:
            self.criticScheduler.step()

        self.GammaScheduler.step()
        self.GAMMA = self.GammaScheduler.get_variable()


    def update_actor_hyperParam(self):
        if self.actorOptimizer.state_dict()['param_groups'][0]['lr'] <= self.LR_A_END:
            for param_group in self.actorOptimizer.param_groups:
                param_group['lr'] = self.LR_A_END
        else:
            self.actorScheduler.step()


    def updateHyperParam(self):
        self.update_critic_hyperParam()
        self.update_actor_hyperParam()


    def update_target_networks(self):
        soft_update(self.criticTarget, self.critic, self.TAU)
        if self.actorType == 'TD3':
            soft_update(self.actorTarget, self.actor, self.TAU)


    def update_critic(self, batch, addBias=False): # in child class
        raise NotImplementedError


    def update_actor(self): # in child class
        raise NotImplementedError


    def update(self, timer, update_period=2):
        if len(self.memory) < self.start_updates:
            return 0.0, 0.0

        #== EXPERIENCE REPLAY ==
        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for detailed explanation).
        # This converts batch-array of Transitions to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        loss_q = self.update_critic(batch)
        loss_pi = 0.0
        if timer % update_period == 0:
            loss_pi = self.update_actor(batch)
            print('\r{:d}: (q, pi) = ({:3.5f}/{:3.5f}).'.format(
                self.cntUpdate, loss_q, loss_pi), end=' ')

        self.update_target_networks()

        return loss_q, loss_pi


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
        trainingRecords = []
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
                if self.cntUpdate > max(warmupIter, self.start_updates):
                    with torch.no_grad():
                        a, _ = self.actor.sample(
                            torch.from_numpy(s).float().to(self.device))
                        a = a.cpu().numpy()
                else:
                    a = env.action_space.sample()
                    # a = self.genRandomActions(1)[0]

                # Interact with env
                s_, r, done, info = env.step(a)
                s_ = None if done else s_
                # env.render()
                epCost = max(info["g_x"], min(epCost, info["l_x"]))

                # Store the transition in memory
                self.store_transition(s, a, r, s_, info)
                s = s_

                # Check after fixed number of gradient updates
                if self.cntUpdate != 0 and self.cntUpdate % checkPeriod == 0:
                    actor_sim = self.actor
                    # if self.actorType == 'SAC':
                    #     actor_sim = lambda x: self.actor.sample(x,deterministic=True)
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
                loss_q, loss_pi = 0, 0
                update_every = 100
                if self.cntUpdate % update_every == 0:
                    for timer in range(update_every):
                        loss_q, loss_pi = self.update(timer)
                        trainingRecords.append([loss_q, loss_pi])
                self.cntUpdate += 1
                # Update gamma, lr etc.
                self.updateHyperParam()

                # Terminate early
                if done:
                    break

            # Rollout report
            # runningCost = runningCost * 0.9 + epCost * 0.1
            # trainingRecords.append(TrainingRecord(ep, runningCost, epCost, loss_q, loss_pi))
            # if verbose:
            #     print('\r{:3.0f}: This episode gets running/episode cost = ({:3.2f}/{:.2f}) and losses = ({:3.2f}/{:.2f}) after {:d} steps.'.format(\
            #         ep, runningCost, epCost, loss_q, loss_pi, step_num+1), end=' ')
            #     print('The agent currently updates {:d} times.'.format(self.cntUpdate), end='\t\t')

            # # Check stopping criteria
            # if runningCostThr != None:
            #     if runningCost <= runningCostThr:
            #         print("\n At Updates[{:3.0f}] Solved! Running cost is now {:3.2f}!".format(self.cntUpdate, runningCost))
            #         env.close()
            #         break
        endLearning = time.time()
        timeInitBuffer = endInitBuffer - startInitBuffer
        timeInitQ = endInitQ - startInitQ
        timeLearning = endLearning - startLearning
        self.save(self.cntUpdate, modelFolder)
        print('\nInitBuffer: {:.1f}, InitQ: {:.1f}, Learning: {:.1f}'.format(
            timeInitBuffer, timeInitQ, timeLearning))

        trainingRecords = np.array(trainingRecords)
        trainProgress = np.array(trainProgress)
        return trainingRecords, trainProgress
    # * LEARN ENDS


    # * OTHERS STARTS
    def store_transition(self, *args):
        self.memory.update(Transition(*args))


    def save(self, step, logs_path):
        logs_path_critic = os.path.join(logs_path, 'critic/')
        logs_path_actor = os.path.join(logs_path, 'actor/')
        save_model(self.critic, step, logs_path_critic, 'critic', self.MAX_MODEL)
        save_model(self.actor,  step, logs_path_actor, 'actor',  self.MAX_MODEL)
        if not self.saved:
            config_path = os.path.join(logs_path, "CONFIG.pkl")
            pickle.dump(self.CONFIG, open(config_path, "wb"))
            self.saved = True


    def restore(self, step, logs_path):
        logs_path_critic = os.path.join(logs_path, 'model/critic/critic-{}.pth'.format(step))
        logs_path_actor  = os.path.join(logs_path, 'model/actor/actor-{}.pth'.format(step))
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
        print('  <= Restore {}' .format(logs_path))


    def genRandomActions(self, num_actions):
        UB = self.actionSpace.high
        LB = self.actionSpace.low
        dim = UB.shape[0]
        actions = (UB - LB) * np.random.rand(num_actions, dim) + LB
        return actions


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
    # * OTHERS ENDS