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
#       + __init__ (o)
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

from .model import StepLR, StepLRMargin, TwinnedQNetwork
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
        self.actorType = actorType
        self.memory = ReplayMemory(CONFIG.MEMORY_CAPACITY)

        #== ENV PARAM ==
        self.actionSpace = actionSpace

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
    def build_network(self, dimList, actType=['Tanh', 'Tanh']):
        self.build_critic(dimList[0], actType[0])
        self.build_actor(dimList[1], actType[1])
        self.build_optimizer()


    def build_actor(self, dimList, actType='Tanh'): # in child class
        raise NotImplementedError


    def build_critic(self, dimList, actType='Tanh'):
        self.critic = TwinnedQNetwork(dimList, actType, self.device)
        self.criticTarget = TwinnedQNetwork(dimList, actType, self.device)


    def build_optimizer(self):
        self.criticOptimizer = AdamW(self.critic.parameters(), lr=self.LR_C,
            weight_decay=1e-3)
        self.ActorOptimizer = AdamW(self.actor.parameters(), lr=self.LR_A,
            weight_decay=1e-3)
        self.criticScheduler = lr_scheduler.StepLR(self.criticOptimizer,
            step_size=self.LR_C_PERIOD, gamma=self.LR_C_DECAY)
        self.ActorScheduler = lr_scheduler.StepLR(self.ActorOptimizer,
            step_size=self.LR_A_PERIOD, gamma=self.LR_A_DECAY)
        # self.Q2Optimizer = AdamW(self.critic.Q2.parameters(), lr=self.LR_C,
        #   weight_decay=1e-3)
        # self.Q2Scheduler = lr_scheduler.StepLR(self.Q2Optimizer,
        #   step_size=self.LR_C_PERIOD, gamma=self.LR_C_DECAY)
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


    def updateHyperParam(self):
        self.update_critic_hyperParam()
        self.update_actor_hyperParam()


    def update_target_network(self):
        soft_update(self.criticTarget.Q1, self.critic.Q1, self.TAU)
        soft_update(self.criticTarget.Q2, self.critic.Q2, self.TAU)
        if actorType == 'TD3':
            soft_update(self.actorTarget, self.actor, self.TAU)


    def update_critic(self, batch, addBias=False): # in child class
        raise NotImplementedError


    def update_actor(self): # in child class
        raise NotImplementedError


    def update(self, update_period=2):
        if len(self.memory) < self.BATCH_SIZE*20:
            return

        #== EXPERIENCE REPLAY ==
        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for detailed explanation).
        # This converts batch-array of Transitions to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        loss_q = self.update_critic(batch)
        loss_pi = 0.0
        if self.cntUpdate % update_period == 0:
            loss_pi = self.update_actor(batch)
        self.update_target_network()

        return loss_q, loss_pi


    def learn(  self, env, MAX_UPDATES=2000000, MAX_EP_STEPS=100,
                warmupBuffer=True, warmupQ=False, warmupIter=10000,
                addBias=False, doneTerminate=True, runningCostThr=None,
                curUpdates=None, checkPeriod=50000, 
                plotFigure=True, storeFigure=False,
                showBool=False, vmin=-1, vmax=1, numRndTraj=200,
                storeModel=True, storeBest=False, 
                outFolder='RA', verbose=True):
        """
        learn: Learns the value function.

        Args:
            env (gym.Env Obj.): environment.
            MAX_UPDATES (int, optional): the maximal number of gradient 
                updates. Defaults to 2000000.
            MAX_EP_STEPS (int, optional): the number of steps in an episode. 
                Defaults to 100.
            warmupBuffer (bool, optional): fill the replay buffer if True.
                Defaults to True.
            warmupQ (bool, optional): train the Q-network by (l_x, g_x) if 
                True. Defaults to False.
            warmupIter (int, optional): the number of iterations in the 
                Q-network warmup. Defaults to 10000.
            addBias (bool, optional): use biased version of value function if 
                True. Defaults to False.
            doneTerminate (bool, optional): ends the episode when the agent 
                crosses the boundary if True. Defaults to True.
            runningCostThr (float, optional): ends the training if the running 
                cost is smaller than the threshold. Defaults to None.
            curUpdates (int, optional): set the current number of updates 
                (usually used when restoring trained models). Defaults to None.
            checkPeriod (int, optional): the period we check the performance.
                Defaults to 50000.
            plotFigure (bool, optional): plot figures if True. Defaults to True.
            storeFigure (bool, optional): store figures if True. Defaults to 
                False.
            showBool (bool, optional): use bool value function if True. 
                Defaults to False.
            vmin (float, optional): the minimal value in the colorbar. Defaults 
                to -1.
            vmax (float, optional): the maximal value in the colorbar. Defaults 
                to 1.
            numRndTraj (int, optional): the number of random trajectories used 
                to obtain the success ratio. Defaults to 200.
            storeModel (bool, optional): store models if True. Defaults to True.
            storeBest (bool, optional): only store the best model if True. 
                Defaults to False.
            outFolder (str, optional): the relative folder path with respect to 
                models/ and figure/. Defaults to 'RA'.
            verbose (bool, optional): output message if True. Defaults to True.

        Returns:
            trainingRecords (List): each entry consists of  ['ep', 
                'runningCost', 'cost', 'lossC'] after every episode.
            trainProgress (List): each entry consists of the 
                success/failure/unfinished ratio of random trajectories and is
                checked periodically.
        """

        # == Warmup Buffer ==
        startInitBuffer = time.time()
        if warmupBuffer:
            self.initBuffer(env)
        endInitBuffer = time.time()

        # == Warmup Q ==
        startInitQ = time.time()
        if warmupQ:
            self.initQ(env, warmupIter=warmupIter, outFolder=outFolder,
                plotFigure=plotFigure, storeFigure=storeFigure)
        endInitQ = time.time()

        # == Main Training ==
        startLearning = time.time()
        TrainingRecord = namedtuple('TrainingRecord', ['ep', 'runningCost', 'cost', 'loss_q', 'loss_pi'])
        trainingRecords = []
        runningCost = 0.
        trainProgress = []
        checkPointSucc = 0.
        ep = 0
        if curUpdates is not None:
            self.cntUpdate = curUpdates
            print("starting from {:d} updates".format(self.cntUpdate))
        while self.cntUpdate <= MAX_UPDATES:
            s = env.reset()
            epCost = 0.
            ep += 1
            # Rollout
            for step_num in range(MAX_EP_STEPS):
                # Select action
                a, _ = self.actor.sample(s)

                # Interact with env
                s_, r, done, info = env.step(a)
                epCost += r

                # Store the transition in memory
                self.store_transition(s, a, r, s_, info)
                s = s_

                # Check after fixed number of gradient updates
                # if self.cntUpdate != 0 and self.cntUpdate % checkPeriod == 0:
                #     results= env.simulate_trajectories(self.Q_network,
                #         T=MAX_EP_STEPS, num_rnd_traj=numRndTraj,
                #         keepOutOf=False, toEnd=False)[1]
                #     success  = np.sum(results==1) / numRndTraj
                #     failure  = np.sum(results==-1)/ numRndTraj
                #     unfinish = np.sum(results==0) / numRndTraj
                #     trainProgress.append([success, failure, unfinish])
                #     if verbose:
                #         lr = self.optimizer.state_dict()['param_groups'][0]['lr']
                #         print('\nAfter [{:d}] updates:'.format(self.cntUpdate))
                #         print('  - eps={:.2f}, gamma={:.6f}, lr={:.1e}.'.format(
                #             self.EPSILON, self.GAMMA, lr))
                #         print('  - success/failure/unfinished ratio: {:.3f}, {:.3f}, {:.3f}'.format(
                #             success, failure, unfinish))

                #     if storeModel:
                #         if storeBest:
                #             if success > checkPointSucc:
                #                 checkPointSucc = success
                #                 self.save(self.cntUpdate, '{:s}/model/'.format(outFolder))
                #         else:
                #             self.save(self.cntUpdate, '{:s}/model/'.format(outFolder))

                #     if plotFigure or storeFigure:
                #         self.Q_network.eval()
                #         if showBool:
                #             env.visualize(self.Q_network, vmin=0, boolPlot=True, addBias=addBias)
                #         else:
                #             env.visualize(self.Q_network, vmin=vmin, vmax=vmax, cmap='seismic', addBias=addBias)
                #         if storeFigure:
                #             figureFolder = '{:s}/figure/'.format(outFolder)
                #             os.makedirs(figureFolder, exist_ok=True)
                #             plt.savefig('{:s}{:d}.png'.format(figureFolder, self.cntUpdate))
                #         if plotFigure:
                #             plt.show()
                #             plt.pause(0.001)
                #             plt.close()

                # Perform one step of the optimization (on the target network)
                loss_q, loss_pi = self.update()
                self.cntUpdate += 1
                self.updateHyperParam()

                # Terminate early
                if done and doneTerminate:
                    break

            # Rollout report
            runningCost = runningCost * 0.9 + epCost * 0.1
            trainingRecords.append(TrainingRecord(ep, runningCost, epCost, loss_q, loss_pi))
            if verbose:
                print('\r{:3.0f}: This episode gets running/episode cost = ({:3.2f}/{:.2f}) after {:d} steps.'.format(\
                    ep, runningCost, epCost, step_num+1), end=' ')
                print('The agent currently updates {:d} times.'.format(self.cntUpdate), end='\t\t')

            # Check stopping criteria
            if runningCostThr != None:
                if runningCost <= runningCostThr:
                    print("\n At Updates[{:3.0f}] Solved! Running cost is now {:3.2f}!".format(self.cntUpdate, runningCost))
                    env.close()
                    break
        endLearning = time.time()
        timeInitBuffer = endInitBuffer - startInitBuffer
        timeInitQ = endInitQ - startInitQ
        timeLearning = endLearning - startLearning
        self.save(self.cntUpdate, '{:s}/model/'.format(outFolder))
        print('\nInitBuffer: {:.1f}, InitQ: {:.1f}, Learning: {:.1f}'.format(
            timeInitBuffer, timeInitQ, timeLearning))
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


    def genRandomActions(self, num_actions):
        UB = self.actionSpace.high
        LB = self.actionSpace.low
        dim = UB.shape[0]
        actions = (UB - LB) * np.random.rand(num_warmup_samples, dim) + LB
        return actions
    # * OTHERS ENDS