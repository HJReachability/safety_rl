# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )

# Here we aim to minimize the cost. We make the following two modifications:
#  - a' = argmin_a' Q_policy(s', a')
#  - V(s') = Q_tar(s', a')
#  - V(s) = gamma ( max{ g(s), min{ l(s), V(s') } } + (1-gamma) max{ g(s), l(s) }
#  - loss = E[ ( V(f(s,a)) - Q_policy(s,a) )^2 ]

import torch
import torch.nn as nn
from torch.nn.functional import mse_loss, smooth_l1_loss

import numpy as np
import matplotlib.pyplot as plt
import os
import time

from .model import model
from .DDQN import DDQN, Transition

class DDQNSingle(DDQN):
    def __init__(self, CONFIG, numAction, actionList, dimList,
        mode='RA', terminalType='g', verbose=True):
        """
        __init__

        Args:
            CONFIG (object): configuration.
            numAction (int): #actions.
            actionList (list): action set.
            dimList (np.ndarray): dimensions of each layer in the NN.
            mode (str, optional): the learning mode. Defaults to 'RA'.
            terminalType (str, optional): terminal value. Defaults to 'g'.
            verbose (bool, optional): print or not. Defaults to True.
        """
        super(DDQNSingle, self).__init__(CONFIG)

        self.mode = mode # 'normal' or 'RA'
        self.terminalType = terminalType

        #== ENV PARAM ==
        self.numAction = numAction
        self.actionList = actionList

        #== Build NN for (D)DQN ==
        self.dimList = dimList
        self.actType = CONFIG.ACTIVATION
        self.build_network(dimList, self.actType, verbose)
        print("DDQN: mode-{}; terminalType-{}".format(
            self.mode, self.terminalType))


    def build_network(self, dimList, actType='Tanh', verbose=True):
        """
        build_network

        Args:
            dimList (np.ndarray): dimensions of each layer in the NN.
            actType (str, optional): activation function. Defaults to 'Tanh'.
            verbose (bool, optional): print or not. Defaults to True.
        """
        self.Q_network = model(dimList, actType, verbose=verbose)
        self.target_network = model(dimList, actType)

        if self.device == torch.device('cuda'):
            self.Q_network.cuda()
            self.target_network.cuda()

        self.build_optimizer()


    def update(self, addBias=False):
        """
        update: update the critic.

        Args:
            addBias (bool, optional): use biased version of value function if True.
                Defaults to False.

        Returns:
            float: critic loss.
        """
        if len(self.memory) < self.BATCH_SIZE*20:
            return

        #== EXPERIENCE REPLAY ==
        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for detailed explanation).
        # This converts batch-array of Transitions to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))
        non_final_mask, non_final_state_nxt, state, action, reward, g_x, l_x = \
            self.unpack_batch(batch)

        #== get Q(s,a) ==
        # `gather` reguires idx to be Long, input and index should have the same shape
        # with only difference at the dimension we want to extract value
        # out[i][j][k] = input[i][j][ index[i][j][k] ], which has the same dim as index
        # -> state_action_values = Q [ i ][ action[i] ]
        # view(-1): from mtx to vector
        self.Q_network.train()
        state_action_values = self.Q_network(state).gather(dim=1, index=action).view(-1)

        #== get a' by Q_policy: a' = argmin_a' Q_policy(s', a') ==
        with torch.no_grad():
            self.Q_network.eval()
            action_nxt = self.Q_network(non_final_state_nxt).min(1, keepdim=True)[1]

        #== get expected value ==
        state_value_nxt = torch.zeros(self.BATCH_SIZE).to(self.device)

        with torch.no_grad(): # V(s') = Q_tar(s', a'), a' is from Q_policy
            if self.double:
                self.target_network.eval()
                Q_expect = self.target_network(non_final_state_nxt)
            else:
                self.Q_network.eval()
                Q_expect = self.Q_network(non_final_state_nxt)
        state_value_nxt[non_final_mask] = Q_expect.gather(dim=1, index=action_nxt).view(-1)

        #== Discounted Reach-Avoid Bellman Equation (DRABE) ==
        if self.mode == 'RA':
            expected_state_action_values = torch.zeros(self.BATCH_SIZE).float().to(self.device)
            if addBias: # Bias version: V(s) = gamma ( max{ g(s), min{ l(s), V_diff(s') } } - max{ g(s), l(s) } ),
                        # where V_diff(s') = V(s') + max{ g(s'), l(s') }
                min_term = torch.min(l_x, state_value_nxt+torch.max(l_x, g_x))
                terminal = torch.max(l_x, g_x)
                non_terminal = torch.max(min_term, g_x) - terminal
                expected_state_action_values[non_final_mask] = self.GAMMA * non_terminal[non_final_mask]
                expected_state_action_values[torch.logical_not(non_final_mask)] = terminal[torch.logical_not(non_final_mask)]
            else:
                # Better version instead of DRABE on the paper (discussed on Nov. 18, 2020)
                # V(s) = gamma ( max{ g(s), min{ l(s), V_better(s') } } + (1-gamma) max{ g(s), l(s) },
                # where V_better(s') = max{ g(s'), min{ l(s'), V(s') } }
                # Another version (discussed on Feb. 22, 2021):
                    # we want Q(s, u) = V( f(s,u) )
                non_terminal = torch.max(
                    g_x[non_final_mask],
                    torch.min(
                        l_x[non_final_mask],
                        state_value_nxt[non_final_mask]
                    )
                )
                terminal = torch.max(l_x, g_x)

                # normal state
                expected_state_action_values[non_final_mask] = \
                    non_terminal * self.GAMMA + \
                    terminal[non_final_mask] * (1-self.GAMMA)

                # terminal state
                final_mask = torch.logical_not(non_final_mask)
                if self.terminalType == 'g':
                    expected_state_action_values[final_mask] = g_x[final_mask]
                elif self.terminalType == 'max':
                    expected_state_action_values[final_mask] = terminal[final_mask]
                else:
                    raise ValueError("invalid terminalType")
        else: # V(s) = c(s, a) + gamma * V(s')
            expected_state_action_values = state_value_nxt * self.GAMMA + reward

        #== regression: Q(s, a) <- V(s) ==
        loss = smooth_l1_loss(input=state_action_values, target=expected_state_action_values.detach())

        #== backpropagation ==
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.Q_network.parameters(), self.max_grad_norm)
        self.optimizer.step()

        self.update_target_network()

        return loss.item()


    def initBuffer(self, env):
        """
        initBuffer: randomly put some transitions into the memory replay buffer.

        Args:
            env (gym.Env Obj.): environment.
        """
        cnt = 0
        while len(self.memory) < self.memory.capacity:
            cnt += 1
            print('\rWarmup Buffer [{:d}]'.format(cnt), end='')
            s = env.reset()
            a, a_idx = self.select_action(s, explore=True)
            s_, r, done, info = env.step(a_idx)
            s_ = None if done else s_
            self.store_transition(s, a, r, s_, info)
            if done:
                s = env.reset()
            else:
                s = s_
        print(" --- Warmup Buffer Ends")


    def initQ(  self, env, warmupIter, outFolder, num_warmup_samples=200,
                vmin=-1, vmax=1, plotFigure=True, storeFigure=True):
        """
        initQ: initalize Q-network.

        Args:
            env (gym.Env Obj.): environment.
            warmupIter (int, optional): the number of iterations in the
                Q-network warmup.
            outFolder (str, optional): the relative folder path with respect to
                model/ and figure/.
            num_warmup_samples (int, optional): Defaults to 200.
            vmin (float, optional): the minimal value in the colorbar. Defaults to -1.
            vmax (float, optional): the maximal value in the colorbar. Defaults to 1.
            plotFigure (bool, optional): plot figures if True. Defaults to True.
            storeFigure (bool, optional): store figures if True. Defaults to False.

        Returns:
            np.ndarray: loss of fitting Q-values to heuristic values.
        """
        lossList = np.empty(warmupIter, dtype=float)
        for ep_tmp in range(warmupIter):
            states, heuristic_v = env.get_warmup_examples(num_warmup_samples=num_warmup_samples)

            self.Q_network.train()
            heuristic_v = torch.from_numpy(heuristic_v).float().to(self.device)
            states = torch.from_numpy(states).float().to(self.device)
            v = self.Q_network(states)
            # loss = smooth_l1_loss(input=v, target=heuristic_v)
            loss = mse_loss(input=v, target=heuristic_v, reduction='sum')

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.Q_network.parameters(), self.max_grad_norm)
            self.optimizer.step()
            lossList[ep_tmp] = loss.detach().cpu().numpy()
            print('\rWarmup Q [{:d}]. MSE = {:f}'.format(ep_tmp+1, loss),end='')

        print(" --- Warmup Q Ends")
        if plotFigure or storeFigure:
            self.Q_network.eval()
            env.visualize(self.Q_network, vmin=vmin, vmax=vmax, cmap='seismic')
            if storeFigure:
                figureFolder = os.path.join(outFolder, 'figure')
                os.makedirs(figureFolder, exist_ok=True)
                figurePath = os.path.join(figureFolder, 'initQ.png')
                plt.savefig(figurePath)
            if plotFigure:
                plt.show()
                plt.pause(0.001)
                plt.close()
        self.target_network.load_state_dict(self.Q_network.state_dict()) # hard replace
        self.build_optimizer()

        return lossList


    def learn(  self, env, MAX_UPDATES=2000000, MAX_EP_STEPS=100,
                warmupBuffer=True, warmupQ=False, warmupIter=10000,
                addBias=False, doneTerminate=True, runningCostThr=None,
                curUpdates=None, checkPeriod=50000,
                plotFigure=True, storeFigure=False,
                showBool=False, vmin=-1, vmax=1, numRndTraj=200,
                storeModel=True, storeBest=False,
                outFolder='RA', verbose=True):
        """
        learn: Learns the vlaue function.

        Args:
            env (gym.Env Obj.): environment.
            MAX_UPDATES (int, optional): the maximal number of gradient updates.
                Defaults to 2000000.
            MAX_EP_STEPS (int, optional): the number of steps in an episode.
                Defaults to 100.
            warmupBuffer (bool, optional): fill the replay buffer if True.
                Defaults to True.
            warmupQ (bool, optional): train the Q-network by (l_x, g_x) if True.
                Defaults to False.
            warmupIter (int, optional): the number of iterations in the
                Q-network warmup. Defaults to 10000.
            addBias (bool, optional): use biased version of value function if True.
                Defaults to False.
            doneTerminate (bool, optional): ends the episode when the agent
                crosses the boundary if True. Defaults to True.
            runningCostThr (float, optional): ends the training if the running
                cost is smaller than the threshold. Defaults to None.
            curUpdates (int, optional): set the current number of updates
                (usually used when restoring trained models). Defaults to None.
            checkPeriod (int, optional): the period we check the performance.
                Defaults to 50000.
            plotFigure (bool, optional): plot figures if True. Defaults to True.
            storeFigure (bool, optional): store figures if True. Defaults to False.
            showBool (bool, optional): use bool value function if True.
                Defaults to False.
            vmin (float, optional): the minimal value in the colorbar. Defaults to -1.
            vmax (float, optional): the maximal value in the colorbar. Defaults to 1.
            numRndTraj (int, optional): the number of random trajectories used
                to obtain the success ratio. Defaults to 200.
            storeModel (bool, optional): store models if True. Defaults to True.
            storeBest (bool, optional): only store the best model if True.
                Defaults to False.
            outFolder (str, optional): the relative folder path with respect to
                model/ and figure/. Defaults to 'RA'.
            verbose (bool, optional): output message if True. Defaults to True.

        Returns:
            trainingRecords (np.ndarray): loss_critic for every update.
            trainProgress (np.ndarray): each entry consists of the
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
                    plotFigure=plotFigure, storeFigure=storeFigure,
                    vmin=vmin, vmax=vmax)
        endInitQ = time.time()

        # == Main Training ==
        startLearning = time.time()
        trainingRecords = []
        runningCost = 0.
        trainProgress = []
        checkPointSucc = 0.
        ep = 0

        if curUpdates is not None:
            self.cntUpdate = curUpdates
            print("starting from {:d} updates".format(self.cntUpdate))

        if storeModel:
            modelFolder = os.path.join(outFolder, 'model')
            os.makedirs(modelFolder, exist_ok=True)
        if storeFigure:
            figureFolder = os.path.join(outFolder, 'figure')
            os.makedirs(figureFolder, exist_ok=True)

        while self.cntUpdate <= MAX_UPDATES:
            s = env.reset()
            epCost = 0.
            ep += 1
            # Rollout
            for step_num in range(MAX_EP_STEPS):
                # Select action
                a, a_idx = self.select_action(s, explore=True)

                # Interact with env
                s_, r, done, info = env.step(a_idx)
                s_ = None if done else s_
                epCost += r

                # Store the transition in memory
                self.store_transition(s, a_idx, r, s_, info)
                s = s_

                # Check after fixed number of gradient updates
                if self.cntUpdate != 0 and self.cntUpdate % checkPeriod == 0:
                    results= env.simulate_trajectories(self.Q_network,
                        T=MAX_EP_STEPS, num_rnd_traj=numRndTraj,
                        keepOutOf=False, toEnd=False)[1]
                    success  = np.sum(results==1) / results.shape[0]
                    failure  = np.sum(results==-1)/ results.shape[0]
                    unfinish = np.sum(results==0) / results.shape[0]
                    trainProgress.append([success, failure, unfinish])
                    if verbose:
                        lr = self.optimizer.state_dict()['param_groups'][0]['lr']
                        print('\nAfter [{:d}] updates:'.format(self.cntUpdate))
                        print('  - eps={:.2f}, gamma={:.6f}, lr={:.1e}.'.format(
                            self.EPSILON, self.GAMMA, lr))
                        print('  - success/failure/unfinished ratio:', end=' ')
                        with np.printoptions(formatter={'float': '{: .3f}'.format}):
                            print(np.array([success, failure, unfinish]))
                        # print('{:.3f}, {:.3f}, {:.3f}'.format(success, failure, unfinish))

                    if storeModel:
                        if storeBest:
                            if success > checkPointSucc:
                                checkPointSucc = success
                                self.save(self.cntUpdate, modelFolder)
                        else:
                            self.save(self.cntUpdate, modelFolder)

                    if plotFigure or storeFigure:
                        self.Q_network.eval()
                        if showBool:
                            env.visualize(self.Q_network, vmin=0,
                                boolPlot=True, addBias=addBias)
                        else:
                            env.visualize(self.Q_network, vmin=vmin,
                                vmax=vmax, cmap='seismic', addBias=addBias)
                        if storeFigure:
                            figurePath = os.path.join(figureFolder,
                                '{:d}.png'.format(self.cntUpdate))
                            plt.savefig(figurePath)
                        if plotFigure:
                            plt.show()
                            plt.pause(0.001)
                        plt.close()

                # Perform one step of the optimization (on the target network)
                lossC = self.update(addBias=addBias)
                trainingRecords.append(lossC)
                self.cntUpdate += 1
                self.updateHyperParam()

                # Terminate early
                if done and doneTerminate:
                    break

            # Rollout report
            runningCost = runningCost * 0.9 + epCost * 0.1
            if verbose:
                print('\r[{:d}-{:d}]: This episode gets running/episode cost = ({:3.2f}/{:.2f}) after {:d} steps.'.format(\
                    ep, self.cntUpdate, runningCost, epCost, step_num+1), end='')

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
        self.save(self.cntUpdate, modelFolder)
        print('\nInitBuffer: {:.1f}, InitQ: {:.1f}, Learning: {:.1f}'.format(
            timeInitBuffer, timeInitQ, timeLearning))
        trainingRecords = np.array(trainingRecords)
        trainProgress = np.array(trainProgress)
        return trainingRecords, trainProgress


    def select_action(self, state, explore=False):
        """
        select_action

        Args:
            state (np.ndarray): state
            explore (bool, optional): randomized the deterministic action by
                epsilon-greedy. Defaults to False.

        Returns:
            np.ndarray: action
            int: action index
        """
        self.Q_network.eval()
        # tensor.min() returns (value, indices), which are in tensor form
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        if (np.random.rand() < self.EPSILON) and explore:
            action_index = np.random.randint(0, self.numAction)
        else:
            action_index = self.Q_network(state).min(dim=1)[1].item()
        return self.actionList[action_index], action_index


    # def unpack_batch(self, batch):
    #     """
    #     unpack_batch: decompose batch into different variables.

    #     Args:
    #         batch (object): Transition of batch-arrays.

    #     Returns:
    #         tuple of torch.Tensor.
    #     """
    #     # `non_final_mask` is used for environments that have next state to be None
    #     non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.s_)),
    #         dtype=torch.bool).to(self.device)
    #     non_final_state_nxt = torch.FloatTensor([s for s in batch.s_ if s is not None]).to(self.device)
    #     state  = torch.FloatTensor(batch.s).to(self.device)
    #     action = torch.LongTensor(batch.a).to(self.device).view(-1,1)
    #     reward = torch.FloatTensor(batch.r).to(self.device)

    #     g_x = torch.FloatTensor([info['g_x'] for info in batch.info]).to(self.device).view(-1)
    #     l_x = torch.FloatTensor([info['l_x'] for info in batch.info]).to(self.device).view(-1)

    #     return non_final_mask, non_final_state_nxt, state, action, reward, g_x, l_x