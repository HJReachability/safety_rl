# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )

# TODO
# Here we aim to minimize the cost. We make the following two modifications:
#  - u', d' = argmin_u' argmax_d' Q_policy(s', u', d'), 
#  
#  - loss = E[ ( y - Q_policy(s,a) )^2 ]

# // - a' = argmin_a' Q_policy(s', a')
# // - V(s') = Q_tar(s', a')
# // - V(s) = gamma ( max{ g(s), min{ l(s), V_better(s') } } + (1-gamma) max{ g(s), l(s) },
# //   where V_better(s') = max{ g(s'), min{ l(s'), V(s') } }
# // - loss = E[ ( V(s) - Q_policy(s,a) )^2 ]

import torch
import torch.nn as nn
from torch.nn.functional import mse_loss, smooth_l1_loss

from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
import os

from .model import model
from .DDQN import DDQN, Transition

def actionIndexInt2Tuple(actionIdx, numActionList):
    numJoinAction = int(numActionList[0] * numActionList[1])
    assert actionIdx < numJoinAction, \
        "The size of joint action set is {:d} but get index {:d}".format(
        numJoinAction, actionIdx)
    rowIdx = actionIdx // numActionList[1]
    colIdx = actionIdx % numActionList[1]
    return (rowIdx, colIdx)

def actionIndexTuple2Int(actionIdxTuple, numActionList):
    rowIdx, colIdx = actionIdxTuple
    assert rowIdx < numActionList[0], \
        "The size of evader's action set is {:d} but get index {:d}".format(
        numActionList[0], rowIdx)
    assert colIdx < numActionList[1], \
        "The size of pursuer's action set is {:d} but get index {:d}".format(
        numActionList[1], colIdx)

    actionIdx = numActionList[1] * rowIdx + colIdx
    return actionIdx


class DDQNPursuitEvasion(DDQN):
    def __init__(self, CONFIG, numActionList, dimList, mode='RA', actType='Tanh'):
        """
        __init__

        Args:
            CONFIG ([type]): configuration.
            numActionList ([int]): the dimensions of the action sets of the evader
                and the pursuer.
            dimList ([int]): dimensions of each layer in the NN
            mode (str, optional): the learning mode. Defaults to 'RA'.
            actType (str, optional): the type of activation function in the NN.
                Defaults to 'Tanh'.
        """                    
        super(DDQNPursuitEvasion, self).__init__(CONFIG)
        
        self.mode = mode # 'normal' or 'RA'

        #== ENV PARAM ==
        self.numJoinAction = int(numActionList[0] * numActionList[1])
        self.numActionList = numActionList

        #== Build NN for (D)DQN ==
        assert dimList is not None, "Define the architecture"
        assert dimList[-1] == self.numJoinAction, \
            "We expect the dim of the last layer to be {:d}, but get {:d}".format(self.numJoinAction, dimList[-1])
        self.dimList = dimList
        self.actType = actType
        self.build_network(dimList, actType)


    def build_network(self, dimList, actType='Tanh'):
        self.Q_network = model(dimList, actType, verbose=True)
        self.target_network = model(dimList, actType)

        if self.device == torch.device('cuda'):
            self.Q_network.cuda()
            self.target_network.cuda()

        self.build_optimizer()


    def update(self, addBias=False):
        if len(self.memory) < self.BATCH_SIZE*20:
            return

        #== EXPERIENCE REPLAY ==
        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for detailed explanation).
        # This converts batch-array of Transitions to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # `non_final_mask` is used for environments that have next state to be None
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.s_)),
            dtype=torch.bool).to(self.device)
        non_final_state_nxt = torch.FloatTensor([s for s in batch.s_ if s is not None]).to(self.device)
        state  = torch.FloatTensor(batch.s).to(self.device)
        action = torch.LongTensor(batch.a).to(self.device).view(-1,1)
        reward = torch.FloatTensor(batch.r).to(self.device)
        if self.mode == 'RA':
            g_x = torch.FloatTensor([info['g_x'] for info in batch.info]).to(self.device).view(-1)
            l_x = torch.FloatTensor([info['l_x'] for info in batch.info]).to(self.device).view(-1)
            g_x_nxt = torch.FloatTensor([info['g_x_nxt'] for info in batch.info]).to(self.device).view(-1)
            l_x_nxt = torch.FloatTensor([info['l_x_nxt'] for info in batch.info]).to(self.device).view(-1)

        #== get Q(s,a) ==
        # `gather` reguires idx to be Long, input and index should have the same shape
        # with only difference at the dimension we want to extract value
        # out[i][j][k] = input[i][j][ index[i][j][k] ], which has the same dim as index
        # -> state_action_values = Q [ i ][ action[i] ]
        # view(-1): from mtx to vector
        state_action_values = self.Q_network(state).gather(dim=1, index=action).view(-1)

        # ? >>> CHECK IF THIS IS CORRECT
        #== get a' ==
        # u', d' = argmin_u' argmax_d' Q_policy(s', u', d')
        # a' = tuple2Int(u', d')
        with torch.no_grad():
            num_non_final = non_final_state_nxt.shape[0]
            state_nxt_action_values = self.Q_network(non_final_state_nxt)
            Q_mtx = state_nxt_action_values.detach().reshape(num_non_final, self.numActionList[0], self.numActionList[1])
            # minmax values and indices
            pursuerValues, colIndices = Q_mtx.max(dim=-1)
            minmaxValue, rowIdx = pursuerValues.min(dim=-1)
            colIdx = colIndices[np.arange(num_non_final), rowIdx]
            action_nxt = [actionIndexTuple2Int((r,c), self.numActionList) for r, c in zip(rowIdx, colIdx)]
            action_nxt = torch.LongTensor(action_nxt).to(self.device).view(-1,1)
        # ? <<<

        #== get expected value ==
        state_value_nxt = torch.zeros(self.BATCH_SIZE).to(self.device)

        with torch.no_grad(): # V(s') = Q_tar(s', a'), a' is from Q_policy
            if self.double:
                Q_expect = self.target_network(non_final_state_nxt)
            else:
                Q_expect = self.Q_network(non_final_state_nxt)
        state_value_nxt[non_final_mask] = Q_expect.gather(dim=1, index=action_nxt).view(-1)

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
                V_better = torch.max( g_x_nxt, torch.min(l_x_nxt, state_value_nxt))
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


    def initBuffer(self, env):
        cnt = 0
        while len(self.memory) < self.memory.capacity:
            cnt += 1
            print('\rWarmup Buffer [{:d}]'.format(cnt), end='')
            s = env.reset()
            actionIdx, actionIdxTuple = self.select_action(s, explore=True)
            s_, r, done, info = env.step(actionIdxTuple)
            self.store_transition(s, actionIdx, r, s_, info)
        print(" --- Warmup Buffer Ends")


    def initQ(self, env, warmupIter, num_warmup_samples=200, vmin=-1, vmax=1):
        lossList = []
        for iterIdx in range(warmupIter):
            print('\rWarmup Q [{:d}]'.format(iterIdx+1), end='')
            states, heuristic_v = env.get_warmup_examples(num_warmup_samples=num_warmup_samples)

            self.Q_network.train()
            heuristic_v = torch.from_numpy(heuristic_v).float().to(self.device)
            states = torch.from_numpy(states).float().to(self.device)
            v = self.Q_network(states)
            loss = smooth_l1_loss(input=v, target=heuristic_v)
            lossList.append(loss.data.cpu().numpy())

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.Q_network.parameters(), self.max_grad_norm)
            self.optimizer.step()

            if (iterIdx+1) % 10000 == 0:
                self.Q_network.eval()
                print()
                fig, axes = plt.subplots(1,4, figsize=(16, 4))

                xPursuerList=[.1, .3, .5, .7]
                yPursuerList=[.1, .3, .5, .7]
                for i, (ax, xPursuer, yPursuer) in enumerate(zip(axes, xPursuerList, yPursuerList)):
                    cbarPlot = i==3
                    env.plot_formatting(ax=ax)
                    env.plot_target_failure_set(ax=ax, xPursuer=xPursuer, yPursuer=yPursuer)
                    env.plot_v_values(self.Q_network, ax=ax, fig=fig, cbarPlot=cbarPlot,
                                            xPursuer=xPursuer, yPursuer=yPursuer, cmap='seismic', vmin=-1, vmax=1)
                plt.pause(0.001)

        print(" --- Warmup Q Ends")
        # self.Q_network.eval()
        # env.visualize(self.Q_network, vmin=vmin, vmax=vmax, cmap='seismic')
        # plt.pause(0.001)
        self.target_network.load_state_dict(self.Q_network.state_dict()) # hard replace
        self.build_optimizer()
        lossList = np.array(lossList)
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
        if warmupBuffer:
            self.initBuffer(env)

        # == Warmup Q ==
        if warmupQ:
            self.initQ(env, warmupIter=warmupIter, num_warmup_samples=200, vmin=vmin, vmax=vmax)

        # == Main Training ==
        TrainingRecord = namedtuple('TrainingRecord', ['ep', 'runningCost', 'cost', 'lossC'])
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
                # ? >>> CHECK IF THIS IS CORRECT
                # Select action
                actionIdx, actionIdxTuple = self.select_action(s, explore=True)

                # Interact with env
                s_, r, done, info = env.step(actionIdxTuple)
                epCost += r

                # Store the transition in memory
                self.store_transition(s, actionIdx, r, s_, info)
                s = s_
                # ? <<<

                # Check after fixed number of gradient updates
                if self.cntUpdate != 0 and self.cntUpdate % checkPeriod == 0:
                    self.Q_network.eval()
                    _, results = env.simulate_trajectories( self.Q_network, T=MAX_EP_STEPS, 
                                                            num_rnd_traj=numRndTraj,
                                                            keepOutOf=False, toEnd=False)
                    success  = np.sum(results==1) / numRndTraj
                    failure  = np.sum(results==-1)/ numRndTraj
                    unfinish = np.sum(results==0) / numRndTraj
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
                        self.Q_network.eval()
                        if showBool:
                            env.visualize(self.Q_network, vmin=0, boolPlot=True, addBias=addBias)
                        else:
                            env.visualize(self.Q_network, vmin=vmin, vmax=vmax, cmap='seismic', addBias=addBias)
                        if storeFigure:
                            figureFolder = 'figure/{:s}/'.format(outFolder)
                            os.makedirs(figureFolder, exist_ok=True)
                            plt.savefig('{:s}/{:d}.png'.format(figureFolder, self.cntUpdate))
                        if plotFigure:
                            plt.pause(0.001)

                # Perform one step of the optimization (on the target network)
                lossC = self.update(addBias=addBias)
                self.cntUpdate += 1
                self.updateHyperParam()

                # Terminate early
                if done and doneTerminate:
                    break

            # Rollout report
            runningCost = runningCost * 0.9 + epCost * 0.1
            trainingRecords.append(TrainingRecord(ep, runningCost, epCost, lossC))
            if verbose:
                print('\r{:3.0f}: This episode gets running/episode cost = ({:3.2f}/{:.2f}) after {:d} steps.'.format(\
                    ep, runningCost, epCost, step_num+1), end=' ')
                print('The agent currently updates {:d} times'.format(self.cntUpdate), end='\t\t')

            # Check stopping criteria
            if runningCostThr != None:
                if runningCost <= runningCostThr:
                    print("\n At Updates[{:3.0f}] Solved! Running cost is now {:3.2f}!".format(self.cntUpdate, runningCost))
                    env.close()
                    break
        print()
        self.save(self.cntUpdate, 'models/{:s}/'.format(outFolder))
        return trainingRecords, trainProgress


    def select_action(self, state, explore=False):
        if (np.random.rand() < self.EPSILON) and explore:
            actionIdx = np.random.randint(0, self.numJoinAction)
            actionIdxTuple = actionIndexInt2Tuple(actionIdx, self.numActionList)
        else:
            self.Q_network.eval()
            state = torch.from_numpy(state).float().to(self.device)
            state_action_values = self.Q_network(state)
            Q_mtx = state_action_values.detach().cpu().reshape(self.numActionList[0], self.numActionList[1])
            pursuerValues, colIndices = Q_mtx.max(dim=1)
            minmaxValue, rowIdx = pursuerValues.min(dim=0)
            colIdx = colIndices[rowIdx]
            actionIdxTuple = (np.array(rowIdx), np.array(colIdx))
            actionIdx = actionIndexTuple2Int(actionIdxTuple, self.numActionList)
        return actionIdx, actionIdxTuple