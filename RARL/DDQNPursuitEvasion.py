"""
Please contact the author(s) of this library if you have any questions.
Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )

This module implements a special case of reach-avoid reinforcement learning
with double deep Q-network, called pursuit-evasion (or attack-defense) game.

Here we aim to minimize the upper value of the game.
    - u', d' = argmin_u' argmax_d' Q_policy(s', u', d'),
    - loss = E[ ( y - Q_policy(s,a) )^2 ]
"""

import torch
from torch import nn
from torch.nn.functional import smooth_l1_loss

import numpy as np
import matplotlib.pyplot as plt
import os

from .model import Model
from .DDQN import DDQN, Transition

import time


# == local functions ==
def actionIndexInt2Tuple(actionIdx, numActionList):
  """Transforms an action index to tuple of action indices.

  Args:
      actionIdx (int): action index of the discrete action set.
      numActionList (list): consists of the number of actions in evader and
          pursuer's action sets.

  Returns:
      tuple of int: indices of the Q-matrix.
  """
  numJoinAction = int(numActionList[0] * numActionList[1])
  assert (actionIdx < numJoinAction), (
      "The size of joint action set is "
      "{:d} but get index {:d}".format(numJoinAction, actionIdx)
  )
  rowIdx = actionIdx // numActionList[1]
  colIdx = actionIdx % numActionList[1]
  return (rowIdx, colIdx)


def actionIndexTuple2Int(actionIdxTuple, numActionList):
  """Transforms tuple of action indices to an action index.

  Args:
      actionIdxTuple (tuple of int): indices of the Q-matrix.
      numActionList (list): consists of the number of actions in evader and
          pursuer's action sets.

  Returns:
      int: action index of the discrete action set.
  """
  rowIdx, colIdx = actionIdxTuple
  assert (
      rowIdx < numActionList[0]
  ), "The size of evader's action set is {:d} but get index {:d}".format(
      numActionList[0], rowIdx
  )
  assert (
      colIdx < numActionList[1]
  ), "The size of pursuer's action set is {:d} but get index {:d}".format(
      numActionList[1], colIdx
  )

  actionIdx = numActionList[1] * rowIdx + colIdx
  return actionIdx


class DDQNPursuitEvasion(DDQN):
  """
  Implements the double deep Q-network algorithm. Supports minimizing the upper
  value of the pursuit-evasion game.

  Args:
      DDQN (object): an object implementing the basic utils functions.
  """

  def __init__(
      self, CONFIG, numActionList, dimList, mode="RA", terminalType="g",
      verbose=True
  ):
    """
    Initializes with a configuration object, environment information, neural
    network architecture, reinforcement learning algorithm type and type of
    the terminal value for reach-avoid reinforcement learning.

    Args:
        CONFIG (object): configuration.
        numActionList (list): the number of actions in the evader and pursuer's
            action sets.
        dimList (np.ndarray): dimensions of each layer in the neural network.
        mode (str, optional): the reinforcement learning mode.
            Defaults to 'RA'.
        terminalType (str, optional): type of the terminal value.
            Defaults to 'g'.
        verbose (bool, optional): print the messages if True. Defaults to True.
    """
    super(DDQNPursuitEvasion, self).__init__(CONFIG)

    self.mode = mode  # only supports 'RA'
    self.terminalType = terminalType

    # == ENV PARAM ==
    self.numJoinAction = int(numActionList[0] * numActionList[1])
    self.numActionList = numActionList

    # == Build neural network for (D)DQN ==
    assert dimList is not None, "Define the architecture"
    assert dimList[-1] == self.numJoinAction, (
        "We expect the dim of the last layer to be "
        + "{:d}, but get {:d}".format(self.numJoinAction, dimList[-1])
    )
    self.dimList = dimList
    self.actType = CONFIG.ACTIVATION
    self.build_network(dimList, self.actType, verbose)
    print(
        "DDQN: mode-{}; terminalType-{}".format(self.mode, self.terminalType)
    )

  def build_network(self, dimList, actType="Tanh", verbose=True):
    """Builds a neural network for the Q-network.

    Args:
        dimList (np.ndarray): dimensions of each layer in the neural network.
        actType (str, optional): activation function. Defaults to 'Tanh'.
        verbose (bool, optional): print the messages if True. Defaults to True.
    """
    self.Q_network = Model(dimList, actType, verbose=verbose)
    self.target_network = Model(dimList, actType)

    if self.device == torch.device("cuda"):
      self.Q_network.cuda()
      self.target_network.cuda()

    self.build_optimizer()

  def update(self):
    """Updates the Q-network using a batch of sampled replay transitions.

    Returns:
        float: critic loss.
    """
    if len(self.memory) < self.BATCH_SIZE * 20:
      return

    # == EXPERIENCE REPLAY ==
    transitions = self.memory.sample(self.BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043
    # for detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))
    (non_final_mask, non_final_state_nxt, state, action, reward, g_x,
     l_x) = self.unpack_batch(batch)

    # == get Q(s,a) ==
    # `gather` reguires that idx is Long and input and index should have the
    # same shape with only difference at the dimension we want to extract.
    # value out[i][j][k] = input[i][j][ index[i][j][k] ], which has the
    # same dim as index
    # -> state_action_values = Q [ i ][ action[i] ]
    # view(-1): from mtx to vector
    state_action_values = (
        self.Q_network(state).gather(dim=1, index=action).view(-1)
    )

    # == get a' ==
    # u', d' = argmin_u' argmax_d' Q_policy(s', u', d')
    # a' = tuple2Int(u', d')
    with torch.no_grad():
      num_non_final = non_final_state_nxt.shape[0]
      state_nxt_action_values = self.Q_network(non_final_state_nxt)
      Q_mtx = state_nxt_action_values.detach().reshape(
          num_non_final, self.numActionList[0], self.numActionList[1]
      )
      # minmax values and indices
      pursuerValues, colIndices = Q_mtx.max(dim=-1)
      _, rowIdx = pursuerValues.min(dim=-1)
      colIdx = colIndices[np.arange(num_non_final), rowIdx]
      action_nxt = [
          actionIndexTuple2Int((r, c), self.numActionList)
          for r, c in zip(rowIdx, colIdx)
      ]
      action_nxt = (torch.LongTensor(action_nxt).to(self.device).view(-1, 1))

    # == get expected value ==
    state_value_nxt = torch.zeros(self.BATCH_SIZE).to(self.device)

    with torch.no_grad():  # V(s') = Q_tar(s', a'), a' is from Q_policy
      if self.double_network:
        Q_expect = self.target_network(non_final_state_nxt)
      else:
        Q_expect = self.Q_network(non_final_state_nxt)
    state_value_nxt[non_final_mask] = \
        Q_expect.gather(dim=1, index=action_nxt).view(-1)

    # == Discounted Reach-Avoid Bellman Equation (DRABE) ==
    if self.mode == "RA":
      expected_state_action_values = (
          torch.zeros(self.BATCH_SIZE).float().to(self.device)
      )
      # Q(s, u) = V( f(s,u) )
      non_terminal = torch.max(
          g_x[non_final_mask],
          torch.min(l_x[non_final_mask], state_value_nxt[non_final_mask]),
      )
      terminal = torch.max(l_x, g_x)

      # normal state
      expected_state_action_values[non_final_mask] = (
          non_terminal * self.GAMMA + terminal[non_final_mask] *
          (1 - self.GAMMA)
      )
      # terminal state
      final_mask = torch.logical_not(non_final_mask)
      if self.terminalType == "g":
        expected_state_action_values[final_mask] = g_x[final_mask]
      elif self.terminalType == "max":
        expected_state_action_values[final_mask] = terminal[final_mask]
      else:
        raise ValueError("invalid terminalType")

    # == regression: Q(s, a) <- V(s) ==
    self.Q_network.train()
    loss = smooth_l1_loss(
        input=state_action_values,
        target=expected_state_action_values.detach(),
    )

    # == backpropagation ==
    self.optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(self.Q_network.parameters(), self.max_grad_norm)
    self.optimizer.step()

    self.update_target_network()

    return loss.item()

  def initBuffer(self, env):
    """Adds some transitions to the replay memory (buffer) randomly.

    Args:
        env (gym.Env): the environment we interact with.
    """
    cnt = 0
    while len(self.memory) < self.memory.capacity:
      cnt += 1
      print("\rWarmup Buffer [{:d}]".format(cnt), end="")
      s = env.reset()
      actionIdx, actionIdxTuple = self.select_action(s, explore=True)
      s_, r, done, info = env.step(actionIdxTuple)
      self.store_transition(s, actionIdx, r, s_, info)
    print("\n  => Warmup Buffer Ends")

  def initQ(
      self, env, warmupIter, outFolder, num_warmup_samples=200, vmin=-1,
      vmax=1, plotFigure=True, storeFigure=True
  ):
    """
    Initalizes the Q-network given that the environment can provide warmup
    examples with heuristic values.

    Args:
        env (gym.Env): the environment we interact with.
        warmupIter (int, optional): the number of iterations in the
            Q-network warmup.
        outFolder (str, optional): the path of the parent folder of model/ and
            figure/.
        num_warmup_samples (int, optional): the number of warmup samples.
            Defaults to 200.
        vmin (float, optional): the minmum value in the colorbar.
            Defaults to -1.
        vmax (float, optional): the maximum value in the colorbar.
            Defaults to 1.
        plotFigure (bool, optional): plot figures if True.
            Defaults to True.
        storeFigure (bool, optional): store figures if True.
            Defaults to False.

    Returns:
        np.ndarray: loss of fitting Q-values to heuristic values.
    """
    lossList = []
    for iterIdx in range(warmupIter):
      print("\rWarmup Q [{:d}]".format(iterIdx + 1), end="")
      states, heuristic_v = env.get_warmup_examples(
          num_warmup_samples=num_warmup_samples
      )

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

    if plotFigure or storeFigure:
      self.Q_network.eval()
      env.visualize(self.Q_network, vmin=vmin, vmax=vmax, cmap="seismic")
      if storeFigure:
        figureFolder = os.path.join(outFolder, "figure")
        os.makedirs(figureFolder, exist_ok=True)
        figurePath = os.path.join(figureFolder, "initQ.png")
        plt.savefig(figurePath)
      if plotFigure:
        plt.show()
        plt.pause(0.001)
        plt.close()
    self.target_network.load_state_dict(
        self.Q_network.state_dict()
    )  # hard replace
    self.build_optimizer()
    lossList = np.array(lossList)
    print("\n  => Warmup Q Ends")
    return lossList

  def learn(
      self, env, MAX_UPDATES=2000000, MAX_EP_STEPS=100, warmupBuffer=True,
      warmupQ=False, warmupIter=10000, doneTerminate=True, runningCostThr=None,
      curUpdates=None, checkPeriod=50000, plotFigure=True, storeFigure=False,
      showBool=False, vmin=-1, vmax=1, numRndTraj=200, storeModel=True,
      storeBest=False, outFolder="RA", verbose=True
  ):
    """Learns the Q function given the training hyper-parameters.

    Args:
        env (gym.Env): the environment we interact with.
        MAX_UPDATES (int, optional): the maximum number of gradient
            updates. Defaults to 2000000.
        MAX_EP_STEPS (int, optional): the number of steps in an episode.
            Defaults to 100.
        warmupBuffer (bool, optional): fill the replay buffer if True.
            Defaults to True.
        warmupQ (bool, optional): train the Q-network by (l_x, g_x) if
            True. Defaults to False.
        warmupIter (int, optional): the number of iterations in the
            Q-network warmup. Defaults to 10000.
        doneTerminate (bool, optional): end the episode when the agent
            crosses the boundary if True. Defaults to True.
        runningCostThr (float, optional): end the training if the running
            cost is smaller than the threshold. Defaults to None.
        curUpdates (int, optional): set the current number of updates
            (usually used when restoring trained models). Defaults to None.
        checkPeriod (int, optional): the period we check the performance.
            Defaults to 50000.
        plotFigure (bool, optional): plot figures if True. Defaults to True.
        storeFigure (bool, optional): store figures if True. Defaults to False.
        showBool (bool, optional): plot the sign of value function if True.
            Defaults to False.
        vmin (float, optional): the minimum value in the colorbar.
            Defaults to -1.
        vmax (float, optional): the maximum value in the colorbar.
            Defaults to 1.
        numRndTraj (int, optional): the number of random trajectories used
            to obtain the success ratio. Defaults to 200.
        storeModel (bool, optional): store models if True. Defaults to True.
        storeBest (bool, optional): only store the best model if True.
            Defaults to False.
        outFolder (str, optional): the path of the parent folder of model/ and
            figure/. Defaults to 'RA'.
        verbose (bool, optional): print the messages if True. Defaults to True.

    Returns:
        trainingRecords (np.ndarray): loss for every Q-network update.
        trainProgress (np.ndarray): each entry consists of the
            (success, failure, unfinished) ratio of random trajectories, which
            are checked periodically.
    """
    # == Warmup Buffer ==
    startInitBuffer = time.time()
    if warmupBuffer:
      self.initBuffer(env)
    endInitBuffer = time.time()

    # == Warmup Q ==
    startInitQ = time.time()
    if warmupQ:
      self.initQ(
          env, warmupIter=warmupIter, outFolder=outFolder,
          plotFigure=plotFigure, storeFigure=storeFigure, vmin=vmin, vmax=vmax
      )
    endInitQ = time.time()

    # == Main Training ==
    startLearning = time.time()
    trainingRecords = []
    runningCost = 0.0
    trainProgress = []
    checkPointSucc = 0.0
    ep = 0
    if curUpdates is not None:
      self.cntUpdate = curUpdates
      print("starting from {:d} updates".format(self.cntUpdate))

    if storeModel:
      modelFolder = os.path.join(outFolder, "model")
      os.makedirs(modelFolder, exist_ok=True)
    if storeFigure:
      figureFolder = os.path.join(outFolder, "figure")
      os.makedirs(figureFolder, exist_ok=True)

    while self.cntUpdate <= MAX_UPDATES:
      s = env.reset()
      epCost = 0.0
      ep += 1
      # Rollout
      for step_num in range(MAX_EP_STEPS):
        # Select action
        actionIdx, actionIdxTuple = self.select_action(s, explore=True)

        # Interact with env
        s_, r, done, info = env.step(actionIdxTuple)
        epCost += r

        # Store the transition in memory
        self.store_transition(s, actionIdx, r, s_, info)
        s = s_

        # Check after fixed number of gradient updates
        if self.cntUpdate != 0 and self.cntUpdate % checkPeriod == 0:
          self.Q_network.eval()
          _, results, _ = env.simulate_trajectories(
              self.Q_network, T=MAX_EP_STEPS, num_rnd_traj=numRndTraj,
              keepOutOf=False, toEnd=False
          )
          success = np.sum(results == 1) / numRndTraj
          failure = np.sum(results == -1) / numRndTraj
          unfinish = np.sum(results == 0) / numRndTraj
          trainProgress.append([success, failure, unfinish])
          if verbose:
            lr = self.optimizer.state_dict()["param_groups"][0]["lr"]
            print("\n\nAfter [{:d}] updates:".format(self.cntUpdate))
            print(
                "  - eps={:.2f}, gamma={:.6f}, lr={:.1e}.".format(
                    self.EPSILON, self.GAMMA, lr
                )
            )
            print(
                "  - success/failure/unfinished ratio: "
                + "{:.3f}, {:.3f}, {:.3f}".format(success, failure, unfinish)
            )

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
              env.visualize(self.Q_network, vmin=0, boolPlot=True)
            else:
              env.visualize(
                  self.Q_network, vmin=vmin, vmax=vmax, cmap="seismic"
              )
            if storeFigure:
              figurePath = os.path.join(
                  figureFolder, "{:d}.png".format(self.cntUpdate)
              )
              plt.savefig(figurePath)
            if plotFigure:
              plt.show()
              plt.pause(0.001)
            plt.clf()
            plt.close('all')

        # Perform one step of the optimization (on the target network)
        lossC = self.update()
        trainingRecords.append(lossC)
        self.cntUpdate += 1
        self.updateHyperParam()

        # Terminate early
        if done and doneTerminate:
          break

      # Rollout report
      runningCost = runningCost*0.9 + epCost*0.1
      if verbose:
        print(
            "\r[{:d}-{:d}]: ".format(ep, self.cntUpdate)
            + "This episode gets running/episode cost = "
            + "({:3.2f}/{:.2f}) after {:d} steps.".
            format(runningCost, epCost, step_num + 1),
            end="",
        )

      # Check stopping criteria
      if runningCostThr is not None:
        if runningCost <= runningCostThr:
          print(
              "\n At Updates[{:3.0f}] Solved!".format(self.cntUpdate)
              + " Running cost is now {:3.2f}!".format(runningCost)
          )
          env.close()
          break
    endLearning = time.time()
    timeInitBuffer = endInitBuffer - startInitBuffer
    timeInitQ = endInitQ - startInitQ
    timeLearning = endLearning - startLearning
    self.save(self.cntUpdate, modelFolder)
    print(
        "\nInitBuffer: {:.1f}, InitQ: {:.1f}, Learning: {:.1f}".format(
            timeInitBuffer, timeInitQ, timeLearning
        )
    )
    trainingRecords = np.array(trainingRecords)
    trainProgress = np.array(trainProgress)
    return trainingRecords, trainProgress

  def select_action(self, state, explore=False):
    """Selects an action given the state and conditioned on `explore` flag.

    Args:
        state (np.ndarray): the state of the environment.
        explore (bool, optional): randomize the deterministic action by
            epsilon-greedy algorithm if True. Defaults to False.

    Returns:
        int: action index
        tuple of int: indices to access the Q-value matrix.
    """
    if (np.random.rand() < self.EPSILON) and explore:
      actionIdx = np.random.randint(0, self.numJoinAction)
      actionIdxTuple = actionIndexInt2Tuple(actionIdx, self.numActionList)
    else:
      self.Q_network.eval()
      state = torch.from_numpy(state).float().to(self.device)
      state_action_values = self.Q_network(state)
      Q_mtx = (
          state_action_values.detach().cpu().reshape(
              self.numActionList[0], self.numActionList[1]
          )
      )
      pursuerValues, colIndices = Q_mtx.max(dim=1)
      _, rowIdx = pursuerValues.min(dim=0)
      colIdx = colIndices[rowIdx]
      actionIdxTuple = (np.array(rowIdx), np.array(colIdx))
      actionIdx = actionIndexTuple2Int(actionIdxTuple, self.numActionList)
    return actionIdx, actionIdxTuple
