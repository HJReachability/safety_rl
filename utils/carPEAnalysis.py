"""
Please contact the author(s) of this library if you have any questions.
Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import pickle
import gym
import os

import sys

sys.path.append("..")
from RARL.config import dqnConfig
from RARL.DDQNPursuitEvasion import DDQNPursuitEvasion

# == PLOTTING ==
tiffany = "#0abab5"


def plotTrajStep(
    state, env, agent, c=[tiffany, "y"], lw=2, nx=101, ny=101, toEnd=False,
    T=100
):
  """Plots the trajectory step-by-step.

  Args:
      state (np.ndarray): the initial state of the trajectory.
      env (gym.Env): the environment.
      agent (RARL.DDQNPursuitEvasion): the agent.
      c (list, optional): the colors of the evader and the pursuer.
          Defaults to [tiffany, 'y'].
      lw (int, optional): the linewidth of the trajectory. Defaults to 2.
      nx (int, optional): the number of grids in the x direction.
          Defaults to 101.
      ny (int, optional): the number of grids in the y direction.
          Defaults to 101.
      toEnd (bool, optional): end the rollout only after the agent goes
          beyond the boundary. Defaults to False.
      T (int, optional): the maximum number of steps in a trajectory.
          Defaults to 100.

  Returns:
      tuple of np.ndarrays: the first array stores the reach-avoid outcome so
          far, the second array stores the target margin at each time step,
          and the third array stores the safety margin at each time step.

  Example:
      state = np.array([-.9, 0., 0., -.5, -.3, .75*np.pi])
      valueList, lxList, gxList = plotTrajStep(state, env, agent)
  """
  trajEvader, trajPursuer, result, minV, info = env.simulate_one_trajectory(
      agent.Q_network, T=T, state=state, toEnd=toEnd
  )
  valueList = info["valueList"]
  gxList = info["gxList"]
  lxList = info["lxList"]
  print("trajectory length is {:d}".format(trajEvader.shape[0]))

  # == PLOT ==
  trajEvaderX = trajEvader[:, 0]
  trajEvaderY = trajEvader[:, 1]
  trajPursuerX = trajPursuer[:, 0]
  trajPursuerY = trajPursuer[:, 1]
  numCol = 5
  numRow = int(np.ceil(trajEvader.shape[0] / numCol))
  numAx = int(numRow * numCol)
  fig, axes = plt.subplots(numRow, numCol, figsize=(4 * numCol, 4 * numRow))

  for i in range(trajEvader.shape[0]):
    print("{:d}/{:d}".format(i + 1, numAx), end="\r")
    rowIdx = int(i / numCol)
    colIdx = i % numCol
    if numRow > 1:
      ax = axes[rowIdx][colIdx]
    else:
      ax = axes[colIdx]
    xPursuer = trajPursuer[i, 0]
    yPursuer = trajPursuer[i, 1]
    thetaPursuer = trajPursuer[i, 2]
    theta = trajEvader[i, 2]

    cbarPlot = (i % numCol) == (numCol - 1)
    ax.scatter(trajEvaderX[i], trajEvaderY[i], s=48, c=c[0], zorder=3)
    ax.plot(
        trajEvaderX[i:], trajEvaderY[i:], color=c[0], linewidth=lw, zorder=2
    )
    ax.scatter(trajPursuerX[i], trajPursuerY[i], s=48, c=c[1], zorder=3)
    ax.plot(
        trajPursuerX[i:], trajPursuerY[i:], color=c[1], linewidth=lw, zorder=2
    )

    env.plot_formatting(ax=ax)
    env.plot_target_failure_set(ax=ax, xPursuer=xPursuer, yPursuer=yPursuer)
    env.plot_v_values(
        agent.Q_network, ax=ax, fig=fig, cbarPlot=cbarPlot, theta=theta,
        xPursuer=xPursuer, yPursuer=yPursuer, thetaPursuer=thetaPursuer,
        cmap="seismic", vmin=-0.5, vmax=0.5, nx=nx, ny=ny
    )
    ax.set_title(r"$v={:.3f}$".format(valueList[i]), fontsize=16)
  return valueList, lxList, gxList


def plotCM(
    fig, ax, cm, target_names=["0", "1"], labels=["", ""], fontsize=20,
    thresh=0.5, cmap="viridis", cbarPlot=False
):
  """Plots the confusion matrix.

  Args:
      fig (matplotlib.Figure): the figure to plot.
      ax (matplotlib.axes.Axes): the ax to plot.
      cm (np.ndarray): the confusion matrix.
      target_names (list, optional): the laels of targets.
          Defaults to ["0", "1"].
      labels (list, optional): the labels of x and y axes.
          Defaults to ["", ""].
      fontsize (int, optional): Defaults to 20.
      thresh (float, optional): the threshold to change font color.
          Defaults to 0.5.
      cmap (str, optional): the colormap. Defaults to "viridis".
      cbarPlot (bool, optional): plot the color bar if True. Defaults to False.
  """

  im = ax.imshow(cm, interpolation="none", cmap=cmap, vmin=0, vmax=1.0)
  if cbarPlot:
    cbar = fig.colorbar(
        im, ax=ax, pad=0.01, fraction=0.05, shrink=0.75, ticks=[0, 0.5, 1.0]
    )
    cbar.ax.set_yticklabels(labels=[0, 0.5, 1.0], fontsize=fontsize - 4)

  if target_names is not None:
    tick_marks = np.arange(len(target_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(target_names, rotation=0, fontsize=fontsize - 4)
    ax.set_yticklabels(target_names, rotation=90, fontsize=fontsize - 4)

  it = np.nditer(cm, flags=["multi_index"])
  while not it.finished:
    i, j = it.multi_index

    ax.text(
        j, i, "{:0.3f}".format(cm[i, j]), horizontalalignment="center",
        color="k" if cm[i, j] > thresh else "w", fontsize=fontsize
    )
    it.iternext()
  ax.set_xlabel(labels[0], fontsize=fontsize)
  ax.set_ylabel(labels[1], fontsize=fontsize)


def plotAndObtainValueDictIdx(
    env, dictList, testIdxList, indices, instantList=None, showCapture=False,
    maxCol=10, maxRow=2, width=2, height=2, lw=1.5, s=48
):
  """Plots the trajectories and get the reach-avoid values for each index.

  Args:
      env (gym.Env): environment.
      dictList (list): list of dictionaries storing the rollout results.
      testIdxList (list): the index of states by `genValSamples.py`
      indices (list): list of indices to plot.
      instantList (list, optional): list of instants to specify.
          Defaults to None.
      showCapture (bool, optional): plot the captured/failure instant if True.
          Defaults to False.
      maxCol (int, optional): the maximum number of columns in the figure.
          Defaults to 10.
      maxRow (int, optional): the maximum number of rows in the figure.
          Defaults to 2.
      width (int, optional): the width of each subfigure. Defaults to 2.
      height (int, optional): the height of each subfigure. Defaults to 2.
      lw (float, optional): linewidth of the trajectories. Defaults to 1.5.
      s (int, optional): size of scatter plot. Defaults to 48.
  """
  numCol = min(len(indices), maxCol)
  numRow = min(int(np.ceil(len(indices) / numCol)), maxRow)
  numAx = int(numRow * numCol)

  figWidth = width * numCol
  figHeight = height * numRow
  fig, axes = plt.subplots(numRow, numCol, figsize=(figWidth, figHeight))
  valueList = np.empty(shape=(len(indices),), dtype=float)

  for i, pick in enumerate(indices):
    print("{:d}/{:d}".format(i + 1, len(indices)), end="\r")
    if instantList is not None:
      instant = instantList[i]
    dictTmp = dictList[pick]
    testIdx = testIdxList[pick]
    maxminV = dictTmp["maxminV"]
    valueList[i] = maxminV

    # = PLOT =
    if i < numAx:
      rowIdx = int(i / numCol)
      colIdx = i % numCol
      if numRow > 1:
        ax = axes[rowIdx][colIdx]
      elif numCol > 1:
        ax = axes[colIdx]
      else:
        ax = axes
      trajEvaderTmp = dictTmp["trajEvader"]
      trajPursuerTmp = dictTmp["trajPursuer"]

      traj_x = trajEvaderTmp[:, 0]
      traj_y = trajEvaderTmp[:, 1]
      ax.scatter(traj_x[0], traj_y[0], s=s, c="#0abab5")
      ax.plot(traj_x, traj_y, color="#0abab5", linewidth=lw)
      if instantList is not None:
        markerColor = "b" if showCapture else "r"
        ax.scatter(
            traj_x[instant], traj_y[instant], marker="x", s=s, c=markerColor,
            zorder=4
        )

      traj_x = trajPursuerTmp[:, 0]
      traj_y = trajPursuerTmp[:, 1]
      ax.scatter(traj_x[0], traj_y[0], s=s, c="y")
      ax.plot(traj_x, traj_y, color="y", linewidth=lw)
      if instantList is not None:
        if showCapture:
          env.plot_target_failure_set(
              ax=ax, xPursuer=traj_x[instant], yPursuer=traj_y[instant], lw=lw
          )
          ax.scatter(
              traj_x[instant], traj_y[instant], marker="x", s=s, c="b",
              zorder=4
          )
        else:
          env.plot_target_failure_set(ax, showCapture=False, lw=lw)
      else:
        env.plot_target_failure_set(ax, showCapture=False, lw=lw)
      env.plot_formatting(ax=ax)
      ax.set_title("[{:d}]: {:.3f}".format(testIdx, maxminV), fontsize=12)
      ax.set_xticklabels([])
      ax.set_yticklabels([])
  plt.tight_layout()
  plt.show()

  return valueList


# == DATA ANALYSIS ==
def generateCM(labelValue, predictValue):
  """Generates the confusion matrix and rteturn it.

  Args:
      labelValue (np.ndarray): true values.
      predictValue (np.ndarray): predicted values.
  """
  FPMtx = np.logical_and((labelValue <= 0), (predictValue > 0))
  FPIndices = np.argwhere(FPMtx)
  FPNum = np.sum(FPMtx)

  FNMtx = np.logical_and((labelValue > 0), (predictValue <= 0))
  FNIndices = np.argwhere(FNMtx)
  FNNum = np.sum(FNMtx)

  TPMtx = np.logical_and((labelValue > 0), (predictValue > 0))
  TPIndices = np.argwhere(TPMtx)
  TPNum = np.sum(TPMtx)

  TNMtx = np.logical_and((labelValue <= 0), (predictValue <= 0))
  TNIndices = np.argwhere(TNMtx)
  TNNum = np.sum(TNMtx)

  accuracy = (TPNum+TNNum) / (TPNum+TNNum+FPNum+FNNum)
  FPrate = FPNum / (FPNum+TNNum)
  FNrate = FNNum / (TPNum+FNNum)
  TNrate = TNNum / (FPNum+TNNum)
  TPrate = TPNum / (TPNum+FNNum)

  print(
      "TP: {:.0f}, FN: {:.0f}, FP: {:.0f}, TN: {:.0f}".format(
          TPNum, FNNum, FPNum, TNNum
      )
  )
  cm = np.array([[TPrate, FNrate], [FPrate, TNrate]])

  return cm, accuracy, TPIndices, FNIndices, FPIndices, TNIndices


# == GENERATE TRAJECTORIES ==
def pursuerResponse(env, agent, statePursuer, trajEvader):
  """
  Rollouts the pursuer's policy given the initial state and the trajectory of
  the evader.

  Args:
      env (gym.Env): the environment.
      agent (RARL.DDQNPursuitEvasion): the agent.
      statePursuer (np.ndarray): the initial state of the pursuer.
      trajEvader (np.ndarray): the trajectory of the evader.

  Returns:
      np.ndarray: trajectory of the pursuer.
      int: the (categorical) outcome of the reach-avoid game.
      float: the (numerical) outcome of the reach-avoid game.
      dict: the information of the trajectory.
  """
  trajPursuer = []
  result = 0  # not finished

  valueList = []
  gxList = []
  lxList = []
  for t in range(trajEvader.shape[0]):
    stateEvader = trajEvader[t]
    trajPursuer.append(statePursuer)
    state = np.concatenate((stateEvader, statePursuer), axis=0)
    donePursuer = not env.pursuer.check_within_bounds(statePursuer)

    g_x = env.safety_margin(state)
    l_x = env.target_margin(state)

    # = Rollout Record
    if t == 0:
      maxG = g_x
      current = max(l_x, maxG)
      minV = current
    else:
      maxG = max(maxG, g_x)
      current = max(l_x, maxG)
      minV = min(current, minV)

    valueList.append(minV)
    gxList.append(g_x)
    lxList.append(l_x)

    # = Dynamics
    stateTensor = torch.FloatTensor(state).to(env.device)
    with torch.no_grad():
      state_action_values = agent.Q_network(stateTensor)
    Q_mtx = state_action_values.reshape(
        env.numActionList[0], env.numActionList[1]
    )
    pursuerValues, colIndices = Q_mtx.max(dim=1)
    minmaxValue, rowIdx = pursuerValues.min(dim=0)
    colIdx = colIndices[rowIdx]

    # If cars are within the boundary, we update their states according
    # to the controls
    if not donePursuer:
      uPursuer = env.pursuer.discrete_controls[colIdx]
      statePursuer = env.pursuer.integrate_forward(statePursuer, uPursuer)

  trajPursuer = np.array(trajPursuer)
  info = {"valueList": valueList, "gxList": gxList, "lxList": lxList}
  return trajPursuer, result, minV, info


def exhaustiveDefenderSearch(env, agent, state, actionSeq, maxLength=40):
  """
  Verifies the evader's policy given a pursuer's defensive strategy. It is a
  subprocess of validateEvaderPolicy.

  Args:
      env (gym.Env): the environment.
      agent (RARL.DDQNPursuitEvasion): the agent.
      state (np.ndarray): the initial state of the pursuer and the evader.
      actionSeq (list of ints): the indices of the pursuer's action set. It
          controls pursuer's action within each chunk.
      maxLength (int, optional): the maximum length of the trajectory.
          Defaults to 40.

  Returns:
      np.ndarray: trajectory of the evader.
      np.ndarray: trajectory of the pursuer.
      float: the (numerical) outcome of the reach-avoid game.
      dict: the information of the trajectory.
  """
  numChunk = actionSeq.shape[0]
  chunkLength = int(np.ceil(maxLength / numChunk))
  stateEvader = state[:3]
  statePursuer = state[3:]
  trajPursuer = [statePursuer]
  trajEvader = [stateEvader]
  valueList = []
  gxList = []
  lxList = []
  pursuerActionSeqIdx = 0

  for t in range(maxLength):
    state = np.concatenate((stateEvader, statePursuer), axis=0)

    g_x = env.safety_margin(state)
    l_x = env.target_margin(state)

    # = Rollout Record
    if t == 0:
      maxG = g_x
      current = max(l_x, maxG)
      minV = current
    else:
      maxG = max(maxG, g_x)
      current = max(l_x, maxG)
      minV = min(current, minV)

    valueList.append(minV)
    gxList.append(g_x)
    lxList.append(l_x)

    # = Dynamics
    stateTensor = torch.FloatTensor(state).to(env.device)
    with torch.no_grad():
      state_action_values = agent.Q_network(stateTensor)
    Q_mtx = state_action_values.reshape(
        env.numActionList[0], env.numActionList[1]
    )
    pursuerValues, _ = Q_mtx.max(dim=1)
    _, rowIdx = pursuerValues.min(dim=0)

    uEvader = env.evader.discrete_controls[rowIdx]
    stateEvader = env.evader.integrate_forward(stateEvader, uEvader)
    actionIdx = actionSeq[pursuerActionSeqIdx]
    uPursuer = env.pursuer.discrete_controls[actionIdx]
    statePursuer = env.pursuer.integrate_forward(statePursuer, uPursuer)

    trajPursuer.append(statePursuer)
    trajEvader.append(stateEvader)
    if (t+1) % chunkLength == 0:
      pursuerActionSeqIdx += 1

  trajEvader = np.array(trajEvader)
  trajPursuer = np.array(trajPursuer)
  info = {"valueList": valueList, "gxList": gxList, "lxList": lxList}
  return trajEvader, trajPursuer, minV, info


def exhaustiveAttackerSearch(env, agent, state, actionSeq, maxLength=40):
  """Verifies the pursuer's policy given an evader's attacking strategy.

  Args:
      env (gym.Env): the environment.
      agent (RARL.DDQNPursuitEvasion): the agent.
      state (np.ndarray): the initial state of the pursuer and the evader.
      actionSeq (list of ints): the indices of the pursuer's action set. It
          controls pursuer's action within each chunk.
      maxLength (int, optional): the maximum length of the trajectory.
          Defaults to 40.

  Returns:
      np.ndarray: trajectory of the evader.
      np.ndarray: trajectory of the pursuer.
      float: the (numerical) outcome of the reach-avoid game.
      dict: the information of the trajectory.
  """
  numChunk = actionSeq.shape[0]
  chunkLength = int(np.ceil(maxLength / numChunk))
  stateEvader = state[:3]
  statePursuer = state[3:]
  trajPursuer = [statePursuer]
  trajEvader = [stateEvader]
  valueList = []
  gxList = []
  lxList = []
  pursuerActionSeqIdx = 0

  for t in range(maxLength):
    state = np.concatenate((stateEvader, statePursuer), axis=0)
    doneEvader = not env.evader.check_within_bounds(stateEvader)
    donePursuer = not env.pursuer.check_within_bounds(statePursuer)

    g_x = env.safety_margin(state)
    l_x = env.target_margin(state)

    # = Rollout Record
    if t == 0:
      maxG = g_x
      current = max(l_x, maxG)
      minV = current
    else:
      maxG = max(maxG, g_x)
      current = max(l_x, maxG)
      minV = min(current, minV)

    valueList.append(minV)
    gxList.append(g_x)
    lxList.append(l_x)

    # = Dynamics
    stateTensor = torch.FloatTensor(state).to(env.device)
    with torch.no_grad():
      state_action_values = agent.Q_network(stateTensor)
    Q_mtx = state_action_values.reshape(
        env.numActionList[0], env.numActionList[1]
    )
    pursuerValues, colIndices = Q_mtx.max(dim=1)
    minmaxValue, rowIdx = pursuerValues.min(dim=0)
    # colIdx = colIndices[rowIdx]

    # If cars are within the boundary, we update their states according to
    # the controls
    if not doneEvader:
      uEvader = env.evader.discrete_controls[rowIdx]
      stateEvader = env.evader.integrate_forward(stateEvader, uEvader)
    if not donePursuer:
      actionIdx = actionSeq[pursuerActionSeqIdx]
      uPursuer = env.pursuer.discrete_controls[actionIdx]
      statePursuer = env.pursuer.integrate_forward(statePursuer, uPursuer)

    trajPursuer.append(statePursuer)
    trajEvader.append(stateEvader)
    if (t+1) % chunkLength == 0:
      pursuerActionSeqIdx += 1

  trajEvader = np.array(trajEvader)
  trajPursuer = np.array(trajPursuer)
  info = {"valueList": valueList, "gxList": gxList, "lxList": lxList}
  return trajEvader, trajPursuer, minV, info


def validateEvaderPolicy(env, agent, state, maxLength=40, numChunk=10):
  """
  Validates the evader's policy by trying out pursuer's strategies exaustively.

  Args:
      env (gym.Env): the environment.
      agent (RARL.DDQNPursuitEvasion): the agent.
      state (np.ndarray): the initial state of the pursuer and the evader.
      maxLength (int, optional): the maximum length of the trajectory.
          Defaults to 40.
      numChunk (int, optional): the number of chunks in action sequence.
          Defaults to 10.

  Returns:
      dict: contains the information in this exhaustive search.
  """
  actionSet = np.empty(shape=(env.numActionList[1], numChunk), dtype=int)
  for i in range(numChunk):
    actionSet[:, i] = np.arange(env.numActionList[1])

  shapeTmp = np.ones(numChunk, dtype=int) * env.numActionList[1]
  rolloutResult = np.empty(shape=shapeTmp, dtype=int)
  it = np.nditer(rolloutResult, flags=["multi_index"])
  responseDict = {
      "state": state,
      "maxLength": maxLength,
      "numChunk": numChunk,
  }
  flag = True
  while not it.finished:
    idx = it.multi_index
    actionSeq = actionSet[idx, np.arange(numChunk)]
    print(actionSeq, end="\r")
    trajEvader, trajPursuer, minV, _ = exhaustiveDefenderSearch(
        env, agent, state, actionSeq, maxLength
    )
    info = {
        "trajEvader": trajEvader,
        "trajPursuer": trajPursuer,
        "minV": minV,
    }
    responseDict[idx] = info
    it.iternext()
    if flag:
      maxminV = minV
      maxminIdx = idx
      flag = False
    elif minV > maxminV:
      maxminV = minV
      maxminIdx = idx
  responseDict["maxminV"] = maxminV
  responseDict["maxminIdx"] = maxminIdx
  return responseDict


def checkCapture(env, trajEvader, trajPursuer):
  """Checks if the evader is captured by the pursuer.

  Args:
      env (gym.Env): the environment.
      trajEvader (np.ndarray): the trajectory of the evader.
      trajPursuer (np.ndarray): the trajectory of the pursuer.

  Returns:
      bool: True if the evader is captured by the pursuer.
      int: the time step at which the evader is captured by the pursuer.
  """
  numStep = trajEvader.shape[0]
  captureFlag = False
  captureInstant = None
  for t in range(numStep):
    posEvader = trajEvader[t, :2]
    posPursuer = trajPursuer[t, :2]
    dist_evader_pursuer = np.linalg.norm(posEvader - posPursuer, ord=2)
    capture_g_x = env.capture_range - dist_evader_pursuer
    # however, the value can be lower than this captureValue because we
    # care about the minimum value along the trajectory.
    if capture_g_x > 0:
      captureFlag = True
      captureInstant = t
      break
      # if not captureFlag:
      #     captureInstant = t
      #     captureValue = capture_g_x
      #     captureFlag = True
      # elif capture_g_x > captureValue:
      #     captureInstant = t
      #     captureValue = capture_g_x
  return captureFlag, captureInstant


def checkCrossConstraint(env, trajEvader, trajPursuer):
  """Checks if the evader crosses the boundary.

  Args:
      env (gym.Env): the environment.
      trajEvader (np.ndarray): the trajectory of the evader.
      trajPursuer (np.ndarray): the trajectory of the pursuer.

  Returns:
      bool: True if crossing the boundary.
      int: the time step at which the evader hits the obstacle.
  """
  numStep = trajEvader.shape[0]
  crossConstraintFlag = False
  crossConstraintInstant = None
  for t in range(numStep):
    posEvader = trajEvader[t, :2]
    evader_g_x = env.evader.safety_margin(posEvader)
    if not crossConstraintFlag and evader_g_x > 0:
      crossConstraintInstant = t
      crossConstraintFlag = True
  return crossConstraintFlag, crossConstraintInstant


def analyzeValidationResult(validationFile, env, verbose=True):
  """
  Analyzes the performance of evader's policy given the exhaustive search of
  pursuer's strategies.

  Args:
      validationFile (str): the location of the validation results.
      env (gym.Env): the environment.
      verbose (bool, optional): print messages if True. Defaults to True.

  Returns:
      valDict (dict): the results of exhaustive search.
      successList (list): the list of testing indices where the evader
          succeeds.
      failureList (list): the list of testing indices where the evader fails
          (captured or hit the obstacle).
      captureList (list): the list of testing indices where the evader is
          captured.
      captureInstantList (list): the list of time steps at which the evader is
          captured.
      crossConstraintList (list): the list of testing indices where the evader
          hits the obstacles.
      crossConstraintInstantList (list): the list of time steps at which the
          evader hits the obstacles.
      unfinishedList (list): the list of testing indices where the evader is
          either succeeds or fails.
  """
  print("<-- Load from {:s} ...".format(validationFile))
  valDict = np.load(validationFile, allow_pickle="TRUE").item()

  dictList = valDict["dictList"]
  failureList = []
  successList = []
  for i, dictTmp in enumerate(dictList):
    maxminV = dictTmp["maxminV"]
    if maxminV > 0:
      failureList.append(i)
    else:
      successList.append(i)
  if verbose:
    print(
        "Failure Ratio: {:d} / {:d} = {:.2%}.".format(
            len(failureList),
            len(dictList),
            len(failureList) / len(dictList),
        )
    )

  # == ANALYZE FAILED STATES ==
  captureList = []
  captureInstantList = []
  crossConstraintList = []
  crossConstraintInstantList = []
  unfinishedList = []
  for i, pick in enumerate(failureList):
    if verbose:
      print("{:d}/{:d}".format(i + 1, len(failureList)), end="\r")
    dictTmp = dictList[pick]
    trajEvaderTmp = dictTmp["trajEvader"]
    trajPursuerTmp = dictTmp["trajPursuer"]
    captureFlag, captureInstant = checkCapture(
        env, trajEvaderTmp, trajPursuerTmp
    )
    crossConstraintFlag, crossConstraintInstant = checkCrossConstraint(
        env, trajEvaderTmp, trajPursuerTmp
    )
    if captureFlag:
      if crossConstraintFlag:
        if captureInstant < crossConstraintInstant:
          captureList.append(pick)
          captureInstantList.append(captureInstant)
        else:
          crossConstraintList.append(pick)
          crossConstraintInstantList.append(crossConstraintInstant)
      else:
        captureList.append(pick)
        captureInstantList.append(captureInstant)
    elif crossConstraintFlag:
      crossConstraintList.append(pick)
      crossConstraintInstantList.append(crossConstraintInstant)
    else:
      unfinishedList.append(pick)
  if verbose:
    print(
        "{:d} captured, {:d} cross, {:d} unfinished, {:d} succeed".format(
            len(captureList),
            len(crossConstraintList),
            len(unfinishedList),
            len(successList),
        )
    )
  return (
      valDict, successList, failureList, captureList, captureInstantList,
      crossConstraintList, crossConstraintInstantList, unfinishedList
  )


def colUnfinishedSamples(unfinishedList, valDict, valSamplesDict):
  """Collects unfinished testing indices.

  Args:
      unfinishedList (list): the test indices of unfinished samples.
      valDict (dict): includes
          'dictList'
          'stateIdxList': the index of states by `genEstSamples.py`
          'testIdxList': the index of states by `genValSamples.py`
      valSamplesDict (dict): includes
          'idxList': the index of states by `genEstSamples.py`
          'rollvalList': the rollout values of states by `genEstSamples.py`
          'ddqnList': the DDQN values of states by `genEstSamples.py`
  """
  # == add to valSamplesTN ==
  unfinishedStateIdxList = []
  unfinishedStateList = np.empty(shape=(len(unfinishedList), 6), dtype=float)
  newRolloutValueList = np.empty(shape=(len(unfinishedList),), dtype=float)
  newDdqnValueList = np.empty(shape=(len(unfinishedList),), dtype=float)
  unfinishedValueList = np.empty(shape=(len(unfinishedList),), dtype=float)

  dictList = valDict["dictList"]
  stateIdxList = valDict["stateIdxList"]
  testIdxList = valDict["testIdxList"]

  for i, pick in enumerate(unfinishedList):
    print("{:d}/{:d}".format(i + 1, len(unfinishedList)), end="\r")

    testIdx = testIdxList[pick]
    dictTmp = dictList[pick]
    stateIdx = stateIdxList[pick]
    maxminV = dictTmp["maxminV"]
    # maxminIdx = dictTmp["maxminIdx"]
    trajEvaderTmp = dictTmp["trajEvader"]
    trajPursuerTmp = dictTmp["trajPursuer"]

    state = np.empty(shape=(6,), dtype=float)
    state[:3] = trajEvaderTmp[-1, :]
    state[3:] = trajPursuerTmp[-1, :]

    rolloutValue = valSamplesDict["rollvalList"][testIdx]
    ddqnValue = valSamplesDict["ddqnList"][testIdx]

    unfinishedValueList[i] = maxminV
    unfinishedStateIdxList.append(stateIdx)
    unfinishedStateList[i, :] = state
    newRolloutValueList[i] = rolloutValue
    newDdqnValueList[i] = ddqnValue

  # == RECORD ==
  finalDict = {}
  finalDict["states"] = unfinishedStateList
  finalDict["idxList"] = unfinishedStateIdxList
  finalDict["ddqnList"] = newDdqnValueList
  finalDict["rollvalList"] = newRolloutValueList
  finalDict["unfinishedValueList"] = unfinishedValueList
  finalDict["pickList"] = unfinishedList  # indices of validation samples

  return finalDict


# == LOADING ==
def loadEnv(args, verbose=True):
  """Constructs the environmnet given the arguments and return it.

  Args:
      args (Namespace): it contains
          - forceCPU (bool): use PyTorch with CPU only
          - cpf (bool): consider Pursuer's collision to the obstacle.
      verbose (bool, optional): print messages if True. Defaults to True.

  """
  env_name = "dubins_car_pe-v0"
  if args.forceCPU:
    device = torch.device("cpu")
  else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  env = gym.make(env_name, device=device, mode="RA", doneType="toEnd")
  env.set_considerPursuerFailure(args.cpf)
  if verbose:
    print("\n== Environment Information ==")
    env.report()
    print()
  return env


def loadAgent(
    args, device, state_dim, action_num, numActionList, verbose=True
):
  """Constructs the agent with arguments and return it.

  Args:
      args (Namespace): it contains
          - modelFolder (str): the parent folder to get the stored models.
      device (torch.device): the device used by PyTorch.
      state_dim (int): the dimension of the state.
      action_num (int): the number of actions in the action set.
      numActionList (list): the number of actions in the evader and pursuer's
          action sets.
      verbose (bool, optional): print messages if True. Defaults to True.
  """
  if verbose:
    print("\n== Agent Information ==")
  modelFolder = os.path.join(args.modelFolder, "model")
  configFile = os.path.join(modelFolder, "CONFIG.pkl")
  with open(configFile, "rb") as handle:
    tmpConfig = pickle.load(handle)
  CONFIG = dqnConfig()
  for key, _ in tmpConfig.__dict__.items():
    CONFIG.__dict__[key] = tmpConfig.__dict__[key]
  CONFIG.DEVICE = device
  CONFIG.SEED = 0

  dimList = [state_dim] + CONFIG.ARCHITECTURE + [action_num]
  agent = DDQNPursuitEvasion(
      CONFIG, numActionList, dimList, CONFIG.ACTIVATION, verbose=verbose
  )
  agent.restore(1000000, args.modelFolder)

  if verbose:
    print(vars(CONFIG))
    print("agent's device:", agent.device)

  return agent
