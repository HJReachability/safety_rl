"""
Please contact the author(s) of this library if you have any questions.
Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )

This experiment runs double deep Q-network with the discounted reach-avoid
Bellman equation (DRABE) proposed in [RSS21] on a 6-dimensional Dubins car
attack-defense problem. We use this script to generate Fig. 7 in the paper.

Examples:
    RA: python3 sim_car_pe.py -sf -of scratch -w -wi 30000 -g 0.9999 -n 9999
    test: python3 sim_car_pe.py -sf -of scratch -n tmp -mu 100 -cp 40
"""

import os
import argparse
import time
from warnings import simplefilter
import gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator
import torch

from RARL.DDQNPursuitEvasion import DDQNPursuitEvasion
from RARL.config import dqnConfig
from RARL.utils import save_obj
from gym_reachability import gym_reachability  # Custom Gym env.

matplotlib.use('Agg')
simplefilter(action='ignore', category=FutureWarning)
timestr = time.strftime("%Y-%m-%d-%H_%M")

# == ARGS ==
parser = argparse.ArgumentParser()

# environment parameters
parser.add_argument(
    "-dt", "--doneType", help="when to raise done flag", default='toEnd',
    type=str
)
parser.add_argument(
    "-ct", "--costType", help="cost type", default='sparse', type=str
)
parser.add_argument(
    "-rnd", "--randomSeed", help="random seed", default=0, type=int
)
parser.add_argument(
    "-cpf", "--cpf", help="consider pursuer failure set", action="store_true"
)

# car dynamics
parser.add_argument(
    "-cr", "--consRadius", help="constraint radius", default=1., type=float
)
parser.add_argument(
    "-tr", "--targetRadius", help="target radius", default=.5, type=float
)
parser.add_argument(
    "-turn", "--turnRadius", help="turning radius", default=.25, type=float
)
parser.add_argument("-s", "--speed", help="speed", default=.75, type=float)

# training scheme
parser.add_argument(
    "-w", "--warmup", help="warmup Q-network", action="store_true"
)
parser.add_argument(
    "-wi", "--warmupIter", help="warmup iteration", default=30000, type=int
)
parser.add_argument(
    "-mu", "--maxUpdates", help="maximal #gradient updates", default=4000000,
    type=int
)
parser.add_argument(
    "-ut", "--updateTimes", help="#hyper-param. steps", default=20, type=int
)
parser.add_argument(
    "-mc", "--memoryCapacity", help="memoryCapacity", default=50000, type=int
)
parser.add_argument(
    "-cp", "--checkPeriod", help="check period", default=200000, type=int
)

# hyper-parameters
parser.add_argument(
    "-a", "--annealing", help="gamma annealing", action="store_true"
)
parser.add_argument(
    "-arc", "--architecture", help="NN architecture", default=[512, 512, 512],
    nargs="*", type=int
)
parser.add_argument(
    "-lr", "--learningRate", help="learning rate", default=1e-3, type=float
)
parser.add_argument(
    "-g", "--gamma", help="contraction coeff.", default=0.8, type=float
)
parser.add_argument(
    "-act", "--actType", help="activation type", default='Tanh', type=str
)

# RL type
parser.add_argument("-m", "--mode", help="mode", default='RA', type=str)
parser.add_argument(
    "-tt", "--terminalType", help="terminal value", default='g', type=str
)

# file
parser.add_argument(
    "-st", "--showTime", help="show timestr", action="store_true"
)
parser.add_argument("-n", "--name", help="extra name", default='', type=str)
parser.add_argument(
    "-of", "--outFolder", help="output file", default='experiments', type=str
)
parser.add_argument(
    "-pf", "--plotFigure", help="plot figures", action="store_true"
)
parser.add_argument(
    "-sf", "--storeFigure", help="store figures", action="store_true"
)

args = parser.parse_args()
print(args)

# == CONFIGURATION ==
env_name = "dubins_car_pe-v0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
maxUpdates = args.maxUpdates
updateTimes = args.updateTimes
updatePeriod = int(maxUpdates / updateTimes)
updatePeriodHalf = int(updatePeriod / 2)
maxSteps = 200

fn = args.name + '-' + args.doneType
if args.showTime:
  fn = fn + '-' + timestr

outFolder = os.path.join(args.outFolder, 'car-pe-DDQN', fn)
print(outFolder)
figureFolder = os.path.join(outFolder, 'figure')
os.makedirs(figureFolder, exist_ok=True)

# == Environment ==
print("\n== Environment Information ==")
env = gym.make(
    env_name, device=device, mode='RA', doneType=args.doneType,
    sample_inside_obs=False, considerPursuerFailure=args.cpf
)
stateDim = env.state.shape[0]
actionNum = env.action_space.n
action_list = np.arange(actionNum)
env.set_seed(args.randomSeed)
env.report()

# == Get and Plot max{l_x, g_x} ==
if args.plotFigure or args.storeFigure:
  nx, ny = 101, 101
  theta, thetaPursuer = 0., 0.
  v = np.zeros((4, nx, ny))
  l_x = np.zeros((4, nx, ny))
  g_x = np.zeros((4, nx, ny))
  xs = np.linspace(env.bounds[0, 0], env.bounds[0, 1], nx)
  ys = np.linspace(env.bounds[1, 0], env.bounds[1, 1], ny)

  xPursuerList = [.1, .3, .5, .8]
  yPursuerList = [.1, .3, .5, .8]
  for i, (xPursuer, yPursuer) in enumerate(zip(xPursuerList, yPursuerList)):
    it = np.nditer(l_x[0], flags=['multi_index'])

    while not it.finished:
      idx = it.multi_index
      x = xs[idx[0]]
      y = ys[idx[1]]

      state = np.array([x, y, theta, xPursuer, yPursuer, thetaPursuer])
      l_x[i][idx] = env.target_margin(state)
      g_x[i][idx] = env.safety_margin(state)

      v[i][idx] = np.maximum(l_x[i][idx], g_x[i][idx])
      it.iternext()

  axStyle = env.get_axes()
  fig, axes = plt.subplots(1, 4, figsize=(16, 4))
  for i, ax in enumerate(axes):
    xPursuer = xPursuerList[i]
    yPursuer = yPursuerList[i]
    f = ax.imshow(
        v[i].T, interpolation='none', extent=axStyle[0], origin="lower",
        cmap="seismic", vmin=-1, vmax=1
    )
    env.plot_target_failure_set(ax, xPursuer=xPursuer, yPursuer=yPursuer)
    if i == 3:
      fig.colorbar(
          f, ax=ax, pad=0.01, fraction=0.05, shrink=0.95, ticks=[-1, 0, 1]
      )
    env.plot_formatting(ax)
  fig.tight_layout()
  if args.storeFigure:
    figurePath = os.path.join(figureFolder, 'env.png')
    fig.savefig(figurePath)
  if args.plotFigure:
    plt.show()
    plt.pause(0.001)
  plt.close()

# == Agent CONFIG ==
print("\n== Agent Information ==")
if args.annealing:
  GAMMA_END = 0.9999
  EPS_PERIOD = int(updatePeriod / 10)
  EPS_RESET_PERIOD = updatePeriod
else:
  GAMMA_END = args.gamma
  EPS_PERIOD = updatePeriod
  EPS_RESET_PERIOD = maxUpdates

CONFIG = dqnConfig(
    DEVICE=device, ENV_NAME=env_name, SEED=args.randomSeed,
    MAX_UPDATES=maxUpdates, MAX_EP_STEPS=maxSteps, BATCH_SIZE=64,
    MEMORY_CAPACITY=args.memoryCapacity, ARCHITECTURE=args.architecture,
    ACTIVATION=args.actType, GAMMA=args.gamma, GAMMA_PERIOD=updatePeriod,
    GAMMA_END=GAMMA_END, EPS_PERIOD=EPS_PERIOD, EPS_DECAY=0.7,
    EPS_RESET_PERIOD=EPS_RESET_PERIOD, LR_C=args.learningRate,
    LR_C_PERIOD=updatePeriod, LR_C_DECAY=0.8, MAX_MODEL=50
)
# print(vars(CONFIG))

# == AGENT ==
numActionList = env.numActionList
numJoinAction = int(numActionList[0] * numActionList[1])
dimList = [stateDim] + CONFIG.ARCHITECTURE + [actionNum]
agent = DDQNPursuitEvasion(
    CONFIG, numActionList, dimList, mode='RA', terminalType='g'
)
print("We want to use: {}, and Agent uses: {}".format(device, agent.device))
print("Critic is using cuda: ", next(agent.Q_network.parameters()).is_cuda)

vmin = -1
vmax = 1
if args.warmup:
  print("\n== Warmup Q ==")
  lossList = agent.initQ(
      env, args.warmupIter, outFolder, num_warmup_samples=200, vmin=vmin,
      vmax=vmax, plotFigure=args.plotFigure, storeFigure=args.storeFigure
  )

  if args.plotFigure or args.storeFigure:
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    tmp = np.arange(500, args.warmupIter)
    ax.plot(tmp, lossList[tmp], 'b-')

    ax.set_xlim(500, args.warmupIter)
    ax.set_xlabel('Iteration', fontsize=18)
    ax.set_ylabel('Loss', fontsize=18)
    ax.xaxis.set_major_locator(LinearLocator(5))
    ax.xaxis.set_major_formatter('{x:.1f}')
    ax.yaxis.set_major_locator(LinearLocator(5))
    ax.yaxis.set_major_formatter('{x:.1f}')
    fig.tight_layout()

    if args.storeFigure:
      figurePath = os.path.join(figureFolder, 'initQ_Loss.png')
      fig.savefig(figurePath)
    if args.plotFigure:
      plt.show()
      plt.pause(0.001)
    plt.close()

print("\n== Training Information ==")
vmin = -1
vmax = 1
trainRecords, trainProgress = agent.learn(
    env, MAX_UPDATES=maxUpdates, MAX_EP_STEPS=maxSteps, warmupQ=False,
    doneTerminate=True, vmin=vmin, vmax=vmax, showBool=False,
    checkPeriod=args.checkPeriod, outFolder=outFolder,
    plotFigure=args.plotFigure, storeFigure=args.storeFigure
)

trainDict = {}
trainDict['trainRecords'] = trainRecords
trainDict['trainProgress'] = trainProgress
filePath = os.path.join(outFolder, 'train')

if args.plotFigure or args.storeFigure:
  # region: loss
  fig, axes = plt.subplots(1, 2, figsize=(8, 4))

  data = trainRecords
  ax = axes[0]
  ax.plot(data, 'b:')
  ax.set_xlabel('Iteration (x 1e5)', fontsize=18)
  ax.set_xticks(np.linspace(0, maxUpdates, 5))
  ax.set_xticklabels(np.linspace(0, maxUpdates, 5) / 1e5)
  ax.set_title('loss_critic', fontsize=18)
  ax.set_xlim(left=0, right=maxUpdates)

  data = trainProgress[:, 0]
  ax = axes[1]
  x = np.arange(data.shape[0]) + 1
  ax.plot(x, data, 'b-o')
  ax.set_xlabel('Index', fontsize=18)
  ax.set_xticks(x)
  ax.set_title('Success Rate', fontsize=18)
  ax.set_xlim(left=1, right=data.shape[0])
  ax.set_ylim(0, 0.8)

  fig.tight_layout()
  if args.storeFigure:
    figurePath = os.path.join(figureFolder, 'train_loss_success.png')
    fig.savefig(figurePath)
  if args.plotFigure:
    plt.show()
    plt.pause(0.001)
  plt.close()
  # endregion

  # region: value_rollout_action
  idx = np.argmax(trainProgress[:, 0]) + 1
  successRate = np.amax(trainProgress[:, 0])
  print('We pick model with success rate-{:.3f}'.format(successRate))
  agent.restore(idx * args.checkPeriod, outFolder)

  nx = 101
  ny = nx
  xs = np.linspace(env.bounds[0, 0], env.bounds[0, 1], nx)
  ys = np.linspace(env.bounds[1, 0], env.bounds[1, 1], ny)

  resultMtx = np.empty((nx, ny), dtype=int)
  actDistMtx = np.empty((nx, ny), dtype=int)
  it = np.nditer(resultMtx, flags=['multi_index'])

  while not it.finished:
    idx = it.multi_index
    print(idx, end='\r')
    x = xs[idx[0]]
    y = ys[idx[1]]

    state = np.array([x, y, 0., -0.2, -0.3, 0.75 * np.pi])
    stateTensor = torch.FloatTensor(state).to(agent.device).unsqueeze(0)
    state_action_values = agent.Q_network(stateTensor)
    Q_mtx = state_action_values.reshape(
        env.numActionList[0], env.numActionList[1]
    )
    # Q_mtx = Q_mtx.detach().cpu()
    pursuerValues, colIndices = Q_mtx.max(dim=1)
    _, rowIdx = pursuerValues.min(dim=0)
    colIdx = colIndices[rowIdx]

    uEvader = env.evader.discrete_controls[rowIdx]
    uPursuer = env.pursuer.discrete_controls[colIdx]
    actDistMtx[idx] = rowIdx

    _, _, result, _, _ = env.simulate_one_trajectory(
        agent.Q_network, T=250, state=state, toEnd=False
    )
    resultMtx[idx] = result
    it.iternext()

  fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
  axStyle = env.get_axes()

  # = Action
  ax = axes[2]
  im = ax.imshow(
      actDistMtx.T, interpolation='none', extent=axStyle[0], origin="lower",
      cmap='seismic', vmin=0, vmax=actionNum - 1, zorder=-1
  )
  ax.set_xlabel('Action', fontsize=24)

  # = Rollout
  ax = axes[1]
  im = ax.imshow(
      resultMtx.T != 1, interpolation='none', extent=axStyle[0],
      origin="lower", cmap='seismic', vmin=0, vmax=1, zorder=-1
  )
  env.plot_trajectories(
      agent.Q_network, states=[env.visual_initial_states[1]], toEnd=False,
      ax=ax, lw=1.5, T=200
  )
  ax.set_xlabel('Rollout RA', fontsize=24)

  # = Value
  ax = axes[0]
  im = ax.imshow(
      v.T, interpolation='none', extent=axStyle[0], origin="lower",
      cmap='seismic', vmin=vmin, vmax=vmax, zorder=-1
  )
  ax.set_xlabel('Value', fontsize=24)

  for ax in axes:
    env.plot_target_failure_set(ax=ax, xPursuer=-0.2, yPursuer=-0.3)
    env.plot_formatting(ax=ax)

  fig.tight_layout()
  if args.storeFigure:
    figurePath = os.path.join(figureFolder, 'value_rollout_action.png')
    fig.savefig(figurePath)
  if args.plotFigure:
    plt.show()
    plt.pause(0.001)
  # endregion

  trainDict['resultMtx'] = resultMtx
  trainDict['actDistMtx'] = actDistMtx

save_obj(trainDict, filePath)
