from warnings import simplefilter 
simplefilter(action='ignore', category=FutureWarning)

from gym_reachability import gym_reachability  # Custom Gym env.
import gym
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from collections import namedtuple
import pickle
import os
import argparse

from KC_DQN.DDQNPursuitEvasion import DDQNPursuitEvasion
from KC_DQN.DDQN import Transition
from KC_DQN.config import dqnConfig

import time
timestr = time.strftime("%Y-%m-%d-%H_%M")


#== ARGS ==
# e.g., python3 sim_car_PE.py -te -w 
# test: python3 sim_car_PE.py -te -w -mu 100 -ut 2 -wi 1 -of scratch/
parser = argparse.ArgumentParser()
# parser.add_argument("-nt",  "--num_test",       help="the number of tests",         default=1,      type=int)
# parser.add_argument("-nw",  "--num_worker",     help="the number of workers",       default=1,      type=int)

# training scheme
parser.add_argument("-te",  "--toEnd",          help="stop until reaching boundary",    action="store_true")
parser.add_argument("-ab",  "--addBias",        help="add bias term for RA",            action="store_true")
parser.add_argument("-w",   "--warmup",         help="warmup Q-network",                action="store_true")
parser.add_argument("-rnd", "--randomSeed",     help="random seed",                     default=0,      type=int)
parser.add_argument("-mu",  "--maxUpdates",     help="maximal #gradient updates",       default=4e6,    type=int)
parser.add_argument("-mc",  "--memoryCapacity", help="memoryCapacity",                  default=1e4,    type=int)
parser.add_argument("-ut",  "--updateTimes",    help="#hyper-param. steps",             default=20,     type=int)
parser.add_argument("-wi",  "--warmupIter",     help="warmup iteration",                default=20000,  type=int)

# hyper-parameters
parser.add_argument("-arc", "--architecture",   help="NN architecture",     default=[512, 512, 512],  nargs="*", type=int)
parser.add_argument("-lr",  "--learningRate",   help="learning rate",       default=1e-3,   type=float)
parser.add_argument("-g",   "--gamma",          help="contraction coeff.",  default=0.8,    type=float)
parser.add_argument("-act", "--actType",        help="activation type",     default='Tanh', type=str)

# file
parser.add_argument("-n",   "--name",           help="extra name",      default='',                         type=str)
parser.add_argument("-of",  "--outFolder",      help="output file",     default='/scratch/gpfs/kaichieh/',  type=str)
parser.add_argument("-pf",  "--plotFigure",     help="plot figures",    action="store_true")
parser.add_argument("-sf",  "--storeFigure",    help="store figures",   action="store_true")

args = parser.parse_args()
print("\n== Arguments ==")
print(args)


#== CONFIGURATION ==
toEnd = args.toEnd
env_name = "dubins_car_pe-v0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
maxUpdates = args.maxUpdates
updateTimes = args.updateTimes
updatePeriod = int(maxUpdates / updateTimes)
updatePeriodHalf = int(updatePeriod/2)
maxSteps = 100

outFolder = args.outFolder + 'carPE/' + args.name + timestr
figureFolder = '{:s}/figure/'.format(outFolder)
os.makedirs(figureFolder, exist_ok=True)


#== Environment ==
print("\n== Environment Information ==")
if toEnd:
    env = gym.make(env_name, device=device, mode='RA', doneType='toEnd')
else:
    env = gym.make(env_name, device=device, mode='RA', doneType='TF')

stateNum = env.state.shape[0]
actionNum = env.action_space.n
action_list = np.arange(actionNum)
print("State Dimension: {:d}, ActionSpace Dimension: {:d}".format(stateNum, actionNum))


#== Setting in this Environment ==
print("Dynamic parameters:")
print("  EVADER")
print("    Constraint radius: {:.1f}, Target radius: {:.1f}, Turn radius: {:.2f}, Maximum speed: {:.1f}, Maximum angular speed: {:.3f}".format(
    env.evader.constraint_radius, env.evader.target_radius, env.evader.R_turn, env.evader.speed, env.evader.max_turning_rate))
print("  PURSUER")
print("    Constraint radius: {:.1f}, Turn radius: {:.2f}, Maximum speed: {:.2f}, Maximum angular speed: {:.3f}".format(
    env.pursuer.constraint_radius, env.pursuer.R_turn, env.pursuer.speed, env.pursuer.max_turning_rate))
if 2*env.evader.R_turn-env.evader.constraint_radius > env.evader.target_radius:
    print("Type II Reach-Avoid Set")
else:
    print("Type I Reach-Avoid Set")


#== Get and Plot max{l_x, g_x} ==
nx, ny = 101, 101
theta, thetaPursuer = 0., 0.
v = np.zeros((4, nx, ny))
l_x = np.zeros((4, nx, ny))
g_x = np.zeros((4, nx, ny))
xs = np.linspace(env.bounds[0,0], env.bounds[0,1], nx)
ys =np.linspace(env.bounds[1,0], env.bounds[1,1], ny)

xPursuerList=[.1, .3, .5, .7]
yPursuerList=[.1, .3, .5, .7]
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
fig, axes = plt.subplots(1,4, figsize=(16, 4))
for i, (ax, xPursuer, yPursuer) in enumerate(zip(axes, xPursuerList, yPursuerList)):
    f = ax.imshow(v[i].T, interpolation='none', extent=axStyle[0], origin="lower", cmap="seismic", vmin=-1, vmax=1)
    ax.axis(axStyle[0])
    ax.grid(False)
    ax.set_aspect(axStyle[1])  # makes equal aspect ratio
    env.plot_target_failure_set(ax, xPursuer=xPursuer, yPursuer=yPursuer)
    if i == 3:
        fig.colorbar(f, ax=ax, pad=0.01, fraction=0.05, shrink=.95, ticks=[-1, 0, 1])
plt.tight_layout()
fig.savefig('{:s}env.png'.format(figureFolder))
plt.close()


#== Agent CONFIG ==
print("\n== Agent Information ==")
CONFIG = dqnConfig(DEVICE=device, ENV_NAME=env_name, SEED=args.randomSeed,
    MAX_UPDATES=maxUpdates, MAX_EP_STEPS=maxSteps,
    BATCH_SIZE=100, MEMORY_CAPACITY=args.memoryCapacity,
    ARCHITECTURE=args.architecture, ACTIVATION=args.actType,
    GAMMA=args.gamma, GAMMA_PERIOD=updatePeriod, GAMMA_END=0.999999,
    EPS_PERIOD=updatePeriod, EPS_DECAY=0.6,
    LR_C=args.learningRate, LR_C_PERIOD=updatePeriod, LR_C_DECAY=0.8,
    MAX_MODEL=50)
# print(vars(CONFIG))
picklePath = outFolder+'/CONFIG.pkl'
with open(picklePath, 'wb') as handle:
    pickle.dump(CONFIG, handle, protocol=pickle.HIGHEST_PROTOCOL)


#== AGENT ==
numActionList = env.numActionList
numJoinAction = int(numActionList[0] * numActionList[1])
dimList = [stateNum] + CONFIG.ARCHITECTURE + [actionNum]
agent = DDQNPursuitEvasion(CONFIG, numActionList, dimList, actType=args.actType)
print(agent.device)

print("\n== Training Information ==")
vmin = -1
vmax = 1
checkPeriod = updatePeriod
training_records, trainProgress = agent.learn(env,
    MAX_UPDATES=maxUpdates, MAX_EP_STEPS=CONFIG.MAX_EP_STEPS, addBias=args.addBias,
    warmupQ=args.warmup, warmupIter=args.warmupIter, doneTerminate=True,
    vmin=vmin, vmax=vmax, showBool=False,
    checkPeriod=checkPeriod, outFolder=outFolder,
    plotFigure=args.plotFigure, storeFigure=args.storeFigure)