from warnings import simplefilter 
simplefilter(action='ignore', category=FutureWarning)

from gym_reachability import gym_reachability  # Custom Gym env.
import gym
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import pickle
import os
import argparse

from KC_DQN.DDQNPursuitEvasion import DDQNPursuitEvasion
from KC_DQN.config import dqnConfig
from KC_DQN.utils import save_obj

import time
timestr = time.strftime("%Y-%m-%d-%H_%M")


#== ARGS ==
# python3 sim_car_pe.py -sf -of scratch -w -wi 5000 -g 0.9999 -n 9999
parser = argparse.ArgumentParser()

# environment parameters
parser.add_argument("-dt",  "--doneType",       help="when to raise done flag",
    default='toEnd',    type=str)
parser.add_argument("-ct",  "--costType",       help="cost type",
    default='sparse',   type=str)
parser.add_argument("-rnd", "--randomSeed",     help="random seed",
    default=0,          type=int)
parser.add_argument("-cpf", "--cpf",            help="consider pursuer failure set",
    action="store_true")

# car dynamics
parser.add_argument("-cr",      "--constraintRadius",   help="constraint radius",
    default=1., type=float)
parser.add_argument("-tr",      "--targetRadius",       help="target radius",
    default=.5, type=float)
parser.add_argument("-turn",    "--turnRadius",         help="turning radius",
    default=.6, type=float)
parser.add_argument("-s",       "--speed",              help="speed",
    default=.5, type=float)

# training scheme
parser.add_argument("-w",   "--warmup",         help="warmup Q-network",
    action="store_true")
parser.add_argument("-wi",  "--warmupIter",     help="warmup iteration",
    default=10000,  type=int)
parser.add_argument("-mu",  "--maxUpdates",     help="maximal #gradient updates",
    default=400000, type=int)
parser.add_argument("-ut",  "--updateTimes",    help="#hyper-param. steps",
    default=20,     type=int)
parser.add_argument("-mc",  "--memoryCapacity", help="memoryCapacity",
    default=50000,  type=int)
parser.add_argument("-cp",  "--checkPeriod",    help="check period",
    default=20000,  type=int)

# hyper-parameters
parser.add_argument("-a",   "--annealing",      help="gamma annealing",
    action="store_true")
parser.add_argument("-arc", "--architecture",   help="NN architecture",
    default=[100, 100],  nargs="*", type=int)
parser.add_argument("-lr",  "--learningRate",   help="learning rate",
    default=1e-3,   type=float)
parser.add_argument("-g",   "--gamma",          help="contraction coeff.",
    default=0.8,    type=float)
parser.add_argument("-act", "--actType",        help="activation type",
    default='Tanh', type=str)

# RL type
parser.add_argument("-m",   "--mode",           help="mode",
    default='RA',       type=str)
parser.add_argument("-tt",  "--terminalType",   help="terminal value",
    default='g',        type=str)

# file
parser.add_argument("-st",  "--showTime",       help="show timestr",
    action="store_true")
parser.add_argument("-n",   "--name",           help="extra name",
    default='',                         type=str)
parser.add_argument("-of",  "--outFolder",      help="output file",
    default='/scratch/gpfs/kaichieh/',  type=str)
parser.add_argument("-pf",  "--plotFigure",     help="plot figures",
    action="store_true")
parser.add_argument("-sf",  "--storeFigure",    help="store figures",
    action="store_true")

args = parser.parse_args()
print(args)


#== CONFIGURATION ==
env_name = "dubins_car_pe-v0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
maxUpdates = args.maxUpdates
updateTimes = args.updateTimes
updatePeriod = int(maxUpdates / updateTimes)
updatePeriodHalf = int(updatePeriod/2)
maxSteps = 200

fn = args.name + '-' + args.doneType
if args.showTime:
    fn = fn + '-' + timestr

outFolder = os.path.join(args.outFolder, 'car-pe-DDQN', fn)
print(outFolder)
figureFolder = os.path.join(outFolder, 'figure')
os.makedirs(figureFolder, exist_ok=True)


#== Environment ==
print("\n== Environment Information ==")
env = gym.make(env_name, device=device, mode='RA', doneType=args.doneType,
    sample_inside_obs=False, considerPursuerFailure=args.cpf)
stateDim = env.state.shape[0]
actionNum = env.action_space.n
action_list = np.arange(actionNum)
env.set_seed(args.randomSeed)
env.report()


#== Get and Plot max{l_x, g_x} ==
if args.plotFigure or args.storeFigure:
    nx, ny = 101, 101
    theta, thetaPursuer = 0., 0.
    v = np.zeros((4, nx, ny))
    l_x = np.zeros((4, nx, ny))
    g_x = np.zeros((4, nx, ny))
    xs = np.linspace(env.bounds[0,0], env.bounds[0,1], nx)
    ys =np.linspace(env.bounds[1,0], env.bounds[1,1], ny)

    xPursuerList=[.1, .3, .5, .8]
    yPursuerList=[.1, .3, .5, .8]
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
    fig.tight_layout()
    if args.storeFigure:
        figurePath = os.path.join(figureFolder, 'env.png')
        fig.savefig(figurePath)
    if args.plotFigure:
        plt.show()
        plt.pause(0.001)
    plt.close()


#== Agent CONFIG ==
print("\n== Agent Information ==")
if args.annealing:
    GAMMA_END = 0.9999
    EPS_PERIOD = int(updatePeriod/10)
    EPS_RESET_PERIOD = updatePeriod
else:
    GAMMA_END = args.gamma
    EPS_PERIOD = updatePeriod
    EPS_RESET_PERIOD = maxUpdates

CONFIG = dqnConfig(DEVICE=device, ENV_NAME=env_name, SEED=args.randomSeed,
    MAX_UPDATES=maxUpdates, MAX_EP_STEPS=maxSteps,
    BATCH_SIZE=64, MEMORY_CAPACITY=args.memoryCapacity,
    ARCHITECTURE=args.architecture, ACTIVATION=args.actType,
    GAMMA=args.gamma, GAMMA_PERIOD=updatePeriod, GAMMA_END=GAMMA_END,
    EPS_PERIOD=EPS_PERIOD, EPS_DECAY=0.7, EPS_RESET_PERIOD=EPS_RESET_PERIOD,
    LR_C=args.learningRate, LR_C_PERIOD=updatePeriod, LR_C_DECAY=0.8,
    MAX_MODEL=50)
print(vars(CONFIG))


# #== AGENT ==
# numActionList = env.numActionList
# numJoinAction = int(numActionList[0] * numActionList[1])
# dimList = [stateDim] + CONFIG.ARCHITECTURE + [actionNum]
# agent = DDQNPursuitEvasion(CONFIG, numActionList, dimList, actType=args.actType)
# print(agent.device)

# print("\n== Training Information ==")
# vmin = -1
# vmax = 1
# checkPeriod = updatePeriod
# training_records, trainProgress = agent.learn(env,
#     MAX_UPDATES=maxUpdates, MAX_EP_STEPS=CONFIG.MAX_EP_STEPS, addBias=args.addBias,
#     warmupQ=args.warmup, warmupIter=args.warmupIter, doneTerminate=True,
#     vmin=vmin, vmax=vmax, showBool=False,
#     checkPeriod=checkPeriod, outFolder=outFolder,
#     plotFigure=args.plotFigure, storeFigure=args.storeFigure)