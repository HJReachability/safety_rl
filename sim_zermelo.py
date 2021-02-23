# Examples:
    # RA: python3 sim_zermelo.py -te -w -sf
    # Lagrange: python3 sim_zermelo.py -te -w -sf -m lagrange -of scratch


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

from KC_DQN.DDQNSingle import DDQNSingle
from KC_DQN.config import dqnConfig

import time
timestr = time.strftime("%Y-%m-%d-%H_%M")


#== ARGS ==
parser = argparse.ArgumentParser()

# training scheme
parser.add_argument("-te",  "--toEnd",          help="stop until reaching boundary",    action="store_true")
parser.add_argument("-ab",  "--addBias",        help="add bias term for RA",            action="store_true")
parser.add_argument("-w",   "--warmup",         help="warmup Q-network",                action="store_true")
parser.add_argument("-rnd", "--randomSeed",     help="random seed",                     default=0,      type=int)
parser.add_argument("-mu",  "--maxUpdates",     help="maximal #gradient updates",       default=4e6,    type=int)
parser.add_argument("-mc",  "--memoryCapacity", help="memoryCapacity",                  default=1e4,    type=int)
parser.add_argument("-ut",  "--updateTimes",    help="#hyper-param. steps",             default=20,     type=int)
parser.add_argument("-wi",  "--warmupIter",     help="warmup iteration",                default=10000,  type=int)
parser.add_argument("-cp",  "--checkPeriod",    help="check period",                    default=200000, type=int)

# hyper-parameters
parser.add_argument("-arc", "--architecture",   help="NN architecture",             default=[100],  nargs="*", type=int)
parser.add_argument("-lr",  "--learningRate",   help="learning rate",               default=1e-3,   type=float)
parser.add_argument("-g",   "--gamma",          help="contraction coeff.",          default=0.9,    type=float)
parser.add_argument("-r",   "--reward",         help="when entering target set",    default=-1,     type=float)
parser.add_argument("-p",   "--penalty",        help="when entering failure set",   default=1,      type=float)
parser.add_argument("-s",   "--scaling",        help="scaling of ell/g",            default=1,      type=float)
parser.add_argument("-act", "--actType",        help="activation type",             default='Tanh', type=str)

# RL type
parser.add_argument("-m",   "--mode",           help="mode",            default='RA',       type=str)
parser.add_argument("-ct",  "--costType",       help="cost type",       default='sparse',   type=str)

# file
parser.add_argument("-n",   "--name",           help="extra name",      default='',                         type=str)
parser.add_argument("-of",  "--outFolder",      help="output file",     default='/scratch/gpfs/kaichieh/',  type=str)
parser.add_argument("-pf",  "--plotFigure",     help="plot figures",    action="store_true")
parser.add_argument("-sf",  "--storeFigure",    help="store figures",   action="store_true")

args = parser.parse_args()
print(args)


#== CONFIGURATION ==
if args.mode == 'lagrange':
    envMode = 'normal'
    agentMode = 'normal'
    GAMMA_END=0.9
elif args.mode == 'mayer':
    envMode = 'extend'
    agentMode = 'normal'
    GAMMA_END=0.9
elif args.mode == 'RA':
    envMode = 'RA'
    agentMode = 'RA'
    GAMMA_END=0.999999

toEnd = args.toEnd
env_name = "zermelo_kc-v0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
maxUpdates = args.maxUpdates
updateTimes = args.updateTimes
updatePeriod = int(maxUpdates / updateTimes)
maxSteps = 120

outFolder = os.path.join(args.outFolder, 'naive/' + args.name + timestr)
figureFolder = os.path.join(outFolder, 'figure/')
os.makedirs(figureFolder, exist_ok=True)


#== Environment ==
print("\n== Environment Information ==")
if toEnd:
    env = gym.make(env_name, device=device, mode=envMode, doneType='toEnd')
else:
    env = gym.make(env_name, device=device, mode=envMode, doneType='TF')

stateNum = env.state.shape[0]
actionNum = env.action_space.n
action_list = np.arange(actionNum)
print("State Dimension: {:d}, ActionSpace Dimension: {:d}".format(stateNum, actionNum))


#== Setting in this Environment ==
env.set_costParam(args.penalty, args.reward, args.costType, args.scaling)
env.set_seed(args.randomSeed)


#== Get and Plot max{l_x, g_x} ==
if args.plotFigure or args.storeFigure:
    nx, ny = 81, 241
    v = np.zeros((nx, ny))
    l_x = np.zeros((nx, ny))
    g_x = np.zeros((nx, ny))
    xs = np.linspace(env.bounds[0,0], env.bounds[0,1], nx)
    ys =np.linspace(env.bounds[1,0], env.bounds[1,1], ny)

    it = np.nditer(v, flags=['multi_index'])

    while not it.finished:
        idx = it.multi_index
        x = xs[idx[0]]
        y = ys[idx[1]]
        
        l_x[idx] = env.target_margin(np.array([x, y]))
        g_x[idx] = env.safety_margin(np.array([x, y]))

        v[idx] = np.maximum(l_x[idx], g_x[idx])
        it.iternext()

    axStyle = env.get_axes()
    fig, axes = plt.subplots(1,3, figsize=(12, 4))
    ax = axes[0]
    f = ax.imshow(l_x.T, interpolation='none', extent=axStyle[0],
        origin="lower", cmap="seismic")
    ax.axis(axStyle[0])
    ax.grid(False)
    ax.set_aspect(axStyle[1])  # makes equal aspect ratio
    env.plot_target_failure_set(ax)
    ax.set_title(r'$\ell(x)$')
    cbar = fig.colorbar(f, ax=ax, pad=0.01, fraction=0.05, shrink=.9)
    env.plot_formatting(ax=ax)

    ax = axes[1]
    f = ax.imshow(g_x.T, interpolation='none', extent=axStyle[0],
        origin="lower", cmap="seismic")
    ax.axis(axStyle[0])
    ax.grid(False)
    ax.set_aspect(axStyle[1])  # makes equal aspect ratio
    env.plot_target_failure_set(ax)
    ax.set_title(r'$g(x)$')
    cbar = fig.colorbar(f, ax=ax, pad=0.01, fraction=0.05, shrink=.9)
    env.plot_formatting(ax=ax)

    ax = axes[2]
    f = ax.imshow(v.T, interpolation='none', extent=axStyle[0],
        origin="lower", cmap="seismic", vmin=-.5, vmax=.5)
    ax = plt.gca()
    ax.axis(axStyle[0])
    ax.grid(False)
    ax.set_aspect(axStyle[1])  # makes equal aspect ratio
    env.plot_target_failure_set(ax)
    ax.set_title(r'$v(x)$')
    cbar = fig.colorbar(f, ax=ax, pad=0.01, fraction=0.05, shrink=.9)
    env.plot_formatting(ax=ax)
    plt.tight_layout()
    fig.savefig('{:s}env.png'.format(figureFolder))
    plt.close()


#== Agent CONFIG ==
print("\n== Agent Information ==")
CONFIG = dqnConfig(DEVICE=device, ENV_NAME=env_name, SEED=args.randomSeed,
    MAX_UPDATES=maxUpdates, MAX_EP_STEPS=maxSteps,
    BATCH_SIZE=100, MEMORY_CAPACITY=args.memoryCapacity,
    ARCHITECTURE=args.architecture, ACTIVATION=args.actType,
    GAMMA=args.gamma, GAMMA_PERIOD=updatePeriod, GAMMA_END=GAMMA_END,
    EPS_PERIOD=updatePeriod, EPS_DECAY=0.6,
    LR_C=args.learningRate, LR_C_PERIOD=updatePeriod, LR_C_DECAY=0.8,
    MAX_MODEL=120)
print(vars(CONFIG))
picklePath = os.path.join(outFolder, 'CONFIG.pkl')
with open(picklePath, 'wb') as handle:
    pickle.dump(CONFIG, handle, protocol=pickle.HIGHEST_PROTOCOL)


#== AGENT ==
dimList = [stateNum] + CONFIG.ARCHITECTURE + [actionNum]
agent = DDQNSingle(CONFIG, actionNum, action_list, dimList=dimList,
    mode=agentMode, actType='Tanh')
print(device)
# print(agent.Q_network.moduleList[0].weight.type())
# print(agent.optimizer, '\n')


print("\n== Training Information ==")
vmin = -1
vmax = 1
checkPeriod = args.checkPeriod
training_records, trainProgress = agent.learn(env,
    MAX_UPDATES=maxUpdates, MAX_EP_STEPS=maxSteps, addBias=args.addBias,
    warmupQ=args.warmup, warmupIter=args.warmupIter, doneTerminate=True,
    vmin=vmin, vmax=vmax, showBool=False,
    checkPeriod=checkPeriod, outFolder=outFolder,
    plotFigure=args.plotFigure, storeFigure=args.storeFigure)