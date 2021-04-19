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
parser.add_argument("-mu",  "--maxUpdates",     help="maximal #gradient updates",       default=1.5e6,  type=int)
parser.add_argument("-mc",  "--memoryCapacity", help="memoryCapacity",                  default=1e4,    type=int)
parser.add_argument("-ut",  "--updateTimes",    help="#hyper-param. steps",             default=10,     type=int)
parser.add_argument("-wi",  "--warmupIter",     help="warmup iteration",                default=1000,   type=int)
parser.add_argument("-cp",  "--checkPeriod",    help="check period",                    default=75000,  type=int)

# hyper-parameters
parser.add_argument("-arc", "--architecture",   help="NN architecture",             default=[100],  nargs="*", type=int)
parser.add_argument("-lr",  "--learningRate",   help="learning rate",               default=1e-3,   type=float)
parser.add_argument("-g",   "--gamma",          help="contraction coeff.",          default=0.9,    type=float)
parser.add_argument("-t",   "--thickness",      help="thickness of the obstacle",   default=-1,     type=float)
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
toEnd = args.toEnd
env_name = "zermelo_show-v0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
maxUpdates = args.maxUpdates
updateTimes = args.updateTimes
updatePeriod = int(maxUpdates / updateTimes)
maxSteps = 120

if args.mode == 'lagrange':
    envMode = 'normal'
    agentMode = 'normal'
    GAMMA_END = args.gamma
    EPS_PERIOD = updatePeriod
    EPS_RESET_PERIOD = 1e8
elif args.mode == 'mayer':
    envMode = 'extend'
    agentMode = 'normal'
    GAMMA_END = args.gamma
    EPS_PERIOD = updatePeriod
    EPS_RESET_PERIOD = 1e8
elif args.mode == 'RA':
    envMode = 'RA'
    agentMode = 'RA'
    GAMMA_END = 0.9999
    EPS_PERIOD = int(updatePeriod/10)
    EPS_RESET_PERIOD = updatePeriod

outFolder = os.path.join(args.outFolder, 'show/' + args.name + timestr)
figureFolder = os.path.join(outFolder, 'figure/')
os.makedirs(figureFolder, exist_ok=True)


#== Environment ==
print("\n== Environment Information ==")
if toEnd:
    env = gym.make(env_name, device=device, mode=envMode, doneType='toEnd',
        thickness=args.thickness)
else:
    env = gym.make(env_name, device=device, mode=envMode, doneType='TF',
        thickness=args.thickness)

stateNum = env.state.shape[0]
actionNum = env.action_space.n
action_list = np.arange(actionNum)
print("State Dimension: {:d}, ActionSpace Dimension: {:d}".format(stateNum, actionNum))


#== Setting in this Environment ==
env.set_costParam(args.penalty, args.reward, args.costType, args.scaling)
env.set_seed(args.randomSeed)


#== Agent CONFIG ==
print("\n== Agent Information ==")
CONFIG = dqnConfig(DEVICE=device, ENV_NAME=env_name, SEED=args.randomSeed,
    MAX_UPDATES=maxUpdates, MAX_EP_STEPS=maxSteps,
    BATCH_SIZE=100, MEMORY_CAPACITY=args.memoryCapacity,
    ARCHITECTURE=args.architecture, ACTIVATION=args.actType,
    GAMMA=args.gamma, GAMMA_PERIOD=updatePeriod, GAMMA_END=GAMMA_END,
    EPS_PERIOD=EPS_PERIOD, EPS_DECAY=0.7, EPS_RESET_PERIOD=EPS_RESET_PERIOD,
    LR_C=args.learningRate, LR_C_PERIOD=updatePeriod, LR_C_DECAY=0.8,
    MAX_MODEL=50)

for key, value in CONFIG.__dict__.items():
    if key[:1] != '_': print(key, value)

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