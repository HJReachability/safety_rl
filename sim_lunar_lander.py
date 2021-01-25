#!/usr/bin/env python
# coding: utf-8


from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

from gym_reachability import gym_reachability  # Custom Gym env.
import gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
from collections import namedtuple
import argparse
from multiprocessing import Pool

from KC_DQN.DDQNSingle import DDQNSingle
from KC_DQN.config import dqnConfig

import time
timestr = time.strftime("%Y-%m-%d-%H_%M_%S")

#== ARGS ==
# e.g., python3 sim_zermelo.py -te -m lagrange -nt 50 -of lagrange_low_50 -ma 1500000 -p .1
# e.g., python3 sim_zermelo.py -te -m RA -nt 50 -of RA -ma 1500000
parser = argparse.ArgumentParser()
parser.add_argument("-nt",  "--num_test",       help="the number of tests",         default=1,      type=int)
parser.add_argument("-nw",  "--num_worker",     help="the number of workers",       default=1,      type=int)

# training scheme
parser.add_argument("-te",  "--toEnd",          help="stop until reaching boundary",    action="store_true")
parser.add_argument("-ab",  "--addBias",        help="add bias term for RA",            action="store_true")
parser.add_argument("-ma",  "--maxAccess",      help="maximal number of access",        default=4e6,  type=int)
parser.add_argument("-ms",  "--maxSteps",       help="maximal length of rollouts",      default=100,  type=int)
parser.add_argument("-cp",  "--check_period",   help="check the success ratio",         default=50000,  type=int)
parser.add_argument("-up",  "--update_period",  help="update period for scheduler",     default=int(4e6/20),  type=int)

# hyper-parameters
parser.add_argument("-r",   "--reward",         help="when entering target set",    default=-1,     type=float)
parser.add_argument("-p",   "--penalty",        help="when entering failure set",   default=1,      type=float)
parser.add_argument("-s",   "--scaling",        help="scaling of ell/g",            default=1,      type=float)
parser.add_argument("-lr",  "--learningRate",   help="learning rate",               default=1e-3,   type=float)
parser.add_argument("-g",   "--gamma",          help="contraction coefficient",     default=0.999,  type=float)
parser.add_argument("-arc", "--architecture",   help="neural network architecture", default=[512, 512, 512],  nargs="*", type=int)
parser.add_argument("-act", "--activation",     help="activation function",         default='Tanh', type=str)

# RL type
parser.add_argument("-m",   "--mode",           help="mode",            default='RA',       type=str)
parser.add_argument("-ct",  "--costType",       help="cost type",       default='sparse',   type=str)
parser.add_argument("-of",  "--outFolder",      help="output file",     default='RA' + timestr,       type=str)

args = parser.parse_args()
print(args)


# == CONFIGURATION ==
# todo(vrubies) change to "one_player_reach_avoid_lunar_lander-v0"
env_name = "lunar_lander_reachability-v0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
maxSteps = args.maxSteps  # Length limit for one episode.
maxUpdates = args.maxAccess
update_period = args.update_period  # int(maxEpisodes / 10)
update_period_half = int(update_period/2)

# if args.mode == 'lagrange':
#     envMode = 'normal'
#     agentMode = 'normal'
#     gammaInit = .9
#     gamma_period = 1000000
# elif args.mode == 'mayer':
#     envMode = 'extend'
#     agentMode = 'normal'
#     gammaInit = .9
#     gamma_period = 1000000
# elif args.mode == 'RA':
envMode = 'RA'
agentMode = 'RA'
gammaInit = args.gamma
gamma_period = update_period

CONFIG = dqnConfig(DEVICE=device,
                   ENV_NAME=env_name,
                   MAX_UPDATES=maxUpdates,
                   MAX_EP_STEPS=maxSteps,
                   BATCH_SIZE=100,
                   MEMORY_CAPACITY=10000,
                   GAMMA=gammaInit,
                   GAMMA_PERIOD=gamma_period,
                   GAMMA_END=0.999999,
                   EPS_PERIOD=1000,
                   EPS_DECAY=0.6,
                   LR_C=args.learningRate,
                   LR_C_PERIOD=update_period,
                   LR_C_DECAY=0.8)
                   # MAX_MODEL=50)
# == REPORT ==
for key, value in CONFIG.__dict__.items():
    if key[:1] != '_': print(key, value)


# == ENVIRONMENT ==
env = gym.make(env_name, device=device, mode=envMode, doneType='toEnd')
env.set_costParam(args.penalty, args.reward, args.costType, args.scaling)

# == EXPERIMENT ==
def multi_experiment(seedNum, args, CONFIG, env, report_period):
    # == AGENT ==
    s_dim = env.observation_space.shape[0]
    numAction = env.action_space.n
    actionList = np.arange(numAction)
    dimList = [s_dim] + args.architecture + [numAction]

    env.set_seed(seedNum)
    np.random.seed(seedNum)
    agent = DDQNSingle(CONFIG, numAction, actionList, dimList,
                       mode=agentMode, actType=args.activation)

    # If *true* episode ends when gym environment gives done flag.
    # If *false* end
    # == TRAINING ==
    _, trainProgress = agent.learn(
        env,
        MAX_UPDATES=maxUpdates,  # 6000000 for Dubins
        MAX_EP_STEPS=CONFIG.MAX_EP_STEPS,
        warmupBuffer=True,
        warmupQ=False,  # Need to implement inside env.
        warmupIter=1000,
        addBias=False,  # args.addBias,
        doneTerminate=True,
        runningCostThr=None,
        curUpdates=None,
        # toEnd=args.toEnd,
        # reportPeriod=report_period,  # How often to report Value function figs.
        plotFigure=False,  # Display value function while learning.
        showBool=False,  # Show boolean reach avoid set 0/1.
        vmin=-1,
        vmax=1,
        checkPeriod=args.check_period,  # How often to compute Safe vs. Unsafe.
        storeFigure=False,  # Store the figure in an eps file.
        storeModel=True,
        storeBest=False,
        # randomPlot=True,  # Plot from random starting points.
        outFolder=args.outFolder,
        verbose=True)
    return trainProgress


def plot_experiment(args, CONFIG, env, path):
    # == AGENT ==
    s_dim = env.observation_space.shape[0]
    numAction = env.action_space.n
    actionList = np.arange(numAction)
    dimList = [s_dim] + args.architecture + [numAction]
    agent = DDQNSingle(CONFIG, numAction, actionList, dimList,
                       mode=agentMode, actType=args.activation)
    agent.restore(path)

    env.visualize(agent.Q_network, True, nx=91, ny=91, boolPlot=False, trueRAZero=False,
        addBias=False, lvlset=0)

    # env.visualize(agent.Q_network, cmap='seismic', addBias=False)
    # env.visualize(agent.Q_network, vmin=0, boolPlot=True, addBias=False)
    plt.show()

path1 = "models/RA2021-01-21-21_37_49/model-1500000.pth"
# path2 = "models/RA2020-11-20-06_58_53/model-1500000.pth"
# plot_experiment(args, CONFIG, env, path1)
multi_experiment(0, args, CONFIG, env, update_period)

# == TESTING ==

# trainProgressList = []
# L = args.num_test
# nThr = args.num_worker
# for ith in range( int(L/(nThr+1e-6))+1 ):
#     print('{} / {}'.format(ith+1, int(L/(nThr+1e-6))+1) )
#     with Pool(processes = nThr) as pool:
#         seedList = list(range(ith*nThr, min(L, (ith+1)*nThr) ))
#         argsList = [args]*len(seedList)
#         configList = [CONFIG]*len(seedList)
#         envList = [env]*len(seedList)
#         reportPeriodList = [update_period]*len(seedList)
#         trainProgress_i = pool.starmap(multi_experiment, zip(seedList, argsList, configList, envList, reportPeriodList))
#     trainProgressList = trainProgressList + trainProgress_i
# print(trainProgressList)


# == RECORD ==
# import pickle
# with open("figure/{:s}/{:s}.txt".format(args.outFolder,
#                                         args.outFolder), "wb") as fp:
#     pickle.dump(trainProgressList, fp)