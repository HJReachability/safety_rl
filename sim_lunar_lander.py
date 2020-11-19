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

from KC_DQN.DDQN import DDQN
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
parser.add_argument("-ma",  "--maxAccess",      help="maximal number of access",        default=1.5e6,  type=int)
parser.add_argument("-cp",  "--check_period",   help="check the success ratio",         default=50000,  type=int)
parser.add_argument("-vp",  "--vis_period",   help="visualize period",                  default=5000,  type=int)

# hyper-parameters
parser.add_argument("-r",   "--reward",         help="when entering target set",    default=-1,     type=float)
parser.add_argument("-p",   "--penalty",        help="when entering failure set",   default=1,      type=float)
parser.add_argument("-s",   "--scaling",        help="scaling of ell/g",            default=1,      type=float)
parser.add_argument("-lr",  "--learningRate",   help="learning rate",               default=1e-3,   type=float)
parser.add_argument("-g",   "--gamma",          help="contraction coefficient",     default=0.999,  type=float)

# RL type
parser.add_argument("-m",   "--mode",           help="mode",            default='RA',       type=str)
parser.add_argument("-ct",  "--costType",       help="cost type",       default='sparse',   type=str)
parser.add_argument("-of",  "--outFolder",        help="output file",     default='RA' + timestr,       type=str)

args = parser.parse_args()
print(args)


# == CONFIGURATION ==
env_name = "lunar_lander_reachability-v0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
maxSteps = 120
if args.toEnd:
    maxEpisodes = int(args.maxAccess / maxSteps * 2)
else:
    maxEpisodes = 60000
update_period = args.vis_period  # int(maxEpisodes / 10)
update_period_half = int(update_period/2)

if args.mode == 'lagrange':
    envMode = 'normal'
    agentMode = 'normal'
    gammaInit = .9
    gamma_period = 1000000
elif args.mode == 'mayer':
    envMode = 'extend'
    agentMode = 'normal'
    gammaInit = .9
    gamma_period = 1000000
elif args.mode == 'RA':
    envMode = 'RA'
    agentMode = 'RA'
    gammaInit = args.gamma
    gamma_period = update_period

CONFIG = dqnConfig(DEVICE=device,
                   ENV_NAME=env_name,
                   MAX_EPISODES=maxEpisodes,
                   MAX_EP_STEPS=maxSteps,
                   BATCH_SIZE=100,
                   MEMORY_CAPACITY=10000,
                   GAMMA=gammaInit,
                   GAMMA_PERIOD=gamma_period,
                   EPS_PERIOD=1000,
                   EPS_DECAY=0.6,
                   LR_C=args.learningRate,
                   LR_C_PERIOD=update_period,
                   LR_C_DECAY=0.8)
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
    action_num = env.action_space.n
    action_list = np.arange(action_num)

    env.set_seed(seedNum)
    np.random.seed(seedNum)
    agent = DDQN(s_dim, action_num, CONFIG, action_list, mode=agentMode,
                 RA_scaling=args.scaling)

    # == TRAINING ==
    _, trainProgress = agent.learn(
        env,
        # MAX_EPISODES=CONFIG.MAX_EPISODES,
        MAX_EP_STEPS=CONFIG.MAX_EP_STEPS,
        addBias=True,  # args.addBias,
        toEnd=args.toEnd,
        reportPeriod=report_period,  # How often to report Value function figs.
        plotFigure=True,  # Display value function while learning.
        showBool=False,  # Show boolean reach avoid set 0/1.
        checkPeriod=args.check_period,  # How often to compute Safe vs. Unsafe.
        storeFigure=True,  # Store the figure in an eps file.
        storeModel=False,
        randomPlot=True,  # Plot from random starting points.
        outFolder=args.outFolder,
        verbose=True)
    return trainProgress


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
import pickle
with open("figure/{:s}/{:s}.txt".format(args.outFolder,
                                        args.outFolder), "wb") as fp:
    pickle.dump(trainProgressList, fp)
