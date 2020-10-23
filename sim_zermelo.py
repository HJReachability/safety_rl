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

from KC_DQN.DDQN import DDQN
from KC_DQN.config import dqnConfig


#== ==
parser = argparse.ArgumentParser()
parser.add_argument("-nt", "--num_test",     help="the number of tests",      default=1,     type=int)
parser.add_argument("-ma", "--maxAccess",    help="maximal number of access", default=2e6,   type=int)
parser.add_argument("-cp", "--check_period", help="check the success rate",   default=50000, type=int)

parser.add_argument("-r", "--reward",  help="when entering target set",  default=-1, type=float)
parser.add_argument("-p", "--penalty", help="when entering failure set", default=1,  type=float)
parser.add_argument("-s", "--scaling", help="scaling of l_x",            default=1,  type=float)

parser.add_argument("-m",  "--mode",     help="mode",      default='RA',     type=str)
parser.add_argument("-ct", "--costType", help="cost type", default='sparse', type=str)
parser.add_argument("-of", "--outFile",  help="output file", default='RA', type=str)

parser.add_argument("-te", "--toEnd", help="stops until to the end", action="store_true")

args = parser.parse_args()
print(args)

#== CONFIGURATION ==
toEnd = args.toEnd
env_name = "zermelo_kc-v0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
maxAccess = args.maxAccess
maxSteps = 120
if toEnd:
    maxEpisodes = int(maxAccess / maxSteps * 2)
else:
    maxEpisodes = 60000
update_period = int(maxEpisodes / 10)

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
    gammaInit = .999
    gamma_period = 4000

CONFIG = dqnConfig(DEVICE=device, ENV_NAME=env_name, 
                   MAX_EPISODES=maxEpisodes, MAX_EP_STEPS=maxSteps,
                   BATCH_SIZE=100, MEMORY_CAPACITY=10000,
                   GAMMA=gammaInit, GAMMA_PERIOD=gamma_period,
                   EPS_PERIOD=update_period, EPS_DECAY=0.6,
                   LR_C=1e-2, LR_C_PERIOD=4000, LR_C_DECAY=0.8)

#== REPORT ==
for key, value in CONFIG.__dict__.items():
    if key[:1] != '_': print(key, value)
#print(CONFIG.MAX_EPISODES, CONFIG.MAX_EP_STEPS)


# == Environment ==
if toEnd:
    env = gym.make(env_name, device=device, mode=envMode, doneType='toEnd')
else:
    env = gym.make(env_name, device=device, mode=envMode)
env.set_costParam(args.penalty, args.reward, args.costType, args.scaling)

# == Discretization ==
grid_cells = (41, 121)
num_states = np.cumprod(grid_cells)[-1]
state_bounds = env.bounds
env.set_discretization(grid_cells, state_bounds)

s_dim = env.observation_space.shape[0]
action_num = env.action_space.n

action_list = np.arange(action_num)


#== AGENT ==
plt.figure()
vmin = -1
vmax = 1
#report_period = int(update_period / 2)
report_period = update_period

trainProgressList = []
for test_idx in range(args.num_test):
    print("== TEST --- {:d} ==".format(test_idx))
    env.set_seed(test_idx)
    agent=DDQN(s_dim, action_num, CONFIG, action_list, mode=agentMode, RA_scaling=args.scaling)
    _, trainProgress = agent.learn(env, MAX_EPISODES=CONFIG.MAX_EPISODES, MAX_EP_STEPS=CONFIG.MAX_EP_STEPS,
                                    report_period=report_period, vmin=vmin, vmax=vmax, warmupQ=False, toEnd=toEnd,
                                    check_period=args.check_period, storeModel=False)
    trainProgressList.append(trainProgress)


print(trainProgressList)
import pickle
with open("data/{:s}.txt".format(args.outFile), "wb") as fp:   #Pickling
    pickle.dump(trainProgressList, fp)

'''
matplotlib.style.use('default')
fig, ax = plt.subplots(1,1,figsize=(3,3))
trainProgress = np.array(trainProgress)
ax.plot(np.arange(trainProgress.shape[0])*args.check_period/1e5, trainProgress[:,0])
ax.plot([0, 20], [0.677, 0.677], 'r--')
ax.axis([0, 20, 0, 0.7])
ax.set_xlabel(r'Number of accesses ($\times10^5$)')
fig.tight_layout()

if toEnd:
    fig.savefig('figure/RA_toend_train.eps')
else:
    fig.savefig('figure/RA_train.eps')

plt.show()
'''