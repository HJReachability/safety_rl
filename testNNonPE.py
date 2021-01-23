# -*- coding: utf-8 -*-
from warnings import simplefilter 
simplefilter(action='ignore', category=FutureWarning)

from gym_reachability import gym_reachability  # Custom Gym env.
import gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
from torch.nn.functional import mse_loss, smooth_l1_loss
import torch.nn as nn
from collections import namedtuple
import os
import argparse

from KC_DQN.DDQNPursuitEvasion import DDQNPursuitEvasion
from KC_DQN.config import dqnConfig

parser = argparse.ArgumentParser()
parser.add_argument("-figF", "--figureFolder", help="figure folder", default='RA', type=str)
parser.add_argument("-s", "--server", help="server", default='adroit', type=str)
args = parser.parse_args()
print(args)

if args.server == 'adroit':
    preFigureFolder = '/scratch/network/kaichieh/'
elif args.server == 'della':
    preFigureFolder = '/scratch/gpfs/kaichieh/'

figureFolder = preFigureFolder + args.figureFolder
os.makedirs(figureFolder, exist_ok=True)

np.set_printoptions(precision=4)

'''
MAIN
'''

env_name = "dubins_car_pe-v0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
print("Let's use", torch.cuda.device_count(), "GPUs!")

#== Environment ==
env = gym.make(env_name, device=device, mode='RA', doneType='toEnd')

#== Setting in this Environment ==
print("Dynamic parameters:")
print("  EVADER")
print("    Constraint radius: {:.1f}, Target radius: {:.1f}, Turn radius: {:.2f}, Maximum speed: {:.2f}, Maximum angular speed: {:.3f}".format(
        env.evader.constraint_radius, env.evader.target_radius, env.evader.R_turn, env.evader.speed, env.evader.max_turning_rate))
print("  PURSUER")
print("    Constraint radius: {:.1f}, Turn radius: {:.2f}, Maximum speed: {:.2f}, Maximum angular speed: {:.3f}".format(
        env.pursuer.constraint_radius, env.pursuer.R_turn, env.pursuer.speed, env.pursuer.max_turning_rate))
print(env.evader.discrete_controls)
if 2*env.evader.R_turn-env.evader.constraint_radius > env.evader.target_radius:
    print("Type II Reach-Avoid Set")
else:
    print("Type I Reach-Avoid Set")

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
    
print(np.max(g_x), np.min(g_x), np.max(l_x), np.min(l_x))

axStyle = env.get_axes()

fig, ax = plt.subplots(1, 3, figsize=(16,4), sharex=True, sharey=True)

f0 = ax[0].imshow(l_x[0].T, interpolation='none', extent=axStyle[0], origin="lower", cmap="seismic", vmin=-1, vmax=1)
ax[0].axis(axStyle[0])
ax[0].grid(False)
ax[0].set_aspect(axStyle[1])  # makes equal aspect ratio
# env.plot_target_failure_set(ax[0], xPursuer=xPursuer, yPursuer=yPursuer)
ax[0].set_title(r'$\ell(x)$')
fig.colorbar(f0, ax=ax[0], pad=0.01, shrink=0.9)

f1 = ax[1].imshow(g_x[0].T, interpolation='none', extent=axStyle[0], origin="lower", cmap="seismic", vmin=-1, vmax=1)
ax[1].axis(axStyle[0])
ax[1].grid(False)
ax[1].set_aspect(axStyle[1])  # makes equal aspect ratio
ax[1].set_title(r'$g(x)$')
# env.plot_target_failure_set(ax[1], xPursuer=xPursuer, yPursuer=yPursuer)
fig.colorbar(f1, ax=ax[1], pad=0.01, shrink=0.9)

f2 = ax[2].imshow(v[0].T, interpolation='none', extent=axStyle[0], origin="lower", cmap="seismic", vmin=-1, vmax=1)
ax[2].axis(axStyle[0])
ax[2].grid(False)
ax[2].set_aspect(axStyle[1])  # makes equal aspect ratio
# env.plot_target_failure_set(ax[2], xPursuer=xPursuer, yPursuer=yPursuer)
ax[2].set_title(r'$v(x)$')
fig.colorbar(f2, ax=ax[2], pad=0.01, shrink=0.9)

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

fig, axes = plt.subplots(1,4, figsize=(16, 4))

xPursuerList=[.1, .3, .5, .7]
yPursuerList=[.1, .3, .5, .7]
for i, (ax, xPursuer, yPursuer) in enumerate(zip(axes, xPursuerList, yPursuerList)):
    states, heuristic_v = env.get_warmup_examples(num_warmup_samples=1000,
        theta=0, thetaPursuer=0, xPursuer=xPursuer, yPursuer=yPursuer)

    ax.scatter(states[:,0], states[:,1], c=heuristic_v[:,0], cmap='seismic', vmin=-1, vmax=1)

"""### Test on NN"""

numActionList = [3,3]
numJoinAction = int(numActionList[0] * numActionList[1])
stateNum = 6
CONFIG = dqnConfig(LR_C=1e-3, DEVICE=device)
agent = DDQNPursuitEvasion(CONFIG, numActionList, [stateNum, 512, 512, 512, numJoinAction], actType='Tanh')
# agent = DDQNPursuitEvasion(CONFIG, numActionList, [stateNum, 200, numJoinAction], actType='Tanh')

print(agent.Q_network.moduleList[0].weight.type())
print(agent.optimizer)

warmupIter=50000
num_warmup_samples=500
lossList = agent.initQ(env, warmupIter=warmupIter, num_warmup_samples=200, vmin=-1, vmax=1)


fig, ax = plt.subplots(1,1, figsize=(4, 4))
lossList = np.array(lossList)
ax.plot(lossList[:], 'b-')
print(lossList[-10:])
fig.savefig('loss.jpg')
#==

fig, axes = plt.subplots(1,4, figsize=(16, 4))
xPursuerList=[.1, .3, .5, .7]
yPursuerList=[.1, .3, .5, .7]
for i, (ax, xPursuer, yPursuer) in enumerate(zip(axes, xPursuerList, yPursuerList)):
    cbarPlot = i==3
    env.plot_formatting(ax=ax)
    env.plot_target_failure_set(ax=ax, xPursuer=xPursuer, yPursuer=yPursuer)
    env.plot_v_values(agent.Q_network, ax=ax, fig=fig, cbarPlot=cbarPlot,
                            xPursuer=xPursuer, yPursuer=yPursuer, cmap='seismic', vmin=-1, vmax=1)
plt.pause(0.001)
fig.savefig('initQ.jpg')