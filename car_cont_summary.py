from warnings import simplefilter 
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)

from gym_reachability import gym_reachability  # Custom Gym envHigh.
import gym
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import os
import pickle
import argparse

from KC_DQN.SAC import SAC
from KC_DQN.TD3 import TD3

parser = argparse.ArgumentParser()
parser.add_argument("-n",  "--n", default=201,  type=int)
parser.add_argument("-o",   "--orientation",    default=0.,     type=float)
parser.add_argument("-f",   "--figname",        default='value_rollout_action',     type=str)


args = parser.parse_args()
print(args)

def reportEnv(env):
    print("Dynamic parameters:")
    print("  CAR")
    print("    Constraint radius: {:.1f}, Target radius: {:.1f}, Turn radius: {:.2f}, Maximum speed: {:.2f}, Maximum angular speed: {:.2f}".format(
        env.car.constraint_radius, env.car.target_radius, env.car.R_turn, env.car.speed, env.car.max_turning_rate[0]))
    print("  ENV")
    print("    Constraint radius: {:.1f}, Target radius: {:.1f}, Turn radius: {:.2f}, Maximum speed: {:.2f}".format(
        env.constraint_radius, env.target_radius, env.R_turn, env.speed))


def getModelInfo(path):
    dataFolder = os.path.join('scratch', path)
    modelFolder = os.path.join(dataFolder, 'model')
    picklePath = os.path.join(modelFolder, 'CONFIG.pkl')
    with open(picklePath, 'rb') as fp:
        config = pickle.load(fp)
    config.DEVICE = 'cpu'

    dimListActor = [stateDim] + config.ARCHITECTURE + [actionDim]
    dimListCritic = [stateDim + actionDim] + config.ARCHITECTURE + [1]
    dimLists = [dimListCritic, dimListActor]
    return dataFolder, config, dimLists


#== ENVIRONMENT ==
env_name = "dubins_car_cont-v0"

envHigh = gym.make(env_name, device='cpu', mode='RA')
envHigh.set_radius(target_radius=.5, constraint_radius=1., R_turn=.6)
reportEnv(envHigh)
print()

envLow = gym.make(env_name, device='cpu', mode='RA')
envLow.set_radius(target_radius=.4, constraint_radius=1., R_turn=.75)
reportEnv(envLow)

envList = [envHigh, envLow]
stateDim = envHigh.state.shape[0]
actionDim = envHigh.action_space.shape[0]

#== AGENT ==
dataFolder, config, dimLists = getModelInfo(os.path.join('car-TD3', 'H0-2021-05-16-03_42'))
agentT_H = TD3(config, envHigh.action_space, dimLists, verbose=False)
agentT_H.restore(1200000, dataFolder)

dataFolder, config, dimLists = getModelInfo(os.path.join('car-SAC', 'H-A-2021-05-16-09_16'))
agentS_H_A = SAC(config, envHigh.action_space, dimLists, verbose=False)
agentS_H_A.restore(1200000, dataFolder)

dataFolder, config, dimLists = getModelInfo(os.path.join('car-SAC', 'H-F-2021-05-16-04_03'))
agentS_H_F = SAC(config, envHigh.action_space, dimLists, verbose=False)
agentS_H_F.restore(1200000, dataFolder)

dataFolder, config, dimLists = getModelInfo(os.path.join('car-TD3', 'L0-2021-05-16-03_42'))
agentT_L = TD3(config, envLow.action_space, dimLists, verbose=False)
agentT_L.restore(1200000, dataFolder)

dataFolder, config, dimLists = getModelInfo(os.path.join('car-SAC', 'L-A-2021-05-16-03_34'))
agentS_L_A = SAC(config, envLow.action_space, dimLists, verbose=False)
agentS_L_A.restore(1200000, dataFolder)

dataFolder, config, dimLists = getModelInfo(os.path.join('car-SAC', 'L-F-2021-05-16-04_03'))
agentS_L_F = SAC(config, envLow.action_space, dimLists, verbose=False)
agentS_L_F.restore(1200000, dataFolder)

agentArray = [  [agentT_H, agentS_H_A, agentS_H_F],
                [agentT_L, agentS_L_A, agentS_L_F]]


#== Rollout Reach-Avoid Set ==
nx=args.n
ny=args.n
orientation = args.orientation / 180 * np.pi 
resultMtxArray = np.empty((2, 3, nx, ny),  dtype=int)
actDistArray = np.empty((2, 3, nx, ny), dtype=float)
xs = np.linspace(envHigh.bounds[0,0], envHigh.bounds[0,1], nx)
ys = np.linspace(envHigh.bounds[1,0], envHigh.bounds[1,1], ny)

for i in range(2):
    env = envList[i]

    for j in range(3):
        print("== {}, {} ==".format(i, j))
        agent = agentArray[i][j]

        resultMtx  = np.empty((nx, ny), dtype=int)
        actDistMtx = np.empty((nx, ny), dtype=float)

        it = np.nditer(resultMtx, flags=['multi_index'])

        while not it.finished:
            idx = it.multi_index
            print(idx, end='\r')
            x = xs[idx[0]]
            y = ys[idx[1]]

            state = np.array([x, y, orientation])
            stateTensor = torch.FloatTensor(state)
            u = agent.actor(stateTensor).detach().cpu().numpy()[0]
            actDistMtx[idx] = u

            _, result, _, _ = env.simulate_one_trajectory(agent.actor, T=100, state=state, toEnd=False)
            resultMtx[idx] = result
            it.iternext()
            
        resultMtxArray[i, j] = resultMtx
        actDistArray[i, j] = actDistMtx
        print()


#== FIGURE ==
vmin = -1
vmax = 1

nArray = [  ['TD3-High', 'SAC-High-Auto', 'SAC-High-Fixed'],
            ['TD3-Low', 'SAC-Low-Auto', 'SAC-Low-Fixed']]

figureFolder = os.path.join('figure', 'car-cont')
os.makedirs(figureFolder, exist_ok=True)

fig, axes = plt.subplots(3, 6, figsize=(24, 12), sharex=True, sharey=True)

for i in range(2):
    env = envList[i]
    axStyle = env.get_axes()
    amax = env.car.max_turning_rate[0]

    for j in range(3):
        print("== {}, {} ==".format(i, j))
        agent = agentArray[i][j]
        idx = 3*i+j
        resultMtx = resultMtxArray[i, j]
        actDistMtx = actDistArray[i, j]

        #= Action
        ax = axes[2][idx]
        im = ax.imshow(actDistMtx.T, interpolation='none', extent=axStyle[0],
            origin="lower", cmap='seismic', vmin=-amax, vmax=amax, zorder=-1)
        if idx == 0:
            ax.set_ylabel('Action', fontsize=24)

        #= Rollout
        ax = axes[1][idx]
        im = ax.imshow(resultMtx.T != 1, interpolation='none', extent=axStyle[0],
            origin="lower", cmap='coolwarm', vmin=0, vmax=1, zorder=-1)
        if idx == 0:
            ax.set_ylabel('Rollout', fontsize=24)

        #= Value
        ax = axes[0][idx]
        v = env.get_value(agent.critic.Q1, agent.actor, orientation, nx, ny)
        # Plot V
        im = ax.imshow(v.T, interpolation='none', extent=axStyle[0],
            origin="lower", cmap='seismic', vmin=vmin, vmax=vmax, zorder=-1)
#         cbar = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.05, shrink=.95,
#             ticks=[vmin, 0, vmax])
#         cbar.ax.set_yticklabels(labels=[vmin, 0, vmax], fontsize=24)
        CS = ax.contour(xs, ys, v.T, levels=[0], colors='k', linewidths=2,
            linestyles='dashed')
        if idx == 0:
            ax.set_ylabel('Value', fontsize=24)
        ax.set_title(nArray[i][j], fontsize=24)

        for k in range(3):
            env.plot_target_failure_set(ax=axes[k][idx])
            env.plot_reach_avoid_set(ax=axes[k][idx], orientation=orientation)
            env.plot_formatting(ax=axes[k][idx])

fig.tight_layout()
fig.savefig(os.path.join(figureFolder,  args.figname + '.png'), doi=200)