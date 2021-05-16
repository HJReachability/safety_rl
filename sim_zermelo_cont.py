from warnings import simplefilter 
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)

from gym_reachability import gym_reachability  # Custom Gym env.
import gym
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import os
import argparse

from KC_DQN.SAC import SAC
from KC_DQN.TD3 import TD3
from KC_DQN.config import SACConfig, actorCriticConfig

import time
timestr = time.strftime("%Y-%m-%d-%H_%M")

# test
    # python3 sim_zermelo_cont.py -w -wi 5 -mu 200 -ut 2 -cp 100 -of scratch/tmp -sf
# default
    # python3 sim_zermelo_cont.py -w -sf -of scratch -la
    # python3 sim_zermelo_cont.py -w -sf -of scratch -at TD3


#== ARGS ==
parser = argparse.ArgumentParser()

# training scheme
parser.add_argument("-w",   "--warmup",         help="warmup Q-network",            action="store_true")
parser.add_argument("-rnd", "--randomSeed",     help="random seed",                 default=0,          type=int)
parser.add_argument("-mu",  "--maxUpdates",     help="maximal #gradient updates",   default=600000,     type=int)
parser.add_argument("-mc",  "--memoryCapacity", help="memoryCapacity",              default=10000,      type=int)
parser.add_argument("-ut",  "--updateTimes",    help="#hyper-param. steps",         default=12,         type=int)
parser.add_argument("-wi",  "--warmupIter",     help="warmup iteration",            default=5000,       type=int)
parser.add_argument("-cp",  "--checkPeriod",    help="check period",                default=25000,      type=int)
parser.add_argument("-at",  "--agentType",      help="agent type",                  default='SAC',      type=str)
parser.add_argument("-dt",  "--doneType",       help="when to raise done flag",     default='fail',     type=str)
parser.add_argument("-tt",  "--terminalType",   help="terminal value",              default='g',        type=str)

# hyper-parameters
parser.add_argument("-arc",  "--architecture",  help="NN architecture",         default=[100, 20],          nargs="*", type=int)
parser.add_argument("-act",  "--actType",       help="activation type",         default=['Sin', 'ReLU'],   nargs=2,   type=str)
parser.add_argument("-lrA",  "--lrA",           help="learning rate actor",     default=1e-3,   type=float)
parser.add_argument("-lrC",  "--lrC",           help="learning rate critic",    default=1e-3,   type=float)
parser.add_argument("-lrAl", "--lrAl",          help="learning rate alpha",     default=5e-4,   type=float)
parser.add_argument("-g",    "--gamma",         help="contraction coeff.",      default=0.98,    type=float)
parser.add_argument("-a",    "--alpha",         help="initial temperature",     default=0.05,    type=float)
parser.add_argument("-la",   "--learnAlpha",    help="learnable temperature",   action="store_true")

# Lagrange RL
parser.add_argument("-r",   "--reward",         help="when entering target set",    default=-1,     type=float)
parser.add_argument("-p",   "--penalty",        help="when entering failure set",   default=1,      type=float)
parser.add_argument("-s",   "--scaling",        help="scaling of ell/g",            default=4,      type=float)

# file
parser.add_argument("-n",   "--name",           help="extra name",      default='',                         type=str)
parser.add_argument("-of",  "--outFolder",      help="output file",     default='/scratch/gpfs/kaichieh/',  type=str)
parser.add_argument("-pf",  "--plotFigure",     help="plot figures",    action="store_true")
parser.add_argument("-sf",  "--storeFigure",    help="store figures",   action="store_true")

args = parser.parse_args()
print(args)


#== CONFIGURATION ==
env_name = "zermelo_cont-v0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
maxUpdates = args.maxUpdates
updateTimes = args.updateTimes
updatePeriod = int(maxUpdates / updateTimes)
maxSteps = 120
storeFigure = args.storeFigure
plotFigure = args.plotFigure
vmin = -5
vmax = 5

outFolder = os.path.join(args.outFolder, 'naive', args.agentType, args.name + timestr)
figureFolder = os.path.join(outFolder, 'figure/')
os.makedirs(figureFolder, exist_ok=True)


#== Environment ==
print("\n== Environment Information ==")
env = gym.make(env_name, device=device, mode="RA", doneType='fail')
env.set_costParam(penalty=args.penalty, reward=args.reward, scaling=args.scaling)

stateDim = env.observation_space.shape[0]
actionDim = env.action_space.shape[0]

env.set_seed(args.randomSeed)
print("State Dimension: {:d}, ActionSpace Dimension: {:d}".format(stateDim, actionDim))
print("Margin scaling: {:.1f}, Reward: {:.1f}, Penalty: {:.1f}".format(
    env.scaling, env.reward, env.penalty))


#== Get and Plot max{l_x, g_x} ==
if plotFigure or storeFigure:
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

    fig, axes = plt.subplots(1,3, figsize=(12,6))

    ax = axes[0]
    im = ax.imshow(l_x.T, interpolation='none', extent=axStyle[0],
        origin="lower", cmap="seismic", vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.05, shrink=.95,
        ticks=[vmin, 0, vmax])
    cbar.ax.set_yticklabels(labels=[vmin, 0, vmax], fontsize=24)
    ax.set_title(r'$\ell(x)$', fontsize=18)

    ax = axes[1]
    im = ax.imshow(g_x.T, interpolation='none', extent=axStyle[0],
        origin="lower", cmap="seismic", vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.05, shrink=.95,
        ticks=[vmin, 0, vmax])
    cbar.ax.set_yticklabels(labels=[vmin, 0, vmax], fontsize=24)
    ax.set_title(r'$g(x)$', fontsize=18)

    ax = axes[2]
    im = ax.imshow(v.T, interpolation='none', extent=axStyle[0],
        origin="lower", cmap="seismic", vmin=vmin, vmax=vmax)
    env.plot_reach_avoid_set(ax)
    cbar = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.05, shrink=.95,
        ticks=[vmin, 0, vmax])
    cbar.ax.set_yticklabels(labels=[vmin, 0, vmax], fontsize=24)
    ax.set_title(r'$v(x)$', fontsize=18)

    for ax in axes:
        env.plot_target_failure_set(ax=ax)
        env.plot_formatting(ax=ax)

    fig.tight_layout()
    if storeFigure:
        figurePath = os.path.join(figureFolder, 'env.png')
        fig.savefig(figurePath)
    if plotFigure:
        plt.show()
        plt.pause(0.001)
    plt.close()


#== Agent CONFIG ==
print("\n== Agent Information ==")
actType={'critic': args.actType[0], 'actor': args.actType[1]}
if args.agentType == 'SAC':
    CONFIG = SACConfig(
        ENV_NAME=env_name,
        DEVICE=device,
        MAX_UPDATES=maxUpdates,
        MAX_EP_STEPS=maxSteps,
        # =================== LEARNING RATE PARAMS.
        LR_C=args.lrC,
        LR_C_END=args.lrC/10,
        LR_C_PERIOD=updatePeriod,
        LR_C_DECAY=0.9,
        LR_A=args.lrA,
        LR_A_END=args.lrA/10,
        LR_A_PERIOD=updatePeriod,
        LR_A_DECAY=0.9,
        LR_Al=5e-4, 
        LR_Al_END=1e-5,
        LR_Al_PERIOD=updatePeriod,
        LR_Al_DECAY=0.9,
        # =================== LEARNING RATE .
        GAMMA=args.gamma,
        GAMMA_END=0.9999,
        GAMMA_PERIOD=updatePeriod,
        GAMMA_DECAY=0.5,
        # ===================
        ALPHA=args.alpha,
        LEARN_ALPHA=args.learnAlpha,
        # ===================
        MEMORY_CAPACITY=args.memoryCapacity,
        ARCHITECTURE=args.architecture,
        ACTIVATION=actType,
        REWARD=args.reward,
        PENALTY=args.penalty)
elif args.agentType == 'TD3':
    CONFIG = actorCriticConfig(
        ENV_NAME=env_name,
        DEVICE=device,
        MAX_UPDATES=maxUpdates,
        MAX_EP_STEPS=maxSteps,
        # =================== LEARNING RATE PARAMS.
        LR_C=args.lrC,
        LR_C_END=args.lrC/10,
        LR_C_PERIOD=updatePeriod,
        LR_C_DECAY=0.9,
        LR_A=args.lrA,
        LR_A_END=args.lrA/10,
        LR_A_PERIOD=updatePeriod,
        LR_A_DECAY=0.9,
        # =================== LEARNING RATE .
        GAMMA=args.gamma,
        GAMMA_END=0.9999,
        GAMMA_PERIOD=updatePeriod,
        GAMMA_DECAY=0.5,
        # ===================
        MEMORY_CAPACITY=args.memoryCapacity,
        ARCHITECTURE=args.architecture,
        ACTIVATION=actType,
        REWARD=args.reward,
        PENALTY=args.penalty)
else:
    raise ValueError("{} is not supported. We only support SAC and TD3.".format(args.agentType))

# for key, value in CONFIG.__dict__.items():
#     if key[:1] != '_': print(key, value)
print(CONFIG.ACTIVATION)


#== AGENT ==
dimListActor = [stateDim] + args.architecture + [actionDim]
dimListCritic = [stateDim + actionDim] + args.architecture + [1]
dimLists = [dimListCritic, dimListActor]

if args.agentType == 'SAC':
    agent = SAC(CONFIG, env.action_space, dimLists, args.terminalType)
elif args.agentType == 'TD3':
    agent = TD3(CONFIG, env.action_space, dimLists, args.terminalType)
print('Agent has terminal type:', agent.terminalType)
print("We want to use: {}, and Agent uses: {}".format(device, agent.device))
print("Critic is using cuda: ", next(agent.critic.parameters()).is_cuda)
print("Actor is using cuda: ", next(agent.actor.parameters()).is_cuda)

if args.warmup:
    print("\n== Warmup Q ==")
    lossList = agent.initQ(env, args.warmupIter, outFolder,
        num_warmup_samples=200, vmin=vmin, vmax=vmax,
        plotFigure=plotFigure, storeFigure=storeFigure)

    if plotFigure or storeFigure:
        fig, ax = plt.subplots(1,1, figsize=(4, 4))

        ax.plot(lossList, 'b-')
        ax.set_xlabel('Iteration', fontsize=18)
        ax.set_ylabel('Loss', fontsize=18)
        plt.tight_layout()

        if storeFigure:
            figurePath = os.path.join(figureFolder, 'initQ_Loss.png')
            fig.savefig(figurePath)
        if plotFigure:
            plt.show()
            plt.pause(0.001)
        plt.close()


print("\n== Training Information ==")
trainRecords, trainProgress = agent.learn(env,
    MAX_UPDATES=maxUpdates, MAX_EP_STEPS=maxSteps,
    warmupQ=False, warmupIter=args.warmupIter,
    vmin=vmin, vmax=vmax, checkPeriod=args.checkPeriod, outFolder=outFolder,
    plotFigure=plotFigure, storeFigure=storeFigure, saveBest=False)

if plotFigure or storeFigure:
    #= Train Progress
    cList = ['b', 'r', 'g', 'k']
    nList = ['loss_q', 'loss_pi', 'loss_entropy', 'loss_alpha']

    if args.agentType == 'SAC':
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        for i in range(4):
            ax = axes[i]
            data = trainRecords[:, i]
            c = cList[i]

            ax.plot(data, ':', color=c)
            ax.set_xlabel('Iteration (x 1e5)', fontsize=18)
            ax.set_xticks(np.linspace(0, maxUpdates, 5))
            ax.set_xticklabels(np.linspace(0, maxUpdates, 5) / 1e5)
            ax.set_title(nList[i], fontsize=18)
            ax.set_xlim(left=0, right=maxUpdates)
    elif args.agentType == 'TD3':
        fig, axes = plt.subplots(1, 2, figsize=(6,3))

        for i in range(2):
            ax = axes[i]
            data = trainRecords[:, i]
            c = cList[i]
            
            ax.plot(data, ':', color=c)
            ax.set_xlabel('Iteration (x 1e5)', fontsize=18)
            ax.set_xticks(np.linspace(0, maxUpdates, 5))
            ax.set_xticklabels(np.linspace(0, maxUpdates, 5) / 1e5)
            ax.set_title(nList[i], fontsize=18)
            ax.set_xlim(left=0, right=maxUpdates)

    fig.tight_layout()
    if storeFigure:
        figurePath = os.path.join(figureFolder, 'training_Loss.png')
        fig.savefig(figurePath)
    if plotFigure:
        plt.show()
        plt.pause(0.001)
    plt.close()

    #= Rollout Reach-Avoid Set
    nx=41
    ny=121

    resultMtx = np.empty((nx, ny), dtype=int)
    xs = np.linspace(env.bounds[0,0], env.bounds[0,1], nx)
    ys = np.linspace(env.bounds[1,0], env.bounds[1,1], ny)

    it = np.nditer(resultMtx, flags=['multi_index'])

    while not it.finished:
        idx = it.multi_index
        print(idx, end='\r')
        x = xs[idx[0]]
        y = ys[idx[1]]

        state = np.array([x, y])
        _, _, result = env.simulate_one_trajectory(agent.actor, T=150, state=state, toEnd=False)
        resultMtx[idx] = result

        it.iternext()

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
    ax = axes[0]
    axStyle = env.get_axes()
    im = ax.imshow(resultMtx.T != 1, interpolation='none', extent=axStyle[0], origin="lower",
                cmap='coolwarm', vmin=0, vmax=1, zorder=-1)
    ax.set_xlabel('Rollout', fontsize=24)

    ax = axes[1]

    xs = np.linspace(env.bounds[0,0], env.bounds[0,1], nx)
    ys = np.linspace(env.bounds[1,0], env.bounds[1,1], ny)
    v = env.get_value(agent.critic.Q1, agent.actor, nx, ny)

    # Plot V
    im = ax.imshow(v.T, interpolation='none', extent=axStyle[0],
                    origin="lower", cmap='seismic', vmin=vmin, vmax=vmax)

    cbar = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.05,
                        shrink=.95, ticks=[vmin, 0, vmax])
    cbar.ax.set_yticklabels(labels=[vmin, 0, vmax], fontsize=24)

    CS = ax.contour(xs, ys, v.T, levels=[0], colors='k', linewidths=2, linestyles='dashed')

    # Plot Trajectories
    env.plot_trajectories(agent.actor, states=env.visual_initial_states, toEnd=False, ax=ax)

    ax.set_xlabel('Value', fontsize=24)

    # Formatting
    for ax in axes:
        env.plot_target_failure_set(ax=ax)
        env.plot_reach_avoid_set(ax)
        env.plot_formatting(ax=ax)

    if storeFigure:
        figurePath = os.path.join(figureFolder, 'rollout.png')
        fig.savefig(figurePath)
    if plotFigure:
        plt.show()
        plt.pause(0.001)
    plt.close()