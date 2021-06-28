from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

from gym_reachability import gym_reachability  # Custom Gym env.
import gym
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import os
import argparse

from KC_DQN.TD3 import TD3
from KC_DQN.config import actorCriticConfig

import time
timestr = time.strftime("%Y-%m-%d-%H_%M")


#== ARGS ==
# test
    # python3 sim_car_one_TD3.py -w -wi 5 -mu 200 -cp 100 -arc 20 -of scratch -sf -n tmp
# default
    # python3 sim_car_one_TD3.py -sf -of scratch -g 0.9999 -n 9999
parser = argparse.ArgumentParser()

# training scheme
parser.add_argument("-w",   "--warmup",         help="warmup Q-network",
    action="store_true")
parser.add_argument("-rnd", "--randomSeed",     help="random seed",
    default=0,          type=int)
parser.add_argument("-mu",  "--maxUpdates",     help="maximal #gradient updates",
    default=400000,     type=int)
parser.add_argument("-mc",  "--memoryCapacity", help="memoryCapacity",
    default=10000,      type=int)
parser.add_argument("-ut",  "--updateTimes",    help="#hyper-param. steps",
    default=10,         type=int)
parser.add_argument("-wi",  "--warmupIter",     help="warmup iteration",
    default=5000,       type=int)
parser.add_argument("-cp",  "--checkPeriod",    help="check period",
    default=20000,      type=int)
parser.add_argument("-dt",  "--doneType",       help="when to raise done flag",
    default='fail',     type=str)
parser.add_argument("-tt",  "--terminalType",   help="terminal value",
    default='g',        type=str)

# hyper-parameters
parser.add_argument("-a",   "--annealing",      help="gamma annealing",
    action="store_true")
parser.add_argument("-arc", "--architecture",   help="NN architecture",
    default=[100, 20],  nargs="*",  type=int)
parser.add_argument("-act", "--actType",        help="activation type",
    default=['Tanh', 'ReLU'],   nargs=2,    type=str)
parser.add_argument("-lrA", "--lrA",            help="learning rate actor",
    default=1e-3,   type=float)
parser.add_argument("-lrC", "--lrC",            help="learning rate critic",
    default=1e-3,   type=float)
parser.add_argument("-g",   "--gamma",          help="contraction coeff.",
    default=0.99,   type=float)

# car dynamics
parser.add_argument("-cr",      "--constraintRadius",   help="constraint radius",
    default=1., type=float)
parser.add_argument("-tr",      "--targetRadius",       help="target radius",
    default=.5, type=float)
parser.add_argument("-turn",    "--turnRadius",         help="turning radius",
    default=.6, type=float)
parser.add_argument("-s",       "--speed",              help="speed",
    default=.5, type=float)

# Lagrange RL
parser.add_argument("-r",   "--reward",         help="when entering target set",
    default=-1, type=float)
parser.add_argument("-p",   "--penalty",        help="when entering failure set",
    default=1,  type=float)
parser.add_argument("-sc",  "--scaling",        help="scaling of ell/g",
    default=4,  type=float)

# file
parser.add_argument("-st",  "--showTime",       help="show timestr",
    action="store_true")
parser.add_argument("-n",   "--name",           help="extra name",
    default='', type=str)
parser.add_argument("-of",  "--outFolder",      help="output file",
    default='/scratch/gpfs/kaichieh/',  type=str)
parser.add_argument("-pf",  "--plotFigure",     help="plot figures",
    action="store_true")
parser.add_argument("-sf",  "--storeFigure",    help="store figures",
    action="store_true")

args = parser.parse_args()
print(args)


#== CONFIGURATION ==
env_name = "dubins_car_cont-v0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
maxUpdates = args.maxUpdates
updateTimes = args.updateTimes
updatePeriod = int(maxUpdates / updateTimes)
maxSteps = 100
storeFigure = args.storeFigure
plotFigure = args.plotFigure

fn = args.name + '-' + args.doneType
if args.showTime:
    fn = fn + '-' + timestr

outFolder = os.path.join(args.outFolder, 'car-TD3', fn)
print(outFolder)
figureFolder = os.path.join(outFolder, 'figure')
os.makedirs(figureFolder, exist_ok=True)


#== Environment ==
print("\n== Environment Information ==")
env = gym.make(env_name, device=device, mode='RA', doneType=args.doneType)

stateDim = env.state.shape[0]
actionDim = env.action_space.shape[0]
print("State Dimension: {:d}, Action Dimension: {:d}".format(
    stateDim, actionDim))


#== Setting in this Environment ==
env.set_speed(speed=args.speed)
env.set_target(radius=args.targetRadius)
env.set_constraint(radius=args.constraintRadius)
env.set_radius_rotation(R_turn=args.turnRadius)
print("Dynamic parameters:")
print("  CAR")
print("    Constraint radius: {:.1f}, Target radius: {:.1f}, Turn radius: {:.2f}, Maximum speed: {:.2f}, Maximum angular speed: {:.2f}".format(
    env.car.constraint_radius, env.car.target_radius, env.car.R_turn, env.car.speed, env.car.max_turning_rate[0]))
print("  ENV")
print("    Constraint radius: {:.1f}, Target radius: {:.1f}, Turn radius: {:.2f}, Maximum speed: {:.2f}".format(
    env.constraint_radius, env.target_radius, env.R_turn, env.speed))

print(env.car.action_space)
if 2*env.R_turn-env.constraint_radius > env.target_radius:
    print("Type II Reach-Avoid Set")
else:
    print("Type I Reach-Avoid Set")

env.set_seed(args.randomSeed)
print('env seed:{}, car seed: {}'.format(env.seed_val, env.car.seed_val))


#== Get and Plot max{l_x, g_x} ==
if plotFigure or storeFigure:
    nx, ny = 101, 101
    theta, thetaPursuer = 0., 0.
    v = np.zeros((nx, ny))
    l_x = np.zeros((nx, ny))
    g_x = np.zeros((nx, ny))
    xs = np.linspace(env.bounds[0,0], env.bounds[0,1], nx)
    ys =np.linspace(env.bounds[1,0], env.bounds[1,1], ny)

    it = np.nditer(l_x, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        x = xs[idx[0]]
        y = ys[idx[1]]

        l_x[idx] = env.target_margin(np.array([x, y]))
        g_x[idx] = env.safety_margin(np.array([x, y]))

        v[idx] = np.maximum(l_x[idx], g_x[idx])
        it.iternext()

    axStyle = env.get_axes()
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))

    f = ax.imshow(v.T, interpolation='none', extent=axStyle[0],
        origin="lower", cmap="seismic", vmin=-1, vmax=1, zorder=-1)
    ax.axis(axStyle[0])
    ax.grid(False)
    ax.set_aspect(axStyle[1])  # makes equal aspect ratio
    env.plot_target_failure_set(ax)
    env.plot_reach_avoid_set(ax, orientation=0)
    fig.colorbar(f, ax=ax, pad=0.01, fraction=0.1, shrink=.9, ticks=[-1, 0, 1])
    plt.tight_layout()

    if storeFigure:
        figurePath = os.path.join(figureFolder, 'env.png')
        fig.savefig(figurePath)
    if plotFigure:
        plt.show()
        plt.pause(0.001)
    plt.close()


#== Agent CONFIG ==
print("\n== Agent Information ==")
if args.annealing:
    GAMMA_END = 0.9999
else:
    GAMMA_END = args.gamma

actType={'critic': args.actType[0], 'actor': args.actType[1]}
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
    GAMMA_END=GAMMA_END,
    GAMMA_PERIOD=updatePeriod,
    GAMMA_DECAY=0.5,
    # ===================
    MEMORY_CAPACITY=args.memoryCapacity,
    ARCHITECTURE=args.architecture,
    ACTIVATION=actType,
    REWARD=args.reward,
    PENALTY=args.penalty)


#== AGENT ==
vmin = -1
vmax = 1

dimListActor = [stateDim] + args.architecture + [actionDim]
dimListCritic = [stateDim + actionDim] + args.architecture + [1]
dimLists = [dimListCritic, dimListActor]
agent = TD3(CONFIG, env.action_space, dimLists, terminalType=args.terminalType)
print('Agent has terminal type:', agent.terminalType)
print("We want to use: {}, and Agent uses: {}".format(device, agent.device))
print("Critic is using cuda: ", next(agent.critic.parameters()).is_cuda)
print("Actor is using cuda: ", next(agent.actor.parameters()).is_cuda)

if args.warmup:
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
    warmupQ=False, warmupIter=args.warmupIter, doneTerminate=True,
    vmin=vmin, vmax=vmax, checkPeriod=args.checkPeriod, outFolder=outFolder,
    plotFigure=plotFigure, storeFigure=storeFigure, saveBest=False)


if plotFigure or storeFigure:
    #region: loss
    cList = ['b', 'r']
    fig, axes = plt.subplots(1, 2, figsize=(6,3))

    for i in range(2):
        ax = axes[i]
        data = trainRecords[:, i]
        c = cList[i]

        ax.plot(data, '-', color=c)
        ax.set_xlabel('Iteration', fontsize=18)
        ax.set_ylabel('Loss', fontsize=18)

    if storeFigure:
        figurePath = os.path.join(figureFolder, 'training_Loss.png')
        fig.savefig(figurePath)
    if plotFigure:
        plt.show()
        plt.pause(0.001)
    plt.close()
    #endregion

    #region: Rollout Reach-Avoid Set
    nx=101
    ny=101
    orientation = 0.

    resultMtx = np.empty((nx, ny), dtype=int)
    actionMtx = np.empty((nx, ny), dtype=float)
    xs = np.linspace(env.bounds[0,0], env.bounds[0,1], nx)
    ys = np.linspace(env.bounds[1,0], env.bounds[1,1], ny)
    it = np.nditer(resultMtx, flags=['multi_index'])

    while not it.finished:
        idx = it.multi_index
        print(idx, end='\r')
        x = xs[idx[0]]
        y = ys[idx[1]]

        state = np.array([x, y, orientation])
        _, result, _, _ = env.simulate_one_trajectory(agent.actor, T=100,
                            state=state, toEnd=False)
        state_tensor = torch.FloatTensor(state).to(agent.device)
        u = agent.actor(state_tensor).detach().cpu().numpy()[0]
        resultMtx[idx] = result
        actionMtx[idx] = u
        it.iternext()

    fig2, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)

    #= Rollout
    ax = axes[0]
    axStyle = env.get_axes()
    im = ax.imshow(resultMtx.T != 1, interpolation='none', extent=axStyle[0],
        origin="lower", cmap='coolwarm', vmin=0, vmax=1, zorder=-1)
    env.plot_trajectories(agent.actor, states=env.visual_initial_states,
        toEnd=False, ax=ax, c='w', lw=1.5, T=100, orientation=-np.pi/2)
    ax.set_xlabel('Rollout', fontsize=24)

    #= Value
    ax = axes[1]
    v = env.get_value(agent.critic.Q1, agent.actor, orientation, nx, ny)
    # Plot V
    im = ax.imshow(v.T, interpolation='none', extent=axStyle[0],
        origin="lower", cmap='seismic', vmin=vmin, vmax=vmax, zorder=-1)
    CS = ax.contour(xs, ys, v.T, levels=[0], colors='k', linewidths=2,
        linestyles='dashed')
    ax.set_xlabel('Value', fontsize=24)

    #= Action
    ax = axes[2]
    omega = env.car.max_turning_rate[0]
    vticks = [-omega, 0, omega]
    vticklabels = [ '{:.2f}'.format(x) for x in vticks]
    im = ax.imshow(actionMtx.T, interpolation='none', extent=axStyle[0],
        origin="lower", cmap='seismic', vmin=-omega, vmax=omega, zorder=-1)
    cbar = fig2.colorbar(im, ax=ax, pad=0.02, fraction=0.05, shrink=.8, ticks=vticks)
    cbar.ax.set_yticklabels(labels=vticklabels, fontsize=16)
    ax.set_xlabel('Action', fontsize=24)

    # Formatting
    for ax in axes:
        env.plot_target_failure_set(ax=ax)
        env.plot_reach_avoid_set(ax)
        env.plot_formatting(ax=ax)
    fig.tight_layout()

    if storeFigure:
        figurePath = os.path.join(figureFolder, 'value_rollout_action.png')
        fig2.savefig(figurePath)
    if plotFigure:
        plt.show()
        plt.pause(0.001)
    plt.close()
    #endregion