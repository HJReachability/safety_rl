from warnings import simplefilter 
simplefilter(action='ignore', category=FutureWarning)

from gym_reachability import gym_reachability  # Custom Gym env.
import gym
import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import pickle
import os
import argparse

from KC_DQN.TD3 import TD3
from KC_DQN.config import actorCriticConfig

import time
timestr = time.strftime("%Y-%m-%d-%H_%M")


#== ARGS ==
# test
    # python3 sim_car_one_cont.py -w -wi 5 -mu 200 -ut 2 -cp 100 -arc 100 20 -of scratch/tmp -sf -dt fail -tt g
# default
    # python3 sim_car_one_cont.py -w -sf -of scratch/tmp
parser = argparse.ArgumentParser()

# training scheme
parser.add_argument("-w",   "--warmup",         help="warmup Q-network",            action="store_true")
parser.add_argument("-rnd", "--randomSeed",     help="random seed",                 default=0,          type=int)
parser.add_argument("-mu",  "--maxUpdates",     help="maximal #gradient updates",   default=2400000,    type=int)
parser.add_argument("-mc",  "--memoryCapacity", help="memoryCapacity",              default=1e4,        type=int)
parser.add_argument("-ut",  "--updateTimes",    help="#hyper-param. steps",         default=12,         type=int)
parser.add_argument("-wi",  "--warmupIter",     help="warmup iteration",            default=5000,       type=int)
parser.add_argument("-cp",  "--checkPeriod",    help="check period",                default=200000,     type=int)
parser.add_argument("-dt",  "--doneType",       help="when to raise done flag",     default='fail',     type=str)
parser.add_argument("-tt",  "--terminalType",   help="terminal value",              default='g',        type=str)

# hyper-parameters
parser.add_argument("-arc", "--architecture",   help="NN architecture",         default=[100, 20],          nargs="*", type=int)
parser.add_argument("-act", "--actType",        help="activation type",         default=['Tanh', 'ReLU'],   nargs=2, type=str)
parser.add_argument("-lrA", "--lrA",            help="learning rate actor",     default=1e-3,   type=float)
parser.add_argument("-lrC", "--lrC",            help="learning rate critic",    default=1e-3,   type=float)
parser.add_argument("-g",   "--gamma",          help="contraction coeff.",      default=0.8,    type=float)

# car dynamics
parser.add_argument("-cr",      "--constraintRadius",   help="constraint radius",   default=1., type=float)
parser.add_argument("-tr",      "--targetRadius",       help="target radius",       default=.5, type=float)
parser.add_argument("-turn",    "--turnRadius",         help="turning radius",      default=.6, type=float)
parser.add_argument("-s",       "--speed",              help="speed",               default=.5, type=float)

# file
parser.add_argument("-n",   "--name",           help="extra name",      default='',                         type=str)
parser.add_argument("-of",  "--outFolder",      help="output file",     default='/scratch/gpfs/kaichieh/',  type=str)
parser.add_argument("-pf",  "--plotFigure",     help="plot figures",    action="store_true")
parser.add_argument("-sf",  "--storeFigure",    help="store figures",   action="store_true")

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

outFolder = os.path.join(args.outFolder, 'car-TD3', args.name + timestr)
figureFolder = os.path.join(outFolder, 'figure/')
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
CONFIG = actorCriticConfig(
    ENV_NAME=env_name,
    DEVICE=device,
    MAX_UPDATES=maxUpdates,  # Number of grad updates.
    MAX_EP_STEPS=maxSteps,   # Max number of steps per episode.
    # =================== LEARNING RATE PARAMS.
    LR_C=args.lrC,  # Learning rate.
    LR_C_END=args.lrC/10,           # Final learning rate.
    LR_C_PERIOD=updatePeriod,  # How often to update lr.
    LR_C_DECAY=0.9,          # Learning rate decay rate.
    LR_A=args.lrA,
    LR_A_END=args.lrA/10,
    LR_A_PERIOD=updatePeriod,
    LR_A_DECAY=0.9,
    # =================== LEARNING RATE .
    GAMMA=0.8,# args.gamma,         # Inital gamma
    GAMMA_END=0.9999,    # Final gamma.
    GAMMA_PERIOD=updatePeriod,  # How often to update gamma.
    GAMMA_DECAY=0.5,         # Rate of decay of gamma.
    # ===================
    ALPHA=0.2,
    TAU=0.05,
    HARD_UPDATE=1,
    SOFT_UPDATE=True,
    MEMORY_CAPACITY=args.memoryCapacity,   # Number of transitions in replay buffer.
    BATCH_SIZE=128,          # Number of examples to use to update Q.
    RENDER=False,
    MAX_MODEL=50,            # How many models to store while training.
    # ADDED by vrubies
    ARCHITECTURE=args.architecture,
    ACTIVATION=args.actType,
    SKIP=False,
    REWARD=-1.,
    PENALTY=1.)
# picklePath = outFolder+'/CONFIG.pkl'
# with open(picklePath, 'wb') as handle:
#     pickle.dump(CONFIG, handle, protocol=pickle.HIGHEST_PROTOCOL)


#== AGENT ==
vmin = -1
vmax = 1

dimListActor = [stateDim] + args.architecture + [actionDim]
dimListCritic = [stateDim + actionDim] + args.architecture + [1]
dimLists = [dimListCritic, dimListActor]
agent = TD3(CONFIG, env.action_space, dimLists,
    actType={'critic': args.actType[0], 'actor': args.actType[1]}, verbose=True,
    terminalType=args.terminalType)
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