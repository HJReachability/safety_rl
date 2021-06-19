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
from KC_DQN.utils import save_obj

import time
timestr = time.strftime("%Y-%m-%d-%H_%M")


#== ARGS ==
# python3 sim_car_one.py -sf -of scratch -w -wi 5000 -g 0.9999 -n 9999
parser = argparse.ArgumentParser()

# environment parameters
parser.add_argument("-dt",  "--doneType",       help="when to raise done flag",
    default='toEnd',    type=str)
parser.add_argument("-ct",  "--costType",       help="cost type",
    default='sparse',   type=str)
parser.add_argument("-rnd", "--randomSeed",     help="random seed",
    default=0,          type=int)

# car dynamics
parser.add_argument("-cr",  "--consRadius",     help="constraint radius",
    default=1., type=float)
parser.add_argument("-tr",  "--targetRadius",   help="target radius",
    default=.5, type=float)
parser.add_argument("-turn","--turnRadius",     help="turning radius",
    default=.6, type=float)
parser.add_argument("-s",   "--speed",          help="speed",
    default=.5, type=float)

# training scheme
parser.add_argument("-w",   "--warmup",         help="warmup Q-network",
    action="store_true")
parser.add_argument("-wi",  "--warmupIter",     help="warmup iteration",
    default=10000,  type=int)
parser.add_argument("-mu",  "--maxUpdates",     help="maximal #gradient updates",
    default=400000, type=int)
parser.add_argument("-ut",  "--updateTimes",    help="#hyper-param. steps",
    default=20,     type=int)
parser.add_argument("-mc",  "--memoryCapacity", help="memoryCapacity",
    default=1e4,    type=int)
parser.add_argument("-cp",  "--checkPeriod",    help="check period",
    default=20000, type=int)

# hyper-parameters
parser.add_argument("-a",   "--annealing",      help="gamma annealing",
    action="store_true")
parser.add_argument("-arc", "--architecture",   help="NN architecture",
    default=[100, 100],  nargs="*", type=int)
parser.add_argument("-lr",  "--learningRate",   help="learning rate",
    default=1e-3,   type=float)
parser.add_argument("-g",   "--gamma",          help="contraction coeff.",
    default=0.8,    type=float)
parser.add_argument("-act", "--actType",        help="activation type",
    default='Tanh', type=str)

# RL type
parser.add_argument("-m",   "--mode",           help="mode",
    default='RA',       type=str)
parser.add_argument("-tt",  "--terminalType",   help="terminal value",
    default='g',        type=str)

# file
parser.add_argument("-st",  "--showTime",       help="show timestr",
    action="store_true")
parser.add_argument("-n",   "--name",           help="extra name",
    default='',                         type=str)
parser.add_argument("-of",  "--outFolder",      help="output file",
    default='/scratch/gpfs/kaichieh/',  type=str)
parser.add_argument("-pf",  "--plotFigure",     help="plot figures",
    action="store_true")
parser.add_argument("-sf",  "--storeFigure",    help="store figures",
    action="store_true")

args = parser.parse_args()
print(args)


#== CONFIGURATION ==
env_name = "dubins_car-v1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
maxUpdates = args.maxUpdates
updateTimes = args.updateTimes
updatePeriod = int(maxUpdates / updateTimes)
updatePeriodHalf = int(updatePeriod/2)
maxSteps = 100

fn = args.name + '-' + args.doneType
if args.showTime:
    fn = fn + '-' + timestr

outFolder = os.path.join(args.outFolder, 'car-DDQN', fn)
print(outFolder)
figureFolder = os.path.join(outFolder, 'figure')
os.makedirs(figureFolder, exist_ok=True)


#== Environment ==
print("\n== Environment Information ==")
if args.doneType == 'toEnd':
    sample_inside_obs=True
elif args.doneType == 'TF' or args.doneType == 'fail':
    sample_inside_obs=False

env = gym.make(env_name, device=device, mode=args.mode, doneType=args.doneType,
    sample_inside_obs=sample_inside_obs)

stateNum = env.state.shape[0]
actionNum = env.action_space.n
action_list = np.arange(actionNum)
print("State Dimension: {:d}, ActionSpace Dimension: {:d}".format(
    stateNum, actionNum))


#== Setting in this Environment ==
env.set_speed(speed=args.speed)
env.set_target(radius=args.targetRadius)
env.set_constraint(radius=args.consRadius)
env.set_radius_rotation(R_turn=args.turnRadius)
print("Dynamic parameters:")
print("  CAR")
print("    Constraint radius: {:.1f}, Target radius: {:.1f}, Turn radius: {:.2f}, Maximum speed: {:.2f}, Maximum angular speed: {:.2f}".format(
    env.car.constraint_radius, env.car.target_radius, env.car.R_turn, env.car.speed, env.car.max_turning_rate))
print("  ENV")
print("    Constraint radius: {:.1f}, Target radius: {:.1f}, Turn radius: {:.2f}, Maximum speed: {:.2f}".format(
    env.constraint_radius, env.target_radius, env.R_turn, env.speed))
print(env.car.discrete_controls)
if 2*env.R_turn-env.constraint_radius > env.target_radius:
    print("Type II Reach-Avoid Set")
else:
    print("Type I Reach-Avoid Set")
env.set_seed(args.randomSeed)


#== Get and Plot max{l_x, g_x} ==
if args.plotFigure or args.storeFigure:
    nx, ny = 101, 101
    vmin = -1
    vmax = 1

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
        origin="lower", cmap="seismic", vmin=vmin, vmax=vmax, zorder=-1)
    cbar = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.05, shrink=.95,
        ticks=[vmin, 0, vmax])
    cbar.ax.set_yticklabels(labels=[vmin, 0, vmax], fontsize=24)
    ax.set_title(r'$\ell(x)$', fontsize=18)

    ax = axes[1]
    im = ax.imshow(g_x.T, interpolation='none', extent=axStyle[0],
        origin="lower", cmap="seismic", vmin=vmin, vmax=vmax, zorder=-1)
    cbar = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.05, shrink=.95,
        ticks=[vmin, 0, vmax])
    cbar.ax.set_yticklabels(labels=[vmin, 0, vmax], fontsize=24)
    ax.set_title(r'$g(x)$', fontsize=18)

    ax = axes[2]
    im = ax.imshow(v.T, interpolation='none', extent=axStyle[0],
        origin="lower", cmap="seismic", vmin=vmin, vmax=vmax, zorder=-1)
    env.plot_reach_avoid_set(ax)
    cbar = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.05, shrink=.95,
        ticks=[vmin, 0, vmax])
    cbar.ax.set_yticklabels(labels=[vmin, 0, vmax], fontsize=24)
    ax.set_title(r'$v(x)$', fontsize=18)

    for ax in axes:
        env.plot_target_failure_set(ax=ax)
        env.plot_formatting(ax=ax)

    fig.tight_layout()
    if args.storeFigure:
        figurePath = os.path.join(figureFolder, 'env.png')
        fig.savefig(figurePath)
    if args.plotFigure:
        plt.show()
        plt.pause(0.001)
    plt.close()


#== Agent CONFIG ==
print("\n== Agent Information ==")
if args.annealing:
    GAMMA_END = 0.9999
    EPS_PERIOD = int(updatePeriod/10)
    EPS_RESET_PERIOD = updatePeriod
else:
    GAMMA_END = args.gamma
    EPS_PERIOD = updatePeriod
    EPS_RESET_PERIOD = maxUpdates

CONFIG = dqnConfig(DEVICE=device, ENV_NAME=env_name, SEED=args.randomSeed,
    MAX_UPDATES=maxUpdates, MAX_EP_STEPS=maxSteps,
    BATCH_SIZE=64, MEMORY_CAPACITY=args.memoryCapacity,
    ARCHITECTURE=args.architecture, ACTIVATION=args.actType,
    GAMMA=args.gamma, GAMMA_PERIOD=updatePeriod, GAMMA_END=GAMMA_END,
    EPS_PERIOD=EPS_PERIOD, EPS_DECAY=0.7, EPS_RESET_PERIOD=EPS_RESET_PERIOD,
    LR_C=args.learningRate, LR_C_PERIOD=updatePeriod, LR_C_DECAY=0.8,
    MAX_MODEL=50)
print(CONFIG.EPS_PERIOD, CONFIG.EPS_RESET_PERIOD)
picklePath = outFolder+'/CONFIG.pkl'
with open(picklePath, 'wb') as handle:
    pickle.dump(CONFIG, handle, protocol=pickle.HIGHEST_PROTOCOL)


#== AGENT ==
dimList = [stateNum] + CONFIG.ARCHITECTURE + [actionNum]
agent = DDQNSingle(CONFIG, actionNum, action_list, dimList=dimList,
    mode=args.mode, terminalType=args.terminalType)
print("We want to use: {}, and Agent uses: {}".format(device, agent.device))
print("Critic is using cuda: ", next(agent.Q_network.parameters()).is_cuda)

vmin = -1
vmax = 1
if args.warmup:
    print("\n== Warmup Q ==")
    lossList = agent.initQ(env, args.warmupIter, outFolder,
        num_warmup_samples=200, vmin=vmin, vmax=vmax,
        plotFigure=args.plotFigure, storeFigure=args.storeFigure)

    if args.plotFigure or args.storeFigure:
        fig, ax = plt.subplots(1,1, figsize=(4, 4))
        tmp = np.arange(500, args.warmupIter)
        # tmp = np.arange(args.warmupIter)
        ax.plot(tmp, lossList[tmp], 'b-')
        ax.set_xlabel('Iteration', fontsize=18)
        ax.set_ylabel('Loss', fontsize=18)
        plt.tight_layout()

        if args.storeFigure:
            figurePath = os.path.join(figureFolder, 'initQ_Loss.png')
            fig.savefig(figurePath)
        if args.plotFigure:
            plt.show()
            plt.pause(0.001)
        plt.close()


print("\n== Training Information ==")
vmin = -1
vmax = 1
trainRecords, trainProgress = agent.learn(env,
    MAX_UPDATES=maxUpdates, MAX_EP_STEPS=maxSteps,
    warmupQ=False, doneTerminate=True,
    vmin=vmin, vmax=vmax, showBool=False,
    checkPeriod=args.checkPeriod, outFolder=outFolder,
    plotFigure=args.plotFigure, storeFigure=args.storeFigure)

trainDict = {}
trainDict['trainRecords'] = trainRecords
trainDict['trainProgress'] = trainProgress
filePath = os.path.join(outFolder, 'train')

if args.plotFigure or args.storeFigure:
    # region: loss
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    data = trainRecords
    ax = axes[0]
    ax.plot(data, 'b:')
    ax.set_xlabel('Iteration (x 1e5)', fontsize=18)
    ax.set_xticks(np.linspace(0, maxUpdates, 5))
    ax.set_xticklabels(np.linspace(0, maxUpdates, 5) / 1e5)
    ax.set_title('loss_critic', fontsize=18)
    ax.set_xlim(left=0, right=maxUpdates)

    data = trainProgress[:, 0]
    ax = axes[1]
    x = np.arange(data.shape[0]) + 1
    ax.plot(x, data, 'b-o')
    ax.set_xlabel('Index', fontsize=18)
    ax.set_xticks(x)
    # ax.set_xticklabels(np.arange(data.shape[0]) + 1)
    ax.set_title('Success Rate', fontsize=18)
    ax.set_xlim(left=1, right=data.shape[0])
    ax.set_ylim(0, 0.8)

    fig.tight_layout()
    if args.storeFigure:
        figurePath = os.path.join(figureFolder, 'train_loss_success.png')
        fig.savefig(figurePath)
    if args.plotFigure:
        plt.show()
        plt.pause(0.001)
    plt.close()
    # endregion

    # region: value_rollout_action
    idx = np.argmax(trainProgress[:, 0]) + 1
    successRate = np.amax(trainProgress[:, 0]) 
    print('We pick model with success rate-{:.3f}'.format(successRate))
    agent.restore(idx*args.checkPeriod, outFolder)

    nx=101
    ny=nx
    xs = np.linspace(env.bounds[0,0], env.bounds[0,1], nx)
    ys = np.linspace(env.bounds[1,0], env.bounds[1,1], ny)

    resultMtx  = np.empty((nx, ny), dtype=int)
    actDistMtx = np.empty((nx, ny), dtype=int)
    it = np.nditer(resultMtx, flags=['multi_index'])

    while not it.finished:
        idx = it.multi_index
        print(idx, end='\r')
        x = xs[idx[0]]
        y = ys[idx[1]]

        state = np.array([x, y, 0.])
        stateTensor = torch.FloatTensor(state).unsqueeze(0)
        action_index = agent.Q_network(stateTensor).min(dim=1)[1].item()
        # u = env.discrete_controls[action_index]
        actDistMtx[idx] = action_index

        _, result, _, _ = env.simulate_one_trajectory(agent.Q_network, T=250, state=state, toEnd=False)
        resultMtx[idx] = result
        it.iternext()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
    axStyle = env.get_axes()

    #= Action
    ax = axes[2]
    im = ax.imshow(actDistMtx.T, interpolation='none', extent=axStyle[0],
        origin="lower", cmap='seismic', vmin=0, vmax=actionNum-1, zorder=-1)
    ax.set_xlabel('Action', fontsize=24)

    #= Rollout
    ax = axes[1]
    im = ax.imshow(resultMtx.T != 1, interpolation='none', extent=axStyle[0],
        origin="lower", cmap='seismic', vmin=0, vmax=1, zorder=-1)
    env.plot_trajectories(agent.Q_network, states=env.visual_initial_states,
        toEnd=False, ax=ax, c='w', lw=1.5, T=100)
    ax.set_xlabel('Rollout RA', fontsize=24)

    #= Value
    ax = axes[0]
    v = env.get_value(agent.Q_network, theta=0, nx=nx, ny=ny)
    im = ax.imshow(v.T, interpolation='none', extent=axStyle[0],
        origin="lower", cmap='seismic', vmin=vmin, vmax=vmax, zorder=-1)
    CS = ax.contour(xs, ys, v.T, levels=[0], colors='k', linewidths=2,
        linestyles='dashed')
    ax.set_xlabel('Value', fontsize=24)

    for ax in axes:
        env.plot_target_failure_set(ax=ax)
        env.plot_reach_avoid_set(ax=ax)
        env.plot_formatting(ax=ax)
        
    fig.tight_layout()
    if args.storeFigure:
        figurePath = os.path.join(figureFolder, 'value_rollout_action.png')
        fig.savefig(figurePath)
    if args.plotFigure:
        plt.show()
        plt.pause(0.001)
    # endregion

    trainDict['resultMtx'] = resultMtx
    trainDict['actDistMtx'] = actDistMtx

save_obj(trainDict, filePath)