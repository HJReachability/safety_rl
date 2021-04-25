#!/usr/bin/env python
# coding: utf-8


from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

from gym_reachability import gym_reachability  # Custom Gym env.
import gym
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation
import torch
from collections import namedtuple
import argparse
from multiprocessing import Pool
import pickle
import os.path
import glob

from KC_DQN.TD3 import TD3
from KC_DQN.config import actorCriticConfig

import time
timestr = time.strftime("%Y-%m-%d-%H_%M_%S")

#== ARGS ==
# e.g., python3 sim_zermelo.py -te -m lagrange -nt 50 -of lagrange_low_50 -ma 1500000 -p .1
# e.g., python3 sim_zermelo.py -te -m RA -nt 50 -of RA -ma 1500000
parser = argparse.ArgumentParser()
parser.add_argument("-nt",  "--num_test",       help="the number of tests",         default=1,      type=int)
parser.add_argument("-nw",  "--num_worker",     help="the number of workers",       default=1,      type=int)
parser.add_argument("-test", "--test",          help="test a neural network",       action="store_true")

# training scheme
parser.add_argument("-te",  "--toEnd",          help="stop until reaching boundary",    action="store_true")
parser.add_argument("-ab",  "--addBias",        help="add bias term for RA",            action="store_true")
parser.add_argument("-ma",  "--maxAccess",      help="maximal number of access",        default=4e6,  type=int)
parser.add_argument("-ms",  "--maxSteps",       help="maximal length of rollouts",      default=100,  type=int)
parser.add_argument("-cp",  "--check_period",   help="check the success ratio",         default=10000,  type=int)
parser.add_argument("-upe",  "--update_period_eps",    help="update period for eps scheduler",     default=int(4e6/20),  type=int)
parser.add_argument("-upg",  "--update_period_gamma",  help="update period for gamma scheduler",   default=int(4e6/20),  type=int)
parser.add_argument("-upl",  "--update_period_lr",     help="update period for lr cheduler",       default=int(4e6/20),  type=int)

# hyper-parameters
parser.add_argument("-r",   "--reward",         help="when entering target set",    default=-1,     type=float)
parser.add_argument("-p",   "--penalty",        help="when entering failure set",   default=1,      type=float)
parser.add_argument("-s",   "--scaling",        help="scaling of ell/g",            default=1,      type=float)
parser.add_argument("-lr",  "--learningRate",   help="learning rate",               default=1e-3,   type=float)
parser.add_argument("-g",   "--gamma",          help="contraction coeff.",          default=0.9,    type=float)
parser.add_argument("-e",   "--eps",            help="exploration coeff.",          default=0.5,   type=float)
parser.add_argument("-arc", "--architecture",   help="neural network architecture", default=[100,20],  nargs="*", type=int)
parser.add_argument("-act", "--activation",     help="activation function",         default='Tanh', type=str)
parser.add_argument("-skp", "--skip",           help="skip connections",            action="store_true")
parser.add_argument("-dbl", "--double",         help="double DQN",                  action="store_true")
parser.add_argument("-bs",  "--batchsize",      help="batch size",                  default=100,    type=int)

# RL type
parser.add_argument("-m",   "--mode",           help="mode",            default='RA',       type=str)
parser.add_argument("-ct",  "--costType",       help="cost type",       default='sparse',   type=str)
parser.add_argument("-of",  "--outFolder",      help="output file",     default='RA' + timestr,       type=str)

args = parser.parse_args()
print(args)

def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(
      plt.gcf(), animate, frames=len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=40)

# == CONFIGURATION ==
env_name = "zermelo_cont-v0"
device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")

reward = -1
penalty = 0.1

CONFIG = actorCriticConfig(
            ENV_NAME=env_name,
            DEVICE=device,
            MAX_UPDATES=args.maxAccess,  # Number of grad updates.
            MAX_EP_STEPS=args.maxSteps,   # Max number of steps per episode.
            # =================== EXPLORATION PARAMS.
            EPSILON=args.eps,            # Initial exploration rate.
            EPS_END=0.05,            # Final explortation rate.
            EPS_PERIOD=args.update_period_gamma / 10,  # How often to update EPS.
            EPS_DECAY=0.8,  # !!!      # Rate of decay.
            EPS_RESET_PERIOD=args.update_period_gamma,
            # =================== LEARNING RATE PARAMS.
            LR_C=args.learningRate,  # Learning rate.
            LR_C_END=args.learningRate,           # Final learning rate.
            LR_C_PERIOD=args.update_period_lr,  # How often to update lr.
            LR_C_DECAY=0.9,          # Learning rate decay rate.
            LR_A=args.learningRate,
            LR_A_END=args.learningRate,
            LR_A_PERIOD=args.update_period_lr,
            LR_A_DECAY=0.9,
            # =================== LEARNING RATE .
            GAMMA=0.999,# args.gamma,         # Inital gamma.
            GAMMA_END=0.999,    # Final gamma.
            GAMMA_PERIOD=args.update_period_gamma,  # How often to update gamma.
            GAMMA_DECAY=0.9,         # Rate of decay of gamma.
            # ===================
            TAU=0.05,
            HARD_UPDATE=1,
            SOFT_UPDATE=True,
            MEMORY_CAPACITY=10000,   # Number of transitions in replay buffer.
            BATCH_SIZE=args.batchsize,          # Number of examples to use to update Q.
            RENDER=False,
            MAX_MODEL=10,            # How many models to store while training.
            # ADDED by vrubies
            ARCHITECTURE=args.architecture,
            ACTIVATION=args.activation,
            SKIP=args.skip,
            REWARD=reward,
            PENALTY=penalty)

# == REPORT ==
def report_config(CONFIG):
    for key, value in CONFIG.__dict__.items():
        if key[:1] != '_': print(key, value)


# == ENVIRONMENT ==
env = gym.make(env_name, device=device, mode="RA")
env.set_costParam(penalty=CONFIG.PENALTY, reward=CONFIG.REWARD)

# == EXPERIMENT ==
def multi_experiment(seedNum, args, CONFIG, env, report_period=1000, skip=False):
    # == AGENT ==
    s_dim = env.observation_space.shape[0]
    numAction = env.action_space.shape[0]
    actionList = np.arange(numAction)

    dimListActor = [s_dim] + args.architecture + [numAction]
    dimListCritic = [s_dim + numAction] + args.architecture + [1]
    dimLists = [dimListCritic, dimListActor]

    env.set_seed(seedNum)
    np.random.seed(seedNum)
    torch.manual_seed(seedNum)

    agent = TD3(CONFIG, env.action_space, dimLists, actType={'critic':'Sin', 'actor':'ReLU'},
                verbose=True)

    # If *true* episode ends when gym environment gives done flag.
    # If *false* end
    # == TRAINING ==
    report_config(CONFIG)
    _, trainProgress = agent.learn(
        env,
        MAX_UPDATES=CONFIG.MAX_UPDATES,  # 6000000 for Dubins
        MAX_EP_STEPS=CONFIG.MAX_EP_STEPS,
        warmupBuffer=True,
        warmupQ=False,  # Need to implement inside env.
        warmupIter=20000,
        addBias=False,  # args.addBias,
        doneTerminate=True,
        runningCostThr=None,
        curUpdates=None,
        # toEnd=args.toEnd,
        # reportPeriod=report_period,  # How often to report Value function figs.
        plotFigure=True,  # Display value function while learning.
        showBool=False,  # Show boolean reach avoid set 0/1.
        vmin=-1,
        vmax=1,
        checkPeriod=args.check_period,  # How often to compute Safe vs. Unsafe.
        storeFigure=True,  # Store the figure in an eps file.
        storeModel=True,
        # randomPlot=True,  # Plot from random starting points.
        outFolder=args.outFolder,
        verbose=True)
    return trainProgress


def test_experiment(args, CONFIG, env, path, doneType='toFailureOrSuccess',
                    sim_only=False):
    s_dim = env.observation_space.shape[0]
    numAction = env.action_space.n
    actionList = np.arange(numAction)
    if ".pth" not in path:
        model_list = glob.glob(os.path.join(path, '*.pth'))
        max_step = max([int(li.split('/')[-1][6:-4]) for li in model_list])
        path += 'model-{}.pth'.format(max_step)

    # If CONFIG.pkl file exists overwrite CONFIG object.
    config_path = path.split(path.split("/")[-1])[0] + "CONFIG.pkl"
    if os.path.isfile(config_path):
        CONFIG_ = pickle.load(open(config_path, 'rb'))
        for k in CONFIG_.__dict__:
            CONFIG.__dict__[k] = CONFIG_.__dict__[k]
        CONFIG.DEVICE = device
    report_config(CONFIG)

    env.doneType = doneType
    env.set_costParam(penalty=CONFIG.PENALTY, reward=CONFIG.REWARD)

    dimList = [s_dim] + CONFIG.ARCHITECTURE + [numAction]
    agent = DDQNSingle(CONFIG, numAction, actionList, dimList,
                       mode='RA', actType=CONFIG.ACTIVATION,
                       skip=CONFIG.SKIP)
    agent.restore(path)

    # Show policy in velocity 0 space.
    env.scatter_actions(agent.Q_network, num_states=20000)
    plt.show()

    if not sim_only:
        # Print confusion matrix.
        print(env.confusion_matrix(agent.Q_network, num_states=100))

        # Visualize value function.
        env.visualize(agent.Q_network, True, nx=91, ny=91, boolPlot=False, trueRAZero=False,
            addBias=False, lvlset=0)
        plt.show()

    my_images = []
    fig, ax = plt.subplots(figsize=(12, 7))
    s_trajs = []
    total_reward = 0
    s = env.reset()
    tmp_int = 0
    tmp_ii = 0
    while True:
        state_tensor = torch.FloatTensor(s).to(device).unsqueeze(0)
        action_index = agent.Q_network(state_tensor).min(dim=1)[1].item()
        s, r, done, info = env.step(action_index)
        s_trajs.append([s[0], s[1]])
        total_reward += r
        tmp_ii += 1

        my_images.append(env.render(mode="rgb_array"))

        if done or tmp_ii > 1000:
          tmp_ii = 0
          env.reset()
          if tmp_int > 20:
            break
          else:
            tmp_int += 1
    env.close()
    # save_frames_as_gif(my_images)

path1 = "models/RA2021-03-15-09_58_46/"
if args.test:
    test_experiment(args, CONFIG, env, path1,
                    doneType='toThreshold', sim_only=False)
else:
    multi_experiment(1, args, CONFIG, env, skip=args.skip)
