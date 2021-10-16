"""
Please contact the author(s) of this library if you have any questions.
Authors: Vicenc Rubies-Royo ( vrubies@berkeley.edu )

This experiment runs double deep Q-network with the discounted reach-avoid
Bellman equation (DRABE) proposed in [RSS21] on a 6-dimensional Lunar Lander
problem. We use this script to generate Fig. 1 in the paper.

Examples:
  python3 sim_lunar_lander.py -sf
"""

from warnings import simplefilter
import time
import os.path
import pickle
import argparse

import gym
import numpy as np
import matplotlib.pyplot as plt
import torch

from RARL.DDQNSingle import DDQNSingle
from RARL.config import dqnConfig
from gym_reachability import gym_reachability  # Custom Gym env.

timestr = time.strftime("%Y-%m-%d-%H_%M_%S")
simplefilter(action='ignore', category=FutureWarning)

# == ARGS ==
parser = argparse.ArgumentParser()

parser.add_argument(
    "-nt", "--num_test", help="the number of tests", default=1, type=int
)
parser.add_argument(
    "-nw", "--num_worker", help="the number of workers", default=1, type=int
)
parser.add_argument(
    "-test", "--test", help="test a neural network", action="store_true"
)
parser.add_argument(
    "-rnd", "--randomSeed", help="random seed", default=0, type=int
)
parser.add_argument(
    "-dt", "--doneType", help="when to raise done flag", default='toThreshold',
    type=str
)
parser.add_argument(
    "-p", "--path", help="path to model", default='toThreshold', type=str
)
parser.add_argument(
    "-cfg", "--config_path", help="path to CONFIG.pkl file",
    default='toThreshold', type=str
)

# training scheme
parser.add_argument(
    "-w", "--warmup", help="warmup Q-network", action="store_true"
)
parser.add_argument(
    "-wi", "--warmupIter", help="warmup iteration", default=10000, type=int
)
parser.add_argument(
    "-ab", "--addBias", help="add bias term for RA", action="store_true"
)
parser.add_argument(
    "-mu", "--maxUpdates", help="maximal #gradient updates", default=5e6,
    type=int
)
parser.add_argument(
    "-ms", "--maxSteps", help="maximal length of rollouts", default=200,
    type=int
)
parser.add_argument(
    "-mc", "--memoryCapacity", help="memoryCapacity", default=150000, type=int
)
parser.add_argument(
    "-cp", "--checkPeriod", help="check the success ratio", default=50000,
    type=int
)
parser.add_argument(
    "-upe", "--update_period_eps", help="update period for eps scheduler",
    default=500000, type=int
)
parser.add_argument(
    "-upg", "--update_period_gamma", help="update period for gamma scheduler",
    default=500000, type=int
)
parser.add_argument(
    "-upl", "--update_period_lr", help="update period for lr cheduler",
    default=250000, type=int
)

# hyper-parameters
parser.add_argument(
    "-lr", "--learningRate", help="learning rate", default=1e-4, type=float
)
parser.add_argument(
    "-g", "--gamma", help="contraction coeff.", default=0.9, type=float
)
parser.add_argument(
    "-e", "--eps", help="exploration coeff.", default=0.5, type=float
)
parser.add_argument(
    "-arc", "--architecture", help="neural network architecture",
    default=[512, 512], nargs="*", type=int
)
parser.add_argument(
    "-act", "--actType", help="activation type", default='ReLU', type=str
)
parser.add_argument("-dbl", "--double", help="double DQN", action="store_true")
parser.add_argument(
    "-bs", "--batchsize", help="batch size", default=100, type=int
)

# RL type
parser.add_argument("-m", "--mode", help="mode", default='RA', type=str)

# file
parser.add_argument(
    "-st", "--showTime", help="show timestr", action="store_true"
)
parser.add_argument("-n", "--name", help="extra name", default='', type=str)
parser.add_argument(
    "-of", "--outFolder", help="output file",
    default='experiments/LunarLander' + timestr, type=str
)
parser.add_argument(
    "-pf", "--plotFigure", help="plot figures", action="store_true"
)
parser.add_argument(
    "-sf", "--storeFigure", help="store figures", action="store_true"
)

args = parser.parse_args()
print(args)

# == CONFIGURATION ==
env_name = "one_player_reach_avoid_lunar_lander-v0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

fn = args.name + '-' + args.doneType
if args.showTime:
  fn = fn + '-' + timestr

outFolder = os.path.join(args.outFolder, 'LunarLander-DDQN', fn)
print(outFolder)

CONFIG = dqnConfig(
    ENV_NAME=env_name,
    DEVICE=device,
    SEED=args.randomSeed,
    MAX_UPDATES=args.maxUpdates,  # Number of grad updates.
    MAX_EP_STEPS=args.maxSteps,  # Max number of steps per episode.
    # =================== EXPLORATION PARAMS.
    EPSILON=args.eps,  # Initial exploration rate.
    EPS_END=0.05,  # Final explortation rate.
    EPS_PERIOD=args.update_period_eps / 10,  # How often to update EPS.
    EPS_DECAY=0.8,  # Rate of decay.
    EPS_RESET_PERIOD=args.update_period_eps,
    # =================== LEARNING RATE PARAMS.
    LR_C=args.learningRate,  # Learning rate.
    LR_C_END=args.learningRate * 0.5,  # Final learning rate.
    LR_C_PERIOD=args.update_period_lr,  # How often to update lr.
    LR_C_DECAY=0.8,  # Learning rate decay rate.
    # =================== LEARNING RATE .
    GAMMA=args.gamma,  # Inital gamma.
    GAMMA_END=0.999,  # Final gamma.
    GAMMA_PERIOD=args.update_period_gamma,  # How often to update gamma.
    GAMMA_DECAY=0.1,  # Rate of decay of gamma.
    # ===================
    TAU=0.01,
    HARD_UPDATE=1,
    SOFT_UPDATE=True,
    MEMORY_CAPACITY=args.memoryCapacity,
    BATCH_SIZE=args.batchsize,  # Number of examples to use to update Q.
    RENDER=False,
    MAX_MODEL=10,  # How many models to store while training.
    DOUBLE=args.double,
    ARCHITECTURE=args.architecture,
    ACTIVATION=args.actType,
    REWARD=-1,
    PENALTY=0.1
)


# == REPORT ==
def report_config(CONFIG):
  for key, value in CONFIG.__dict__.items():
    if key[:1] != '_':
      print(key, value)


# == ENVIRONMENT ==
env = gym.make(
    env_name, device=device, mode="RA", doneType='toThreshold',
    obstacle_sampling=True
)
env.set_costParam(penalty=CONFIG.PENALTY, reward=CONFIG.REWARD)


# == EXPERIMENT ==
def run_experiment(args, CONFIG, env):
  """Run the reach-avoid training algorithm.

  Args:
      args: parsed arguments.
      CONFIG (config, object): configuration parameters.
      env (gym.Env): environment used for training.
  """
  # == AGENT ==
  s_dim = env.observation_space.shape[0]
  numAction = env.action_space.n
  actionList = np.arange(numAction)
  dimList = [s_dim] + args.architecture + [numAction]

  env.set_seed(args.randomSeed)
  np.random.seed(args.randomSeed)
  agent = DDQNSingle(CONFIG, numAction, actionList, dimList, mode='RA')

  # == TRAINING ==
  report_config(CONFIG)
  _, trainProgress = agent.learn(
      env,
      MAX_UPDATES=CONFIG.MAX_UPDATES,
      MAX_EP_STEPS=CONFIG.MAX_EP_STEPS,
      warmupBuffer=True,
      warmupQ=args.warmup,
      warmupIter=args.warmupIter,
      addBias=args.addBias,
      doneTerminate=True,
      runningCostThr=None,
      curUpdates=None,
      plotFigure=args.plotFigure,  # Display value function while learning.
      showBool=False,  # Show boolean reach avoid set 0/1.
      vmin=-1,
      vmax=1,
      checkPeriod=args.checkPeriod,  # How often to compute Safe vs. Unsafe.
      storeFigure=args.storeFigure,  # Store the figure in an eps file.
      storeModel=True,
      storeBest=False,
      outFolder=outFolder,
      verbose=True
  )
  return trainProgress


def test_experiment(path, config_path, env, doneType='toFailureOrSuccess'):
  """Plot the value function slices.

  Args:
      path (string): path to the model file *.pth.
      config_path (string): path to the CONFIG.pkl file of the experiment.
      env (gym.Env): environment used for training.
      doneType (string, optional): termination type for episodes.
  """
  s_dim = env.observation_space.shape[0]
  numAction = env.action_space.n
  actionList = np.arange(numAction)

  if os.path.isfile(config_path):
    CONFIG_ = pickle.load(open(config_path, 'rb'))
    for k in CONFIG_.__dict__:
      CONFIG.__dict__[k] = CONFIG_.__dict__[k]
    CONFIG.DEVICE = device
  report_config(CONFIG)

  env.doneType = doneType
  env.set_costParam(penalty=CONFIG.PENALTY, reward=CONFIG.REWARD)

  dimList = [s_dim] + CONFIG.ARCHITECTURE + [numAction]
  agent = DDQNSingle(
      CONFIG, numAction, actionList, dimList, mode='RA',
      actType=CONFIG.ACTIVATION
  )
  agent.restore(path)

  # Visualize value function.
  env.visualize(
      agent.Q_network, True, nx=91, ny=91, boolPlot=False, trueRAZero=False,
      addBias=False, lvlset=0
  )
  plt.show()


if args.test:
  test_experiment(args.path, args.config_path, env)
else:
  run_experiment(args, CONFIG, env)
