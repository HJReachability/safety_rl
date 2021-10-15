"""
Please contact the author(s) of this library if you have any questions.
Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )
"""

import torch
import pickle
import gym
import os
import sys

sys.path.append("..")
from RARL.config import dqnConfig
from RARL.DDQNSingle import DDQNSingle

tiffany = "#0abab5"


def loadEnv(args, verbose=True):
  """Constructs the environmnet given the arguments and return it.

  Args:
      args (Namespace): it contains
          - forceCPU (bool): use PyTorch with CPU only
          - low (bool): load environment of low turning rate.
      verbose (bool, optional): print messages if True. Defaults to True.

  """
  print("\n== Environment Information ==")
  env_name = "dubins_car-v1"
  if args.forceCPU:
    device = torch.device("cpu")
  else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  env = gym.make(env_name, device=device, mode="RA", doneType="toEnd")

  if args.low:
    env.set_target(radius=0.4)
    env.set_radius_rotation(R_turn=0.75, verbose=False)
  else:
    env.set_target(radius=0.5)
    env.set_radius_rotation(R_turn=0.6, verbose=False)

  state_dim = env.state.shape[0]
  action_num = env.action_space.n
  print(
      "State Dimension: {:d}, ActionSpace Dimension: {:d}".format(
          state_dim, action_num
      )
  )

  print("Dynamic parameters:")
  print("  CAR")
  print(
      "    Constraint radius: {:.1f}, ".format(env.car.constraint_radius)
      + "Target radius: {:.1f}, ".format(env.car.target_radius)
      + "Turn radius: {:.2f}, ".format(env.car.R_turn)
      + "Maximum speed: {:.2f}, ".format(env.car.speed)
      + "Maximum angular speed: {:.3f}".format(env.car.max_turning_rate)
  )
  print("  ENV")
  print(
      "    Constraint radius: {:.1f}, ".format(env.constraint_radius)
      + "Target radius: {:.1f}, ".format(env.target_radius)
      + "Turn radius: {:.2f}, ".format(env.R_turn)
      + "Maximum speed: {:.2f}".format(env.speed)
  )
  print(env.car.discrete_controls)
  if 2 * env.car.R_turn - env.car.constraint_radius > env.car.target_radius:
    print("Type II Reach-Avoid Set")
  else:
    print("Type I Reach-Avoid Set")
  return env


def loadAgent(args, device, state_dim, action_num, action_list, verbose=True):
  """Constructs the agent with arguments and return it.

  Args:
      args (Namespace): it contains
          - modelFolder (str): the parent folder to get the stored models.
      device (torch.device): the device used by PyTorch.
      state_dim (int): the dimension of the state.
      action_num (int): the number of actions in the action set.
      action_list (list): the action set.
      verbose (bool, optional): print messages if True. Defaults to True.
  """
  print("\n== Agent Information ==")
  modelFolder = os.path.join(args.modelFolder, "model")
  configFile = os.path.join(modelFolder, "CONFIG.pkl")
  with open(configFile, "rb") as handle:
    tmpConfig = pickle.load(handle)
  CONFIG = dqnConfig()
  for key, _ in tmpConfig.__dict__.items():
    CONFIG.__dict__[key] = tmpConfig.__dict__[key]
  CONFIG.DEVICE = device
  CONFIG.SEED = 0

  dimList = [state_dim] + CONFIG.ARCHITECTURE + [action_num]
  agent = DDQNSingle(CONFIG, action_num, action_list, dimList, verbose=verbose)
  agent.restore(400000, args.modelFolder)

  return agent
