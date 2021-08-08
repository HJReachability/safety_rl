# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )

import torch
import pickle
from gym_reachability import gym_reachability  # Custom Gym env.
import gym
import os

from RARL.config import dqnConfig
from RARL.DDQNSingle import DDQNSingle

tiffany = '#0abab5'

def loadEnv(args, verbose=True):
    print("\n== Environment Information ==")
    env_name = "dubins_car-v1"
    if args.forceCPU:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make(env_name, device=device, mode='RA', doneType='toEnd')

    if args.low:
        env.set_target(radius=.4)
        env.set_radius_rotation(R_turn=.75, verbose=False)
    else:
        env.set_target(radius=.5)
        env.set_radius_rotation(R_turn=.6, verbose=False)

    stateNum = env.state.shape[0]
    actionNum = env.action_space.n
    print("State Dimension: {:d}, ActionSpace Dimension: {:d}".format(stateNum, actionNum))

    print("Dynamic parameters:")
    print("  CAR")
    print("    Constraint radius: {:.1f}, Target radius: {:.1f}, Turn radius: {:.2f}, Maximum speed: {:.2f}, Maximum angular speed: {:.3f}".format(
        env.car.constraint_radius, env.car.target_radius, env.car.R_turn, env.car.speed, env.car.max_turning_rate))
    print("  ENV")
    print("    Constraint radius: {:.1f}, Target radius: {:.1f}, Turn radius: {:.2f}, Maximum speed: {:.2f}".format(
        env.constraint_radius, env.target_radius, env.R_turn, env.speed))
    print(env.car.discrete_controls)
    if 2*env.car.R_turn-env.car.constraint_radius > env.car.target_radius:
        print("Type II Reach-Avoid Set")
    else:
        print("Type I Reach-Avoid Set")
    return env


def loadAgent(args, device, stateNum, actionNum, actionList, verbose=True):
    print("\n== Agent Information ==")
    modelFolder = os.path.join(args.modelFolder, 'model')
    configFile = os.path.join(modelFolder, 'CONFIG.pkl')
    with open(configFile, 'rb') as handle:
        tmpConfig = pickle.load(handle)
    CONFIG = dqnConfig()
    for key, _ in tmpConfig.__dict__.items():
        CONFIG.__dict__[key] = tmpConfig.__dict__[key]
    CONFIG.DEVICE = device
    CONFIG.SEED = 0

    dimList = [stateNum] + CONFIG.ARCHITECTURE + [actionNum]
    agent = DDQNSingle(CONFIG, actionNum, actionList, dimList, verbose=verbose)
    agent.restore(400000, args.modelFolder)

    return agent