from warnings import simplefilter 
simplefilter(action='ignore', category=FutureWarning)

from gym_reachability import gym_reachability  # Custom Gym env.
import gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
from collections import namedtuple
import os
import time
import pickle
from multiprocessing import Pool

from KC_DQN.DDQNSingle import DDQNSingle
from KC_DQN.config import dqnConfig

import argparse

# EXECUTION TIME
# Basic Setting:
#   - 101 samples per dimension, 6 workers, maxLength = 100
#   - NN: 1-layer with 100 neurons per leayer
# Results
#   - 4000 seconds (toEnd = True)
#   - 1351 seconds (toEnd = False)

# EXAMPLES
# toEnd, low turning rate: 
#   python3 sim_approx_error_single.py -te -l -of carOneLow -mf models/store_best/car/RA/small/tanh
# TF, high turning rate:
#   python3 sim_approx_error_single.py -of carOneHighTF -mf models/store_best/car/RA/big/tanh

def multi_experiment(env, agent, firstIdx, numSample, maxLength, toEnd):
    stateBound = np.array([ [-1, 1],
                            [-1, 1],
                            [0, 2*np.pi*(1-1/numSample)]])
    samples = np.linspace(start=stateBound[:,0], stop=stateBound[:,1], num=numSample)

    freeCoordNum = 2
    rolloutResult  = np.empty(shape=np.ones(freeCoordNum, dtype=int)*numSample, dtype=int)
    trajLength     = np.empty(shape=np.ones(freeCoordNum, dtype=int)*numSample, dtype=int)
    ddqnValue      = np.empty(shape=np.ones(freeCoordNum, dtype=int)*numSample, dtype=float)
    rolloutValue   = np.empty(shape=np.ones(freeCoordNum, dtype=int)*numSample, dtype=float)
    it = np.nditer(rolloutResult, flags=['multi_index'])

    while not it.finished:
        idx = it.multi_index
        stateIdx = idx + (firstIdx,)
        print(stateIdx, end='\r')
        state = samples[stateIdx, np.arange(3)]
        traj, result, minV, _ = env.simulate_one_trajectory(
            agent.Q_network, T=maxLength, state=state, toEnd=toEnd)
        trajLength[idx] = traj.shape[0]
        rolloutResult[idx] = result # result \in { 1, -1}
        rolloutValue[idx] = minV

        agent.Q_network.eval()
        stateTensor = torch.from_numpy(state).float().to(agent.device)
        state_action_values = agent.Q_network(stateTensor)
        Q_vec = state_action_values.detach().cpu().reshape(-1)
        ddqnValue[idx] = Q_vec.min().item()

        it.iternext()

    carOneDict = {}
    carOneDict['rolloutResult'] = rolloutResult
    carOneDict['trajLength']    = trajLength
    carOneDict['ddqnValue']     = ddqnValue
    carOneDict['rolloutValue']  = rolloutValue
    
    print()
    return carOneDict


def run(args):
    #== ENVIRONMENT ==
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
    action_list = np.arange(actionNum)
    print("State Dimension: {:d}, ActionSpace Dimension: {:d}".format(stateNum, actionNum))

    print("Dynamic parameters:")
    print("  CAR")
    print("    Constraint radius: {:.1f}, Target radius: {:.1f}, Turn radius: {:.2f}, Maximum speed: {:.2f}, Maximum angular speed: {:.3f}".format(
        env.car.constraint_radius, env.car.target_radius, env.car.R_turn, env.car.speed, env.car.max_turning_rate))
    print(env.car.discrete_controls)
    if 2*env.car.R_turn-env.car.constraint_radius > env.car.target_radius:
        print("Type II Reach-Avoid Set")
    else:
        print("Type I Reach-Avoid Set")


    #== AGENT ==
    print("\n== Agent Information ==")
    with open('{:s}/CONFIG.pkl'.format(args.modelFolder), 'rb') as handle:
        tmpConfig = pickle.load(handle)
    CONFIG = dqnConfig()
    for key, value in tmpConfig.__dict__.items():
        CONFIG.__dict__[key] = tmpConfig.__dict__[key]
    CONFIG.DEVICE = device
    CONFIG.SEED = 0
    print(vars(CONFIG))

    dimList = [stateNum] + CONFIG.ARCHITECTURE + [actionNum]
    agent = DDQNSingle(CONFIG, actionNum, action_list, dimList, actType=CONFIG.ACTIVATION)
    modelFile = '{:s}/model-{:d}.pth'.format(args.modelFolder + '/model', 4000000)
    agent.restore(modelFile)


    #== ROLLOUT RESULTS ==
    print("\n== Approximate Error Information ==")
    np.set_printoptions(precision=2, suppress=True)
    numSample = args.numSample
    stateBound = np.array([ [-1, 1],
                            [-1, 1],
                            [0, 2*np.pi*(1-1/numSample)]])
    samples = np.linspace(start=stateBound[:,0], stop=stateBound[:,1], num=numSample)
    print(samples)

    maxLength = args.maxLength
    toEnd = args.toEnd
    carPESubDictList = []
    numThread = args.numWorker
    numTurn = int(numSample/(numThread+1e-6))+1
    for ith in range(numTurn):
        print('{} / {}'.format(ith+1, numTurn))
        with Pool(processes = numThread) as pool:
            firstIdxList = list(range(ith*numThread, min(numSample, (ith+1)*numThread) ))
            print(firstIdxList)
            numExp = len(firstIdxList)
            envList       = [env]       * numExp
            agentList     = [agent]     * numExp
            numSampleList = [numSample] * numExp
            maxLengthList = [maxLength] * numExp
            toEndList     = [toEnd]     * numExp

            carPESubDict_i = pool.starmap(multi_experiment, zip(
                envList, agentList, firstIdxList, numSampleList, maxLengthList, toEndList))
        carPESubDictList = carPESubDictList + carPESubDict_i

    #== COMBINE RESULTS ==
    rolloutResult  = np.empty(shape=np.ones(3, dtype=int)*numSample, dtype=int)
    trajLength     = np.empty(shape=np.ones(3, dtype=int)*numSample, dtype=int)
    ddqnValue      = np.empty(shape=np.ones(3, dtype=int)*numSample, dtype=float)
    rolloutValue   = np.empty(shape=np.ones(3, dtype=int)*numSample, dtype=float)

    for i, carPESubDict_i in enumerate(carPESubDictList):
        rolloutResult[:, :, i] = carPESubDict_i['rolloutResult']
        trajLength[:, :, i]    = carPESubDict_i['trajLength']
        ddqnValue[:, :, i]     = carPESubDict_i['ddqnValue']
        rolloutValue[:, :, i]  = carPESubDict_i['rolloutValue']

    carOneDict = {}
    carOneDict['numSample']     = numSample
    carOneDict['maxLength']     = maxLength
    carOneDict['toEnd']         = toEnd
    carOneDict['rolloutResult'] = rolloutResult
    carOneDict['trajLength']    = trajLength
    carOneDict['ddqnValue']     = ddqnValue
    carOneDict['rolloutValue']  = rolloutValue

    outFolder = args.modelFolder + '/data/'
    os.makedirs(outFolder, exist_ok=True)
    outFile = outFolder + args.outFile + '.npy'
    np.save('{:s}'.format(outFile), carOneDict)
    print('Save to {:s} ...'.format(outFile))

    #== Plot Reach-Avoid Set based on analytic solutions and approximate value function ==
    fig, axes = plt.subplots(1, 2, figsize=(8,4), sharex=True, sharey=True)

    axStyle = [np.array([-1., 1., -1., 1.])]
            
    ax = axes[0]
    im = ax.imshow(carOneDict['ddqnValue'][:,:,0].T, interpolation='none', extent=axStyle[0], origin="lower",
                cmap='seismic', vmin=-1, vmax=1)
    env.plot_reach_avoid_set(ax, c='g', lw=3, orientation=0)
    ax.set_xlabel(r'$\theta={:.0f}^\circ$'.format(np.pi/2*180/np.pi), fontsize=24)

    #== Rollout ==
    ax = axes[1]
    im = ax.imshow(carOneDict['rolloutValue'][:,:,0].T <= 0, interpolation='none', extent=axStyle[0], origin="lower",
                cmap='coolwarm', vmin=0, vmax=1)
    ax.set_xlabel('Rollout', fontsize=24)

    #== Formatting ==
    for ax in axes:
        env.plot_target_failure_set(ax=ax)
        env.plot_formatting(ax=ax)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f",   "--forceCPU",       help="force CPU",           action="store_true")
    parser.add_argument("-low", "--low",            help="lowOmega",            action="store_true")
    parser.add_argument("-te",  "--toEnd",          help="stop until boundary", action="store_true")

    parser.add_argument("-ns",  "--numSample",      help="#samples",    default=101,            type=int)
    parser.add_argument("-nw",  "--numWorker",      help="#workers",    default=6,              type=int)
    parser.add_argument("-ml",  "--maxLength",      help="max length",  default=100,            type=int)
    parser.add_argument("-of",  "--outFile",        help="output file", default='carOneDict',   type=str)
    parser.add_argument("-mf",  "--modelFolder",    help="model folder", 
        default='models/store_best/car/RA/big/tanh', type=str)

    args = parser.parse_args()
    print("\n== Arguments ==")
    print(args)

    start = time.time()
    run(args)
    print('Execution time: {:.1f}'.format(time.time()-start))