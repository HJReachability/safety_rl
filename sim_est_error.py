# == ESTIMATION ERROR ==
# We want to evaluate how well we learned from the data.
# We compare the DDQN-predicted value vs. the rollout value by DDQN-induced 
# policies.

from warnings import simplefilter 
simplefilter(action='ignore', category=FutureWarning)
from gym_reachability import gym_reachability  # Custom Gym env.
import gym
import numpy as np
import torch
import os
import time
import pickle

from utils.carPEAnalysis import *

import argparse

# 27554 seconds
#   11 samples per dimension with 6 workers
#   NN: 2-layer with 512 neurons per leayer
# ex: python3 sim_est_error.py -of largeBuffer-3-512-new
#       -mf scratch/carPE/largeBuffer-3-512-2021-02-07-01_51
# ex: py3 sim_est_error.py -mf scratch/carPE/largeBuffer-3-512-2021-02-07-01_51

def multi_experiment(env, agent, samples, firstIdx, numSample, maxLength, toEnd):
    print("I'm process", os.getpid())
    # R = env.evader_constraint_radius
    # r = env.evader_target_radius
    # bounds = np.array([ [r, R],
    #                     [0., 2*np.pi*(1-1/numSample)],
    #                     [0., 2*np.pi*(1-1/numSample)],
    #                     [0.01, R],
    #                     [np.pi*(1/numSample), np.pi*(2-1/numSample)],
    #                     [0., 2*np.pi*(1-1/numSample)]])
    # samples = np.linspace(start=bounds[:,0], stop=bounds[:,1], num=numSample)

    freeCoordNum = 5
    shapeTmp = np.ones(freeCoordNum, dtype=int)*numSample
    rolloutResult   = np.empty(shape=shapeTmp, dtype=int)
    trajLength      = np.empty(shape=shapeTmp, dtype=int)
    ddqnValue       = np.empty(shape=shapeTmp, dtype=float)
    rolloutValue    = np.empty(shape=shapeTmp, dtype=float)
    it = np.nditer(rolloutResult, flags=['multi_index'])

    while not it.finished:
        idx = it.multi_index
        stateIdx = (firstIdx,) + idx
        print(stateIdx, end='\r')
        state = samples[stateIdx, np.arange(6)]
        dist, phi = state[[0, 1]]
        state[0] = dist * np.cos(phi)
        state[1] = dist * np.sin(phi)
        dist, phi = state[[3, 4]]
        state[3] = dist * np.cos(phi)
        state[4] = dist * np.sin(phi)
        traj, _, result, minV, _ = env.simulate_one_trajectory(
            agent.Q_network, T=maxLength, state=state, toEnd=toEnd)
        trajLength[idx] = traj.shape[0]
        rolloutResult[idx] = result # result \in { 1, -1}
        rolloutValue[idx] = minV

        agent.Q_network.eval()
        state = torch.from_numpy(state).float().to(agent.device)
        state_action_values = agent.Q_network(state)
        Q_mtx = state_action_values.detach().cpu().reshape(
            agent.numActionList[0], agent.numActionList[1])
        pursuerValues, _ = Q_mtx.max(dim=1)
        minmaxValue, _ = pursuerValues.min(dim=0)
        ddqnValue[idx] = minmaxValue

        it.iternext()

    carPEDict = {}
    carPEDict['rolloutResult'] = rolloutResult
    carPEDict['trajLength']    = trajLength
    carPEDict['ddqnValue']     = ddqnValue
    carPEDict['rolloutValue']  = rolloutValue
    
    print()
    return carPEDict


def run(args):
    #== ENVIRONMENT ==
    env = loadEnv(args)
    stateNum = env.state.shape[0]
    actionNum = env.action_space.n
    numActionList = env.numActionList
    device = env.device

    #== AGENT ==
    configFile = '{:s}/CONFIG.pkl'.format(args.modelFolder)
    agent = loadAgent(
        args, configFile, device, stateNum, actionNum, numActionList)

    #== ROLLOUT RESULTS ==
    print("\n== Approximate Error Information ==")
    np.set_printoptions(precision=2, suppress=True)
    numSample = args.numSample
    R = env.evader_constraint_radius - 0.01
    r = env.evader_target_radius + 0.01
    bounds = np.array([ [r, R],
                        [0., 2*np.pi*(1-1/numSample)],
                        [0., 2*np.pi*(1-1/numSample)],
                        [0.01, R],
                        [np.pi*(1/numSample), np.pi*(2-1/numSample)],
                        [0., 2*np.pi*(1-1/numSample)]])
    samples = np.linspace(start=bounds[:,0], stop=bounds[:,1], num=numSample)
    print(samples)

    from multiprocessing import Pool
    maxLength = args.maxLength
    toEnd = args.toEnd
    carPESubDictList = []
    numThread = args.numWorker
    numTurn = int(numSample/(numThread+1e-6))+1
    for ith in range(numTurn):
        print('{} / {}'.format(ith+1, numTurn))
        with Pool(processes = numThread) as pool:
            startIdx = ith*numThread
            endIdx = min(numSample, (ith+1)*numThread)
            firstIdxList = list(range(startIdx, endIdx))
            print(firstIdxList)
            numExp = len(firstIdxList)
            envList       = [env]       * numExp
            agentList     = [agent]     * numExp
            samplesList   = [samples]   * numExp
            numSampleList = [numSample] * numExp
            maxLengthList = [maxLength] * numExp
            toEndList     = [toEnd]     * numExp

            carPESubDict_i = pool.starmap(multi_experiment, zip(
                envList, agentList, samplesList, firstIdxList, numSampleList, 
                maxLengthList, toEndList))
        carPESubDictList = carPESubDictList + carPESubDict_i

    #== COMBINE RESULTS ==
    shapeTmp = np.ones(6, dtype=int)*numSample
    rolloutResult  = np.empty(shape=shapeTmp, dtype=int)
    trajLength     = np.empty(shape=shapeTmp, dtype=int)
    ddqnValue      = np.empty(shape=shapeTmp, dtype=float)
    rolloutValue   = np.empty(shape=shapeTmp, dtype=float)

    for i, carPESubDict_i in enumerate(carPESubDictList):
        rolloutResult[i, :, :, :, :, :] = carPESubDict_i['rolloutResult']
        trajLength[i, :, :, :, :, :]    = carPESubDict_i['trajLength']
        ddqnValue[i, :, :, :, :, :]     = carPESubDict_i['ddqnValue']
        rolloutValue[i, :, :, :, :, :]  = carPESubDict_i['rolloutValue']

    carPEDict = {}
    carPEDict['numSample']     = numSample
    carPEDict['maxLength']     = maxLength
    carPEDict['toEnd']         = toEnd
    carPEDict['rolloutResult'] = rolloutResult
    carPEDict['trajLength']    = trajLength
    carPEDict['ddqnValue']     = ddqnValue
    carPEDict['rolloutValue']  = rolloutValue
    carPEDict['samples']       = samples

    outFolder = args.modelFolder + '/data/'
    os.makedirs(outFolder, exist_ok=True)
    outFile = outFolder + args.outFile + '.npy'
    np.save('{:s}'.format(outFile), carPEDict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f",  "--forceCPU",    help="force CPU",
        action="store_true")
    parser.add_argument("-te", "--toEnd",       help="to end",
        action="store_true")
    parser.add_argument("-cpf", "--cpf",        help="consider pursuer failure",
        action="store_true")
    parser.add_argument("-ml", "--maxLength",   help="max length",
        default=150, type=int)
    parser.add_argument("-ns", "--numSample",   help="#samples",
        default=11, type=int)
    parser.add_argument("-nw", "--numWorker",   help="#workers",
        default=6, type=int)
    parser.add_argument("-of", "--outFile",     help="output file",
        default='estError', type=str)
    parser.add_argument("-mf", "--modelFolder", help="model folder", 
        default='scratch/carPE/largeBuffer-2021-02-04-23_02', type=str)

    args = parser.parse_args()
    print("\n== Arguments ==")
    print(args)

    start = time.time()
    run(args)
    print('Execution time: {:.1f}'.format(time.time()-start))