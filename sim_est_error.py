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
# ex: python3 sim_est_error.py -ns 11 -nw 6 -of largeBuffer-3-512
#   -mf scratch/carPE/largeBuffer-3-512-2021-02-07-01_51 

def multi_experiment(env, agent, firstIdx, numSample, maxLength, toEnd):
    print("I'm process", os.getpid())
    bounds = np.array([ [-1, 1],
                        [-1, 1],
                        [0, 2*np.pi]])
    stateBound = np.concatenate((bounds, bounds), axis=0)
    samples = np.linspace(start=stateBound[:,0], stop=stateBound[:,1],
        num=numSample)

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
    bounds = np.array([ [-1, 1],
                        [-1, 1],
                        [0, 2*np.pi]])
    stateBound = np.concatenate((bounds, bounds), axis=0)
    samples = np.linspace(start=stateBound[:,0], stop=stateBound[:,1],
        num=numSample)
    print(samples)

    from multiprocessing import Pool
    maxLength = 150
    toEnd = True
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
            numSampleList = [numSample] * numExp
            maxLengthList = [maxLength] * numExp
            toEndList     = [toEnd]     * numExp

            carPESubDict_i = pool.starmap(multi_experiment, zip(
                envList, agentList, firstIdxList, numSampleList, maxLengthList, 
                toEndList))
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

    outFile = 'data/' + args.outFile + '.npy'
    np.save('{:s}'.format(outFile), carPEDict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f",  "--forceCPU",    help="force CPU",
        action="store_true")
    parser.add_argument("-ns", "--numSample",   help="#samples",
        default=3, type=int)
    parser.add_argument("-nw", "--numWorker",   help="#workers",
        default=6, type=int)
    parser.add_argument("-of", "--outFile",     help="output file",
        default='carPEDict', type=str)
    parser.add_argument("-mf", "--modelFolder", help="model folder", 
        default='scratch/carPE/largeBuffer-2021-02-04-23_02', type=str)

    args = parser.parse_args()
    print("\n== Arguments ==")
    print(args)

    start = time.time()
    run(args)
    print('Execution time: {:.1f}'.format(time.time()-start))