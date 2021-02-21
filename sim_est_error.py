# == ESTIMATION ERROR ==
# 1. We want to evaluate how well we learned from the data.
# 2. We compare the DDQN-predicted value vs. the rollout value by DDQN-induced
    # policies.
# 3. Pre-processing:
    # we need to run `genEstSamples.py` beforehand to get state samples

# EXECUTION TIME: 32.9 seconds for
    # one attacker position
    # 10 attacker heading angles
    # 100 defender simulations
    # 10 defender heading angles

# EXAMPLES
    # test:
        # python3 sim_est_error.py -of tmp
    # default:
        # python3 sim_est_error.py
    # toEnd:
        # python3 sim_est_error.py -te
    # consider pursuer failure set:
        # python3 sim_est_error.py -cpf
        #   -mf scratch/carPE/largeBuffer-3-512-cpf-2021-02-14-23_24

from warnings import simplefilter 
simplefilter(action='ignore', category=FutureWarning)
from gym_reachability import gym_reachability  # Custom Gym env.
import gym
import numpy as np
import torch
import os
import time
import pickle
import argparse
from multiprocessing import Pool

from utils.carPEAnalysis import *


def multiExp(args, posAtt, thetaAttIdx, samplesDef, thetas,
    maxLength, toEnd, verbose=False):

    np.set_printoptions(precision=3, suppress=True, floatmode='fixed')
    #== ENVIRONMENT ==
    env = loadEnv(args, verbose)
    stateNum = env.state.shape[0]
    actionNum = env.action_space.n
    numActionList = env.numActionList
    device = env.device

    #== AGENT ==
    configFile = '{:s}/CONFIG.pkl'.format(args.modelFolder)
    agent = loadAgent(
        args, device, stateNum, actionNum, numActionList, verbose)

    # print("I'm process", os.getpid())
    numTheta = thetas.shape[0]
    numDef = samplesDef.shape[0]
    shapeTmp = np.array([numDef, numTheta])
    trajLength   = np.empty(shape=shapeTmp, dtype=int)
    ddqnValue    = np.empty(shape=shapeTmp, dtype=float)
    rolloutValue = np.empty(shape=shapeTmp, dtype=float)
    it = np.nditer(ddqnValue, flags=['multi_index'])

    while not it.finished:
        idx = it.multi_index
        print(idx, end='\r')
        state = np.empty(shape=(6,), dtype=float)
        state[:2] = posAtt
        state[2] = thetas[thetaAttIdx]
        state[3:5] = samplesDef[idx[0], :]
        state[5] = thetas[idx[1]]
        traj, _, result, minV, _ = env.simulate_one_trajectory(
            agent.Q_network, T=maxLength, state=state, toEnd=toEnd)
        trajLength[idx] = traj.shape[0]
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
    carPEDict['trajLength']    = trajLength
    carPEDict['ddqnValue']     = ddqnValue
    carPEDict['rolloutValue']  = rolloutValue
    carPEDict['thetaAttIdx']  = thetaAttIdx
    return carPEDict


def run(args):
    startTime = time.time()
    dataFolder = os.path.join(args.modelFolder, 'data/')
    dataFile = os.path.join(dataFolder, 'samplesEst.npy')
    print('Load from {:s} ...'.format(dataFile))
    read_dictionary = np.load(dataFile, allow_pickle='TRUE').item()
    samples = read_dictionary['samples']
    [samplesAtt, samplesDef, thetas] = samples
    numTheta = thetas.shape[0]
    numDef = samplesDef.shape[0]
    numAtt = samplesAtt.shape[0]
    posAtt = samplesAtt[args.index, :]
    print(posAtt, numTheta, numDef)

    #== ROLLOUT RESULTS ==
    print("\n== Estimation Error Information ==")
    maxLength = args.maxLength
    toEnd = args.toEnd
    carPESubDictList = []
    numThread = args.numWorker
    numTest = thetas.shape[0]
    numTurn = int(numTest/(numThread+1e-6))+1
    for ith in range(numTurn):
        print('{} / {}'.format(ith+1, numTurn), end=': ')
        with Pool(processes = numThread) as pool:
            startIdx = ith*numThread
            endIdx = min(numTest, (ith+1)*numThread)
            print('{:.0f}-{:.0f}'.format(startIdx, endIdx-1))
            # stateAttList = []
            # for j in range(startIdx, endIdx):
            #     stateTmp = np.empty(shape=(3,), dtype=float)
            #     stateTmp[:2] = posAtt
            #     stateTmp[2] = thetas[j]
            #     stateAttList.append(stateTmp)
            thetaIdxAttList = [j for j in range(startIdx, endIdx)]
            numExp = len(thetaIdxAttList)
            posAttList      = [posAtt]      * numExp
            argsList        = [args]        * numExp
            samplesDefList  = [samplesDef]  * numExp
            thetasList      = [thetas]      * numExp
            maxLengthList   = [maxLength]   * numExp
            toEndList       = [toEnd]       * numExp
            verboseList     = [False]       * numExp

            carPESubDict_i = pool.starmap(multiExp, zip( argsList,
                posAttList, thetaIdxAttList, samplesDefList, thetasList, 
                maxLengthList, toEndList, verboseList))
        carPESubDictList = carPESubDictList + carPESubDict_i

    #== COMBINE RESULTS ==
    shapeTmp = np.array([numTheta, numDef, numTheta])
    trajLength     = np.empty(shape=shapeTmp, dtype=int)
    ddqnValue      = np.empty(shape=shapeTmp, dtype=float)
    rolloutValue   = np.empty(shape=shapeTmp, dtype=float)

    for i, carPESubDict_i in enumerate(carPESubDictList):
        thetaAttIdx = carPESubDict_i['thetaAttIdx']
        trajLength[thetaAttIdx, :, :]    = carPESubDict_i['trajLength']
        ddqnValue[thetaAttIdx, :, :]     = carPESubDict_i['ddqnValue']
        rolloutValue[thetaAttIdx, :, :]  = carPESubDict_i['rolloutValue']
    print(ddqnValue.shape)

    endTime = time.time()
    execTime = endTime - startTime
    print('--> Execution time: {:.1f}'.format(execTime))

    carPEDict = {}
    carPEDict['maxLength']     = maxLength
    carPEDict['toEnd']         = toEnd
    carPEDict['samples']       = samples
    carPEDict['idx']           = args.index
    carPEDict['trajLength']    = trajLength
    carPEDict['ddqnValue']     = ddqnValue
    carPEDict['rolloutValue']  = rolloutValue

    outFolder = os.path.join(args.modelFolder, 'data/', 'est/')
    os.makedirs(outFolder, exist_ok=True)
    outFile = outFolder + args.outFile + str(args.index) + '.npy'
    np.save('{:s}'.format(outFile), carPEDict)
    print('--> Save to {:s} ...'.format(outFile))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Environment Parameters
    parser.add_argument("-cpf", "--cpf",        help="consider pursuer failure",
        action="store_true")

    # Simulation Parameters
    parser.add_argument("-f", "--forceCPU",     help="force CPU",
        action="store_true")
    parser.add_argument("-te", "--toEnd",       help="to end",
        action="store_true")
    parser.add_argument("-ml", "--maxLength",   help="max length",
        default=150, type=int)
    parser.add_argument("-nw", "--numWorker",   help="#workers",
        default=6, type=int)
    parser.add_argument("-idx", "--index", help="the index of state in samples",
        default=0, type=int)

    # File Parameters
    parser.add_argument("-of", "--outFile",     help="output file",
        default='estError', type=str)
    parser.add_argument("-mf", "--modelFolder", help="model folder", 
        default='scratch/carPE/largeBuffer-3-512-2021-02-07-01_51', type=str)

    args = parser.parse_args()
    print("\n== Arguments ==")
    print(args)

    #== Execution ==
    np.set_printoptions(precision=3, suppress=True, floatmode='fixed')
    run(args)