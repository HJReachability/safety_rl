# == APPROXIMATION ERROR ==
# The representability of the proposed method.
# 1. Rollout value vs. (approximately conservative) “optimal value” via sampling
#   about 5 open-loop attacker trajectories
# 2. validate them by simulating a very large number of pursuer (defender) 
#   trajectories, so that we can be confident that our system truly succeeds not 
#   only against the “oracle adversary” predicted by the Q network, but also 
#   against any possible adversary.


from warnings import simplefilter 
simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import argparse
import time
import os

from utils.carPEAnalysis import *

# Example: python3 sim_approx_error.py -nt 10

def multi_experiment(env, agent, state, maxLength=50, numPursuerStep=10):
    print("I'm process", os.getpid())
    responseDict = validateEvaderPolicy(env, agent, state, maxLength, numPursuerStep)
    maxminV = responseDict['maxminV']
    maxminIdx = responseDict['maxminIdx']
    info = responseDict[maxminIdx]

    dictTmp = {}
    dictTmp['maxminV'] = maxminV
    dictTmp['maxminIdx'] = maxminIdx
    dictTmp['trajEvader'] = info['trajEvader']
    dictTmp['trajPursuer'] = info['trajPursuer']
    return dictTmp


def run(args):
    #== ENVIRONMENT ==
    env = loadEnv(args)
    stateNum = env.state.shape[0]
    actionNum = env.action_space.n
    numActionList = env.numActionList
    device = env.device

    #== AGENT ==
    configFile = '{:s}/CONFIG.pkl'.format(args.modelFolder)
    agent = loadAgent(args, configFile, device, stateNum, actionNum, numActionList)

    #== Getting states to be tested ==
    print('\n== Getting states to be tested ==')
    print('Load from {:s} ...'.format(args.resultFile))
    read_dictionary = np.load(args.resultFile, allow_pickle='TRUE').item()
    print(read_dictionary.keys())
    ddqnValue     = read_dictionary['ddqnValue']
    numSample     = read_dictionary['numSample']

    DDQNSucMtx = (ddqnValue <= 0)
    DDQNSucIndices = np.argwhere(DDQNSucMtx)
    length = DDQNSucIndices.shape[0]
    indices = np.random.randint(low=0, high=length, size=(args.numTest,))

    bounds = np.array([ [-1, 1],
                        [-1, 1],
                        [0, 2*np.pi]])
    stateBound = np.concatenate((bounds, bounds), axis=0)
    samples = np.linspace(start=stateBound[:,0], stop=stateBound[:,1],
        num=numSample)
    states = np.empty(shape=(args.numTest, 6), dtype=float)
    for cnt, i in enumerate(indices):
        idx = tuple(DDQNSucIndices[i])
        state = samples[idx, np.arange(6)]
        states[cnt, :] = state

    #! Customize
    from multiprocessing import Pool
    dictList = []
    numThread = args.numWorker
    numTurn = int(args.numTest/(numThread+1e-6))+1
    maxLength = args.maxLength
    numPursuerStep = args.numPursuerStep
    for ith in range(numTurn):
        print('{} / {}'.format(ith+1, numTurn))
        with Pool(processes = numThread) as pool:
            startIdx = ith*numThread
            endIdx = min(numSample, (ith+1)*numThread)
            stateList = []
            for i in range(startIdx, endIdx):
                stateList.append(states[i, :])
            numExp = len(stateList)
            envList            = [env]            * numExp
            agentList          = [agent]          * numExp
            maxLengthList      = [maxLength]      * numExp
            numPursuerStepList = [numPursuerStep] * numExp

            subDictList = pool.starmap(multi_experiment, zip(
                envList, agentList, stateList, maxLengthList, numPursuerStepList))
        dictList = dictList + subDictList

    finalDict = {}
    finalDict['states'] = states
    finalDict['dictList'] = dictList
    finalDict['maxLength'] = maxLength
    finalDict['numPursuerStep'] = numPursuerStep

    outFile = 'data/' + args.outFile + '.npy'
    np.save('{:s}'.format(outFile), finalDict)


if __name__ == '__main__':
    np.set_printoptions(precision=2, suppress=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("-f",  "--forceCPU",    help="force CPU",
        action="store_true")
    parser.add_argument("-nt", "--numTest",   help="#tests",
        default=10, type=int)
    parser.add_argument("-nw", "--numWorker",   help="#workers",
        default=6, type=int)
    parser.add_argument("-ml", "--maxLength",   help="max length",
        default=40, type=int)
    parser.add_argument("-nps", "--numPursuerStep",   help="# pursuer steps",
        default=10, type=int)
    parser.add_argument("-of", "--outFile",     help="output file",
        default='validationDict', type=str)
    parser.add_argument("-mf", "--modelFolder", help="model folder", 
        default='scratch/carPE/largeBuffer-2021-02-04-23_02', type=str)
    parser.add_argument("-rf", "--resultFile", help="result file", 
        default='data/largeBuffer.npy', type=str)


    args = parser.parse_args()
    print("\n== Arguments ==")
    print(args)

    start = time.time()
    run(args)
    print('Execution time: {:.1f}'.format(time.time()-start))