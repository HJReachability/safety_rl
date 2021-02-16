# == APPROXIMATION ERROR ==
# The representability of the proposed method.
# 1. Rollout value vs. (approximately conservative) “optimal value” via sampling
#   about 5 open-loop attacker trajectories
# 2. validate them by simulating a very large number of pursuer (defender) 
#   trajectories, so that we can be confident that our system truly succeeds not 
#   only against the “oracle adversary” predicted by the Q-network, but also 
#   against any possible adversary.


from warnings import simplefilter 
simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import argparse
import time
import os

from utils.carPEAnalysis import *

# Example: python3 sim_approx_error.py -nt 10
# Example: python3 sim_approx_error.py
# 18k seconds
# ex: python3 sim_approx_error.py -of val-large-3-512-new
#       -mf scratch/carPE/largeBuffer-3-512-2021-02-07-01_51
#       -rf data/largeBuffer-2-512-new.npy

def multi_experiment(env, agent, state, maxLength=40, numPursuerStep=10):
    """
    multi_experiment: simulate defender's trajectories and record the maximum
        value among these trajectories.

    Args:
        env (object): environment.
        agent (object): agent (DDQNPursuitEvasion here).
        state (numpy array): initial state.
        maxLength (int, optional): maximal length of the trajectory.
            Defaults to 50.
        numPursuerStep (int, optional): maximal length of action sequence taken
            by the pursuer. Defaults to 10.

    Returns:
        [dict]: records maximal values, the corresponding action indices
            sequence and trajectories of the evader and the pursuer.
    """    
    print("I'm process", os.getpid())
    responseDict = validateEvaderPolicy(
        env, agent, state, maxLength, numPursuerStep)
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
    agent = loadAgent(
        args, configFile, device, stateNum, actionNum, numActionList)

    #== Getting states to be tested ==
    print('\n== Getting states to be tested ==')
    print('Load from {:s} ...'.format(args.resultFile))
    read_dictionary = np.load(args.resultFile, allow_pickle='TRUE').item()
    print(read_dictionary.keys())
    ddqnValue    = read_dictionary['ddqnValue']
    rolloutValue = read_dictionary['rolloutValue']
    samples      = read_dictionary['samples']

    DDQNSucMtx = np.logical_and((rolloutValue <= 0), (ddqnValue <= 0))
    DDQNSucIndices = np.argwhere(DDQNSucMtx)
    length = DDQNSucIndices.shape[0]
    indices = np.random.randint(low=0, high=length, size=(args.numTest,))
    states = np.empty(shape=(args.numTest, 6), dtype=float)
    for cnt, i in enumerate(indices):
        idx = tuple(DDQNSucIndices[i])
        state = samples[idx, np.arange(6)]
        dist, phi = state[[0, 1]]
        state[0] = dist * np.cos(phi)
        state[1] = dist * np.sin(phi)
        dist, phi = state[[3, 4]]
        state[3] = dist * np.cos(phi)
        state[4] = dist * np.sin(phi)
        states[cnt, :] = state
    # print(states)
    #== Estimating Approximation Error in Parallel ==
    from multiprocessing import Pool
    dictList = []
    numThread = args.numWorker
    numTurn = int(args.numTest/(numThread+1e-6))+1
    maxLength = args.maxLength
    numPursuerStep = args.numPursuerStep
    for ith in range(numTurn):
        print('\n{} / {}: '.format(ith+1, numTurn), end='')
        with Pool(processes = numThread) as pool:
            startIdx = ith*numThread
            endIdx = min(args.numTest, (ith+1)*numThread)
            stateList = []
            print('{:.0f}-{:.0f}'.format(startIdx, endIdx-1))
            for i in range(startIdx, endIdx):
                stateList.append(states[i, :])
            numExp = len(stateList)
            envList            = [env]            * numExp
            agentList          = [agent]          * numExp
            maxLengthList      = [maxLength]      * numExp
            numPursuerStepList = [numPursuerStep] * numExp

            subDictList = pool.starmap(multi_experiment, zip( envList,
                agentList, stateList, maxLengthList, numPursuerStepList))
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
    parser.add_argument("-f", "--forceCPU", help="force CPU",
        action="store_true")
    parser.add_argument("-te", "--toEnd", help="to end",
        action="store_true")
    parser.add_argument("-cpf", "--cpf", help="consider pursuer failure",
        action="store_true")
    parser.add_argument("-nt", "--numTest", help="#tests",
        default=100, type=int)
    parser.add_argument("-nw", "--numWorker", help="#workers",
        default=6,  type=int)
    parser.add_argument("-ml", "--maxLength", help="max length",
        default=50, type=int)
    parser.add_argument("-nps", "--numPursuerStep", help="#pursuer steps",
        default=10, type=int)
    parser.add_argument("-of", "--outFile", help="output file",
        default='validationDict', type=str)
    parser.add_argument("-mf", "--modelFolder", help="model folder", 
        default='scratch/carPE/largeBuffer-2021-02-04-23_02', type=str)
    parser.add_argument("-rf", "--resultFile", help="result file", 
        default='data/largeBuffer-2-512-new.npy', type=str)
    args = parser.parse_args()
    print("\n== Arguments ==")
    print(args)

    start = time.time()
    run(args)
    print('Execution time: {:.1f}'.format(time.time()-start))