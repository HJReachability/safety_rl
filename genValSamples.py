# == APPROXIMATION ERROR ==

# EXAMPLES
    # py3 genValSamples.py

from warnings import simplefilter 
simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import argparse
import time
import os


def run(args):
    #== Getting states to be tested ==
    print('\n== Getting states to be tested ==')
    dataFolder = os.path.join(args.modelFolder, 'data/')
    estErrorFile = os.path.join(dataFolder, args.estErrorFile+'.npy')
    print('Load from {:s} ...'.format(estErrorFile))
    read_dictionary = np.load(estErrorFile, allow_pickle='TRUE').item()
    print(read_dictionary.keys())
    ddqnValue    = read_dictionary['ddqnValue']
    rolloutValue = read_dictionary['rolloutValue']
    samples      = read_dictionary['samples']

    if args.sampleType == 0:
        pickMtx = np.logical_and((rolloutValue <= 0), (ddqnValue <= 0))
    elif args.sampleType == 1:
        pickMtx = np.logical_and((rolloutValue > 0), (ddqnValue > 0))
    elif args.sampleType == 2:
        pickMtx = np.logical_and((rolloutValue > 0), (ddqnValue <= 0))
    elif args.sampleType == 3:
        pickMtx = np.logical_and((rolloutValue <= 0), (ddqnValue > 0))
    elif args.sampleType == 4:
        pickMtx = (ddqnValue > 0)
    elif args.sampleType == 5:
        pickMtx = (ddqnValue <= 0)
    sampleTypeList = ['TN', 'TP', 'FN', 'FP', 'POS', 'NEG']
    sampleType = sampleTypeList[args.sampleType]
    print('Type of sampled states:', sampleType)
    pickIndices = np.argwhere(pickMtx)
    length = pickIndices.shape[0]
    indices = np.random.randint(low=0, high=length, size=(args.numTest,))
    states = np.empty(shape=(args.numTest, 6), dtype=float)
    ddqnList = []
    rollvalList = []
    for cnt, i in enumerate(indices):
        idx = tuple(pickIndices[i])
        ddqnList.append(ddqnValue[idx])
        rollvalList.append(rolloutValue[idx])
        state = samples[idx, np.arange(6)]
        dist, phi = state[[0, 1]]
        state[0] = dist * np.cos(phi)
        state[1] = dist * np.sin(phi)
        dist, phi = state[[3, 4]]
        state[3] = dist * np.cos(phi)
        state[4] = dist * np.sin(phi)
        states[cnt, :] = state
    print("The first five indices picked: ", end='')
    print(indices[:5])
    print(states[:5, :])
    # print(ddqnList)
    # print(rollvalList)

    finalDict = {}
    finalDict['states'] = states
    outFolder = os.path.join(dataFolder, sampleType)
    os.makedirs(outFolder, exist_ok=True)
    outFile = os.path.join(outFolder, args.outFile+sampleType+'.npy')
    np.save('{:s}'.format(outFile), finalDict)
    print('--> Save to {:s} ...'.format(outFile))


if __name__ == '__main__':
    #== Arguments ==
    parser = argparse.ArgumentParser()

    # Simulation Parameters
    parser.add_argument("-rnd", "--randomSeed", help="random seed",
        default=0, type=int)
    parser.add_argument("-t", "--sampleType", help="type of sampled states",
        default=0, type=int)
    parser.add_argument("-nt", "--numTest", help="#tests",
        default=100, type=int)

    # File Parameters
    parser.add_argument("-of", "--outFile", help="output file",
        default='samples', type=str)
    parser.add_argument("-mf", "--modelFolder", help="model folder", 
        default='scratch/carPE/largeBuffer-3-512-2021-02-07-01_51', type=str)
    parser.add_argument("-ef", "--estErrorFile", help="estimation error file", 
        default='estError', type=str)

    args = parser.parse_args()
    print("== Arguments ==")
    print(args)

    #== Execution ==
    np.random.seed(args.randomSeed)
    np.set_printoptions(precision=3, suppress=True, floatmode='fixed')
    run(args)