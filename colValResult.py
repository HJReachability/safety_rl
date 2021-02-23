# == APPROXIMATION ERROR ==
# 1. We collect state samples, their worst results and rollout values for all
    # action sequences.
# 2. Each file records state, worst result and rollout values.

# EXAMPLES
    # TN: python3 colValResult.py -t 0 -mf <model path>
    # FP: python3 colValResult.py -t 3 -mf <model path>

import numpy as np
import argparse
import time
import os
import glob


def run(args):
    print('\n== Collecting Results ==')
    sampleTypeList = ['TN', 'TP', 'FN', 'FP', 'POS', 'NEG']
    sampleType = sampleTypeList[args.sampleType]
    dataFolder = os.path.join(args.modelFolder, 'data/', sampleType)
    results = glob.glob(os.path.join(dataFolder, 'valDict'+sampleType+'*'))
    numTest = len(results)
    states = np.empty(shape=(numTest, 6), dtype=float)
    dictList = []
    rolloutValueList = []
    idxList = []
    for i, resultFile in enumerate(results):
        print('Load from {:s} ...'.format(resultFile), end='\r')
        read_dictionary = np.load(resultFile, allow_pickle='TRUE').item()
        states[i, :] = read_dictionary['state']
        dictList.append(read_dictionary['dict'])
        idxList.append(read_dictionary['idx'])
        rolloutValueList.append(read_dictionary['rolloutValue'])
        if i == 0:
            maxLength = read_dictionary['maxLength']
            numPursuerStep = read_dictionary['numPursuerStep']
    print('\nWe collect {:d} results'.format(i))
    finalDict = {}
    finalDict['states'] = states
    finalDict['dictList'] = dictList
    finalDict['idxList'] = idxList
    finalDict['maxLength'] = maxLength
    finalDict['numPursuerStep'] = numPursuerStep
    finalDict['rolloutValueList'] = rolloutValueList
    print(idxList)

    outFolder = os.path.join(args.modelFolder, 'data/')
    outFile = os.path.join(outFolder, args.outFile + sampleType + '.npy')
    print('\n--> Save to {:s} ...'.format(outFile))
    print(finalDict.keys())
    np.save('{:s}'.format(outFile), finalDict)


if __name__ == '__main__':
    #== Arguments ==
    parser = argparse.ArgumentParser()

    # Simulation Parameters
    parser.add_argument("-t", "--sampleType", help="type of sampled states",
        default=0, type=int)

    # File Parameters
    parser.add_argument("-of", "--outFile", help="output file",
        default='valDict', type=str)
    parser.add_argument("-mf", "--modelFolder", help="model folder", 
        default='scratch/carPE/largeBuffer-3-512-2021-02-07-01_51', type=str)

    args = parser.parse_args()
    print("== Arguments ==")
    print(args)

    #== Execution ==
    np.set_printoptions(precision=3, suppress=True, floatmode='fixed')
    run(args)