# == ESTIMATION ERROR ==
# 1. We collect ddqn-predicted values and rollout values from different
    # attacker positions.
# 2. Each file records ddqn-predicted values and rollout values from different
    # attacker heading angles, defender positions and defender heading angles

# EXAMPLES
    # TN: python3 colEstError.py -mf <model path>
    # FP: python3 colEstError.py -mf <model path>

import numpy as np
import argparse
import time
import os
import glob


def run(args):
    print('\n== Collecting Results ==')
    dataFolder = os.path.join(args.modelFolder, 'data/', 'est/')
    results = glob.glob(os.path.join(dataFolder, 'estError*'))
    numTest = len(results)
    for i, resultFile in enumerate(results):
        print('Load from {:s} ...'.format(resultFile), end='\r')
        read_dictionary = np.load(resultFile, allow_pickle='TRUE').item()
        trajLengthTmp   = read_dictionary['trajLength']
        ddqnValueTmp    = read_dictionary['ddqnValue']
        rolloutValueTmp = read_dictionary['rolloutValue']
        if i == 0:
            maxLength = read_dictionary['maxLength']
            toEnd = read_dictionary['toEnd']
            samples = read_dictionary['samples']
            shapeTmp = (numTest,) + trajLengthTmp.shape
            trajLength     = np.empty(shape=shapeTmp, dtype=int)
            ddqnValue      = np.empty(shape=shapeTmp, dtype=float)
            rolloutValue   = np.empty(shape=shapeTmp, dtype=float)
        trajLength[i] = trajLengthTmp
        ddqnValue[i] = ddqnValueTmp
        rolloutValue[i] = rolloutValueTmp
    print()
    finalDict = {}
    finalDict['trajLength'] = trajLength
    finalDict['ddqnValue'] = ddqnValue
    finalDict['rolloutValue'] = rolloutValue

    finalDict['maxLength'] = maxLength
    finalDict['toEnd'] = toEnd
    finalDict['samples'] = samples

    outFolder = os.path.join(args.modelFolder, 'data/')
    outFile = os.path.join(outFolder, args.outFile + '.npy')
    print('\n--> Save to {:s} ...'.format(outFile))
    print(finalDict.keys())
    np.save('{:s}'.format(outFile), finalDict)


if __name__ == '__main__':
    #== Arguments ==
    parser = argparse.ArgumentParser()

    # File Parameters
    parser.add_argument("-of", "--outFile", help="output file",
        default='estError', type=str)
    parser.add_argument("-mf", "--modelFolder", help="model folder", 
        default='scratch/carPE/largeBuffer-3-512-2021-02-07-01_51', type=str)

    args = parser.parse_args()
    print("== Arguments ==")
    print(args)

    #== Execution ==
    np.set_printoptions(precision=3, suppress=True, floatmode='fixed')
    run(args)