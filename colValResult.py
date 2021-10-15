"""
Please contact the author(s) of this library if you have any questions.
Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )

1. We collect state samples, their worst results and rollout values for all
    action sequences.
2. Each file records state, worst result and rollout values.
3. This script collects all `{args.dataFile}{sampleType}*` under
    `{args.modelFolder}/data/{sampleType}/` and genetates
    `{args.outFile}{sampleType}.npy` under `{args.modelFolder}/data/`.
    Each file records exhaustive value from different attacker heading
    angles, defender positions and defender heading angles.

EXAMPLES
    TN: python3 colValResult.py -t 0 -mf <model path>
    FP: python3 colValResult.py -t 3 -mf <model path>
"""

import argparse
import os
import glob
import numpy as np


def run(args):
  print('\n== Collecting Results ==')
  sampleTypeList = ['TN', 'TP', 'FN', 'FP', 'POS', 'NEG']
  sampleType = sampleTypeList[args.sampleType]
  dataFolder = os.path.join(args.modelFolder, 'data', sampleType)
  results = glob.glob(
      os.path.join(dataFolder, args.dataFile + sampleType + '*')
  )
  start = len(args.dataFile + sampleType)
  indices = np.array([int(li.split('/')[-1][start:-4]) for li in results])
  if len(indices) < args.number:
    print(
        "we should get {} results but only get {}, missing:".format(
            args.number, len(indices)
        )
    )
    not_obtain = np.full(shape=(args.number), fill_value=True, dtype=bool)
    for i in indices:
      not_obtain[i] = False
    print(np.arange(args.number)[not_obtain])
    return

  numTest = len(results)
  states = np.empty(shape=(numTest, 6), dtype=float)
  dictList = np.empty(shape=(numTest), dtype=object)
  exhaustiveValueList = np.empty(shape=(numTest), dtype=object)
  stateIdxList = np.empty(shape=(numTest), dtype=object)
  for i, resultFile in enumerate(results):
    print('Load from {:s} ...'.format(resultFile), end='\r')
    read_dictionary = np.load(resultFile, allow_pickle='TRUE').item()
    test_idx = read_dictionary['testIdx']
    states[test_idx, :] = read_dictionary['state']
    dictList[test_idx] = read_dictionary['dict']
    stateIdxList[test_idx] = read_dictionary['stateIdx']
    exhaustiveValueList[test_idx] = read_dictionary['rolloutValue']
    if i == 0:
      maxLength = read_dictionary['maxLength']
      numPursuerStep = read_dictionary['numPursuerStep']
  print('\nWe collect {:d} results'.format(len(dictList)))
  finalDict = {}
  finalDict['states'] = states
  finalDict['dictList'] = dictList
  finalDict['stateIdxList'] = stateIdxList
  finalDict['maxLength'] = maxLength
  finalDict['numPursuerStep'] = numPursuerStep
  finalDict['exhaustiveValueList'] = exhaustiveValueList
  print(stateIdxList[:5])

  outFolder = os.path.join(args.modelFolder, 'data/')
  outFile = os.path.join(outFolder, args.outFile + sampleType + '.npy')
  print('\n--> Save to {:s} ...'.format(outFile))
  print(finalDict.keys())
  np.save('{:s}'.format(outFile), finalDict)


if __name__ == '__main__':
  # == Arguments ==
  parser = argparse.ArgumentParser()

  # Simulation Parameters
  parser.add_argument(
      "-t", "--sampleType", help="type of sampled states", default=0, type=int
  )

  # File Parameters
  parser.add_argument(
      "-n", "--number", help="#files assumed to obtain", default='500',
      type=int
  )
  parser.add_argument(
      "-of", "--outFile", help="output file", default='valDict', type=str
  )
  parser.add_argument("-mf", "--modelFolder", help="model folder", type=str)
  parser.add_argument(
      "-df", "--dataFile", help="samples file", default='valDict', type=str
  )

  args = parser.parse_args()
  print("== Arguments ==")
  print(args)

  # == Execution ==
  np.set_printoptions(precision=3, suppress=True, floatmode='fixed')
  run(args)
