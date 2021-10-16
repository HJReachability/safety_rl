"""
Please contact the author(s) of this library if you have any questions.
Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )

Generate samples to compute approximation error.
1. It supports SIX sample types:
    0-6 corresponds to ['TN', 'TP', 'FN', 'FP', 'POS', 'NEG', 'ALL'].
2. This script uses `{args.modelFolder}/data/{args.dataFile}.npy` to load the
    ddqn-predicted values and rollout values of the sampled initial states.
    Then, it generates the samples for validation under
    `{args.modelFolder}/data/{sampleType}/{args.outFile}{sampleType}.npy`

EXAMPLES
    TN: python3 genValSamples.py -t 0 -mf <model path>
    FP: python3 genValSamples.py -t 3 -mf <model path>
"""

import argparse
import os
from warnings import simplefilter
import numpy as np

simplefilter(action='ignore', category=FutureWarning)


def run(args):
  # == Getting states to be tested ==
  print('\n== Getting states to be tested ==')
  dataFolder = os.path.join(args.modelFolder, 'data')
  dataFile = os.path.join(dataFolder, args.dataFile + '.npy')
  print('Load from {:s} ...'.format(dataFile))
  read_dictionary = np.load(dataFile, allow_pickle='TRUE').item()
  print(read_dictionary.keys())
  ddqnValue = read_dictionary['ddqnValue']
  rolloutValue = read_dictionary['rolloutValue']
  samples = read_dictionary['samples']
  [samplesAtt, samplesDef, thetas] = samples
  print(rolloutValue.shape)

  if args.sampleType == 0:
    pickMtx = np.logical_and((rolloutValue <= 0), (ddqnValue <= 0))
  elif args.sampleType == 1:
    pickMtx = np.logical_and((rolloutValue > 0), (ddqnValue > 0))
  elif args.sampleType == 2:
    pickMtx = np.logical_and((rolloutValue > 0), (ddqnValue <= 0))
  elif args.sampleType == 3:
    pickMtx = np.logical_and((rolloutValue <= 0), (ddqnValue > 0))
  elif args.sampleType == 4:
    pickMtx = (rolloutValue > 0)
  elif args.sampleType == 5:
    pickMtx = (rolloutValue <= 0)
  elif args.sampleType == 6:
    pickMtx = np.full(shape=rolloutValue.shape, fill_value=True)
  sampleTypeList = ['TN', 'TP', 'FN', 'FP', 'POS', 'NEG', 'ALL']
  sampleType = sampleTypeList[args.sampleType]
  print('Type of sampled states:', sampleType)
  pickIndices = np.argwhere(pickMtx)
  length = pickIndices.shape[0]
  indices = np.random.randint(low=0, high=length, size=(args.numTest,))
  states = np.empty(shape=(args.numTest, 6), dtype=float)
  ddqnList = np.empty(shape=(args.numTest), dtype=float)
  rolloutPolicyValueList = np.empty(shape=(args.numTest), dtype=float)
  idxList = []
  for cnt, i in enumerate(indices):
    idx = tuple(pickIndices[i])
    ddqnList[cnt] = ddqnValue[idx]
    rolloutPolicyValueList[cnt] = rolloutValue[idx]
    states[cnt, 0:2] = samplesAtt[idx[0], :]
    states[cnt, 2] = thetas[idx[1]]
    states[cnt, 3:5] = samplesDef[idx[2], :]
    states[cnt, 5] = thetas[idx[3]]
    idxList.append(idx)

  print("The first five indices picked: ")
  endIdx = 10
  print('indices:', idxList[:5])
  print('states:', states[:5, :])
  # print(np.all(ddqnList[:] <= 0))
  # print(np.all(rolloutPolicyValueList[:] <= 0))
  print('DDQN:', ddqnList[:endIdx])
  print('rollout:', rolloutPolicyValueList[:endIdx])

  finalDict = {}
  finalDict['states'] = states
  finalDict['idxList'] = idxList
  finalDict['ddqnList'] = ddqnList
  finalDict['rolloutPolicyValueList'] = rolloutPolicyValueList

  outFolder = os.path.join(dataFolder, sampleType)
  os.makedirs(outFolder, exist_ok=True)
  outFile = os.path.join(outFolder, args.outFile + sampleType + '.npy')
  np.save('{:s}'.format(outFile), finalDict)
  print('--> Save to {:s} ...'.format(outFile))


if __name__ == '__main__':
  # == Arguments ==
  parser = argparse.ArgumentParser()

  # Simulation Parameters
  parser.add_argument(
      "-rnd", "--randomSeed", help="random seed", default=0, type=int
  )
  parser.add_argument(
      "-t", "--sampleType", help="type of sampled states", default=0, type=int
  )
  parser.add_argument("-nt", "--numTest", help="#tests", default=100, type=int)

  # File Parameters
  parser.add_argument(
      "-of", "--outFile", help="output file", default='samples', type=str
  )
  parser.add_argument("-mf", "--modelFolder", help="model folder", type=str)
  parser.add_argument(
      "-df", "--dataFile", help="estimation error file", default='estError',
      type=str
  )

  args = parser.parse_args()
  print("== Arguments ==")
  print(args)

  # == Execution ==
  np.random.seed(args.randomSeed)
  np.set_printoptions(precision=3, suppress=True, floatmode='fixed')
  run(args)
