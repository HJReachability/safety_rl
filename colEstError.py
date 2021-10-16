"""
Please contact the author(s) of this library if you have any questions.
Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )

1. We collect ddqn-predicted values and rollout values from different
    attacker positions.
2. This script collects all `estError*` under `{args.modelFolder}/data/est/`
    and genetates `args.outFile` under the same folder. Each file records
    ddqn-predicted values and rollout values from different attacker heading
    angles, defender positions and defender heading angles.

EXAMPLES
    python3 colEstError.py -mf <model path>
"""

import argparse
import os
import glob
import numpy as np


def run(args):
  print('\n== Collecting Results ==')
  dataFolder = os.path.join(args.modelFolder, 'data', 'est')
  results = glob.glob(os.path.join(dataFolder, 'estError*'))
  start = len('estError')
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
  for i, resultFile in enumerate(results):
    print('Load from {:s} ...'.format(resultFile), end='\r')
    read_dictionary = np.load(resultFile, allow_pickle='TRUE').item()
    trajLengthTmp = read_dictionary['trajLength']
    ddqnValueTmp = read_dictionary['ddqnValue']
    rolloutValueTmp = read_dictionary['rolloutValue']
    idx = read_dictionary['idx']
    if i == 0:
      maxLength = read_dictionary['maxLength']
      toEnd = read_dictionary['toEnd']
      samples = read_dictionary['samples']
      shapeTmp = (numTest,) + trajLengthTmp.shape
      trajLength = np.empty(shape=shapeTmp, dtype=int)
      ddqnValue = np.empty(shape=shapeTmp, dtype=float)
      rolloutValue = np.empty(shape=shapeTmp, dtype=float)
    trajLength[idx] = trajLengthTmp
    ddqnValue[idx] = ddqnValueTmp
    rolloutValue[idx] = rolloutValueTmp
  print()
  print(ddqnValue.shape)
  finalDict = {}
  finalDict['trajLength'] = trajLength
  finalDict['ddqnValue'] = ddqnValue
  finalDict['rolloutValue'] = rolloutValue
  finalDict['maxLength'] = maxLength
  finalDict['toEnd'] = toEnd
  finalDict['samples'] = samples

  outFolder = os.path.join(args.modelFolder, 'data')
  outFile = os.path.join(outFolder, args.outFile + '.npy')
  print('\n--> Save to {:s} ...'.format(outFile))
  print(finalDict.keys())
  np.save('{:s}'.format(outFile), finalDict)


if __name__ == '__main__':
  # == Arguments ==
  parser = argparse.ArgumentParser()

  # File Parameters
  parser.add_argument(
      "-of", "--outFile", help="output file", default='estError', type=str
  )
  parser.add_argument(
      "-n", "--number", help="#files assumed to obtain", default='225',
      type=int
  )
  parser.add_argument("-mf", "--modelFolder", help="model folder", type=str)

  args = parser.parse_args()
  print("== Arguments ==")
  print(args)

  # == Execution ==
  np.set_printoptions(precision=3, suppress=True, floatmode='fixed')
  run(args)
