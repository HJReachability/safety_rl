"""
Please contact the author(s) of this library if you have any questions.
Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )

The representability of the proposed method.
1. Rollout value vs. (approximately conservative) “optimal value” via sampling
  about 5 open-loop attacker trajectories
2. validate them by simulating a very large number of pursuer (defender)
  trajectories, so that we can be confident that our system truly succeeds not
  only against the “oracle adversary” predicted by the Q-network, but also
  against any possible adversary.
3. Pre-processing:
    we need to run `sim_est_error.py` before to get dataFile
4. This script uses samples in
    `{args.modelFolder}/data/{sampleType}/{args.dataFile}{sampleType}.npy`
    as the initial states of testing rollouts. We then specify the arguments of
    the rollout (see help section of arguments for more details). The rollout
    results are stored in
    `{args.modelFolder}/data/{sampleType}/{args.outFile}{sampleType}{args.index}.npy`.

EXAMPLES
    TN: with specific idx
        w/o: python3 sim_approx_one_state.py -idx <idx> -mf <model path>
        cpf: python3 sim_approx_one_state.py -cpf -idx <idx> -mf <model path>
    FP: add -t 3
    TEST: python3 sim_approx_one_state.py -nps 5 -mf <model path>
    unfinished: python3 sim_approx_one_state.py -df samplesThird
        -of valThirdDict -mf <model path>
"""

import argparse
import time
import os
from warnings import simplefilter
import numpy as np
from multiprocessing import Pool
import gym

from utils.carPEAnalysis import loadEnv, loadAgent
from utils.carPEAnalysis import checkCapture, exhaustiveDefenderSearch
from gym_reachability import gym_reachability  # Custom Gym env.

simplefilter(action='ignore', category=FutureWarning)
np.set_printoptions(precision=3, suppress=True, floatmode='fixed')


def multiExp(firstIdx, args, state, maxLength, numPursuerStep, verbose=False):
  """
    multiExp: simulate defender's trajectories and record the maximum
        value among these trajectories.

    Args:
        firstIdx (int): the first indices in the tuple.
        args (object):
        state (numpy array): initial state.
        maxLength (int, optional): maximal length of the trajectory.
            Defaults to 50.
        numPursuerStep (int, optional): maximal length of action sequence taken
            by the pursuer. Defaults to 10.
        verbose (bool, optional): print information or not. Defaults to False.

    Returns:
        [dict]: records maximal values, the corresponding action indices
            sequence and trajectories of the evader and the pursuer.
    """
  # == ENVIRONMENT ==
  env = loadEnv(args, verbose)
  stateNum = env.state.shape[0]
  actionNum = env.action_space.n
  numActionList = env.numActionList
  device = env.device

  # == AGENT ==
  agent = loadAgent(args, device, stateNum, actionNum, numActionList, verbose)

  # == EXPERIMENT ==
  # print("I'm process", os.getpid())
  actionSet = np.empty(shape=(env.numActionList[1], numPursuerStep), dtype=int)
  for i in range(numPursuerStep):
    actionSet[:, i] = np.arange(env.numActionList[1])

  subSeqLength = numPursuerStep - len(firstIdx)
  shapeTmp = np.ones(subSeqLength, dtype=int) * env.numActionList[1]
  rolloutValue = np.empty(shape=shapeTmp, dtype=float)
  it = np.nditer(rolloutValue, flags=['multi_index'])

  firstFlag = True
  firstCaptureFlag = True
  captureFlag = False
  while not it.finished:
    idx = it.multi_index
    actionIdx = firstIdx + idx
    actionSeq = actionSet[actionIdx, np.arange(numPursuerStep)]
    trajEvader, trajPursuer, minV, _ = exhaustiveDefenderSearch(
        env, agent, state, actionSeq, maxLength
    )
    print(actionSeq, end='\r')
    rolloutValue[idx] = minV
    info = {
        'trajEvader': trajEvader,
        'trajPursuer': trajPursuer,
        'maxminV': minV,
        'maxminIdx': actionIdx
    }
    captureFlagTmp, _ = checkCapture(env, trajEvader, trajPursuer)

    # being captured is consider to be worse than standing around, even
    # though standing around may have higher value (far from the target)
    if firstFlag:
      maxminV = minV
      maxminInfo = info
      firstFlag = False
    elif captureFlagTmp:
      captureFlag = True
      if firstCaptureFlag:
        firstCaptureFlag = False
        maxminV = minV
        maxminInfo = info
      elif minV >= maxminV:
        maxminV = minV
        maxminInfo = info
    elif minV > maxminV and not captureFlag:
      maxminV = minV
      maxminInfo = info
    it.iternext()

  maxminInfo['rolloutValue'] = rolloutValue
  maxminInfo['captureFlag'] = captureFlag
  return maxminInfo


def run(args):
  startTime = time.time()

  # == Getting states to be tested ==
  print('\n== Getting states to be tested ==')
  sampleTypeList = ['TN', 'TP', 'FN', 'FP', 'POS', 'NEG']
  sampleType = sampleTypeList[args.sampleType]
  dataFolder = os.path.join(args.modelFolder, 'data', sampleType)
  dataFile = os.path.join(dataFolder, args.dataFile + sampleType + '.npy')
  print('Load from {:s} ...'.format(dataFile))
  read_dictionary = np.load(dataFile, allow_pickle='TRUE').item()
  states = read_dictionary['states']
  stateIdxList = read_dictionary['idxList']
  state = states[args.index]
  stateIdx = stateIdxList[args.index]
  print(stateIdx, state)

  # == Estimating Approximation Error in Parallel ==
  print("\n== Approximation Error Information ==")
  dictList = []
  # the dimension of the action set is 3
  idxTupleList = [(i, j) for i in range(3) for j in range(3)]
  numTask = len(idxTupleList)
  numProcess = args.numWorker
  numTurn = int(numTask / (numProcess+1e-6)) + 1
  maxLength = args.maxLength
  numPursuerStep = args.numPursuerStep
  for ith in range(numTurn):
    print('{} / {}: '.format(ith + 1, numTurn), end='')
    with Pool(processes=numProcess) as pool:
      startIdx = ith * numProcess
      endIdx = min(numTask, (ith+1) * numProcess)
      print('{:.0f}-{:.0f}'.format(startIdx, endIdx - 1))
      firstIdxList = idxTupleList[startIdx:endIdx]
      numExp = len(firstIdxList)
      argsList = [args] * numExp
      stateList = [state] * numExp
      maxLengthList = [maxLength] * numExp
      numPursuerStepList = [numPursuerStep] * numExp
      verboseList = [False] * numExp
      subDictList = pool.starmap(
          multiExp,
          zip(
              firstIdxList, argsList, stateList, maxLengthList,
              numPursuerStepList, verboseList
          )
      )
      print('\n')
    dictList = dictList + subDictList

  # == COMBINE RESULTS ==
  shapeTmp = np.ones(numPursuerStep, dtype=int) * 3
  rolloutValue = np.empty(shape=shapeTmp, dtype=float)
  cnt = 0
  firstCaptureFlag = True
  captureFlag = False
  for i in range(3):
    for j in range(3):
      info = dictList[cnt]
      rolloutValue[i, j] = info['rolloutValue']
      minV = info['maxminV']
      captureFlagTmp = info['captureFlag']
      if cnt == 0:
        maxminV = minV
        maxminInfo = info
      elif captureFlagTmp:
        captureFlag = True
        if firstCaptureFlag:
          firstCaptureFlag = False
          maxminV = minV
          maxminInfo = info
        elif minV >= maxminV:
          maxminV = minV
          maxminInfo = info
      elif minV > maxminV and not captureFlag:
        maxminV = minV
        maxminInfo = info
      cnt += 1

  endTime = time.time()
  execTime = endTime - startTime
  print('--> Execution time: {:.1f}'.format(execTime))

  finalDict = {}
  finalDict['state'] = state
  finalDict['stateIdx'] = stateIdx
  finalDict['dict'] = maxminInfo
  finalDict['rolloutValue'] = rolloutValue
  finalDict['maxLength'] = maxLength
  finalDict['numPursuerStep'] = numPursuerStep
  finalDict['testIdx'] = args.index
  print(maxminInfo['maxminIdx'], maxminInfo['maxminV'])

  outFile = os.path.join(
      dataFolder, args.outFile + sampleType + str(args.index) + '.npy'
  )
  print('--> Save to {:s} ...'.format(outFile))
  print(finalDict.keys())
  np.save('{:s}'.format(outFile), finalDict)


if __name__ == '__main__':
  # == Arguments ==
  parser = argparse.ArgumentParser()
  # Environment Parameters
  parser.add_argument(
      "-cpf", "--cpf", help="consider pursuer failure", action="store_true"
  )

  # Simulation Parameters
  parser.add_argument(
      "-f", "--forceCPU", help="force PyTorch to use CPU", action="store_true"
  )
  parser.add_argument(
      "-nw", "--numWorker", help="#workers", default=5, type=int
  )
  parser.add_argument(
      "-ml", "--maxLength", help="maximum length of rollout episodes",
      default=50, type=int
  )
  parser.add_argument(
      "-nps", "--numPursuerStep", help="#intervals of pursuer action sequence",
      default=10, type=int
  )
  parser.add_argument(
      "-idx", "--index", help="the index of state in samples", default=0,
      type=int
  )
  parser.add_argument(
      "-t", "--sampleType", help="type of sampled states", default=0, type=int
  )

  # File Parameters
  parser.add_argument(
      "-of", "--outFile", help="output file", default='valDict', type=str
  )
  parser.add_argument("-mf", "--modelFolder", help="model folder", type=str)
  parser.add_argument(
      "-df", "--dataFile", help="samples file", default='samples', type=str
  )

  args = parser.parse_args()
  print("\n== Arguments ==")
  print(args)

  # == Execution ==
  run(args)
