"""
Please contact the author(s) of this library if you have any questions.
Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )

1. We want to evaluate how well we learned from the data.
2. We compare the DDQN-predicted value vs. the rollout value by DDQN-induced
    policies.
3. Pre-processing:
    we need to run `genEstSamples.py` beforehand to get state samples
4. This script uses samples from `{args.modelFolder}/data/samplesEst.npy` as
    the initial states of testing rollouts. We then specify the arguments of
    the rollout (see help section of arguments for more details). The rollout
    results are stored in
    `{args.modelFolder}/data/est/{args.outFile}{args.index}.npy`.

EXAMPLES
    test:
        python3 sim_est_error.py -of tmp
    default:
        python3 sim_est_error.py
    toEnd:
        python3 sim_est_error.py -te
    consider pursuer failure set:
        python3 sim_est_error.py -cpf -mf <model_path>
"""

from warnings import simplefilter
import gym
import numpy as np
import torch
import os
import time
import argparse
from multiprocessing import Pool

from utils.carPEAnalysis import loadAgent, loadEnv
from gym_reachability import gym_reachability  # Custom Gym env.

simplefilter(action='ignore', category=FutureWarning)


def multiExp(
    args, posAtt, thetaAttIdx, samplesDef, thetas, maxLength, toEnd,
    verbose=False
):

  np.set_printoptions(precision=3, suppress=True, floatmode='fixed')
  # == ENVIRONMENT ==
  env = loadEnv(args, verbose)
  stateNum = env.state.shape[0]
  actionNum = env.action_space.n
  numActionList = env.numActionList
  device = env.device

  # == AGENT ==
  agent = loadAgent(args, device, stateNum, actionNum, numActionList, verbose)

  # print("I'm process", os.getpid())
  numTheta = thetas.shape[0]
  numDef = samplesDef.shape[0]
  shapeTmp = np.array([numDef, numTheta])
  trajLength = np.empty(shape=shapeTmp, dtype=int)
  ddqnValue = np.empty(shape=shapeTmp, dtype=float)
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
    traj, _, _, minV, _ = env.simulate_one_trajectory(
        agent.Q_network, T=maxLength, state=state, toEnd=toEnd
    )
    trajLength[idx] = traj.shape[0]
    rolloutValue[idx] = minV

    agent.Q_network.eval()
    state = torch.from_numpy(state).float().to(agent.device)
    state_action_values = agent.Q_network(state)
    Q_mtx = state_action_values.detach().cpu().reshape(
        agent.numActionList[0], agent.numActionList[1]
    )
    pursuerValues, _ = Q_mtx.max(dim=1)
    minmaxValue, _ = pursuerValues.min(dim=0)
    ddqnValue[idx] = minmaxValue

    it.iternext()

  carPEDict = {}
  carPEDict['trajLength'] = trajLength
  carPEDict['ddqnValue'] = ddqnValue
  carPEDict['rolloutValue'] = rolloutValue
  carPEDict['thetaAttIdx'] = thetaAttIdx
  return carPEDict


def run(args):
  startTime = time.time()
  dataFolder = os.path.join(args.modelFolder, 'data')
  dataFile = os.path.join(dataFolder, 'samplesEst.npy')
  print('Load from {:s} ...'.format(dataFile))
  read_dictionary = np.load(dataFile, allow_pickle='TRUE').item()
  samples = read_dictionary['samples']
  [samplesAtt, samplesDef, thetas] = samples
  numTheta = thetas.shape[0]
  numDef = samplesDef.shape[0]
  posAtt = samplesAtt[args.index, :]
  print(posAtt, numTheta, numDef)

  # == ROLLOUT RESULTS ==
  print("\n== Estimation Error Information ==")
  maxLength = args.maxLength
  toEnd = args.toEnd
  carPESubDictList = []
  numThread = args.numWorker
  numTest = thetas.shape[0]
  numTurn = int(numTest / (numThread+1e-6)) + 1
  for ith in range(numTurn):
    print('{} / {}'.format(ith + 1, numTurn), end=': ')
    with Pool(processes=numThread) as pool:
      startIdx = ith * numThread
      endIdx = min(numTest, (ith+1) * numThread)
      print('{:.0f}-{:.0f}'.format(startIdx, endIdx - 1))
      thetaIdxAttList = [j for j in range(startIdx, endIdx)]
      numExp = len(thetaIdxAttList)
      posAttList = [posAtt] * numExp
      argsList = [args] * numExp
      samplesDefList = [samplesDef] * numExp
      thetasList = [thetas] * numExp
      maxLengthList = [maxLength] * numExp
      toEndList = [toEnd] * numExp
      verboseList = [False] * numExp

      carPESubDict_i = pool.starmap(
          multiExp,
          zip(
              argsList, posAttList, thetaIdxAttList, samplesDefList,
              thetasList, maxLengthList, toEndList, verboseList
          )
      )
    carPESubDictList = carPESubDictList + carPESubDict_i

  # == COMBINE RESULTS ==
  shapeTmp = np.array([numTheta, numDef, numTheta])
  trajLength = np.empty(shape=shapeTmp, dtype=int)
  ddqnValue = np.empty(shape=shapeTmp, dtype=float)
  rolloutValue = np.empty(shape=shapeTmp, dtype=float)

  for i, carPESubDict_i in enumerate(carPESubDictList):
    thetaAttIdx = carPESubDict_i['thetaAttIdx']
    trajLength[thetaAttIdx, :, :] = carPESubDict_i['trajLength']
    ddqnValue[thetaAttIdx, :, :] = carPESubDict_i['ddqnValue']
    rolloutValue[thetaAttIdx, :, :] = carPESubDict_i['rolloutValue']
  print(ddqnValue.shape)

  endTime = time.time()
  execTime = endTime - startTime
  print('--> Execution time: {:.1f}'.format(execTime))

  carPEDict = {}
  carPEDict['maxLength'] = maxLength
  carPEDict['toEnd'] = toEnd
  carPEDict['samples'] = samples
  carPEDict['idx'] = args.index
  carPEDict['trajLength'] = trajLength
  carPEDict['ddqnValue'] = ddqnValue
  carPEDict['rolloutValue'] = rolloutValue

  outFolder = os.path.join(args.modelFolder, 'data', 'est')
  os.makedirs(outFolder, exist_ok=True)
  outFile = os.path.join(outFolder, args.outFile + str(args.index) + '.npy')
  np.save('{:s}'.format(outFile), carPEDict)
  print('--> Save to {:s} ...'.format(outFile))


if __name__ == '__main__':
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
      "-te", "--toEnd",
      help="continue the rollout until both cars cross the boundary",
      action="store_true"
  )
  parser.add_argument(
      "-ml", "--maxLength", help="maximum length of rollout episodes",
      default=150, type=int
  )
  parser.add_argument(
      "-nw", "--numWorker", help="#workers", default=6, type=int
  )
  parser.add_argument(
      "-idx", "--index", help="the index of state in samples", default=0,
      type=int
  )

  # File Parameters
  parser.add_argument(
      "-of", "--outFile", help="output file", default='estError', type=str
  )
  parser.add_argument("-mf", "--modelFolder", help="model folder", type=str)

  args = parser.parse_args()
  print("\n== Arguments ==")
  print(args)

  # == Execution ==
  np.set_printoptions(precision=3, suppress=True, floatmode='fixed')
  run(args)
