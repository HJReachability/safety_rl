"""
Please contact the author(s) of this library if you have any questions.
Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )

We want to evaluate how well we learned from the data.
We compare the DDQN-predicted value vs. the rollout value by DDQN-induced
policies.

EXECUTION TIME
    Setting:
        101 samples per dimension, 6 workers, maxLength = 100
        NN: 1-layer with 100 neurons per leayer
    Results
        4000 seconds (toEnd = True)
        1000 seconds (toEnd = False)

EXAMPLES
    toEnd, low turning rate:
        python3 sim_est_error_single.py -te -l -of carOneLow
          -mf models/store_best/car/RA/small/tanh
    TF, high turning rate:
        python3 sim_est_error_single.py -of carOneHighTF
          -mf models/store_best/car/RA/big/tanh
    Array:
        python3 sim_est_error_single.py -sf
          -mf scratch/car/highA/highA-0-2021-02-10-21_08
"""

import os
import time
from warnings import simplefilter
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
from multiprocessing import Pool

from utils.carOneAnalysis import loadAgent, loadEnv

simplefilter(action='ignore', category=FutureWarning)
matplotlib.use('Agg')


def multiExp(env, agent, samples, firstIdx, numSample, maxLength, toEnd):
    freeCoordNum = 2
    shapeTmp = np.ones(freeCoordNum, dtype=int) * numSample
    rolloutResult = np.empty(shape=shapeTmp, dtype=int)
    trajLength = np.empty(shape=shapeTmp, dtype=int)
    ddqnValue = np.empty(shape=shapeTmp, dtype=float)
    rolloutValue = np.empty(shape=shapeTmp, dtype=float)
    it = np.nditer(rolloutResult, flags=['multi_index'])

    while not it.finished:
        idx = it.multi_index
        stateIdx = idx + (firstIdx,)
        print(stateIdx, end='\r')
        state = samples[stateIdx, np.arange(3)]
        traj, result, minV, _ = env.simulate_one_trajectory(
            agent.Q_network, T=maxLength, state=state, toEnd=toEnd)
        trajLength[idx] = traj.shape[0]
        rolloutResult[idx] = result  # result \in { 1, -1}
        rolloutValue[idx] = minV

        agent.Q_network.eval()
        stateTensor = torch.from_numpy(state).float().to(agent.device)
        state_action_values = agent.Q_network(stateTensor)
        Q_vec = state_action_values.detach().cpu().reshape(-1)
        ddqnValue[idx] = Q_vec.min().item()

        it.iternext()

    carOneDict = {}
    carOneDict['rolloutResult'] = rolloutResult
    carOneDict['trajLength'] = trajLength
    carOneDict['ddqnValue'] = ddqnValue
    carOneDict['rolloutValue'] = rolloutValue

    print()
    return carOneDict


def run(args):
    startTime = time.time()

    # == ENVIRONMENT ==
    env = loadEnv(args)
    stateNum = env.state.shape[0]
    actionNum = env.action_space.n
    actionList = np.arange(actionNum)
    device = env.device

    # == AGENT ==
    agent = loadAgent(args, device, stateNum, actionNum, actionList)

    # == ROLLOUT RESULTS ==
    print("\n== Estimation Error Information ==")
    np.set_printoptions(precision=2, suppress=True)
    numSample = args.numSample
    bounds = np.array([
        [-1.1, 1.1], [-1.1, 1.1], [0, 2 * np.pi * (1 - 1 / numSample)]])
    samples = np.linspace(start=bounds[:, 0], stop=bounds[:, 1], num=numSample)

    maxLength = args.maxLength
    toEnd = args.toEnd
    carPESubDictList = []
    numThread = args.numWorker
    numTurn = int(numSample / (numThread + 1e-6)) + 1
    for ith in range(numTurn):
        print('{} / {}'.format(ith + 1, numTurn))
        with Pool(processes=numThread) as pool:
            startIdx = ith * numThread
            endIdx = min(numSample, (ith + 1) * numThread)
            firstIdxList = list(range(startIdx, endIdx))
            print(firstIdxList)
            numExp = len(firstIdxList)
            envList = [env] * numExp
            agentList = [agent] * numExp
            samplesList = [samples] * numExp
            numSampleList = [numSample] * numExp
            maxLengthList = [maxLength] * numExp
            toEndList = [toEnd] * numExp

            carPESubDict_i = pool.starmap(multiExp, zip(
                envList, agentList, samplesList, firstIdxList, numSampleList,
                maxLengthList, toEndList))
        carPESubDictList = carPESubDictList + carPESubDict_i

    # == COMBINE RESULTS ==
    shapeTmp = np.ones(3, dtype=int) * numSample
    rolloutResult = np.empty(shape=shapeTmp, dtype=int)
    trajLength = np.empty(shape=shapeTmp, dtype=int)
    ddqnValue = np.empty(shape=shapeTmp, dtype=float)
    rolloutValue = np.empty(shape=shapeTmp, dtype=float)

    for i, carPESubDict_i in enumerate(carPESubDictList):
        rolloutResult[:, :, i] = carPESubDict_i['rolloutResult']
        trajLength[:, :, i] = carPESubDict_i['trajLength']
        ddqnValue[:, :, i] = carPESubDict_i['ddqnValue']
        rolloutValue[:, :, i] = carPESubDict_i['rolloutValue']

    endTime = time.time()
    execTime = endTime - startTime
    print('--> Execution time: {:.1f}'.format(execTime))

    carOneDict = {}
    carOneDict['numSample'] = numSample
    carOneDict['maxLength'] = maxLength
    carOneDict['toEnd'] = toEnd
    carOneDict['rolloutResult'] = rolloutResult
    carOneDict['trajLength'] = trajLength
    carOneDict['ddqnValue'] = ddqnValue
    carOneDict['rolloutValue'] = rolloutValue
    carOneDict['samples'] = samples
    carOneDict['execTime'] = execTime

    outFolder = os.path.join(args.modelFolder, 'data')
    os.makedirs(outFolder, exist_ok=True)
    outFile = outFolder + args.outFile + '.npy'
    np.save('{:s}'.format(outFile), carOneDict)
    print('--> Save to {:s} ...'.format(outFile))

    # == Plot RA Set of the analytic solution and approximate value function ==
    if args.plotFigure or args.storeFigure:
        fig, axes = plt.subplots(
            1, 2, figsize=(8, 4), sharex=True, sharey=True)
        axStyle = np.array([-1.1, 1.1, -1.1, 1.1])

        ax = axes[0]
        ax.imshow(
            carOneDict['ddqnValue'][:, :, 0].T,
            interpolation='none', extent=axStyle, origin="lower",
            cmap='seismic', vmin=-1, vmax=1
        )
        env.plot_reach_avoid_set(ax, c='g', lw=3, orientation=0)
        ax.set_xlabel(r'$\theta={:.0f}^\circ$'.format(0), fontsize=24)

        # == Rollout ==
        ax = axes[1]
        ax.imshow(
            carOneDict['rolloutValue'][:, :, 0].T <= 0,
            interpolation='none', extent=axStyle, origin="lower",
            cmap='coolwarm', vmin=0, vmax=1
        )
        ax.set_xlabel('Rollout', fontsize=24)

        # == Formatting ==
        for ax in axes:
            env.plot_target_failure_set(ax=ax)
            env.plot_formatting(ax=ax)

        if args.storeFigure:
            figureFolder = '{:s}/figure/'.format(args.modelFolder)
            os.makedirs(figureFolder, exist_ok=True)
            plt.savefig('{:s}rollout.png'.format(figureFolder))
        if args.plotFigure:
            plt.show()
            plt.pause(0.001)
            plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Environment Parameters
    parser.add_argument("-low", "--low", help="lowOmega", action="store_true")

    # Simulation Parameters
    parser.add_argument(
        "-te", "--toEnd", help="stop until boundary", action="store_true")
    parser.add_argument(
        "-f", "--forceCPU", help="force CPU", action="store_true")
    parser.add_argument(
        "-ns", "--numSample", help="#samples", default=101, type=int)
    parser.add_argument(
        "-nw", "--numWorker", help="#workers", default=5, type=int)
    parser.add_argument(
        "-ml", "--maxLength", help="max length", default=100, type=int)

    # File Parameters
    parser.add_argument(
        "-pf", "--plotFigure", help="plot figures", action="store_true")
    parser.add_argument(
        "-sf", "--storeFigure", help="store figures", action="store_true")
    parser.add_argument(
        "-of", "--outFile", help="output file", default='estError', type=str)
    parser.add_argument(
        "-mf", "--modelFolder", help="model folder",
        default='models/store_best/car/RA/big/tanh', type=str
    )

    args = parser.parse_args()
    print("\n== Arguments ==")
    print(args)

    # == Execution ==
    run(args)
