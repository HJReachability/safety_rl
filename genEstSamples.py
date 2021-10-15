"""
Please contact the author(s) of this library if you have any questions.
Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )

Generate samples to compute estimation error.
1. For attacker (evader)'s position, we sampled uniformly from the ring
    between (0.51, 0.99).
2. For defender (pursuer)'s position, we sampled uniformly from the ring
    between (0.01, 0.99).
3. For heading angles, we pick [0, 2*pi/(numSamples-1)] with uniform spacing.
4. This script uses a model under `{args.modelFolder}`, creates a subfolder
    `data` under it and generates `{args.outFile}.npy under this subfolder.

EXAMPLES
    TN: python3 genEstSamples.py -ns 10 -mf <model path>
    FP: python3 genEstSamples.py -ns 10 -mf <model path>
"""

import argparse
import os
from warnings import simplefilter
import numpy as np
from numpy.random import normal, uniform
import matplotlib
import matplotlib.pyplot as plt

simplefilter(action='ignore', category=FutureWarning)
matplotlib.use('Agg')


def uniformSampleRing(inner, outer, numSample, dim=2):
  """Sample state(s) from the ring uniformly.

  Args:
      inner (float): the radius of the inner sphere.
      outer (float): the radius of the outer sphere.
      numSample (int): the number of samples.
      dim (int, optional): the dimension of the sphere. Defaults to 2.

  Returns:
      np.ndarray: states sampled from the ring.
  """
  samples = normal(size=(numSample, dim))
  samples /= np.linalg.norm(samples, axis=1, ord=2).reshape(-1, 1)

  shaping = uniform(low=inner**dim, high=outer**dim, size=(numSample, 1))
  samples *= shaping**(1 / dim)

  return samples


def run(args):
  outFolder = os.path.join(args.modelFolder, 'data')
  os.makedirs(outFolder, exist_ok=True)

  # == Getting states to be tested ==
  print('\n== Getting states to be tested ==')
  numSamplePos = args.numSample**2
  numSample = args.numSample
  delta = .01
  R = 1. - delta
  r = .5 + delta
  samplesAtt = uniformSampleRing(inner=r, outer=R, numSample=numSamplePos)
  samplesDef = uniformSampleRing(inner=delta, outer=R, numSample=numSamplePos)
  thetas = np.linspace(
      start=0, stop=2 * np.pi * (1 - 1/numSample), num=numSample
  )
  samples = [samplesAtt, samplesDef, thetas]
  print(samplesAtt[:5, :])

  fig, axes = plt.subplots(1, 2, figsize=(8, 4))
  s = 5
  tiffany = '#0abab5'
  tmp = np.linspace(0, 2 * np.pi, 100)
  xs = np.cos(tmp)
  ys = np.sin(tmp)
  ax = axes[0]
  for sample in samplesAtt:
    x, y = sample
    ax.scatter(x, y, c=tiffany, s=s, alpha=.2)
  ax.plot(R * xs, R * ys, 'k--')
  ax.plot(r * xs, r * ys, 'm--')
  ax = axes[1]
  for sample in samplesDef:
    x, y = sample
    ax.scatter(x, y, c='y', s=s, alpha=.2)
  ax.plot(R * xs, R * ys, 'k--')
  ax.plot(delta * xs, delta * ys, 'r--')
  figFile = outFolder + 'samples.png'
  fig.savefig(figFile)
  plt.close()

  finalDict = {}
  finalDict['samples'] = samples
  outFile = os.path.join(outFolder, args.outFile + '.npy')
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
      "-ns", "--numSample", help="#samples", default=15, type=int
  )

  # File Parameters
  parser.add_argument(
      "-of", "--outFile", help="output file", default='samplesEst', type=str
  )
  parser.add_argument("-mf", "--modelFolder", help="model folder", type=str)

  args = parser.parse_args()
  print("== Arguments ==")
  print(args)

  # == Execution ==
  np.random.seed(args.randomSeed)
  np.set_printoptions(precision=3, suppress=True, floatmode='fixed')
  run(args)
