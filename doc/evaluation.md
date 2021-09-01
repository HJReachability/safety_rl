# Evaluation Flow for Pursuit-Evasion Game between Two Dubins Cars
Here we evaluate two kinds of errors:
* Estimation Error
* Approximation Error

## I. Estimation Error
1. We want to evaluate how well we learned from the data.
2. We compare the DDQN-predicted value vs. the value achieved by rolling out the DDQN policy ("rollout value").

### I-A. Get samples to evaluate the estimation error.
```
python3 genEstSamples.py -ns <numSample> -mf <modelPath>
``` 
* `- rnd seed`: use random seed
* sample attacker positions uniformly in the ring between constraint set and
the target set for `numSample^2` samples
* sample defender positions uniformly in the constraint set for `numSample^2`
samples
* get `numSample` heading angles in `[0, 2*pi/(numSample-1)]` with uniform
spacing
* store the samples in the format of `[posAtt, posDef, thetas]`
* create a directory `data` under `modelPath` and save `samplesEst.npy` under
this directory

### I-B. Run simulations.
```
python3 sim_est_error.py -idx <index> -mf <modelPath>
```
* `-te`: simulate the trajectory until the agent goes beyond the boundary
* `-cpf`: if the environment considers the failure set of the defender
* `-ml maxLength`: the maximum length of the trajectory
* `-nw numWorker`: the number of workers
* get `samples` from `modelPath/data/samplesEst.npy`
* use `index` to pick attacker position and simulate results for different
attacker heading angles, defender positions and defender heading angles based
on the `samples`
* create a directory `est` under `modelPath/data` and save `estError{$idx}.npy`
under this directory

### I-C. Collect results.
```
python3 colEstError.py -mf <modelPath>
```
* collect estimation error from all files under `modelPath/data/est`
* save `estError.npy` under `modelPath/data`


## II. Approximation Error
1. Validate the evader's policy (attacker) by simulating a very large number of pursuer's  (defender) trajectories, so that we can be confident that our system truly succeeds not only against the “oracle adversary” predicted by the Q-network, but also against any possible adversary.

### II-A. Get samples to evaluate the approximation error.
```
python3 genValSamples.py -nt <numTest> -t <sampleType> -mf <modelPath>
```
* `-t sampleType`, SIX sample types:
    * 0 - 6 corresponds to ['TN', 'TP', 'FN', 'FP', 'POS', 'NEG', 'ALL']
    * 'POS'/'NEG' refer to +/- rollout values.
* `-rnd seed`: use random seed
* get `samples`, `rolloutValue` and `ddqnValue` from `modelPath/data/estError.npy`
* use `rolloutValue` and `ddqnValue` to find the indices belong to the specific
`sampleType` and sample uniformly from these indices for `numTest` times
* create a directory `sampleType` (str version) under `modelPath/data` and save
`samples{$sampleType}.npy`

### II-B. Run simulations.
```
python3 sim_approx_defender.py -t <sampleType> -idx <index> -mf <modelPath>
```
* `-cpf`: if the environment considers the failure set of the defender
* `-ml maxLength`: the maximum length of the trajectory
* `-nps numPursuerStep`: the number of chunks in defender's action sequence
* `-nw numWorker`: the number of workers
* get `states` from `modelPath/data/{$sampleType}/samples{$sampleType}.npy`
* use `index` to pick the state from `states` and simulate under different
defender action sequences (exhaustive way)
* save `valDict{$sampleType}{$idx}.npy` under `modelPath/data/{$sampleType}`

### II-C. Collect Results
```
python3 colValResult.py -t <sampleType> -mf <modelPath>
```
* collect validation results from all files under `modelPath/data/{$sampleType}`
* save `valDict{$sampleType}.npy` under `modelPath/data`