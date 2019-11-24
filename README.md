# safety_rl

Jaime F. Fisac*, Neil F. Lugovoy*, Vicenc¸ Rubies-Royo, Shromona Ghosh, Claire J. Tomlin, "[Bridging Hamilton-Jacobi Safety Analysis and Reinforcement Learning](TODO arXiv link here)", ICRA 2019. (* Equal Contribution)

This repository contains the code necessary to run all the experiments in [ICRA 2019] in addition to core functions that can be either directly invoked by other reinforcement learning code or used as illustrative examples of how to adapt typical reinforcement learning methods to work with the Safety Bellman Equation.


## Some quick notes:

1. If you want to use the Safety Bellman Equation with your own reinforcement learning algorithm you will need the functions `sbe_backup` and/or `sbe_outcome` in the `utils.py` file. The former computes the backup for value function learning (Q-learning, DQN, SAC, etc.) from equation (7) in [ICRA19] and the latter computes the outcome of a trajectory from equation (8) in [ICRA19] used in policy optimization methods (policy gradient, TRPO, PPO, etc.).
2. The DQN and Policy Gradient algorithms are modified versions of the implementation from the [Ray library](https://github.com/ray-project/ray). Modifications are enclosed by a line of hashtags marking the start and end of the changes.
3. The Soft Actor Critic algorithm is a modified version from the implementation in [Spinning Up](https://github.com/openai/spinningup) (Ray does not have a working implementation currently). Similarly hashtags indicate changes.
4. The inverted pendulum and lunar lander environments are subclasses of environments implemented in [Open AI's Gym](https://github.com/openai/gym). The major changes from the parent classes are documented in the files.
5. The q learning algorithm is loosely based on [Denny Britz's implementation](https://github.com/dennybritz/reinforcement-learning/blob/master/TD/Q-Learning%20Solution.ipynb).

## Installation:

```
git clone safety_rl
cd safety_rl
pip install -e .
```
 
## Dependencies:

`'numpy', 'matplotlib', 'spinup', 'tensorflow==1.9.0', 'gym', 'scipy', 'ray==0.7.3', 'pandas'`
The `ray` and `tensorflow` dependencies are locked at the current version because the APIs change rapidly and that can break existing code.
 
## Run Experiments:

All of the experiments presented in the paper are scripts in the `experiments` folder. To run an experiment simply run the script with the expeirment you are interested in. All of the experiments will save the data and produce a figure related to the experiment. The scripts are all heavily commented.  

## Proof that l must be computed before environment steps:

Consider a simple environment with two states 0 and 1 with one action and transitions 0 -> 1 and 1 -> 1. If we set gamma = 0 then our value function should be V(0) = l(0) and V(1) = l(1). If l is computed after the environment steps we will only give l(1) to our learning algorithm thus V can never be correct.