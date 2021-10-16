# safety_rl

This repository contains the code necessary to run all the experiments in [ICRA19] in addition to core functions that can be either directly invoked by other reinforcement learning code or used as illustrative examples of how to adapt typical reinforcement learning methods to work with the Safety Bellman Equation.

*[ICRA19]* J. F. Fisac\*, N. Lugovoy\*, V. Rubies-Royo, S. Ghosh, and C. J. Tomlin. “[Bridging Hamilton-Jacobi Safety Analysis and Reinforcement Learning](https://ieeexplore.ieee.org/document/8794107),” IEEE International Conference on Robotics and Automation (ICRA), 2019.


## Some quick notes:

1. If you want to use the Safety Bellman Equation with your own reinforcement learning algorithm you will need the functions `sbe_backup` and/or `sbe_outcome` in the `utils.py` file. The former computes the backup for value function learning (Q-learning, DQN, SAC, etc.) from equation (7) in [ICRA19] and the latter computes the outcome of a trajectory from equation (8) in [ICRA19] used in policy optimization methods (policy gradient, TRPO, PPO, etc.).
2. The DQN and Policy Gradient algorithms are modified versions of the implementation from the [Ray library](https://github.com/ray-project/ray). Modifications are enclosed by a line of hashtags marking the start and end of the changes.
3. The Soft Actor Critic algorithm is a modified version from the implementation in [Spinning Up](https://github.com/openai/spinningup) (Ray does not have a working implementation currently). Similarly hashtags indicate changes.
4. The inverted pendulum and lunar lander environments are subclasses of environments implemented in [Open AI's Gym](https://github.com/openai/gym). The major changes from the parent classes are documented in the files.
5. The Q-learning algorithm is loosely based on [Denny Britz's implementation](https://github.com/dennybritz/reinforcement-learning/blob/master/TD/Q-Learning%20Solution.ipynb).
6. The numerical safety value function used as ground truth in the cart-pole experiment is computed in MATLAB using the [Level Set Toolbox](https://www.cs.ubc.ca/~mitchell/ToolboxLS/), by Prof. Ian Mitchell at the University of British Columbia.

## Installation:

```
git clone safety_rl
cd safety_rl
pip install -e .
```

In order to run Gym experiments requiring MuJoCo, a MuJoCo license is necessary.
Please go to https://www.mujoco.org to download the MuJoCo software and acquire
a free or paid license. Detailed instructions can be found [here]
(https://github.com/openai/mujoco-py#install-mujoco).

Note:As of June 2020, there are known issues with installing MuJoCo 2.0 on
MacOS Catalina (reported [here]())

## Dependencies:

`'numpy', 'matplotlib', 'spinup', 'tensorflow==1.9.0', 'gym', 'scipy', 'requests', 'ray==0.7.3', 'pandas', 'opencv-python', 'psutil', 'lz4', 'Box2D', 'mujoco-py'`
The versions for `ray` and `tensorflow` are locked because the APIs tend to change rapidly and that could break existing code.

## Run Experiments:

All of the experiments presented in the paper are scripts in the `experiments` folder. To replicate a particular experiment simply run the script. All of the experiments will save the data and produce a figure. The scripts are thoroughly commented.
