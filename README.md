# Safety and Liveness Guarantees through Reach-Avoid Reinforcement Learning
[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)
[![Python 3.8](https://img.shields.io/badge/python-3.8-blue)](https://www.python.org/downloads/)

This repository implements a model-free reach-avoid reinforcement learning (RARL) to guarantee safety and liveness, and additionally contains example uses and benchmark evaluations of the proposed algorithm on a range of nonlinear systems.
RARL is primarily developed by Kai-Chieh Hsu, a PhD student in the [Safe Robotics Lab](https://saferobotics.princeton.edu), and Vicenç Rubies-Royo, a postdoc in the [Hybrid Systems Lab](http://hybrid.eecs.berkeley.edu).


The repository also serves as the companion code to our [RSS 2021 paper](https://roboticsconference.org/program/papers/077/), where you can find the theoretical properties of the proposed algorithm as well as the implementation details.
All experiments in the paper are included as examples in this repository, and you can replicate the results by using the commands described in Section II below.
With some simple modification, you can replicate the results in the preceding [ICRA 19 paper](https://ieeexplore.ieee.org/document/8794107), which considers the special case of reachability/safety only.

This tool is designed to work for arbitrary reinforcement learning environments, and uses two scalar signals (a _target margin_ and a _safety margin_) rather than a single scalar _reward_ signal.
You just need to add your environment under `gym_reachability` and register through the standard method in `gym`.
You can refer to some examples provided here.
This tool learns the reach-avoid set by trial-and-error interactions with the environment, so it is not _in itself_ a safe learning algorithm.
However, it can be used in conjunction with an existing safe learning scheme, such as "shielding", to enable learning with safety guarantees (see Script 4 below as well as Section IV.B in the [RSS 2021 paper](https://roboticsconference.org/program/papers/077/) for an example).

## I. Dependencies
If you are using anaconda to control packages, you can use one of the following
command to create an identical environment with the specification file:
```
conda create --name <myenv> --file doc/spec-mac.txt
conda create --name <myenv> --file doc/spec-linux.txt
```
Otherwise, you can install the following packages manually:
1. numpy=1.21.1
2. pytorch=1.9.0
3. gym=0.18.0
4. scipy=1.7.0
5. matplotlib=3.4.2
6. box2d-py=2.3.8
7. shapely=1.7.1

## II. Replicating the results in the [RSS 2021 paper](https://roboticsconference.org/program/papers/077/)
Each script will automatically generate a folder under `experiments/` containing visualizations of the the training process and the weights of trained model.
In addition, the script will generate a `train.pkl` file, which contains the following:
+ training loss 
+ training accuracy
+ trajectory rollout outcome starting from a grid of states
+ action taken from a grid of states

1. Lunar lander in Figure 1
```shell
    python3 sim_lunar_lander.py -sf
```
2. Point object in Figure 2
```shell
    python3 sim_naive.py -w -sf -a -g 0.9 -mu 12000000 -cp 600000 -ut 20 -n anneal
```
3. Point object in Figure 4
```shell
    python3 sim_show.py -sf -g 0.9999 -n 9999
```
4. Dubins car in Figure 5
```shell
    python3 sim_car_one.py -sf -w -wi 5000 -g 0.9999 -n 9999
```
5. Dubins car (attack-defense game) in Figure 7 (Section IV.D):
```shell
    python3 sim_car_pe.py -sf -w -wi 30000 -g 0.9999 -n 9999
```

## Paper Citation
If you use this code or find it helpful, please consider citing the companion [RSS 2021 paper](https://roboticsconference.org/program/papers/077/) as:
```
@INPROCEEDINGS{hsu2021safety,
    AUTHOR    = {Kai-Chieh Hsu$^*$ and Vicenç Rubies-Royo$^*$ and Claire J. Tomlin and Jaime F. Fisac},
    TITLE     = {Safety and Liveness Guarantees through Reach-Avoid Reinforcement Learning},
    BOOKTITLE = {Proceedings of Robotics: Science and Systems},
    YEAR      = {2021},
    ADDRESS   = {Virtual},
    MONTH     = {July},
    DOI       = {10.15607/RSS.2021.XVII.077}
}
```