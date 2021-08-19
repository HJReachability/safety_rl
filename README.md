# Safety and Liveness Guarantees through Reach-Avoid Reinforcement Learning
[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)
[![Python 3.8](https://img.shields.io/badge/python-3.8-blue)](https://www.python.org/downloads/)

We implemented a model-free reach-avoid reinforcement learning (RARL) to guarantee safety and liveness and evaluated our proposed algorithm on a range of nonlinear systems.
RARL is primarily developed by Kai-Chieh Hsu, a PhD student in the [Safe Robotics Lab](https://www.saferobotics.org/), and Vicenç Rubies-Royo, a postdoc in the [Hybrid Systems Lab](http://hybrid.eecs.berkeley.edu).


This is the companion code to our [RSS 2021 paper](https://roboticsconference.org/program/papers/077/), where you can find the theoretical properties of the proposed algorithm as well as the implementation details.
All experiments are included as examples in this repository and you can replicate the results by commands in the section II.
With some simple modification, you can replicate the results in [ICRA 19 paper](https://ieeexplore.ieee.org/document/8794107), where they consider reachability/safety only.

This tool is for arbitrary environments.
You just need to add your environment under `gym_reachability` and register through the standard method in `gym`.
You can refer to some examples provided here.
This tools learns the reach-avoid set by trial-and-error, so it is not a safe learning algorithm.
However, it can be used in conjunction with safe learning scheme, such as "shielding".

## I. Dependencies
If you are using anaconda to control packages, you can use one of the following
command to create an identical environment with the specification file:
```
conda create --name <myenv> --file doc/spec-mac.txt
conda create --name <myenv> --file doc/spec-linux.txt
```
Otherwise, you can install the following packages manually
1. numpy=1.21.1
2. pytorch=1.9.0
3. gym=0.18.0
4. scipy=1.7.0
5. matplotlib=3.4.2
6. box2d-py=2.3.8
7. shapely=1.7.1

## II. Replicating the result in RSS paper
Each script will automatically generate a folder under `experiments/` containing visulaizations during the training and trained models.
In addition, there is `train.pkl`, which consists of
+ training loss 
+ training accuracy
+ trajectory rollout outcome starting from a grid of states
+ action taken from a grid of states

1. for lunar lander in Figure 1
```python
    python3 sim_lunar_lander.py -sf
```
2. for point mass in Figure 2
```python
    python3 sim_naive.py -w -sf -a -g 0.9 -mu 12000000 -cp 600000 -ut 20 -n anneal
```
3. for point mass in Figure 4
```python
    python3 sim_show.py -sf -g 0.9999 -n 9999
```
4. for Dubins car in Figure 5
```python
    python3 sim_car_one.py -sf -w -wi 5000 -g 0.9999 -n 9999
```
5. for Dubins car: Attack-Defense game in Figure 7
```python
    python3 sim_car_pe.py -sf -w -wi 30000 -g 0.9999 -n 9999
```

## Paper Citation
If you use this code or find it helpful, please consider cite this paper with:
```
@INPROCEEDINGS{hsu2021safety,
    AUTHOR    = {Kai-Chieh Hsu$^*$ and Vicenç Rubies-Royo$^*$ and Claire J. Tomlin and Jaime F. Fisac},
    TITLE     = {{Safety and Liveness Guarantees through Reach-Avoid Reinforcement Learning}},
    BOOKTITLE = {Proceedings of Robotics: Science and Systems},
    YEAR      = {2021},
    ADDRESS   = {Virtual},
    MONTH     = {July},
    DOI       = {10.15607/RSS.2021.XVII.077}
}
```