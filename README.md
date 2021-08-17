# Safety and Liveness Guarantees throughReach-Avoid Reinforcement Learning
[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

This is the companion code to RSS 2021 paper:
Kai-Chieh Hsu\*, Vicenç Rubies-Royo\*, Claire J. Tomlin and Jaime F. Fisac,
''[Safety and Liveness Guarantees through Reach-Avoid Reinforcement Learning](https://roboticsconference.org/program/papers/077/)'',
*Proceedings of Robotics: Science and Systems (RSS)*, Jul. 2021.

We implemented reach-avoid Q-learning and evaluated our proposed framework on a
range of nonlinear systems.

## Dependencies
If you are using anaconda to control packages, you can use one of the following
command to create an identical environment with the specification file:
```
conda create --name <myenv> --file doc/spec-mac.txt
conda create --name <myenv> --file doc/spec-linux.txt
```
Otherwise, you can install the following packages manually
1. python=3.8.10
2. numpy=1.21.1
3. pytorch=1.9.0
4. gym=0.18.0
5. scipy=1.7.0
6. matplotlib=3.4.2
7. box2d-py=2.3.8
8. shapely=1.7.1

## Running
1. for lunar lander in Figure 1
```python
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