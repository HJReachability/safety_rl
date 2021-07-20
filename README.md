# Safety and Liveness Guarantees throughReach-Avoid Reinforcement Learning
[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

This is the companion code to RSS 2021 paper:
Kai-Chieh Hsu\*, Vicenç Rubies-Royo\*, Claire J. Tomlin and Jaime F. Fisac,
''[Safety and Liveness Guarantees through Reach-Avoid Reinforcement Learning](https://roboticsconference.org/program/papers/077/)'',
*Proceedings of Robotics: Science and Systems (RSS)*, Jul. 2021.

We implemented reach-avoid Q-learning and evaluated our proposed framework on a
range of nonlinear systems.

## Dependencies
1. PyTorch
2. gym
3. Box2D
4. matplotlib
5. numpy

## Running
1. for lunar lander in Figure 1
```python
```
2. for point mass in Figure 2
```python
    python3 sim_naive.py -w -sf -of scratch -a -g 0.9 -mu 12000000 -cp 600000 -ut 20 -n anneal
```
3. for point mass in Figure 4
```python
    python3 sim_show.py -sf -of scratch -n 9999
```
4. for Dubins car in Figure 5
```python
    python3 sim_car_one.py -sf -of scratch -w -wi 5000 -g 0.9999 -n 9999
```
5. for Dubins car: Attack-Defense game in Figure 7
```python
    python3 sim_car_pe.py -sf -of scratch -w -wi 30000 -g 0.9999 -n 9999
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