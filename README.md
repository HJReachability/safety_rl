# Safety and Liveness Guarantees throughReach-Avoid Reinforcement Learning
This is the companion code to RSS 2021 paper:
Kai-Chieh Hsu\*, Vicenç Rubies-Royo\*, Claire J. Tomlin and Jaime F. Fisac,
''Safety and Liveness Guarantees through Reach-Avoid Reinforcement Learning'',
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
1. for Figure 1
```python
    python3 sim_naive.py -sf -of scratch -a -g 0.9 -mu 12000000 -cp 600000 -ut 20 -n anneal
```
2. for Figure 3
```python
    python3 sim_show.py -sf -of scratch -n 9999
```
3. for Dubins car
```python
    python3 sim_car_one.py -sf -of scratch -w -wi 5000 -g 0.9999 -n 9999
```
4. for lunar lander
```python
```
5. for Dubins car attack-defense game
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