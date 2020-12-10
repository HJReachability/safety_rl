#!/usr/bin/env python
# coding: utf-8


from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

from gym_reachability import gym_reachability  # Custom Gym env.
import gym
import numpy as np

# == ENVIRONMENT ==
env_name = "two_player_pursuit_evasion_lunar_lander-v0"
env = gym.make(env_name)

env.set_seed(1)
total_reward = 0
s = env.reset()
tmp_int = 0
while True:
    s, r, done, info = env.step(np.random.randint(0, 4**2))
    total_reward += r

    still_open = env.render()
    if still_open is False:
        break

    if done:
      env.reset()
    if tmp_int > 2000:
      break
    else:
      tmp_int += 1
# env.render()
print("Done.")
env.close()
# env.imshow_lander()
# plt.pause(10)
