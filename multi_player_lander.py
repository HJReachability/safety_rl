#!/usr/bin/env python
# coding: utf-8


from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

from gym_reachability import gym_reachability  # Custom Gym env.
import gym
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import animation

def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(
      plt.gcf(), animate, frames=len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=20)

# == ENVIRONMENT ==
env_name = "two_player_pursuit_evasion_lunar_lander-v0"
env = gym.make(env_name, rnd_seed=5)

my_images = []
fig, ax = plt.subplots(figsize=(12, 7))

total_reward = 0
s = env.reset()
tmp_int = 0
while True:
    s, r, done, info = env.step(np.random.randint(0, 4**2))
    total_reward += r

    my_images.append(env.render(mode="rgb_array"))
    # img_data = image#[::2, ::3, :]
    # plt.imshow(img_data)
    # plt.pause(0.001)

    # fig.canvas.draw()       # draw the canvas, cache the renderer
    # image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    # image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    # my_images.append(image)
    # plt.clf()

    # env.imshow_lander()
    # if still_open is False:
    #     break

    if done:
      env.reset()
    if tmp_int > 100:
      break
    else:
      tmp_int += 1
# env.render()
print("Done.")
env.close()
save_frames_as_gif(my_images)

# my_images = []
# fig, ax = plt.subplots(figsize=(12, 7))

# env.imshow_lander()
# plt.pause(10)
