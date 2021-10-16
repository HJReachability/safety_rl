"""
Please contact the author(s) of this library if you have any questions.
Authors: Vicenc Rubies-Royo   ( vrubies@berkeley.edu )

This experiment runs tabular Q-learning with the discounted reach-avoid
Bellman equation (DRABE) proposed in [RSS21] on a 2-dimensional point mass
problem. The accuracy of the tabular solution is contingent on the number of
episodes and the degree of exploration set through the opimization parameters
in the code. We use the following setting to generate Fig. 2 in the paper.
"""

from warnings import simplefilter
import gym
import numpy as np
import matplotlib.pyplot as plt

from gym_reachability import gym_reachability  # Custom Gym env.
from tabular_q_learning.q_learning import learn
from utils.utils import make_linear_schedule, make_stepped_schedule, v_from_q
from utils.utils import visualize_matrix
from utils.utils import make_inverse_visit_schedule

simplefilter(action='ignore', category=FutureWarning)

# == Environment ==
max_episode_length = 1
env = gym.make("point_mass-v0")
fictitious_terminal_val = 10

nx, ny = 81, 241
v = np.zeros((nx, ny))
l_x = np.zeros((nx, ny))
g_x = np.zeros((nx, ny))
xs = np.linspace(-4, 4, nx)
ys = np.linspace(-3, 11, ny)

it = np.nditer(v, flags=['multi_index'])

while not it.finished:
  idx = it.multi_index
  x = xs[idx[0]]
  y = ys[idx[1]]

  l_x[idx] = env.target_margin(np.array([x, y]))
  g_x[idx] = env.safety_margin(np.array([x, y]))

  v[idx] = np.maximum(l_x[idx], g_x[idx])
  it.iternext()

axStyle = [[-4, 4, -3, 11], 8 / 14]
fig, axes = plt.subplots(1, 3, figsize=(12, 6))

ax = axes[0]
f = ax.imshow(
    l_x.T, interpolation='none', extent=axStyle[0], origin="lower",
    cmap="seismic"
)
ax.axis(axStyle[0])
ax.grid(False)
ax.set_aspect(axStyle[1])  # makes equal aspect ratio
env.plot_target_failure_set(ax)
ax.set_title(r'$\ell(x)$')
cbar = fig.colorbar(f, ax=ax, pad=0.01, fraction=0.05, shrink=.9)

ax = axes[1]
f = ax.imshow(
    g_x.T, interpolation='none', extent=axStyle[0], origin="lower",
    cmap="seismic"
)
ax.axis(axStyle[0])
ax.grid(False)
ax.set_aspect(axStyle[1])  # makes equal aspect ratio
env.plot_target_failure_set(ax)
ax.set_title(r'$g(x)$')
cbar = fig.colorbar(f, ax=ax, pad=0.01, fraction=0.05, shrink=.9)

ax = axes[2]
f = ax.imshow(
    v.T, interpolation='none', extent=axStyle[0], origin="lower",
    cmap="seismic", vmin=-.5, vmax=.5
)
ax = plt.gca()
ax.axis(axStyle[0])
ax.grid(False)
ax.set_aspect(axStyle[1])  # makes equal aspect ratio
env.plot_target_failure_set(ax)
ax.set_title(r'$v(x)$')
cbar = fig.colorbar(f, ax=ax, pad=0.01, fraction=0.05, shrink=.9)

# == Seeding ==
seed = 1

# == Discretization ==
grid_cells = (81, 241)
num_states = np.cumprod(grid_cells)[-1]
state_bounds = env.bounds
env.set_discretization(grid_cells, state_bounds)

# == Optimization ==
max_episodes = int(12e6) + 1
get_alpha = make_inverse_visit_schedule(max_episodes / num_states)
get_epsilon = make_linear_schedule(0.95, 0.1, max_episodes)
get_gamma = make_stepped_schedule(0.9, int(max_episodes / 20), 0.99999999)

viz_states = [np.array([0, 0])]  # Visualization states.

q, stats = learn(
    get_learning_rate=get_alpha, get_epsilon=get_epsilon, get_gamma=get_gamma,
    max_episodes=max_episodes, env=env, grid_cells=grid_cells,
    state_bounds=state_bounds, seed=seed,
    max_episode_length=max_episode_length,
    fictitious_terminal_val=fictitious_terminal_val, num_rnd_traj=None,
    visualization_states=env.visual_initial_states, save_freq=5e5, vis_T=500
)

v = v_from_q(q)
visualize_matrix(v.T)
visualize_matrix(np.sign(v).T)
