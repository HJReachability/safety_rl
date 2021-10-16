# Copyright (c) 2019–2020, The Regents of the University of California.
# All rights reserved.
#
# This file is subject to the terms and conditions defined in the LICENSE file
# included in this repository.
#
# Please contact the author(s) of this library if you have any questions.
# Authors: Neil Lugovoy   ( nflugovoy@berkeley.edu )

from gym_reachability import gym_reachability  # Custom Gym env.
import gym
import numpy as np
import matplotlib.pyplot as plt

from tabular_q_learning.q_learning import learn
from utils import make_linear_schedule
from utils import make_stepped_schedule
from utils import v_from_q
from utils import visualize_matrix
from utils import make_inverse_polynomial_visit_schedule
from utils import make_stepped_linear_schedule
from utils import make_inverse_visit_schedule

# == Experiment 1 ==
"""
This experiment runs tabular Q-learning with the Safety Bellman Equation (SBE)
backup proposed in [ICRA19] on a 2-dimensional double integrator problem. The
accuracy of the tabular solution is contingent on the number of episodes and
the degree of exploration set through the opimization parameters in the code.
Upon completion, the tabular values are compared to the analytic solution of
the value function.
"""

# == Environment ==
max_episode_length = 1
env = gym.make("double_integrator-v0")
fictitious_terminal_val = -10

# == Seeding ==
seed = 55

# == Discretization ==
grid_cells = (51, 101)
num_states = np.cumprod(grid_cells)[-1]
state_bounds = env.bounds
env.set_grid_cells(grid_cells)
env.set_bounds(state_bounds)
analytic_v = env.analytic_v()
# visualize_matrix(analytic_v)
# visualize_matrix(np.sign(analytic_v))
# env.visualize_analytic_comparison(analytic_v)
# env.visualize_analytic_comparison(np.sign(analytic_v))

# == Optimization ==
max_episodes = int(3e6)
get_alpha = make_inverse_visit_schedule(max_episodes/num_states)#make_linear_schedule(0.9, 0.1, max_episodes)#make_inverse_polynomial_visit_schedule(1.0, 0.51)
get_epsilon = make_linear_schedule(0.95, 0.1, max_episodes)
get_gamma = make_stepped_schedule(0.999, int(max_episodes / 5), 0.9999999)

q, stats = learn(get_learning_rate=get_alpha,
                 get_epsilon=get_epsilon,
                 get_gamma=get_gamma,
                 max_episodes=max_episodes,
                 env=env,
                 grid_cells=grid_cells,
                 state_bounds=state_bounds,
                 seed=seed,
                 max_episode_length=max_episode_length,
                 fictitious_terminal_val=fictitious_terminal_val)

v = v_from_q(q)
print(env.ground_truth_comparison_v(v))
visualize_matrix(v)
visualize_matrix(np.sign(v))
# print(np.shape(v))
# print(np.shape(env.analytic_v()))
# visualize_matrix(env.analytic_v())
# visualize_matrix(np.sign(env.analytic_v()))
env.visualize_analytic_comparison(v)
visualize_matrix(np.sum(stats['state_action_visits'], axis=2))
plt.plot(stats["max_change"])
plt.show()
