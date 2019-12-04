"""
Authors: Neil Lugovoy   ( nflugovoy@berkeley.edu )

See the LICENSE in the root directory of this repo for license info.
"""

from mdr_rl.tabular_q_learning.q_learning import learn
from mdr_rl.utils import (make_linear_schedule,
                          make_stepped_schedule,
                          v_from_q,
                          visualize_matrix,
                          make_inverse_polynomial_visit_schedule)
# needed to be able to make custom gym env
from mdr_rl.gym_reachability import gym_reachability  # noqa
import numpy as np
import gym

# == Experiment 1 ==
"""
This experiment runs tabular q learning with the Safety Bellman Equation backup on a double 
integrator problem.  
"""

# == Environment ==
max_episode_length = 1
env = gym.make("double_integrator-v0")
fictitious_terminal_val = -10

# == Seeding ==
seed = 55

# == Discretization ==
buckets = (50, 101)
state_bounds = env.bounds
env.set_discretization(buckets=buckets, bounds=state_bounds)
visualize_matrix(env.analytic_v())
visualize_matrix(np.sign(env.analytic_v()))
env.visualize_analytic_comparison(env.analytic_v())
env.visualize_analytic_comparison(np.sign(env.analytic_v()))

# == Optimization ==
max_episodes = int(1e6)
get_alpha = make_inverse_polynomial_visit_schedule(1.0, 0.8)
get_epsilon = make_linear_schedule(0.5, 0.2, int(5e3))
get_gamma = make_stepped_schedule(0.5, int(5e5), 0.9999)

q, stats = learn(
    get_learning_rate=get_alpha,
    get_epsilon=get_epsilon,
    get_gamma=get_gamma,
    max_episodes=max_episodes,
    env=env,
    buckets=buckets,
    state_bounds=state_bounds,
    seed=seed,
    max_episode_length=max_episode_length,
    fictitious_terminal_val=fictitious_terminal_val
)

v = v_from_q(q)
print(env.ground_truth_comparison_v(v))
visualize_matrix(v)
visualize_matrix(np.sign(v))
print(np.shape(v))
print(np.shape(env.analytic_v()))
visualize_matrix(env.analytic_v())
visualize_matrix(np.sign(env.analytic_v()))
env.visualize_analytic_comparison(v)
visualize_matrix(np.sum(stats['state_action_visits'], axis=2))

