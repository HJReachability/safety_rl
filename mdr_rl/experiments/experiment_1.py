from mdr_rl.tabular_q_learning.q_learning import learn
from mdr_rl.utils import make_log_decay_schedule, make_linear_schedule, make_stepped_schedule,\
    v_from_q, visualize_matrix
from mdr_rl.gym_reachability import gym_reachability # needed to be able to make custom gym env
import gym

# == Experiment 1 ==
"""
This experiment runs tabular q learning with the Safety Bellman Equation backup on a double 
integrator problem.  
"""

# == env ==
seed = 100
max_episode_length = 1
env = gym.make("double_integrator-v0")
fictitious_terminal_val = -10

# == discretization ==
buckets = (81, 81)
state_bounds = env.bounds

# == optimization ==
get_alpha = make_log_decay_schedule(0.3, 0.5)
get_epsilon = make_linear_schedule(0.5, 0.2, int(5e3))
get_gamma = make_stepped_schedule(0.7, int(5e5), 0.9999)

# == scheduling
max_episodes = int(1e6)

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
