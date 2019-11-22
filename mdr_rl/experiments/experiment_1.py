"""
 * Copyright (c) 2019, The Regents of the University of California (Regents).
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 *    1. Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *
 *    2. Redistributions in binary form must reproduce the above
 *       copyright notice, this list of conditions and the following
 *       disclaimer in the documentation and/or other materials provided
 *       with the distribution.
 *
 *    3. Neither the name of the copyright holder nor the names of its
 *       contributors may be used to endorse or promote products derived
 *       from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS AS IS
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * Please contact the author(s) of this library if you have any questions.
 * Authors: Neil Lugovoy   ( nflugovoy@berkeley.edu )
"""
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

# == Environment ==
max_episode_length = 1
env = gym.make("double_integrator-v0")
fictitious_terminal_val = -10

# == Seeding ==
seed = 100

# == Discretization ==
buckets = (81, 81)
state_bounds = env.bounds

# == Optimization ==
max_episodes = int(1e6)
get_alpha = make_log_decay_schedule(0.3, 0.5)
get_epsilon = make_linear_schedule(0.5, 0.2, int(5e3))
get_gamma = make_stepped_schedule(0.7, int(5e5), 0.9999)

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
