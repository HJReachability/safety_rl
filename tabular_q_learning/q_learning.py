# Copyright (c) 2019–2020, The Regents of the University of California.
# All rights reserved.
#
# This file is based on Denny Britz's implementation of (tabular) Q-Learning,
# available at:
#
# https://github.com/dennybritz/reinforcement-learning/blob/master/TD/Q-Learning%20Solution.ipynb
#
# The code in this file allows using Q-Learning with the Safety Bellman Equation
# (SBE) from Equation (7) in [ICRA19].
#
# This file is subject to the terms and conditions defined in the LICENSE file
# included in this code repository.
#
# Please contact the author(s) of this library if you have any questions.
# Authors: Neil Lugovoy   ( nflugovoy@berkeley.edu )

import sys
import time
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from utils import index_to_state
from utils import state_to_index
from utils import sbe_outcome
from utils import save
from utils import v_from_q
from utils import visualize_matrix


def learn(get_learning_rate, get_epsilon, get_gamma, max_episodes, grid_cells,
          state_bounds, env, max_episode_length=None, q_values=None,
          start_episode=None, suppress_print=False, seed=0,
          fictitious_terminal_val=None, visualization_states=None,
          num_rnd_traj=None, use_sbe=True, save_freq=None):
    """ Computes state-action value function in tabular form.

    Args:
        get_learning_rate: Function of current episode number returns learning
            rate.
        get_epsilon: Function of current episode number returns explore rate.
        get_gamma: A function of current episode returns gamma, remember to set
            gamma to None if using this.
        max_episodes: Maximum number of episodes to run for.
        grid_cells: Tuple of ints where the ith value is the number of
            grid_cells for ith dimension of state.
        state_bounds: List of tuples where ith tuple contains the min and max
            value in that order of ith dimension.
        env: OpenAI gym environment.
        max_episode_length: Number of timesteps that counts as solving an
            episode, also acts as max episode timesteps.
        q_values: Precomputed q_values function for warm start.
        start_episode: What episode to start hyper-parameter schedulers at if
            warmstart.
        suppress_print: Boolean whether to suppress print statements about
            current episode or not.
        seed: Seed for random number generator.
        violation_terminal_val: Whether to use a terminal state with this value
            for the backup when a trajectory *ends with a violation*. This
            avoids the need of adding grid cells inside the failure set. Note
            that we are assuming that gym steps returning done==True correspond
            to violations of the constraints.
        use_sbe: Whether to use the Safety Bellman Equation backup from
            equation (7) in [ICRA19]. If false the standard sum of discounted
            rewards backup is used.
        save_freq: How often to save q_values and stats.

    Returns:
        A numpy tensor of shape (grid_cells + (env.action_space.n,)) that
        contains the q_values value function. For example in cartpole the
        dimensions are q_values[x][x_dot][theta][theta_dot][action].
    """
    # TODO(vrubies): sbe change
    # TODO(vrubies): reach_margin, avoid_margin
    # Time-related variables for performace analysis.
    start = time.process_time()
    now = datetime.now()

    np.random.seed(seed)

    # Argument checks.
    if max_episode_length is None:
        import warnings
        warnings.warn(
            "max_episode_length is None assuming infinite episode length.")

    # Set up state-action value function.
    if q_values is None:
        if start_episode is not None:
            raise ValueError(
                "Start_episode is only to be used with a warmstart q_values.")

        q_values = np.zeros(grid_cells + (env.action_space.n,))
        viz_fail = np.zeros(grid_cells)
        # Iterate over all state-action pairs and initialize Q-values.
        it = np.nditer(q_values, flags=['multi_index'])
        while not it.finished:
            # Initialize Q(s,a) = max{l(s),g(s)}.
            state = it.multi_index[:-1]
            idx_2_state = index_to_state(grid_cells, state_bounds, state)
            g_x = env.safety_margin(idx_2_state)
            l_x = env.target_margin(idx_2_state)
            q_values[it.multi_index] = max(l_x, g_x)
            viz_fail[it.multi_index[:-1]] = g_x
            it.iternext()
        env.visualize_analytic_comparison(viz_fail * (viz_fail < 0) +
                                          (viz_fail > 0), True)
        plt.pause(1)
    elif not np.array_equal(np.shape(q_values)[:-1], grid_cells):
        raise ValueError(
            "The shape of q_values excluding the last dimension must be the "
            "same as the shape of the discretization of states.")
    elif start_episode is None:
        import warnings
        warnings.warn(
            "Used warm start q_values without a start_episode, hyper-parameter "
            "schedulers may produce undesired results.")

    # Initialize experiment log.
    stats = {
        "start_time": datetime.now().strftime("%b_%d_%y %H:%M:%S"),
        "episode_lengths": np.zeros(max_episodes),
        "average_episode_rewards": np.zeros(max_episodes),
        "true_min": np.zeros(max_episodes),
        "episode_outcomes": np.zeros(max_episodes),
        "state_action_visits": np.zeros(np.shape(q_values)),
        "epsilon": np.zeros(max_episodes),
        "learning_rate": np.zeros(max_episodes),
        "state_bounds": state_bounds,
        "grid_cells": grid_cells,
        "gamma": np.zeros(max_episodes),
        "type": "tabular q_values-learning",
        "environment": env.spec.id,
        "seed": seed,
        "episode": 0
    }

    env.set_grid_cells(grid_cells)
    env.set_bounds(state_bounds)

    if start_episode is None:
        start_episode = 0

    # Set starting exploration fraction, learning rate and discount factor.
    epsilon = get_epsilon(start_episode)
    alpha = get_learning_rate(start_episode, 1)
    gamma = get_gamma(start_episode, 1)

    cumulative_time = 0
    # Main learning loop: Run episodic trajectories from random initial states.
    for episode in range(max_episodes):
        if not suppress_print and (episode + 1) % 100 == 0:
            message = "\rEpisode {}/{} alpha:{} gamma:{} epsilon:{} avg_ep:{}."
            print(message.format(
                episode + 1, max_episodes, alpha, gamma, epsilon,
                cumulative_time/(episode + 1.0)), end="")
            sys.stdout.flush()
        if ((num_rnd_traj is not None or visualization_states is not None)
                and (episode + 1) % 10000 == 0):
            env.visualize_analytic_comparison(v_from_q(q_values), True)
            # env.plot_trajectories(q_values, T=100, num_rnd_traj=num_rnd_traj,
            #                       states=visualization_states)
            plt.pause(0.001)
        state = env.reset()
        state_ix = state_to_index(grid_cells, state_bounds, state)
        done = False
        t = 0
        episode_rewards = []

        time_for_episode = time.time()
        # Execute a single rollout.
        while not done:

            # Determine action to use based on epsilon-greedy decision rule.
            action_ix = select_action(q_values, state_ix, env, epsilon)

            # Take step and map state to corresponding grid index.
            next_state, reward, done, info = env.step(action_ix)
            g_x = info['g_x'] if 'g_x' in info else -np.inf
            next_state_ix = state_to_index(grid_cells, state_bounds,
                                           next_state)

            # Update episode experiment log.
            stats['state_action_visits'][state_ix + (action_ix,)] += 1
            num_visits = stats['state_action_visits'][state_ix + (action_ix,)]
            episode_rewards.append(reward)
            t += 1

            # Update exploration fraction, learning rate and discount factor.
            epsilon = get_epsilon(episode + start_episode)
            alpha = get_learning_rate(episode + start_episode, num_visits)
            gamma = get_gamma(episode + start_episode, num_visits)

            # Perform Bellman backup.
            if use_sbe:  # Safety Bellman Equation backup.
                l_x = reward
                if not done:
                    min_term = min(l_x, np.amin(q_values[next_state_ix]))
                    new_q = (
                        (1.0 - gamma) * max(l_x, g_x) +
                        gamma * max(min_term, g_x))
                else:
                    if fictitious_terminal_val:
                        if g_x > 0:  # Safety violation.
                            g_x = fictitious_terminal_val
                        elif l_x < 0:  # Target reached.
                            l_x = -fictitious_terminal_val
                    new_q = max(l_x, g_x)
            else:       # Sum of discounted rewards backup.
                if not done:
                    new_q = (reward + gamma *
                             np.amax(q_values[next_state_ix]))
                else:
                    if fictitious_terminal_val:
                        new_q = reward + gamma * fictitious_terminal_val
                    else:
                        new_q = reward

            # Update state-action values.
            q_values[state_ix + (action_ix,)] = (
                (1 - alpha) * q_values[state_ix + (action_ix,)] + alpha * new_q)
            state_ix = next_state_ix

            # End episode if max episode length is reached.
            if max_episode_length is not None and t >= max_episode_length:
                break

        time_for_episode = time.time() - time_for_episode
        cumulative_time += time_for_episode
        # save episode statistics
        episode_rewards = np.array(episode_rewards)
        outcome = sbe_outcome(episode_rewards, gamma)[0]
        stats["episode_outcomes"][episode] = outcome
        stats["true_min"][episode] = np.min(episode_rewards)
        stats["episode_lengths"][episode] = len(episode_rewards)
        stats["average_episode_rewards"][episode] = np.average(episode_rewards)
        stats["learning_rate"][episode] = alpha
        stats["epsilon"][episode] = epsilon
        stats["gamma"][episode] = gamma
        stats["episode"] = episode

        if save_freq and episode % save_freq == 0:
            save(q_values, stats, env.unwrapped.spec.id)

    stats["time_elapsed"] = time.process_time() - start
    print("\n")

    return q_values, stats


def select_action(q_values, state_ix, env, epsilon=0):
    """ Selects an action at random or based on the state-action value function.

    Args:
        q_values: State-action value function.
        state: State.
        env: OpenAI gym environment.
        epsilon: Exploration rate.

    Returns:
        The action chosen for this state.
    """
    if np.random.random() < epsilon:
        action_ix = env.action_space.sample()
    else:
        action_ix = np.argmin(q_values[state_ix])
    return action_ix


def play(q_values, env, num_episodes, grid_cells, state_bounds,
         suppress_print=False, episode_length=None):
    """ Renders gym environment under greedy policy.

    The gym environment will be rendered and executed according to the greedy
    policy induced by the state-action value function. NOTE: After you use an
    environment to play you'll need to make a new environment instance to
    use it again because the close function is called.

    Args:
        q_values: State-action value function.
        env: OpenAI gym environment that hasn't been closed yet.
        num_episodes: How many episode to run for.
        grid_cells: tuple of ints where the ith value is the number of
            grid_cells for ith dimension of state.
        state_bounds: List of tuples where ith tuple contains the min and max
            value in that order of ith dimension.
        suppress_print: Boolean whether to suppress print statements about
            current episode or not.
        episode_length: Max length of episode.
    """
    for i in range(num_episodes):
        obv = env.reset()
        t = 0
        done = False
        while not done:
            if episode_length and t >= episode_length:
                break
            state = state_to_index(grid_cells, state_bounds, obv)
            action = select_action(q_values, state, env, epsilon=0)
            obv, reward, done, _ = env.step(action)
            env.render()
            t += 1
        if not suppress_print:
            print("episode", i,  "lasted", t, "timesteps.")
    # This is required to prevent the script from crashing after closing the
    # window.
    env.close()
