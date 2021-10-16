# Copyright (c) 2019–2020, The Regents of the University of California.
# All rights reserved.
#
# This file is subject to the terms and conditions defined in the LICENSE file
# included in this code repository.
#
# Please contact the author(s) of this library if you have any questions.
# Authors: Neil Lugovoy   ( nflugovoy@berkeley.edu )

from datetime import datetime

import ray
import ray.tune as tune
from ray.tune import Experiment
from ray.rllib.agents.trainer import Trainer
from ray.tune.registry import register_env
import gym
from dqn.run_dqn_experiment import TrainDQN

from utils import get_save_dir

# == Experiment 2 ==
"""
This experiment runs the DQN reinforcement learning algorithm with the Safety
Bellman Equation (SBE) proposed in [ICRA19] on a 2-dimensional double integrator
and a 4-dimensional cart-pole system. It performs 100 independent training runs
with different random seeds, and evaluates the learned safety policies using the
simulator over the course of the training process to determine the fraction of
trajectories violating safety at the different stages. At the end of training
the learned state-action safety Q-function is compared against policy rollouts
in the simulator. The state safety value function (V) is compared against the
analytic value function (double integrator) or the numerical value function
computed with the Level Set Toolbox (cart-pole).
"""


if __name__ == '__main__':
    # Register environments (need custom gym environments for safety problem).
    def double_int_env_creator(env_config):
        from gym_reachability import gym_reachability
        return gym.make('double_integrator-v0')

    def cartpole_env_creater(env_config):
        from gym_reachability import gym_reachability
        return gym.make('cartpole_reach-v0')

    register_env('double_integrator-v0', double_int_env_creator)
    register_env('cartpole_reach-v0', cartpole_env_creater)

    ray.init()
    now = datetime.now()
    date = now.strftime('%b') + str(now.day)
    dqn_config = {}
    exp_config = {}

    # == Environment ==
    dqn_config['horizon'] = 1
    dqn_config['env'] = 'double_integrator-v0'

    # == Model ==
    dqn_config['num_atoms'] = 1
    dqn_config['noisy'] = False
    dqn_config['dueling'] = False
    dqn_config['double_q'] = False
    dqn_config['hiddens'] = [150, 150]
    dqn_config['n_step'] = 1

    # == Exploration ==
    dqn_config['schedule_max_timesteps'] = int(1e6)
    # How many steps are sampled per call to agent.train().
    dqn_config['timesteps_per_iteration'] = int(1e3)
    dqn_config['exploration_fraction'] = 0.5
    dqn_config['exploration_final_eps'] = 0.3
    dqn_config['target_network_update_freq'] = int(1e4)

    # == Replay buffer ==
    dqn_config['buffer_size'] = int(5e4)
    dqn_config['prioritized_replay'] = False
    dqn_config['compress_observations'] = False

    # == Optimization ==
    dqn_config['lr'] = 0.00025
    dqn_config['grad_norm_clipping'] = None
    dqn_config['learning_starts'] = int(5e3)
    dqn_config['sample_batch_size'] = 1
    dqn_config['train_batch_size'] = int(1e2)

    # == Parallelism ==
    dqn_config['num_workers'] = 1
    dqn_config['num_envs_per_worker'] = 1

    # == Seeding ==
    # TODO Does this need to be in exp config? Check with ray doc about seeding.
    dqn_config['seed'] = tune.grid_search(list(range(100)))

    # == Custom Safety Bellman Equation configs ==
    Trainer._allow_unknown_configs = True     # Needed for SBE config option.
    dqn_config['gamma_schedule'] = 'stepped'
    dqn_config['final_gamma'] = 0.999999
    dqn_config['gamma'] = 0.8                 # Initial discount factor.
    dqn_config['gamma_half_life'] = int(4e4)  # Measured in environment steps.
    dqn_config['use_sbe'] = True

    # == Data Collection Parameters ==

    # Compare to ground truth value function at end of training.
    exp_config['ground_truth_compare'] = True
    exp_config['grid_cells'] = (100, 100)

    # Violations data collected throughout training.
    exp_config['violations_horizon'] = 120
    exp_config['violations_samples'] = 1000
    exp_config['num_violation_collections'] = 10

    # Rollout comparison done at end of training.
    exp_config['rollout_samples'] = int(1e4)
    exp_config['rollout_horizon'] = 100

    # Experiment timing.
    exp_config['max_iterations'] = int(dqn_config['schedule_max_timesteps'] /
                                       dqn_config['timesteps_per_iteration'])
    exp_config['checkpoint_freq'] = int(exp_config['max_iterations'] /
                                        exp_config['num_violation_collections'])

    exp_config['dqn_config'] = dqn_config

    # This Experiment will call TrainDQN._setup() once at the beginning of the
    # experiment and call TrainDQN._train() until the condition specified by the
    # argument stop is met. In this case once the training_iteration reaches
    # exp_config["max_iterations"] the experiment will stop. Every
    # checkpoint_freq it will call TrainDQN._save() and at the end of the
    # experiment. Each trial is a seed in this case since the config specifies
    # to grid search over seeds and no other config options. The data for each
    # trial will be saved in local_dir/name/trial_name where local_dir and name
    # are the arguments to Experiment() and trial_name is produced by ray based
    # on the hyper-parameters of the trial and time of the Experiment.
    train_double_integrator = Experiment(
        name='dqn_double_integrator_' + date,
        config=exp_config,
        run=TrainDQN,
        num_samples=1,
        stop={'training_iteration': exp_config['max_iterations']},
        resources_per_trial={'cpu': 1, 'gpu': 0},
        local_dir=get_save_dir(),
        checkpoint_freq=exp_config['checkpoint_freq'],
        checkpoint_at_end=True)

    # Copying dictionary before making changes. Otherwise the previous
    # experiment would be changed.
    cartpole_exp_config = exp_config.copy()
    cartpole_dqn_config = dqn_config.copy()

    # Cartpole specific parameters.

    # == Environment ==
    cartpole_dqn_config['env'] = 'cartpole_reach-v0'

    # == Data Collection Parameters ==
    cartpole_exp_config['grid_cells'] = (31, 31, 31, 31)

    # == Optimization ==
    cartpole_dqn_config['schedule_max_timesteps'] = int(2e6)
    cartpole_dqn_config['gamma_half_life'] = int(5e4)

    # == Scheduling ==
    cartpole_exp_config['max_iterations'] = int(
        dqn_config['schedule_max_timesteps'] /
        dqn_config['timesteps_per_iteration'])
    cartpole_exp_config['checkpoint_freq'] = int(
        exp_config['max_iterations'] /
        exp_config['num_violation_collections'])

    cartpole_exp_config['dqn_config'] = cartpole_dqn_config

    train_cartpole = Experiment(
        name='dqn_cartpole_' + date,
        config=exp_config,
        run=TrainDQN,
        num_samples=1,
        stop={'training_iteration': exp_config['max_iterations']},
        resources_per_trial={'cpu': 1, 'gpu': 0},
        local_dir=get_save_dir(),
        checkpoint_freq=exp_config['checkpoint_freq'],
        checkpoint_at_end=True)
    ray.tune.run_experiments([train_cartpole, train_cartpole], verbose=2)

# TODO gather data into figures
