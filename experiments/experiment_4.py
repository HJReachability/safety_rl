# Copyright (c) 2019–2020, The Regents of the University of California.
# All rights reserved.
#
# This file is subject to the terms and conditions defined in the LICENSE file
# included in this repository.
#
# Please contact the author(s) of this library if you have any questions.
# Authors: Neil Lugovoy   ( nflugovoy@berkeley.edu )

import ray
from ray import tune
from ray.tune import Experiment
from policy_gradient.pg import PGTrainer as PGTrainerSBE
from ray.rllib.agents.pg.pg import PGTrainer
from datetime import datetime
from ray.tune.registry import register_env
import gym
from utils import get_save_dir

# == Experiment 4 ==
"""
This experiment runs policy gradient optimizing for the Safety Bellman Equation outcome that is
induced by the backup on the cheetah environment and performs hyper-parameter search. This is
compared against policy gradient optimizing for sum of discounted rewards with the same reward
function and sum of discounted rewards with only penalization for falling over.
"""


def cheetah_penalize_env_creator(env_config):
    from gym_reachability import gym_reachability  # needed to use custom gym env
    return gym.make('cheetah_balance-v0')


def cheetah_env_creator(env_config):
    from gym_reachability import gym_reachability  # needed to use custom gym env
    return gym.make('cheetah_balance_penalize-v0')


# register envs
register_env('cheetah_balance-v0', cheetah_penalize_env_creator)
register_env('cheetah_balance_penalize-v0', cheetah_penalize_env_creator)


ray.init()
now = datetime.now()
date = now.strftime('%b') + str(now.day)

# =======  Hyper-Parameter Search =======

search_config = {}

# === Environment ===
search_config['env'] = 'cheetah_balance-v0'
search_config['gamma'] = 0.99
search_config['horizon'] = 45
search_config['seed'] = 0

# == Optimization ==
search_config['lr'] = tune.grid_search([1e-4, 5e-4, 1e-3])
search_config['train_batch_size'] = tune.grid_search([50, 100, 150])
search_config['sample_batch_size'] = 50

# == Scheduling ==
search_config['timesteps_per_iteration'] = int(1e5)

max_iterations = int(1e7 / search_config['timesteps_per_iteration'])
checkpoint_freq = int(1e5)

# This Experiment will call the PGTrainer constructor once at the beginning of the experiment and
# call PGTrainer._train() until the condition specified by the argument stop is met. In this case
# once the training_iteration reaches exp_config["max_iterations"] the experiment will stop.
# Every checkpoint_freq it will call save the policy state and at the end of the experiment. Each
# trial is a different set of hyper-parameter in this case since the config specifies to grid search
# over hyper-parameters. The data for each trial will be saved in local_dir/name/trial_name where
# local_dir and name are the arguments to Experiment() and trial_name is produced by ray based
# on the hyper-parameters of the trial and time of the Experiment.

# searches over hyper-parameters for sum of discounted rewards policy gradient on the cheetah
sum_cheetah_search = Experiment(
        name='pg_sum_cheetah_search_' + date,
        config=search_config,
        run=PGTrainer,
        num_samples=1,
        stop={'training_iteration': max_iterations},
        local_dir=get_save_dir(),
        checkpoint_freq=checkpoint_freq,
        checkpoint_at_end=True)

# searches over hyper-parameters for SBE outcome policy gradient on the cheetah
sbe_cheetah_search = Experiment(
        name='pg_sbe_cheetah_search' + date,
        config=search_config,
        run=PGTrainerSBE,  # note the difference from above
        num_samples=1,
        stop={'training_iteration': max_iterations},
        local_dir=get_save_dir(),
        checkpoint_freq=checkpoint_freq,
        checkpoint_at_end=True)

# crucial to copy dictionary before making changes or else previous experiment will be changed
search_penalize_config = search_config.copy()
search_penalize_config['env'] = 'cheetah_balance_penalize-v0'

# searches over hyper-parameters sum of discounted rewards policy gradient on the cheetah with the
# reward function only penalizing safety violations (see cheetah_balance_penalize.py)
sum_cheetah_penalize_search = Experiment(
        name='pg_sum_cheetah_penalize_search' + date,
        config=search_penalize_config,
        run=PGTrainer,
        num_samples=1,
        stop={'training_iteration': max_iterations},
        local_dir=get_save_dir(),
        checkpoint_freq=checkpoint_freq,
        checkpoint_at_end=True)


tune.run_experiments([sum_cheetah_search,
                      sum_cheetah_penalize_search,
                      sbe_cheetah_search],
                     verbose=2)


# =======  100 Seed Experiment =======

multi_seed_config = search_config.copy()

# === Environment ===
multi_seed_config['env'] = 'cheetah_balance-v0'
multi_seed_config['seed'] = tune.grid_search(list(range(100)))

# == Optimization ==
# TODO need to set these based on results of hyper param search
multi_seed_config['lr'] = 1e-4
multi_seed_config['train_batch_size'] = 200

# runs sum of discounted rewards policy gradient on the cheetah on 100 seeds with the best
# hyper-parameters from the hyper-parameter search
sum_cheetah = Experiment(
        name='pg_sum_cheetah_' + date,
        config=multi_seed_config,
        run=PGTrainer,
        num_samples=1,
        stop={'training_iteration': max_iterations},
        local_dir=get_save_dir(),
        checkpoint_freq=checkpoint_freq,
        checkpoint_at_end=True)

# == Optimization ==
sbe_multi_seed_config = multi_seed_config.copy()

# TODO need to set these based on results of hyper param search
multi_seed_config['lr'] = 1e-4
multi_seed_config['train_batch_size'] = 200

# runs SBE outcome policy gradient on the cheetah on 100 seeds with the best hyper-parameters from
# the hyper-parameter search
sbe_cheetah = Experiment(
        name='pg_sbe_cheetah' + date,
        config=multi_seed_config,
        run=PGTrainerSBE,
        num_samples=1,
        stop={'training_iteration': max_iterations},
        local_dir=get_save_dir(),
        checkpoint_freq=checkpoint_freq,
        checkpoint_at_end=True)

# == Environment ==
penalize_multi_seed_config = multi_seed_config.copy()
penalize_multi_seed_config['env'] = 'cheetah_balance_penalize-v0'

# == Optimization ==
# TODO need to set these based on results of hyper param search
penalize_multi_seed_config['lr'] = 1e-4
penalize_multi_seed_config['train_batch_size'] = 200

# runs sum of discounted rewards policy gradient on the penalize-only cheetah on 100 seeds with the
# best hyper-parameters from the hyper-parameter search
sum_cheetah_penalize = Experiment(
        name='pg_sum_cheetah_penalize' + date,
        config=penalize_multi_seed_config,
        run=PGTrainer,
        num_samples=1,
        stop={'training_iteration': max_iterations},
        local_dir=get_save_dir(),
        checkpoint_freq=checkpoint_freq,
        checkpoint_at_end=True)

tune.run_experiments([sum_cheetah, sum_cheetah_penalize, sbe_cheetah], verbose=2)
