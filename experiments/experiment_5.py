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
from sac.sac import sac
from spinup.algos.sac import core
import gym
import os
from datetime import datetime
from utils import get_save_dir

# == Experiment 5 ==
"""
This experiment runs Soft Actor Critic (SAC) with the Safety Bellman Equation
backup on the cheetah task and searches over hyper-parameters. This is compared
against SAC optimizing for sum of discounted rewards with the same reward
function and sum of discounted rewards with only penalization for falling over.
"""

# Since the SAC implementation in Spinning Up is a function that runs once and
# not object-oriented to run it in Ray a single function that runs the entire
# experiment must be run.
def run_sac(search_config, reporter):

    # Register environments (need custom gym environments for safety problem).
    def env_fn():
        from gym_reachability import gym_reachability
        return gym.make(search_config['env'])

    # Ray changes the working directory to be a folder named the hyperparams
    # used for the file.
    setup_logger_kwargs = {'output_dir': os.getcwd()}

    sac(env_fn=env_fn,
        actor_critic=core.mlp_actor_critic,
        ac_kwargs=dict(hidden_sizes=[64, 32]),
        gamma=search_config['gamma'],
        seed=search_config['seed'],
        alpha=search_config['alpha'],
        batch_size=search_config['batch_size'],
        lr=search_config['lr'],
        steps_per_epoch=search_config['steps_per_epoch'],
        epochs=search_config['epochs'],
        max_ep_len=search_config['max_ep_len'],
        use_sbe=search_config['sbe'],
        logger_kwargs=setup_logger_kwargs)
    # Needed to prevent crash on this version of ray.
    reporter(time_this_iters=0)


ray.init()
now = datetime.now()
save_dir = now.strftime('%b') + str(now.day)

# ======= Hyper-Parameter Search Experiment =======
search_config = {}

# == Environment ==
search_config['env'] = 'cheetah_balance-v0'
search_config['gamma'] = 0.99
search_config['max_ep_len'] = 100
search_config['seed'] = 0

# == Optimization ==
search_config['epochs'] = 5
search_config['steps_per_epoch'] = int(1e4)

search_config['alpha'] = tune.grid_search([1e-3, 1e-2, 1e-1])
search_config['batch_size'] = tune.grid_search([50, 100, 200])
search_config['lr'] = tune.grid_search([1e-4, 5e-4, 1e-3])

search_config['sbe'] = False


# This Experiment will call the function run_sac() until completion. Each trial
# is a different set of hyper-parameter in this case since the config specifies
# to grid search over hyper-parameters. The data for each trial will be saved in
# local_dir/name/trial_name where local_dir and name are the arguments to
# Experiment() and trial_name is produced by ray based on the hyper-parameters
# of the trial and time of the Experiment.

# Searches over hyper-parameters for sum of discounted rewards SAC on the
# cheetah.
sum_cheetah_search = Experiment(
    name='sac_sum_cheetah_search_' + save_dir,
    config=search_config,
    run=run_sac,
    num_samples=1,
    local_dir=get_save_dir(),
    checkpoint_at_end=True)

# Crucial to copy dictionary before making changes or else previous experiment
# will be changed.
sbe_search_config = search_config.copy()
sbe_search_config['sbe'] = True

# Searches over hyper-parameters for SBE outcome SAC on the cheetah.
sbe_cheetah_search = Experiment(
    name='sac_sbe_cheetah_search_' + save_dir,
    config=sbe_search_config,
    run=run_sac,
    num_samples=1,
    local_dir=get_save_dir(),
    checkpoint_at_end=True)

sum_search_config_penalize = search_config.copy()
sum_search_config_penalize['env'] = 'cheetah_balance_penalize-v0'
sum_search_config_penalize['sbe'] = False

# Searches over hyper-parameters sum of discounted rewards SAC on the cheetah
# with the reward function only penalizing safety violations (see
# cheetah_balance_penalize.py).
sum_cheetah_penalize_search = Experiment(
    name='sac_sum_cheetah_penalize_search_' + save_dir,
    config=sum_search_config_penalize,
    run=run_sac,
    num_samples=1,
    local_dir=get_save_dir(),
    checkpoint_at_end=True)

tune.run_experiments([sum_cheetah_search,
                      sbe_cheetah_search,
                      sum_cheetah_penalize_search])

# ======= 100 Seed Experiment =======

multi_seed_config = search_config.copy()

# == Environment ==
multi_seed_config['env'] = 'cheetah_balance-v0'
multi_seed_config['seed'] = tune.grid_search(list(range(100)))

# == Optimization ==
# TODO set according to result of hyper-param search
multi_seed_config['alpha'] = 1e-3
multi_seed_config['batch_size'] = 100
multi_seed_config['lr'] = 1e-4

multi_seed_config['sbe'] = False

# Runs sum of discounted rewards SAC on the cheetah on 100 seeds with the best
# hyper-parameters from the hyper-parameter search.
sum_cheetah = Experiment(
    name='sac_sum_cheetah_' + save_dir,
    config=multi_seed_config,
    run=run_sac,
    num_samples=1,
    local_dir=get_save_dir(),
    checkpoint_at_end=True)

sbe_multi_seed_config = multi_seed_config.copy()

# == Optimization ==
# TODO set according to result of hyper-param search
sbe_multi_seed_config['alpha'] = 1e-3
sbe_multi_seed_config['batch_size'] = 100
sbe_multi_seed_config['lr'] = 1e-4

sbe_multi_seed_config['sbe'] = True

# Runs SBE SAC on the cheetah on 100 seeds with the best hyper-parameters from
# the hyper-parameter search.
sbe_cheetah = Experiment(
    name='sac_sbe_cheetah_' + save_dir,
    config=sbe_multi_seed_config,
    run=run_sac,
    num_samples=1,
    local_dir=get_save_dir(),
    checkpoint_at_end=True)

sum_penalize_multi_seed_config = multi_seed_config.copy()

# == Environment ==
sum_penalize_multi_seed_config['env'] = 'cheetah_balance_penalize-v0'

# == Optimization ==
sum_penalize_multi_seed_config['alpha'] = 1e-3
sum_penalize_multi_seed_config['batch_size'] = 100
sum_penalize_multi_seed_config['lr'] = 1e-4

sum_penalize_multi_seed_config['sbe'] = False

# Runs sum of discounted rewards SAC on the penalize-only cheetah on 100 seeds
# with the best hyper-parameters from the hyper-parameter search.
sum_cheetah_penalize = Experiment(
    name='sac_sum_cheetah_penalize_' + save_dir,
    config=sum_penalize_multi_seed_config,
    run=run_sac,
    num_samples=1,
    local_dir=get_save_dir(),
    checkpoint_at_end=True)

tune.run_experiments([sum_cheetah_search,
                      sbe_cheetah_search,
                      sum_cheetah_penalize_search])
