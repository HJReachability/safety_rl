import ray
from ray import tune
from ray.tune import Experiment
from mdr_rl.sac.sac import sac
from spinup.algos.sac import core
import gym
import os
from datetime import datetime


# == Experiment 5 ==
"""
This experiment runs soft actor critic with the Safety Bellman Equation backup on the cheetah task 
and searches over hyper-parameters. This is compared against sac optimizing for sum of discounted 
rewards with the same reward function and sum of discounted rewards with only penalization for 
falling over.
"""


def run_sac(search_config, reporter):

    def env_fn():
        from mdr_rl.gym_reachability import gym_reachability  # needed for custom env
        return gym.make(search_config['env'])

    # ray changes the working directory to be a folder named the hyperparams used for the file
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
    reporter(time_this_iters=0)  # needed to prevent crash on this version of ray


ray.init()
now = datetime.now()
save_dir = now.strftime('%b') + str(now.day)

# == hyper param search ==

search_config = {}

# == env ==
search_config['env'] = 'cheetah_balance-v0'
search_config['gamma'] = 0.99
search_config['max_ep_len'] = 100
search_config['seed'] = 0

# == optimization ==
search_config['alpha'] = tune.grid_search([1e-3, 1e-2, 1e-1])
search_config['batch_size'] = tune.grid_search([50, 100, 200])
search_config['lr'] = tune.grid_search([1e-4, 5e-4, 1e-3])

# == scheduling ==
search_config['epochs'] = 5
search_config['steps_per_epoch'] = int(1e4)

search_config['sbe'] = False
sum_cheetah_search = Experiment(
        name='sac_sum_cheetah_search_' + save_dir,
        config=search_config,
        run=run_sac,
        num_samples=1,
        local_dir='~/safety_rl/mdr_rl/data',
        checkpoint_at_end=True)

# super important to copy dictionary before making changes or else prev experiment will be
# changed
sbe_search_config = search_config.copy()
sbe_search_config['sbe'] = True

sbe_cheetah_search = Experiment(
        name='sac_sbe_cheetah_search_' + save_dir,
        config=sbe_search_config,
        run=run_sac,
        num_samples=1,
        local_dir='~/safety_rl/mdr_rl/data',
        checkpoint_at_end=True)

sum_search_config_penalize = search_config.copy()
sum_search_config_penalize['env'] = 'cheetah_balance_penalize-v0'
sum_search_config_penalize['sbe'] = False
sum_cheetah_penalize_search = Experiment(
        name='sac_sum_cheetah_penalize_search_' + save_dir,
        config=sum_search_config_penalize,
        run=run_sac,
        num_samples=1,
        local_dir='~/safety_rl/mdr_rl/data',
        checkpoint_at_end=True)

tune.run_experiments([sum_cheetah_search, sbe_cheetah_search, sum_cheetah_penalize_search])


# == 100 seed experiment ==
multi_seed_config = search_config.copy()

# == Environment ==
multi_seed_config['env'] = 'cheetah_balance-v0'
multi_seed_config['seed'] = tune.grid_search(list(range(100)))

# == optimization ==
# TODO set according to result of hyper-param search
multi_seed_config['alpha'] = 1e-3
multi_seed_config['batch_size'] = 100
multi_seed_config['lr'] = 1e-4

multi_seed_config['sbe'] = False
sum_cheetah = Experiment(
        name='sac_sum_cheetah_' + save_dir,
        config=multi_seed_config,
        run=run_sac,
        num_samples=1,
        local_dir='~/safety_rl/mdr_rl/data',
        checkpoint_at_end=True)

sbe_multi_seed_config = multi_seed_config.copy()

# == optimization ==
# TODO set according to result of hyper-param search
sbe_multi_seed_config['alpha'] = 1e-3
sbe_multi_seed_config['batch_size'] = 100
sbe_multi_seed_config['lr'] = 1e-4

sbe_multi_seed_config['sbe'] = True
sbe_cheetah = Experiment(
        name='sac_sbe_cheetah_' + save_dir,
        config=sbe_multi_seed_config,
        run=run_sac,
        num_samples=1,
        local_dir='~/safety_rl/mdr_rl/data',
        checkpoint_at_end=True)

sum_penalize_multi_seed_config = multi_seed_config.copy()

# == Environment ==
sum_penalize_multi_seed_config['env'] = 'cheetah_balance_penalize-v0'

# == Optimization ==
# TODO set according to result of hyper-param search
sum_penalize_multi_seed_config['alpha'] = 1e-3
sum_penalize_multi_seed_config['batch_size'] = 100
sum_penalize_multi_seed_config['lr'] = 1e-4

sum_penalize_multi_seed_config['sbe'] = False
sum_cheetah_penalize = Experiment(
        name='sac_sum_cheetah_penalize_' + save_dir,
        config=sum_penalize_multi_seed_config,
        run=run_sac,
        num_samples=1,
        local_dir='~/safety_rl/mdr_rl/data',
        checkpoint_at_end=True)

tune.run_experiments([sum_cheetah_search, sbe_cheetah_search, sum_cheetah_penalize_search])
