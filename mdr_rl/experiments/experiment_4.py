import ray
from ray import tune
from ray.tune import Experiment
from mdr_rl.policy_gradient.pg import PGTrainer as PGTrainerSBE
from ray.rllib.agents.pg.pg import PGTrainer
from datetime import datetime
from ray.tune.registry import register_env
import gym


# == Experiment 4 ==
"""
This experiment runs policy gradient optimizing for the minimum of discounted rewards (mdr) that is
induced by the backup on the cheetah environment and performs hyper-parameter search. This is
compared against policy gradient optimizing for sum of discounted rewards with the same reward
function and sum of discounted rewards with only penalization for falling over
"""


def cheetah_penalize_env_creator(env_config):
    from mdr_rl.gym_reachability import gym_reachability  # needed to use custom gym env
    return gym.make('cheetah_balance-v0')


def cheetah_env_creator(env_config):
    from mdr_rl.gym_reachability import gym_reachability  # needed to use custom gym env
    return gym.make('cheetah_balance_penalize-v0')


# register envs
register_env('cheetah_balance-v0', cheetah_penalize_env_creator)
register_env('cheetah_balance_penalize-v0', cheetah_penalize_env_creator)


ray.init()
now = datetime.now()
save_dir = now.strftime('%b') + str(now.day)

# == hyper param search ==

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

sum_cheetah_search = Experiment(
        name='pg_sum_cheetah_search_' + save_dir,
        config=search_config,
        run=PGTrainer,
        num_samples=1,
        stop={'training_iteration': max_iterations},
        local_dir='~/safety_rl/mdr_rl/data',
        checkpoint_freq=checkpoint_freq,
        checkpoint_at_end=True)

sbe_cheetah_search = Experiment(
        name='pg_sbe_cheetah_search' + save_dir,
        config=search_config,
        run=PGTrainerSBE,  # note the difference from above
        num_samples=1,
        stop={'training_iteration': max_iterations},
        local_dir='~/safety_rl/mdr_rl/data',
        checkpoint_freq=checkpoint_freq,
        checkpoint_at_end=True)

search_penalize_config = search_config.copy()
search_penalize_config['env'] = 'cheetah_balance_penalize-v0'

sum_cheetah_penalize_search = Experiment(
        name='pg_sum_cheetah_penalize_search' + save_dir,
        config=search_penalize_config,
        run=PGTrainer,
        num_samples=1,
        stop={'training_iteration': max_iterations},
        local_dir='~/safety_rl/mdr_rl/data',
        checkpoint_freq=checkpoint_freq,
        checkpoint_at_end=True)


tune.run_experiments([sum_cheetah_search,
                      sum_cheetah_penalize_search,
                      sbe_cheetah_search],
                     verbose=2)


# == 100 seed experiment ==
# super important to copy dictionary before making changes or else prev experiment will be changed
multi_seed_config = search_config.copy()

# === Environment ===
multi_seed_config['env'] = 'cheetah_balance-v0'
multi_seed_config['seed'] = tune.grid_search(list(range(100)))

# == Optimization ==
# TODO need to set these based on results of hyper param search
multi_seed_config['lr'] = 1e-4
multi_seed_config['train_batch_size'] = 200

sum_cheetah = Experiment(
        name='pg_sum_cheetah_' + save_dir,
        config=multi_seed_config,
        run=PGTrainer,
        num_samples=1,
        stop={'training_iteration': max_iterations},
        local_dir='~/safety_rl/mdr_rl/data',
        checkpoint_freq=checkpoint_freq,
        checkpoint_at_end=True)

# == Optimization ==
sbe_multi_seed_config = multi_seed_config.copy()

# TODO need to set these based on results of hyper param search
multi_seed_config['lr'] = 1e-4
multi_seed_config['train_batch_size'] = 200

sbe_cheetah = Experiment(
        name='pg_sbe_cheetah' + save_dir,
        config=multi_seed_config,
        run=PGTrainerSBE,
        num_samples=1,
        stop={'training_iteration': max_iterations},
        local_dir='~/safety_rl/mdr_rl/data',
        checkpoint_freq=checkpoint_freq,
        checkpoint_at_end=True)

# == Environment ==
penalize_multi_seed_config = multi_seed_config.copy()
penalize_multi_seed_config['env'] = 'cheetah_balance_penalize-v0'

# == Optimization ==
# TODO need to set these based on results of hyper param search
penalize_multi_seed_config['lr'] = 1e-4
penalize_multi_seed_config['train_batch_size'] = 200

sum_cheetah_penalize = Experiment(
        name='pg_sum_cheetah_penalize' + save_dir,
        config=penalize_multi_seed_config,
        run=PGTrainer,
        num_samples=1,
        stop={'training_iteration': max_iterations},
        local_dir='~/safety_rl/mdr_rl/data',
        checkpoint_freq=checkpoint_freq,
        checkpoint_at_end=True)

tune.run_experiments([sum_cheetah, sum_cheetah_penalize, sbe_cheetah], verbose=2)
