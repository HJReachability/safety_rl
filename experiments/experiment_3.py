"""
Authors: Neil Lugovoy   ( nflugovoy@berkeley.edu )

See the LICENSE in the root directory of this repo for license info.
"""

import ray
import ray.tune as tune
from ray.tune import Experiment
import gym
from datetime import datetime
from ray.rllib.agents.trainer import Trainer
from ray.tune.registry import register_env
from dqn.run_dqn_experiment import TrainDQN
from utils import get_save_dir

# == Experiment 3 ==
"""
This experiment runs DQN with the Safety Bellman Equation on lunar lander over 100 random seeds and
compares the resulting policies against the simulator over the course of training to see how many
trajectories violate the safety constraints. At the end of training the q function is compared
against on-policy rollouts in the simulator.
"""

if __name__ == "__main__":
    ray.init()
    now = datetime.now()
    date = now.strftime("%b") + str(now.day)

    # register env
    def double_int_env_creator(env_config):
        from gym_reachability import gym_reachability  # needed to use custom gym env
        return gym.make('lunar_lander_reachability-v0')

    register_env('lunar_lander_reachability-v0', double_int_env_creator)

    dqn_config = {}
    exp_config = {}

    # == Environment ==
    dqn_config["horizon"] = 1
    dqn_config["env"] = "lunar_lander_reachability-v0"

    # == Model ==
    dqn_config["num_atoms"] = 1
    dqn_config["noisy"] = False
    dqn_config["dueling"] = False
    dqn_config["double_q"] = False
    dqn_config["hiddens"] = [150, 200, 150]
    dqn_config["n_step"] = 1

    # == Exploration ==
    dqn_config["schedule_max_timesteps"] = int(3e6)
    dqn_config["timesteps_per_iteration"] = int(1e3)  # num steps sampled per call to agent.train()
    dqn_config["exploration_fraction"] = 2 / 3
    dqn_config["exploration_final_eps"] = 0.1
    dqn_config["target_network_update_freq"] = int(15e3)

    # == Replay buffer ==
    dqn_config["buffer_size"] = int(1e4)
    dqn_config["prioritized_replay"] = False
    dqn_config["compress_observations"] = False

    # == Optimization ==
    dqn_config["lr"] = 0.00025
    dqn_config["grad_norm_clipping"] = None
    dqn_config["learning_starts"] = int(5e3)
    dqn_config["sample_batch_size"] = 1
    dqn_config["train_batch_size"] = 32

    # == Parallelism ==
    dqn_config["num_workers"] = 1
    dqn_config["num_envs_per_worker"] = 1

    # == Seeding ==
    dqn_config["seed"] = tune.grid_search(list(range(100)))

    # == Custom Safety Bellman Equation configs ==
    Trainer._allow_unknown_configs = True  # need to allow use of SBE config option
    dqn_config["gamma_schedule"] = "stepped"
    dqn_config["final_gamma"] = 0.999999
    dqn_config["gamma"] = 0.7  # initial gamma
    dqn_config["gamma_half_life"] = int(6e4)  # measured relative to steps taken in the environment
    dqn_config["sbe"] = True

    # == Data Collection Parameters ==

    # violations data collected throughout training
    exp_config["violations_horizon"] = 120
    exp_config["violations_samples"] = 1000
    exp_config["num_violation_collections"] = 10

    # rollout comparison done at end of training
    exp_config["rollout_samples"] = int(1e4)
    exp_config["rollout_horizon"] = 100

    # experiment timing
    exp_config["max_iterations"] = int(dqn_config["schedule_max_timesteps"]
                                       / dqn_config["timesteps_per_iteration"])
    exp_config["checkpoint_freq"] = int(exp_config["max_iterations"]
                                        / exp_config["num_violation_collections"])

    exp_config["dqn_config"] = dqn_config

    # This Experiment will call TrainDQN._setup() once at the beginning of the experiment and call
    # TrainDQN._train() until the condition specified by the argument stop is met. In this case
    # once the training_iteration reaches exp_config["max_iterations"] the experiment will stop.
    # Every checkpoint_freq it will call TrainDQN._save() and at the end of the experiment. Each
    # trial is a seed in this case since the config specifies to grid search over seeds and no other
    # config options. The data for each trial will be saved in local_dir/name/trial_name where
    # local_dir and name are the arguments to Experiment() and trial_name is produced by ray based
    # on the hyper-parameters of the trial and time of the Experiment.
    train_double_integrator = Experiment(
        name="dqn_lunar_lander_" + date,
        config=exp_config,
        run=TrainDQN,
        num_samples=1,
        stop={"training_iteration": exp_config["max_iterations"]},
        resources_per_trial={"cpu": 1, "gpu": 0},
        local_dir=get_save_dir(),
        checkpoint_freq=exp_config["checkpoint_freq"],
        checkpoint_at_end=True)
    ray.tune.run_experiments([train_double_integrator], verbose=2)


# TODO gather data into figures
