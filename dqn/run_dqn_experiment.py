# Copyright (c) 2019–2020, The Regents of the University of California.
# All rights reserved.
#
# This file is subject to the terms and conditions defined in the LICENSE file
# included in this repository.
#
# Please contact the author(s) of this library if you have any questions.
# Authors: Neil Lugovoy   ( nflugovoy@berkeley.edu )

import ray
from ray.tune import Trainable
from ray.rllib.agents.trainer import Trainer
from dqn import dqn
from dqn.dqn import q_values
from utils import eval_violation, compare_against_rollout
import numpy as np
import json
import gym
import os


class TrainDQN(Trainable):

    def _setup(self, config=None, logger_creator=None):

        # Set up trainer object.
        Trainer._allow_unknown_configs = True  # Needed for SBE config option.
        dqn_config = config['dqn_config']
        self.trainer = dqn.DQNTrainer(config=dqn_config, env=dqn_config['env'])

        self.env = gym.make(dqn_config['env'])

        # Data collection parameters.
        self.violations_horizon = config['violations_horizon']
        self.violations_samples = config['violations_samples']
        self.num_violation_collections = config['num_violation_collections']
        self.rollout_samples = config['rollout_samples']
        self.rollout_horizon = config['rollout_horizon']

        # Whether to compare with ground truth value function after training.
        self.ground_truth_compare = config.get('ground_truth_compare', False)
        if self.ground_truth_compare:
            self.env.set_discretization(config['buckets'], self.env.bounds)

        # Function to evaluate Q-network.
        self.q_func = lambda s: q_values(self.trainer, s)

        # List of data for safety violations.
        self.violations = []

        # Training steps.
        self.total_steps = config['max_iterations']
        self.step = 0

        # Random seed for the experiment.
        # NOTE I need to check if this will see the numpy in env
        np.random.seed(dqn_config['seed'])

    def _train(self):
        result = self.trainer.train()
        if self.step % (self.total_steps //
                        self.num_violation_collections) == 0:
            print('getting violations data')
            # runs policy in environment and gathers data
            num_violations = eval_violation(self.violations_horizon,
                                            self.violations_samples,
                                            self.q_func, self.env)
            self.violations.append(
                (int(self.step), int(num_violations), self.violations_samples))

            # Save violations data to JSON file.
            with open(os.path.join(self.logdir, 'violations.json'),
                      'w') as outfile:
                data = {'data': self.violations}
                json.dump(data, outfile)

        if self.step == self.total_steps - 1:  # end of training
            print('comparing against rollout')
            rollout_data = compare_against_rollout(self.rollout_horizon,
                                                   self.rollout_samples,
                                                   self.q_func,
                                                   self.env)

            # Save Q-function/rollout comparison data to JSON file.
            with open(os.path.join(self.logdir, 'rollout_comparison.json'),
                      'w') as outfile:
                data = {'data': rollout_data}
                json.dump(data, outfile)

            if self.ground_truth_compare:
                print('comparing against ground truth')
                ground_truth_data = self.env.ground_truth_comparison(
                    self.q_func)

                # Save Q-function/ground-truth comparison data to JSON file.
                logpath = os.path.join(self.logdir,
                                       'ground_truth_comparison.json')
                with open(logpath, 'w') as outfile:
                    data = {'data': ground_truth_data}
                    json.dump(data, outfile)

            print('saved to:', self.logdir)
            print('done')

        self.step += 1
        return result

    def _save(self, checkpoint_dir=None):
        return self.trainer.save(checkpoint_dir)

    def _restore(self, path):
        return self.trainer.restore(path)
