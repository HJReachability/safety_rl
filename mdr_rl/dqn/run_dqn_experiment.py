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

import ray
from ray.tune import Trainable
from ray.rllib.agents.trainer import Trainer
from mdr_rl.dqn import dqn
from mdr_rl.dqn.dqn import q_values
from mdr_rl.utils import eval_violation, compare_against_rollout
import numpy as np
import json
import gym
import os


class TrainDQN(Trainable):

    def _setup(self, config=None, logger_creator=None):

        # set up trainer
        Trainer._allow_unknown_configs = True  # need to allow use of SBE config option
        dqn_config = config['dqn_config']
        self.trainer = dqn.DQNTrainer(config=dqn_config, env=dqn_config['env'])

        self.env = gym.make(dqn_config['env'])

        # data collection parameters
        self.violations_horizon = config['violations_horizon']
        self.violations_samples = config['violations_samples']
        self.num_violation_collections = config['num_violation_collections']
        self.rollout_samples = config['rollout_samples']
        self.rollout_horizon = config['rollout_horizon']

        # whether to compare against ground truth value function at end of experiment
        self.ground_truth_compare = config.get('ground_truth_compare', False)
        if self.ground_truth_compare:
            self.env.set_discretization(config['buckets'], self.env.bounds)

        # function to evaluate q network
        self.q_func = lambda s: q_values(self.trainer, s)

        # list of violations data
        self.violations = []

        # steps
        self.total_steps = config['max_iterations']
        self.step = 0

        # seeding
        np.random.seed(dqn_config['seed'])  # NOTE I need to check if this will see the numpy in env

    def _train(self):
        result = self.trainer.train()
        if self.step % (self.total_steps // self.num_violation_collections) == 0:
            print('getting violations data')
            # runs policy in environment and gathers data
            num_violations = eval_violation(self.violations_horizon, self.violations_samples,
                                            self.q_func, self.env)
            self.violations.append((int(self.step), int(num_violations), self.violations_samples))

            # save data to json file
            with open(os.path.join(self.logdir, 'violations.json'), 'w') as outfile:
                data = {'data': self.violations}
                json.dump(data, outfile)

        if self.step == self.total_steps - 1:  # end of training
            print('comparing against rollout')
            rollout_data = compare_against_rollout(self.rollout_horizon,
                                                   self.rollout_samples,
                                                   self.q_func,
                                                   self.env)

            # save data to json file
            with open(os.path.join(self.logdir, 'rollout_comparison.json'), 'w') as outfile:
                data = {'data': rollout_data}
                json.dump(data, outfile)

            if self.ground_truth_compare:
                print('comparing against ground truth')
                ground_truth_data = self.env.ground_truth_comparison(self.q_func)

                # save data to json file
                with open(os.path.join(self.logdir, 'ground_truth_comparison.json'), 'w') as outfile:
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
