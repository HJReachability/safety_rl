# Copyright (c) 2019–2020, The Regents of the University of California.
# All rights reserved.
#
# This file is a modified version of Ray's Policy Gradient (PG) implementation,
# available at:
#
# https://github.com/ray-project/ray/blob/releases/0.7.3/python/ray/rllib/agents/pg/pg_policy.py
#
# The code is modified to allow using PG with the Safety Bellman Equation (SBE)
# outcome from Equation (8) in [ICRA19]. Modifications with respect to the
# original code are enclosed between two lines of asterisks.
#
# This file is subject to the terms and conditions defined in the LICENSE file
# included in this code repository.
#
# Please contact the author(s) of this library if you have any questions.
# Authors: Neil Lugovoy   ( nflugovoy@berkeley.edu )

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ray
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.policy.tf_policy_template import build_tf_policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils import try_import_tf
# ******************************************************************* SBE Begin.
# Import function to compute SBE advantages induced by SBE outcome from Equation
# (8) of [ICRA19] instead of advantages from the sum of discounted rewards.
from policy_gradient.postprocessing import compute_advantages
# ******************************************************************* SBE End.
tf = try_import_tf()


# The basic policy gradients loss
def policy_gradient_loss(policy, model, dist_class, train_batch):
    logits, _ = model.from_batch(train_batch)
    action_dist = dist_class(logits, model)
    return -tf.reduce_mean(
        action_dist.logp(train_batch[SampleBatch.ACTIONS]) *
        train_batch[Postprocessing.ADVANTAGES])


# This adds the "advantages" column to the sampletrain_batch.
def postprocess_advantages(policy,
                           sample_batch,
                           other_agent_batches=None,
                           episode=None):
# ******************************************************************* SBE Begin.
    # Compute advantages using trajectory outcome from Equation (8) of [ICRA19].
    return compute_advantages(sample_batch, 0.0, policy.config["gamma"],
                              use_gae=False, use_sbe=True)
# ******************************************************************* SBE End.


PGTFPolicy = build_tf_policy(
    name="PGTFPolicy",
    get_default_config=lambda: ray.rllib.agents.pg.pg.DEFAULT_CONFIG,
    postprocess_fn=postprocess_advantages,
    loss_fn=policy_gradient_loss)
