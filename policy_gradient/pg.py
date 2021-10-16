# Copyright (c) 2019–2020, The Regents of the University of California.
# All rights reserved.
#
# This file is a modified version of Ray's Policy Gradient (PG) implementation,
# available at:
#
# https://github.com/ray-project/ray/blob/releases/0.7.3/python/ray/rllib/agents/pg/pg.py
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

from ray.rllib.agents.trainer import with_common_config
from ray.rllib.agents.trainer_template import build_trainer
# ******************************************************************* SBE Begin.
# Replace the standard policy gradient policy graph with the one corresponding
# to the SBE outcome from Equation (8) of [ICRA19].
from policy_gradient.pg_policy import PGTFPolicy
# ******************************************************************* SBE End.
# yapf: disable
# __sphinx_doc_begin__
DEFAULT_CONFIG = with_common_config({
    # No remote workers by default
    "num_workers": 0,
    # Learning rate
    "lr": 0.0004,
    # Use PyTorch as backend
    "use_pytorch": False,
})
# __sphinx_doc_end__
# yapf: enable


def get_policy_class(config):
    if config["use_pytorch"]:
        from ray.rllib.agents.pg.torch_pg_policy import PGTorchPolicy
        return PGTorchPolicy
    else:
        return PGTFPolicy


PGTrainer = build_trainer(
    name="PG",
    default_config=DEFAULT_CONFIG,
    default_policy=PGTFPolicy,
    get_policy_class=get_policy_class)
