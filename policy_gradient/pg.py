"""
This file is a modified version of Ray's implementation of Policy Gradient (PG) which can
be found @ https://github.com/ray-project/ray/blob/releases/0.7.3/python/ray/rllib/agents/pg/pg.py

This file is modified such that PG can be used with the Safety Bellman Equation (SBE) outcome from
equation (8) in [ICRA19]. All modifications are marked with a line of hashtags.

Authors: Neil Lugovoy   ( nflugovoy@berkeley.edu )

See the LICENSE in the root directory of this repo for license info.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ray.rllib.agents.trainer import with_common_config
from ray.rllib.agents.trainer_template import build_trainer
###########################################################
# Replace the standard policy gradient policy graph with the policy gradient policy graph
# corresponding to the SBE outcome from Equation (8) of [ICRA19].
from policy_gradient.pg_policy import PGTFPolicy
###########################################################
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