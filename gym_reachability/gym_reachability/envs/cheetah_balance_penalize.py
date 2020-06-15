"""
Authors: Neil Lugovoy   ( nflugovoy@berkeley.edu )

See the LICENSE in the root directory of this repo for license info.
"""

import gym.spaces  # needed to avoid warning from gym
from gym_reachability.gym_reachability.envs.cheetah_balance import CheetahBalanceEnv
from ray.rllib.utils.annotations import override

"""
in this version of the cheetah problem the reward will be -1 for all states in the failure set
(head or front leg touching the ground) and 0 for all other states. thus only providing reward info
through penalization
"""


class CheetahBalancePenalizeEnv(CheetahBalanceEnv):

    @override(CheetahBalanceEnv)
    def l_function(self):
        return -1.0 if self.detect_contact() else 0.0

