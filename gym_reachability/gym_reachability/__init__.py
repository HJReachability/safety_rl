# Copyright (c) 2019–2020, The Regents of the University of California.
# All rights reserved.
#
# This file is subject to the terms and conditions defined in the LICENSE file
# included in this repository.
#
# Please contact the author(s) of this library if you have any questions.
# Authors: Neil Lugovoy   ( nflugovoy@berkeley.edu )
#          Vicenc Rubies-Royo  ( vrubies@berkeley.edu )

from gym.envs.registration import register

register(
    id="cartpole_reach-v0",
    entry_point="gym_reachability.gym_reachability.envs:CartPoleReachabilityEnv",
)

register(
    id="lunar_lander_reachability-v0",
    entry_point="gym_reachability.gym_reachability.envs:LunarLanderReachability"
)

register(
    id="multi_player_lunar_lander_reachability-v0",
    entry_point="gym_reachability.gym_reachability.envs:MultiPlayerLunarLanderReachability"
)

register(
    id="two_player_pursuit_evasion_lunar_lander-v0",
    entry_point="gym_reachability.gym_reachability.envs:TwoPlayerPursuitEvasionLunarLander"
)

register(
    id="one_player_reach_avoid_lunar_lander-v0",
    entry_point="gym_reachability.gym_reachability.envs:OnePlayerReachAvoidLunarLander"
)

register(
    id="double_integrator-v0",
    entry_point="gym_reachability.gym_reachability.envs:DoubleIntegratorEnv"
)


register(
    id="dubins_car-v0",
    entry_point="gym_reachability.gym_reachability.envs:DubinsCarEnv"
)


register(
    id="point_mass-v0",
    entry_point="gym_reachability.gym_reachability.envs:PointMassEnv"
)

register(
    id="zermelo_kc-v0",
    entry_point="gym_reachability.gym_reachability.envs:ZermeloKCEnv"
)

register(
    id="zermelo_show-v0",
    entry_point="gym_reachability.gym_reachability.envs:ZermeloShowEnv"
)

register(
    id="cheetah_balance-v0",
    entry_point="gym_reachability.gym_reachability.envs:CheetahBalanceEnv"
)


register(
    id="cheetah_balance_penalize-v0",
    entry_point="gym_reachability.gym_reachability.envs:CheetahBalancePenalizeEnv"
)
