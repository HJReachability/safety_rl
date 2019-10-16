from gym.envs.registration import register

register(
    id="cartpole_reach-v0",
    entry_point="mdr_rl.gym_reachability.gym_reachability.envs:CartPoleReachabilityEnv",
)

register(
    id="lunar_lander_reachability-v0",
    entry_point="mdr_rl.gym_reachability.gym_reachability.envs:LunarLanderReachability"
)


register(
    id="double_integrator-v0",
    entry_point="mdr_rl.gym_reachability.gym_reachability.envs:DoubleIntegratorEnv"
)

register(
    id="cheetah_balance-v0",
    entry_point="mdr_rl.gym_reachability.gym_reachability.envs:CheetahBalanceEnv"
)

register(
    id="cheetah_balance_penalize-v0",
    entry_point="mdr_rl.gym_reachability.gym_reachability.envs:CheetahBalancePenalizeEnv"
)
