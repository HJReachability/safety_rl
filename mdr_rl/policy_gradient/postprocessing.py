from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.signal
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import DeveloperAPI
###########################################################
# NFL compute advantages by using trajectory outcome from equation 8
from mdr_rl.utils import sbe_outcome
###########################################################


def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


class Postprocessing(object):
    """Constant definitions for postprocessing."""

    ADVANTAGES = "advantages"
    VALUE_TARGETS = "value_targets"


@DeveloperAPI
###########################################################
def compute_advantages(rollout, last_r, gamma=0.9, lambda_=1.0, use_gae=True, 
    use_sbe=False):
    # NFL: added use_sbe option
    ###########################################################
    """Given a rollout, compute its value targets and the advantage.

    Args:
        rollout (SampleBatch): SampleBatch of a single trajectory
        last_r (float): Value estimation for last observation
        gamma (float): Discount factor.
        lambda_ (float): Parameter for GAE
        use_gae (bool): Using Generalized Advantage Estimation
        ###########################################################
        use_sbe (bool): Using Safety Bellman Equation outcome from equation 8
        ###########################################################

    Returns:
        SampleBatch (SampleBatch): Object with experience from rollout and
            processed rewards.
    """

    traj = {}
    trajsize = len(rollout[SampleBatch.ACTIONS])
    for key in rollout:
        traj[key] = np.stack(rollout[key])

    if use_gae:
        ###########################################################
        if use_sbe:  # NFL: added since GAE is not supported yet
            raise NotImplementedError('Generalized Advantage Estimation with' 
                'Safety Bellman Equation is not yet supported')
        ###########################################################
        assert SampleBatch.VF_PREDS in rollout, "Values not found!"
        vpred_t = np.concatenate(
            [rollout[SampleBatch.VF_PREDS],
             np.array([last_r])])
        delta_t = (
            traj[SampleBatch.REWARDS] + gamma * vpred_t[1:] - vpred_t[:-1])
        # This formula for the advantage comes
        # "Generalized Advantage Estimation": https://arxiv.org/abs/1506.02438
        traj[Postprocessing.ADVANTAGES] = discount(delta_t, gamma * lambda_)
        traj[Postprocessing.VALUE_TARGETS] = (
            traj[Postprocessing.ADVANTAGES] +
            traj[SampleBatch.VF_PREDS]).copy().astype(np.float32)
    else:
        rewards_plus_v = np.concatenate(
            [rollout[SampleBatch.REWARDS],
             np.array([last_r])])

        ###########################################################
        if use_sbe:  # NFL: This is the only change needed for Safety Bellman Equation
            traj[Postprocessing.ADVANTAGES] = sbe_outcome(rewards_plus_v,
                                                          gamma)[:-1]
        else:
            traj[Postprocessing.ADVANTAGES] = discount(rewards_plus_v,
                                                       gamma)[:-1]

        # NFL: I have added this block to correctly compute advantages with critic without GAE
        # see issue 3746 on ray for details. This isn't specific to SBE. I will get this
        # merged soon
        if SampleBatch.VF_PREDS in rollout:
            traj[Postprocessing.VALUE_TARGETS] = \
                traj[Postprocessing.ADVANTAGES].copy().astype(np.float32)
            traj[Postprocessing.ADVANTAGES] -= traj[SampleBatch.VF_PREDS]
        else:
            traj[Postprocessing.VALUE_TARGETS] = np.zeros_like(
                traj[Postprocessing.ADVANTAGES])
        ###########################################################
    traj[Postprocessing.ADVANTAGES] = traj[
        Postprocessing.ADVANTAGES].copy().astype(np.float32)

    assert all(val.shape[0] == trajsize for val in traj.values()), \
        "Rollout stacked incorrectly!"
    return SampleBatch(traj)