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

import gym.spaces
import gym
from gym.envs.mujoco.half_cheetah import HalfCheetahEnv
import numpy as np
import math
from ray.rllib.utils.annotations import override


class CheetahBalanceEnv(HalfCheetahEnv):

    @override(HalfCheetahEnv)
    def __init__(self):

        # super init
        super().__init__()

        # initialize cheetah as standing
        self.init_qpos[2] = math.radians(-70.0)
        self.init_qpos[1] += 0.05

    @override(HalfCheetahEnv)
    def reset_model(self):
        """
        resets model to standing state
        :return: observation of new state
        """

        # code for resetting the state from parent env
        q_pos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        q_vel = self.init_qvel + self.np_random.randn(self.model.nv) * .1

        # angle adjust height so it won't go through the floor
        q_pos[1] = self.init_qpos[1] + self.np_random.uniform(low=-0.01, high=0.01)

        self.set_state(q_pos, q_vel)
        return self._get_obs()

    @override(HalfCheetahEnv)
    def step(self, action):
        """
        steps the physics of the environment
        :param action: input action of torques to apply to motors at joints
        :return: observation of new state
        """

        ob, reward, done, info = super(CheetahBalanceEnv, self).step(action)
        r = self.l_function()
        return ob, r, done, info

    def detect_contact(self):
        """
        :return: true if the cheetah's head or front leg is touching the ground, false if not
        """

        # Geom IDs :['floor':0, 'torso':1, 'head':2, 'bthigh':3, 'bshin':4, 'bfoot':5, 'fthigh':6,
        # 'fshin':7, 'ffoot':8]
        return any([self.data.contact[i].geom2 in [2, 7, 8] for i in range(self.data.ncon)])

    def l_function(self):
        """
        :return: the signed distance of the environment at the current state to the failure
         set. For this problem the failure set is the set of all states where the cheetah's head,
         front foot or front shin is touching the ground.

        """

        # calculate the position of the head based on the lengths of the torso, head and thickness
        # of the head these values can be found in the half_cheetah.xml file in openai gym repo
        if self.detect_contact():
            return -1.0

        torso_angle = 2 * math.atan(self.data.body_xquat[1][2] / self.data.body_xquat[1][0])
        head_pos = self.data.body_xpos[-3][-1] + (math.cos(torso_angle + 0.87) * 0.15) - 0.046

        # min of dist from front foot actuator to ground, head to ground, and shin to ground
        return np.min([self.data.body_xpos[-1][-1], head_pos, self.data.body_xpos[-2][-1]])

    @override(HalfCheetahEnv)
    def viewer_setup(self):
        """
        sets up viewer such that the camera elevation is 0 so the cheetah's orientation relative to ground can be seen
        :return: None
        """

        HalfCheetahEnv.viewer_setup(self)
        if self.viewer is not None:
            self.viewer.cam.elevation = 0
