# Copyright (c) 2019–2020, The Regents of the University of California.
# All rights reserved.
#
# This file is subject to the terms and conditions defined in the LICENSE file
# included in this repository.
#
# Please contact the author(s) of this library if you have any questions.
# Authors: Vicenc Rubies Royo   ( vrubies@berkeley.edu )

import numpy as np
import sys
import math

import gym
from gym import spaces
import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape,
                      revoluteJointDef, contactListener)
# from gym.envs.box2d.lunar_lander import LunarLander
from gym.utils import seeding
from gym.utils import EzPickle

# NOTE the overrides cause crashes with ray in this file but I would like to include them for
# clarity in the future
#from ray.rllib.utils.annotations import override
import matplotlib.pyplot as plt
import torch
import random
from shapely.geometry import Polygon, Point
from shapely.affinity import affine_transform
from shapely.ops import triangulate
from gym_reachability.gym_reachability.envs import MultiPlayerLunarLanderReachability


class TwoPlayerPursuitEvasionLunarLander(MultiPlayerLunarLanderReachability):

    def __init__(self,
                 device=torch.device("cpu"),
                 mode='normal',
                 observation_type='default',
                 reach_set=None,  # Used for state and world related reach set.
                 avoid_set=None,  # Used for state (inter-player) avoid set.
                 terrain=None,  # Used for world-related avoid set.
                 doneType='toEnd'):

        super(TwoPlayerPursuitEvasionLunarLander, self).__init__(
            device=device,
            num_players=2,
            observation_type=observation_type)

        # mode: normal or extend (keep track of ell & g)
        self.mode = mode
        if mode == 'extend':
            self.sim_state = np.zeros(self.total_obs_dim + 1)

        self.doneType = doneType

        if mode == 'extend':
            self.visual_initial_states = self.extend_state(
                self.visual_initial_states)

        print("Env: mode---{:s}; doneType---{:s}".format(mode, doneType))

    def reset(self, state_in=None, terrain_polyline=None):
        return super().reset(
            state_in=state_in, terrain_polyline=terrain_polyline)

    def step(self, action):
        return super().step(action)

    def target_margin(self, state):
        # First 6 states are for attacker. Last 6 for defender.
        assert len(state) == 12

        # Attacker target margin.
        x_a = state[0]
        y_a = state[1]
        p_a = Point(x_a, y_a)
        L2_distance_a = self.target_xy_polygon.exterior.distance(p_a)
        inside_a = 2*self.target_xy_polygon.contains(p_a) - 1
        attacker_target_margin = -inside_a*L2_distance_a

        # Defender safety margin to obstacle.
        x_d = state[0+6]
        y_d = state[1+6]
        p_d = Point(x_d, y_d)
        L2_distance_d = self.obstacle_polyline.exterior.distance(p_d)
        inside_d = 2*self.obstacle_polyline.contains(p_d) - 1
        defender_safety_margin = -inside_d*L2_distance_d

        return min(attacker_target_margin,
                   -defender_safety_margin)  # Flip sign.

    def safety_margin(self, state):
        # First 6 states are for attacker. Last 6 for defender.
        assert len(state) == 12
        capture_rad = 1.0

        # Attacker safety margin to obstacle.
        x_a = state[0]
        y_a = state[1]
        p_a = Point(x_a, y_a)
        L2_distance_a = self.obstacle_polyline.exterior.distance(p_a)
        inside_a = 2*self.obstacle_polyline.contains(p_a) - 1
        attacker_safety_margin_to_obstacle = -inside_a*L2_distance_a

        # Attacker safety margin to defender.
        x_d = state[0+6]
        y_d = state[1+6]
        x_r = x_a - x_d
        y_r = y_a - y_d
        distance_a_d = np.sqrt(x_r ** 2 + x_r ** 2)
        attacker_safety_margin_to_defender = capture_rad - distance_a_d

        return max(attacker_safety_margin_to_obstacle,
                   attacker_safety_margin_to_defender)

    def set_doneType(self, doneType):
        self.doneType = doneType

    def set_costParam(self, penalty=1, reward=-1, costType='normal', scaling=4.):
        self.penalty = penalty
        self.reward = reward
        self.costType = costType
        self.scaling = scaling

    def set_seed(self, seed):
        """ Set the random seed.

        Args:
            seed: Random seed.
        """
        self.seed_val = seed
        np.random.seed(self.seed_val)

    # TODO(vrubies) Move to the child.
    def simulate_one_trajectory(self, q_func, T=10, state=None):
        """
        simulates one trajectory in observation scale.
        """
        if state is None:
            state = self.reset()
        else:
            state = self.reset(state)
        traj_x = [state[0]]
        traj_y = [state[1]]
        result = 0  # Not finished.

        for t in range(T):
            if self.safety_margin(
                    self.obs_scale_to_simulator_scale(state)) > 0:
                result = -1  # Failed.
                break
            elif self.target_margin(
                    self.obs_scale_to_simulator_scale(state)) <= 0:
                result = 1  # Succeeded.
                break

            state_tensor = torch.FloatTensor(state,
                                             device=self.device).unsqueeze(0)
            action_index = q_func(state_tensor).min(dim=1)[1].item()

            state, _, done, _ = self.step(action_index)
            traj_x.append(state[0])
            traj_y.append(state[1])
            if done:
                result = -1
                break

        return traj_x, traj_y, result

    # TODO(vrubies) Move to the child.
    def simulate_trajectories(self, q_func, T=10, num_rnd_traj=None,
                              states=None):
        assert ((num_rnd_traj is None and states is not None) or
                (num_rnd_traj is not None and states is None) or
                (len(states) == num_rnd_traj))
        trajectories = []

        if states is None:
            results = np.empty(shape=(num_rnd_traj,), dtype=int)
            for idx in range(num_rnd_traj):
                traj_x, traj_y, result = self.simulate_one_trajectory(
                    q_func, T=T)
                trajectories.append((traj_x, traj_y))
                results[idx] = result
        else:
            results = np.empty(shape=(len(states),), dtype=int)
            for idx, state in enumerate(states):
                traj_x, traj_y, result = self.simulate_one_trajectory(
                    q_func, T=T, state=state)
                trajectories.append((traj_x, traj_y))
                results[idx] = result

        return trajectories, results

    # TODO(vrubies) Move to the child.
    def plot_trajectories(self, q_func, T=10, num_rnd_traj=None, states=None,
                          c='w'):
        # plt.figure(2)
        assert ((num_rnd_traj is None and states is not None) or
                (num_rnd_traj is not None and states is None) or
                (len(states) == num_rnd_traj))
        # plt.clf()
        plt.subplot(len(self.slices_y), len(self.slices_x), 1)
        trajectories, results = self.simulate_trajectories(
            q_func, T=T, num_rnd_traj=num_rnd_traj, states=states)
        for traj in trajectories:
            traj_x, traj_y = traj
            plt.scatter(traj_x[0], traj_y[0], s=48, c=c)
            plt.plot(traj_x, traj_y, color=c, linewidth=2)

        return results

    # TODO(vrubies) Move to the child.
    def get_value(self, q_func, nx=41, ny=121,
                  x_dot=0, y_dot=0, theta=0, theta_dot=0):
        v = np.zeros((nx, ny))
        it = np.nditer(v, flags=['multi_index'])
        xs = np.linspace(self.bounds_observation[0, 0],
                         self.bounds_observation[0, 1], nx)
        ys = np.linspace(self.bounds_observation[1, 0],
                         self.bounds_observation[1, 1], ny)
        # Convert slice simulation variables to observation scale.
        (_, _,
         x_dot, y_dot, theta, theta_dot) = self.simulator_scale_to_obs_scale(
            np.array([0, 0, x_dot, y_dot, theta, theta_dot]))
        # print("Start value collection on grid...")
        while not it.finished:
            idx = it.multi_index

            x = xs[idx[0]]
            y = ys[idx[1]]
            l_x = self.target_margin(
                np.array([x, y, x_dot, y_dot, theta, theta_dot]))
            g_x = self.safety_margin(
                np.array([x, y, x_dot, y_dot, theta, theta_dot]))

            if self.mode == 'normal' or self.mode == 'RA':
                state = torch.FloatTensor(
                    [x, y, x_dot, y_dot, theta, theta_dot],
                    device=self.device).unsqueeze(0)
            else:
                z = max([l_x, g_x])
                state = torch.FloatTensor(
                    [x, y, x_dot, y_dot, theta, theta_dot, z],
                    device=self.device).unsqueeze(0)

            v[idx] = q_func(state).min(dim=1)[0].item()
            it.iternext()
        # print("End value collection on grid.")
        return v, xs, ys
    # TODO(vrubies) Move to the child.
    def get_axes(self):
        """ Gets the bounds for the environment.

        Returns:
            List containing a list of bounds for each state coordinate and a
        """
        aspect_ratio = (
            (self.bounds_observation[0, 1] - self.bounds_observation[0, 0]) /
            (self.bounds_observation[1, 1] - self.bounds_observation[1, 0]))
        axes = np.array([self.bounds_observation[0, 0] - 0.05,
                         self.bounds_observation[0, 1] + 0.05,
                         self.bounds_observation[1, 0] - 0.15,
                         self.bounds_observation[1, 1] + 0.15])
        return [axes, aspect_ratio]
    # TODO(vrubies) Move to the child.
    def imshow_lander(self, extent=None, alpha=0.4):
        if self.img_data is None:
            # todo{vrubies} can we find way to supress gym window?
            img_data = self.render(mode="rgb_array")
            self.close()
            self.img_data = img_data[::2, ::3, :]  # Reduce image size.
        plt.imshow(self.img_data,
                   interpolation='none', extent=extent,
                   origin='upper', alpha=alpha)
    # TODO(vrubies) Move to the child.
    def visualize(self, q_func, no_show=False,
                  vmin=-50, vmax=50, nx=21, ny=21,
                  labels=['', ''],
                  boolPlot=False, plotZero=False,
                  cmap='coolwarm'):
        """ Overlays analytic safe set on top of state value function.

        Args:
            v: State value function.
        """
        # plt.figure(1)
        plt.clf()
        axes = self.get_axes()
        for y_jj, y_dot in enumerate(self.slices_y):
            for x_ii, x_dot in enumerate(self.slices_x):
                plt.subplot(len(self.slices_y), len(self.slices_x),
                            y_jj*len(self.slices_y)+x_ii+1)
                # print("Subplot -> ", y_jj*len(self.slices_y)+x_ii+1)
                v, xs, ys = self.get_value(q_func, nx, ny,
                                           x_dot=x_dot, y_dot=y_dot, theta=0,
                                           theta_dot=0)
                #im = visualize_matrix(v.T, self.get_axes(labels), no_show, vmin=vmin, vmax=vmax)

                if boolPlot:
                    im = plt.imshow(v.T > vmin,
                                    interpolation='none', extent=axes[0],
                                    origin="lower", cmap=cmap)
                else:
                    im = plt.imshow(v.T,
                                    interpolation='none', extent=axes[0],
                                    origin="lower", cmap=cmap)  #,vmin=vmin, vmax=vmax)
                    # cbar = plt.colorbar(im, pad=0.01, shrink=0.95,
                    #                     ticks=[vmin, 0, vmax])
                    # cbar.ax.set_yticklabels(labels=[vmin, 0, vmax],
                    #                         fontsize=24)

                self.imshow_lander(extent=axes[0], alpha=0.4)
                ax = plt.gca()
                # Plot bounadries of constraint set.
                # plt.plot(self.x_box1_pos, self.y_box1_pos, color="black")
                # plt.plot(self.x_box2_pos, self.y_box2_pos, color="black")
                # plt.plot(self.x_box3_pos, self.y_box3_pos, color="black")

                # Plot boundaries of target set.
                # plt.plot(self.x_box4_pos, self.y_box4_pos, color="black")

                # Plot zero level set
                if plotZero:
                    it = np.nditer(v, flags=['multi_index'])
                    while not it.finished:
                        idx = it.multi_index
                        x = xs[idx[0]]
                        y = ys[idx[1]]

                        if v[idx] <= 0:
                            plt.scatter(x, y, c='k', s=48)
                        it.iternext()


                ax.axis(axes[0])
                ax.grid(False)
                ax.set_aspect(axes[1])  # makes equal aspect ratio
                if labels is not None:
                    ax.set_xlabel(labels[0], fontsize=52)
                    ax.set_ylabel(labels[1], fontsize=52)

                ax.tick_params(axis='both', which='both',  # both x and y axes, both major and minor ticks are affected
                               bottom=False, top=False,    # ticks along the top and bottom edges are off
                               left=False, right=False)    # ticks along the left and right edges are off
                ax.set_xticklabels([])
                ax.set_yticklabels([])


        if not no_show:
            plt.show()


# class RandomAlias:
#     # Note: This is a little hacky. The LunarLander uses the instance attribute self.np_random to
#     # pick the moon chunks placements and also determine the randomness in the dynamics and
#     # starting conditions. The size argument is only used for determining the height of the
#     # chunks so this can be used to set the height of the chunks. When low=-1.0 and high=1.0 the
#     # dispersion on the particles is determined on line 247 in step LunarLander which makes the
#     # dynamics probabilistic. Safety Bellman Equation assumes deterministic dynamics so we set that
#     # to be constant

#     @staticmethod
#     def uniform(low, high, size=None):
#         if size is None:
#             if low == -1.0 and high == 1.0:
#                 return 0
#             else:
#                 return np.random.uniform(low=low, high=high)
#         else:
#             return np.ones(12) * HELIPAD_Y * 0.1 # this makes the ground flat
