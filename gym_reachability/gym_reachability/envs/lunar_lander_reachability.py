# Copyright (c) 2019–2020, The Regents of the University of California.
# All rights reserved.
#
# This file is subject to the terms and conditions defined in the LICENSE file
# included in this repository.
#
# Please contact the author(s) of this library if you have any questions.
# Authors: Vicenc Rubies Royo ( vrubies@berkeley.edu )
#          Neil Lugovoy   ( nflugovoy@berkeley.edu )

import numpy as np
import gym
from gym import spaces
from gym.envs.box2d.lunar_lander import LunarLander
from gym.envs.box2d.lunar_lander import SCALE, VIEWPORT_W, VIEWPORT_H, LEG_DOWN, FPS, LEG_AWAY, \
    LANDER_POLY, LEG_H, LEG_W
from Box2D.b2 import edgeShape
# NOTE the overrides cause crashes with ray in this file but I would like to include them for
# clarity in the future
#from ray.rllib.utils.annotations import override
import matplotlib.pyplot as plt
import torch
from shapely.geometry import Polygon, Point
import random
from shapely.affinity import affine_transform
from shapely.ops import triangulate

# these variables are needed to do calculations involving the terrain but are local variables
# in LunarLander reset() unfortunately

W = VIEWPORT_W / SCALE
H = VIEWPORT_H / SCALE
CHUNKS = 17 #11  # number of polygons used to make the lunar surface
HELIPAD_Y = (VIEWPORT_H / SCALE) / 2  # height of helipad in simulator scale

# height of lander body in simulator scale. LANDER_POLY has the (x,y) points that define the
# shape of the lander in pixel scale
LANDER_POLY_X = np.array(LANDER_POLY)[:, 0]
LANDER_POLY_Y = np.array(LANDER_POLY)[:, 1]

LANDER_W = (np.max(LANDER_POLY_X) - np.min(LANDER_POLY_X)) / SCALE
LANDER_H = (np.max(LANDER_POLY_Y) - np.min(LANDER_POLY_Y)) / SCALE

# distance of edge of legs from center of lander body in simulator scale
LEG_X_DIST = LEG_AWAY / SCALE
LEG_Y_DIST = LEG_DOWN / SCALE

# radius around lander to check for collisions
LANDER_RADIUS = ((LANDER_H / 2 + LEG_Y_DIST + LEG_H / SCALE) ** 2 +
                 (LANDER_W / 2 + LEG_X_DIST + LEG_W / SCALE) ** 2) ** 0.5


class LunarLanderReachability(LunarLander):

    # in the LunarLander environment the variables LANDER_POLY, LEG_AWAY, LEG_DOWN, LEG_W, LEG_H
    # SIDE_ENGINE_HEIGHT, SIDE_ENGINE_AWAY, VIEWPORT_W and VIEWPORT_H are measured in pixels
    #
    # the x and y coordinates (and their time derivatives) used for physics calculations in the
    # simulator use those values scaled by 1 / SCALE
    #
    # the observations sent to the learning algorithm when reset() or step() is called use those
    # values scaled by SCALE / (2 * VIEWPORT_H) and SCALE / (2 * VIEWPORT_Y) and centered at
    # (2 * VIEWPORT_W) / SCALE and HELIPAD_Y + LEG_DOWN / SCALE for x and y respectively
    # theta_dot is scaled by 20.0 / FPS
    #
    # this makes reading the lunar_lander.py file difficult so I have tried to make clear what scale
    # is being used here by calling them: pixel scale, simulator scale, and observation scale

    def __init__(self, device=torch.device("cpu"), mode='normal', doneType='toEnd'):

        # in LunarLander init() calls reset() which calls step() so some variables need
        # to be set up before calling init() to prevent problems from variables not being defined

        self.before_parent_init = True

        self.chunk_x = [W/(CHUNKS-1)*i for i in range(CHUNKS)]
        self.helipad_x1 = self.chunk_x[CHUNKS//2-1]
        self.helipad_x2 = self.chunk_x[CHUNKS//2+1]
        self.helipad_y = HELIPAD_Y

        # safety problem limits in --> simulator scale <--

        self.hover_min_y_dot = -0.1
        self.hover_max_y_dot = 0.1
        self.hover_min_x_dot = -0.1
        self.hover_max_x_dot = 0.1

        self.land_min_v = -1.6  # fastest that lander can be falling when it hits the ground

        self.theta_hover_max = np.radians(15.0)  # most the lander can be tilted when landing
        self.theta_hover_min = np.radians(-15.0)

        self.fly_min_x = 0  # first chunk
        self.fly_max_x = W / (CHUNKS - 1) * (CHUNKS - 1)  # last chunk
        self.midpoint_x = (self.fly_max_x + self.fly_min_x) / 2
        self.width_x = (self.fly_max_x - self.fly_min_x)

        self.fly_max_y = VIEWPORT_H / SCALE
        self.fly_min_y = 0
        self.midpoint_y = (self.fly_max_y + self.fly_min_y) / 2
        self.width_y = (self.fly_max_y - self.fly_min_y)

        self.hover_min_x = W / (CHUNKS - 1) * (CHUNKS // 2 - 1)
        self.hover_max_x = W / (CHUNKS - 1) * (CHUNKS // 2 + 1)
        self.hover_min_y = HELIPAD_Y  # calc of edges of landing pad based
        self.hover_max_y = HELIPAD_Y + 2  # on calc in parent reset()

        # set up state space bounds used in evaluating the q value function
        self.vx_bound = 10  # bounds centered at 0 so take negative for lower bound
        self.vy_bound = 10  # this is in simulator scale
        self.theta_bound = np.radians(90)
        self.theta_dot_bound = np.radians(50)

        self.viewer = None

        # Set random seed.
        self.seed_val = 0
        np.random.seed(self.seed_val)

        # Cost Params
        self.penalty = 1
        self.reward = -1
        self.costType = 'dense_ell'
        self.scaling = 1.

        # mode: normal or extend (keep track of ell & g)
        self.mode = mode
        if mode == 'extend':
            self.sim_state = np.zeros(7)
        else:
            self.sim_state = np.zeros(6)
        self.doneType = doneType

        # Visualization params
        self.img_data = None
        self.scaling_factor = 3.0
        self.slices_y = np.array([1, 0, -1]) * self.scaling_factor
        self.slices_x = np.array([-1, 0, 1]) * self.scaling_factor
        self.vis_init_flag = True
        self.visual_initial_states = [
            np.array([self.midpoint_x + self.width_x/4,
                      self.midpoint_y + self.width_y/4,
                      0, 0, 0, 0])]

        if mode == 'extend':
            self.visual_initial_states = self.extend_state(
                self.visual_initial_states)

        print("Env: mode---{:s}; doneType---{:s}".format(mode, doneType))

        # for torch
        self.device = device

        self.bounds_simulation = np.array([[self.fly_min_x, self.fly_max_x],
                                           [self.fly_min_y, self.fly_max_y],
                                           [-self.vx_bound, self.vx_bound],
                                           [-self.vy_bound, self.vy_bound],
                                           [-self.theta_bound,
                                            self.theta_bound],
                                           [-self.theta_dot_bound,
                                            self.theta_dot_bound]])

        self.tested_margins_flag = False
        self.polygon_target = [
            (self.helipad_x1, self.helipad_y),
            (self.helipad_x2, self.helipad_y),
            (self.helipad_x2, self.helipad_y + 2),
            (self.helipad_x1, self.helipad_y + 2),
            (self.helipad_x1, self.helipad_y)]
        self.target_xy_polygon = Polygon(self.polygon_target)

        super(LunarLanderReachability, self).__init__()

        self.before_parent_init = False

        # we don't use the states about whether the legs are touching
        # so 6 dimensions total.
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(6,),
                                            dtype=np.float32)

        # This makes dynamics deterministic.
        self.np_random = RandomAlias

        # Check conversions are ok.
        assert np.all(np.abs(self.obs_scale_to_simulator_scale(
               self.simulator_scale_to_obs_scale(self.bounds_simulation[:, 0]))
                - self.bounds_simulation[:, 0]) < 1e-5)

        # convert to observation scale so network can be evaluated
        self.bounds_observation = np.copy(self.bounds_simulation)
        self.bounds_observation[:, 0] = self.simulator_scale_to_obs_scale(
            self.bounds_simulation[:, 0].T)
        self.bounds_observation[:, 1] = self.simulator_scale_to_obs_scale(
            self.bounds_simulation[:, 1].T)

    def _test_margins(self):
        if self.tested_margins_flag is False:
            self.tested_margins_flag = True
        else:
            return True
        for _ in range(200):
            sampled_state = np.random.uniform(
                low=self.bounds_simulation[:, 0],
                high=self.bounds_simulation[:, 1])
            s_contains_bool = self.obstacle_polyline.contains(
                Point(sampled_state[0], sampled_state[1]))
            t_contains_bool = self.target_xy_polygon.contains(
                Point(sampled_state[0], sampled_state[1]))
            # Check possible mistmatches.
            s_margin = self.safety_margin(sampled_state)
            t_margin = self.target_margin(sampled_state)
            if (((s_margin <= 0) and  # Inside constraint set.
                    s_contains_bool is False)  # Outside constraint set.
                    or
                    ((s_margin > 0) and  # Outside constraint set.
                     s_contains_bool is True)):  # Inside constraint set.
                return False
            elif (((t_margin <= 0) and  # Inside target set.
                    t_contains_bool is False)  # Outside target set.
                    or
                    ((t_margin > 0) and  # Outside target set.
                     t_contains_bool is True)):  # Inside target set.
                return False
        return True


    # found online at:
    # https://codereview.stackexchange.com/questions/69833/..
    # generate-sample-coordinates-inside-a-polygon
    # todo{vrubies} why buggy?
    @staticmethod
    def random_points_in_polygon(polygon, k):
        "Return list of k points uniformly at random inside the polygon."
        areas = []
        transforms = []
        for t in triangulate(polygon):
            areas.append(t.area)
            (x0, y0), (x1, y1), (x2, y2), _ = t.exterior.coords
            transforms.append([x1 - x0, x2 - x0, y2 - y0, y1 - y0, x0, y0])
        points = []
        for transform in random.choices(transforms, weights=areas, k=k):
            x, y = [random.random() for _ in range(2)]
            if x + y > 1:
                p = Point(1 - x, 1 - y)
            else:
                p = Point(x, y)
            points.append(affine_transform(p, transform))
        return points

    def extend_state(self, states):
        new_states = []
        for state in states:
            l_x = self.target_margin(state)
            g_x = self.safety_margin(state)
            new_states.append(np.append(state, max(l_x, g_x)))
        return new_states

    def set_lander_state(self, state):
        # convention is x,y,x_dot,y_dot, theta, theta_dot
        # These internal variables are in --> simulator scale <--
        # changes need to be in np.float64
        self.lander.position = np.array([state[0], state[1]], dtype=np.float64)
        self.lander.linearVelocity = np.array([state[2], state[3]], dtype=np.float64)
        self.lander.angle = np.float64(state[4])
        self.lander.angularVelocity = np.float64(state[5])

        # after lander position is set have to set leg positions to be where
        # new lander position is.
        self.legs[0].position = np.array(
            [self.lander.position.x + LEG_AWAY/SCALE,
             self.lander.position.y], dtype=np.float64)
        self.legs[1].position = np.array(
            [self.lander.position.x - LEG_AWAY/SCALE,
             self.lander.position.y], dtype=np.float64)

    def generate_terrain(self, terrain_polyline=None):
        # Destroy moon.
        self.world.DestroyBody(self.moon)
        self.moon = None

        self.chunk_x = [W/(CHUNKS-1)*i for i in range(CHUNKS)]
        self.helipad_x1 = self.chunk_x[CHUNKS//2-1]
        self.helipad_x2 = self.chunk_x[CHUNKS//2+1]
        self.helipad_y = HELIPAD_Y
        # terrain
        if terrain_polyline is None:
            height = np.ones((CHUNKS+1,))
        else:
            height = terrain_polyline
        height[CHUNKS//2-3] = self.helipad_y + 2.5
        height[CHUNKS//2-2] = self.helipad_y
        height[CHUNKS//2-1] = self.helipad_y
        height[CHUNKS//2+0] = self.helipad_y
        height[CHUNKS//2+1] = self.helipad_y
        height[CHUNKS//2+2] = self.helipad_y
        # smooth_y = [0.33*(height[i-1] + height[i+0] + height[i+1]) for i in range(CHUNKS)]
        smooth_y = list(height[:-1])
        # print(smooth_y)
        # assert len(smooth_y) == len(height)
        # smooth_y = list(height)

        self.moon = self.world.CreateStaticBody(shapes=edgeShape(
            vertices=[(0, 0), (W, 0)]))
        self.sky_polys = []
        obstacle_polyline = [(self.chunk_x[0], smooth_y[0])]
        for i in range(CHUNKS-1):
            p1 = (self.chunk_x[i], smooth_y[i])
            p2 = (self.chunk_x[i+1], smooth_y[i+1])
            obstacle_polyline.append(p2)
            self.moon.CreateEdgeFixture(
                vertices=[p1, p2],
                density=0,
                friction=0.1)
            self.sky_polys.append([p1, p2, (p2[0], H), (p1[0], H)])
        # Enclose terrain within window.
        obstacle_polyline.append((W, H))
        obstacle_polyline.append((0, H))
        obstacle_polyline.append(obstacle_polyline[0])
        self.obstacle_polyline = Polygon(obstacle_polyline)
        assert self._test_margins()

        self.moon.color1 = (0.0, 0.0, 0.0)
        self.moon.color2 = (0.0, 0.0, 0.0)
        # return super(LunarLanderReachability, self).step(
        #     np.array([0, 0]) if self.continuous else 0)[0]

    def rejection_sample(self):
        flag_sample = False
        while not flag_sample:
            xy_sample = np.random.uniform(low=[self.fly_min_x,
                                               self.fly_min_y],
                                          high=[self.fly_max_x,
                                                self.fly_max_y])
            flag_sample = self.obstacle_polyline.contains(
                Point(xy_sample[0], xy_sample[1]))
        return xy_sample

    def reset(self, state_in=None, terrain_polyline=None,
              rejection_sampling=True):
        """
        resets the environment accoring to a uniform distribution.
        state_in assumed to be in simulation scale.
        :return: current state as 6d NumPy array of floats
        """
        # This returns something in --> observation scale <--.
        s = super(LunarLanderReachability, self).reset()
        self.generate_terrain(terrain_polyline=terrain_polyline)

        # Rewrite internal lander variables in --> simulation scale <--.
        if state_in is None:
            state_in = np.copy(self.obs_scale_to_simulator_scale(s))
            # Have to sample uniformly to get good coverage of the state space.
            if rejection_sampling:
                state_in[:2] = self.rejection_sample()
            else:
                point = self.random_points_in_polygon(self.obstacle_polyline, 1)[0]
                state_in[:2] = np.array([point.x, point.y])
            state_in[4] = np.random.uniform(low=-self.theta_bound,
                                            high=self.theta_bound)
        else:
            # Ensure that when specifing a state it is within
            # our simulation bounds.
            for ii in range(len(state_in)):
                state_in[ii] = np.float64(
                    min(state_in[ii], self.bounds_simulation[ii, 1]))
                state_in[ii] = np.float64(
                    max(state_in[ii], self.bounds_simulation[ii, 0]))
        self.set_lander_state(state_in)

        # Convert from simulator scale to observation scale.
        s = self.simulator_scale_to_obs_scale(state_in)

        # Return in --> observation scale <--.
        return s

    def get_state(self):
        """
        gets the current state of the environment
        :return: length 6 array of x, y, x_dot, y_dot, theta, theta_dot in simulator scale
        """
        return np.array([self.lander.position.x,
                         self.lander.position.y,
                         self.lander.linearVelocity.x,
                         self.lander.linearVelocity.y,
                         self.lander.angle,
                         self.lander.angularVelocity])

    def step(self, action):
        if self.before_parent_init:
            cost = None  # Can't be computed.
            obs_s, _, done, info = super(LunarLanderReachability, self).step(
                action)
            return np.copy(obs_s[:-2]), cost, False, {}
        else:
            # Note that l function must be computed before environment
            # steps see reamdme for proof.
            l_x_cur = self.target_margin(self.get_state())
            g_x_cur = self.safety_margin(self.get_state())

        obs_s, _, done, info = super(LunarLanderReachability, self).step(
            action)
        self.obs_state = obs_s[:-2]  # Remove states dealing with contacts.
        self.sim_state = self.obs_scale_to_simulator_scale(self.obs_state)
        l_x_nxt = self.target_margin(self.get_state())
        g_x_nxt = self.safety_margin(self.get_state())

        # cost
        if self.mode == 'extend' or self.mode == 'RA':
            fail = g_x_cur > 0
            success = l_x_cur <= 0
            if fail:
                cost = self.penalty
            elif success:
                cost = self.reward
            else:
                cost = 0.
        else:
            fail = g_x_nxt > 0
            success = l_x_nxt <= 0
            if g_x_nxt > 0 or g_x_cur > 0:
                cost = self.penalty
            elif l_x_nxt <= 0 or l_x_cur <= 0:
                cost = self.reward
            else:
                if self.costType == 'dense_ell':
                    cost = l_x_nxt
                elif self.costType == 'dense_ell_g':
                    cost = l_x_nxt + g_x_nxt
                elif self.costType == 'imp_ell_g':
                    cost = (l_x_nxt-l_x_cur) + (g_x_nxt-g_x_cur)
                elif self.costType == 'imp_ell':
                    cost = (l_x_nxt-l_x_cur)
                elif self.costType == 'sparse':
                    cost = 0. * self.scaling
                elif self.costType == 'max_ell_g':
                    cost = max(l_x_nxt, g_x_nxt)
                else:
                    cost = 0.
        # done
        # if not done and self.doneType == 'toEnd':
        #     outsideTop = (self.sim_state[1] >= self.bounds_simulation[1, 1])
        #     outsideLeft = (self.sim_state[0] <= self.bounds_simulation[0, 0])
        #     outsideRight = (self.sim_state[0] >= self.bounds_simulation[0, 1])
        #     done = outsideTop or outsideLeft or outsideRight
        if not done:
            done = fail or success
            # assert self.doneType == 'TF', 'invalid doneType'

        info = {"g_x": g_x_cur,  "l_x": l_x_cur, "g_x_nxt": g_x_nxt,
                "l_x_nxt": l_x_nxt}
        return np.copy(self.obs_state), cost, done, info

    def safety_margin(self, state):

        x = state[0]
        y = state[1]
        p = Point(x, y)
        L2_distance = self.obstacle_polyline.exterior.distance(p)
        inside = 2*self.obstacle_polyline.contains(p) - 1
        return -inside*L2_distance

    def target_margin(self, state):

        # all in simulation scale
        x = state[0]
        y = state[1]
        x_dot = state[2]
        y_dot = state[3]
        theta = state[4]

        landing_distance = np.min([
            # 10 * (theta - self.theta_hover_min),  # heading error multiply 10
            # 10 * (self.theta_hover_max - theta),  # for similar scale of units
            x - self.helipad_x1,# - LANDER_RADIUS,
            self.helipad_x2 - x,# - LANDER_RADIUS,
            y - self.helipad_y,# - LANDER_RADIUS,
            (self.helipad_y + 2) - y])# - LANDER_RADIUS])
            # ,
            # y_dot - self.hover_min_y_dot,
            # self.hover_max_y_dot - y_dot,
            # x_dot - self.hover_min_x_dot,
            # self.hover_max_x_dot - x_dot])  # speed check

        return -landing_distance

    #@staticmethod
    def simulator_scale_to_obs_scale(self, state):
        """
        converts from simulator scale to observation scale see comment at top of class
        :param state: length 6 array of x, y, x_dot, y_dot, theta, theta_dot in simulator scale
        :return: length 6 array of x, y, x_dot, y_dot, theta, theta_dot in obs scale
        needs to return array with np.float32 precision
        """
        copy_state = np.copy(state)
        chg_dims = self.observation_space.shape[0]
        x, y, x_dot, y_dot, theta, theta_dot = copy_state[:chg_dims]
        copy_state[:chg_dims] = np.array([
            (x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2),
            (y - (HELIPAD_Y + LEG_DOWN/SCALE)) / (VIEWPORT_H / SCALE / 2),
            x_dot * (VIEWPORT_W / SCALE / 2) / FPS,
            y_dot * (VIEWPORT_H / SCALE / 2) / FPS,
            theta,
            20.0*theta_dot / FPS], dtype=np.float32)  # theta_dot])
        return copy_state

    #@staticmethod
    def obs_scale_to_simulator_scale(self, state):
        """
        converts from observation scale to simulator scale see comment at top of class
        :param state: length 6 array of x, y, x_dot, y_dot, theta, theta_dot in simulator scale
        :return: length 6 array of x, y, x_dot, y_dot, theta, theta_dot in obs scale
        needs to return array with np.float64 precision
        """
        copy_state = np.copy(state)
        chg_dims = self.observation_space.shape[0]
        x, y, x_dot, y_dot, theta, theta_dot = copy_state[:chg_dims]
        copy_state[:chg_dims] = np.array([
            (x * (VIEWPORT_W / SCALE / 2)) + (VIEWPORT_W / SCALE / 2),
            (y * (VIEWPORT_H / SCALE / 2)) + (HELIPAD_Y + LEG_DOWN/SCALE),
            x_dot / ((VIEWPORT_W / SCALE / 2) / FPS),
            y_dot / ((VIEWPORT_H / SCALE / 2) / FPS),
            theta,
            theta_dot * FPS / 20.0], dtype=np.float64)  # theta_dot])
        return copy_state

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
            state_sim = self.obs_scale_to_simulator_scale(state)
            s_margin = self.safety_margin(state_sim)
            t_margin = self.target_margin(state_sim)
            if s_margin > 0:
                result = -1  # Failed.
                break
            elif t_margin <= 0:
                result = 1  # Succeeded.
                break
            # print("S_Margin: ", s_margin)
            # print("T_Margin: ", t_margin)

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
            plt.scatter(traj_x[0], traj_y[0], s=24, c=c)
            plt.plot(traj_x, traj_y, color=c, linewidth=2)

        return results

    def get_value(self, q_func, nx=41, ny=121, x_dot=0, y_dot=0, theta=0, theta_dot=0):
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

    def imshow_lander(self, extent=None, alpha=0.4):
        if self.img_data is None:
            # todo{vrubies} can we find way to supress gym window?
            img_data = self.render(mode="rgb_array")
            self.close()
            self.img_data = img_data[::2, ::3, :]  # Reduce image size.
        plt.imshow(self.img_data,
                   interpolation='none', extent=extent,
                   origin='upper', alpha=alpha)

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
                            plt.scatter(x, y, c='k', s=24)
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


class RandomAlias:
    # Note: This is a little hacky. The LunarLander uses the instance attribute self.np_random to
    # pick the moon chunks placements and also determine the randomness in the dynamics and
    # starting conditions. The size argument is only used for determining the height of the
    # chunks so this can be used to set the height of the chunks. When low=-1.0 and high=1.0 the
    # dispersion on the particles is determined on line 247 in step LunarLander which makes the
    # dynamics probabilistic. Safety Bellman Equation assumes deterministic dynamics so we set that
    # to be constant

    @staticmethod
    def uniform(low, high, size=None):
        if size is None:
            if low == -1.0 and high == 1.0:
                return 0
            else:
                return np.random.uniform(low=low, high=high)
        else:
            return np.ones(12) * HELIPAD_Y * 0.1 # this makes the ground flat
