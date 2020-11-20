# Copyright (c) 2019–2020, The Regents of the University of California.
# All rights reserved.
#
# This file is subject to the terms and conditions defined in the LICENSE file
# included in this repository.
#
# Please contact the author(s) of this library if you have any questions.
# Authors: Vicenc Rubies Royo   ( vrubies@berkeley.edu )

import numpy as np
import gym
from gym import spaces
import sys, math
import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape,
                      revoluteJointDef, contactListener)
from gym.envs.box2d.lunar_lander import LunarLander
from gym.utils import seeding, EzPickle
from gym.envs.box2d.lunar_lander import SCALE, VIEWPORT_W, VIEWPORT_H, LEG_DOWN, FPS, LEG_AWAY, \
    LANDER_POLY, LEG_H, LEG_W, LEG_SPRING_TORQUE, SIDE_ENGINE_HEIGHT, SIDE_ENGINE_AWAY
from Box2D.b2 import edgeShape
# NOTE the overrides cause crashes with ray in this file but I would like to include them for
# clarity in the future
from ray.rllib.utils.annotations import override
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

INITIAL_RANDOM = 1000.0   # Set 1500 to make game harder

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


class TwoPlayerLunarLanderReachability(LunarLander):

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
        print("SEG TEST 1")
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
            self.sim_state = np.zeros(12+1)
        else:
            self.sim_state = np.zeros(12)
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

        print("SEG TEST 2")
        # From parent constuctor.
        EzPickle.__init__(self)
        self.seed()
        self.viewer = None
        self.world = Box2D.b2World()
        self.moon = None
        self.lander = {}
        self.legs = {}
        self.particles = []
        self.prev_reward = None

        self.polygon_target = [
            (self.helipad_x1, self.helipad_y),
            (self.helipad_x2, self.helipad_y),
            (self.helipad_x2, self.helipad_y + 2),
            (self.helipad_x1, self.helipad_y + 2),
            (self.helipad_x1, self.helipad_y)]
        self.target_xy_polygon = Polygon(self.polygon_target)

        # we don't use the states about whether the legs are touching
        # so 6 dimensions total.
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(6 * 2,),
                                            dtype=np.float32)

        self.action_space = spaces.Discrete(4 ** 4)

        # this is the hack from above to make the ground flat
        # self.np_random = RandomAlias
        print("SEG TEST 3")
        self.bounds_simulation = np.array([[self.fly_min_x, self.fly_max_x],
                                           [self.fly_min_y, self.fly_max_y],
                                           [-self.vx_bound, self.vx_bound],
                                           [-self.vy_bound, self.vy_bound],
                                           [-self.theta_bound,
                                            self.theta_bound],
                                           [-self.theta_dot_bound,
                                            self.theta_dot_bound],
                                           [self.fly_min_x, self.fly_max_x],
                                           [self.fly_min_y, self.fly_max_y],
                                           [-self.vx_bound, self.vx_bound],
                                           [-self.vy_bound, self.vy_bound],
                                           [-self.theta_bound,
                                            self.theta_bound],
                                           [-self.theta_dot_bound,
                                            self.theta_dot_bound]])

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

        print("SEG TEST 4")
        self.reset()

    # found online at:
    # https://codereview.stackexchange.com/questions/69833/..
    # generate-sample-coordinates-inside-a-polygon
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

    def set_lander_state(self, state, key):
        # convention is x,y,x_dot,y_dot, theta, theta_dot
        # These internal variables are in --> simulator scale <--
        # changes need to be in np.float64
        self.lander[key].position = np.array([state[0], state[1]],
                                             dtype=np.float64)
        self.lander[key].linearVelocity = np.array([state[2], state[3]],
                                                   dtype=np.float64)
        self.lander[key].angle = np.float64(state[4])
        self.lander[key].angularVelocity = np.float64(state[5])

        # after lander position is set have to set leg positions to be where
        # new lander position is.
        self.legs[key][0].position = np.array(
            [self.lander[key].position.x + LEG_AWAY/SCALE,
             self.lander[key].position.y], dtype=np.float64)
        self.legs[key][1].position = np.array(
            [self.lander[key].position.x - LEG_AWAY/SCALE,
             self.lander[key].position.y], dtype=np.float64)

    def generate_lander(self, initial_state, key):
        # Generate Landers
        initial_y = initial_state[0]
        initial_x = initial_state[0]  # VIEWPORT_W/SCALE/2
        self.lander[key] = self.world.CreateDynamicBody(
            position=(initial_x, initial_y),
            angle=0.0,
            fixtures = fixtureDef(
                shape=polygonShape(vertices=[(x/SCALE, y/SCALE) for x, y in LANDER_POLY]),
                density=5.0,
                friction=0.1,
                categoryBits=0x0010,
                maskBits=0x001,   # collide only with ground
                restitution=0.0)  # 0.99 bouncy
                )
        self.lander[key].color1 = (0.5, 0.4, 0.9)
        self.lander[key].color2 = (0.3, 0.3, 0.5)
        self.lander[key].ApplyForceToCenter( (
            self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM),
            self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM)
            ), True)

        self.legs[key] = []
        for i in [-1, +1]:
            leg = self.world.CreateDynamicBody(
                position=(initial_x - i*LEG_AWAY/SCALE, initial_y),
                angle=(i * 0.05),
                fixtures=fixtureDef(
                    shape=polygonShape(box=(LEG_W/SCALE, LEG_H/SCALE)),
                    density=1.0,
                    restitution=0.0,
                    categoryBits=0x0020,
                    maskBits=0x001)
                )
            leg.ground_contact = False
            leg.color1 = (0.5, 0.4, 0.9)
            leg.color2 = (0.3, 0.3, 0.5)
            rjd = revoluteJointDef(
                bodyA=self.lander[key],
                bodyB=leg,
                localAnchorA=(0, 0),
                localAnchorB=(i * LEG_AWAY/SCALE, LEG_DOWN/SCALE),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=LEG_SPRING_TORQUE,
                motorSpeed=+0.3 * i  # low enough not to jump back into the sky
                )
            if i == -1:
                rjd.lowerAngle = +0.9 - 0.5  # The most esoteric numbers here, angled legs have freedom to travel within
                rjd.upperAngle = +0.9
            else:
                rjd.lowerAngle = -0.9
                rjd.upperAngle = -0.9 + 0.5
            leg.joint = self.world.CreateJoint(rjd)
            self.legs[key].append(leg)

    def _destroy(self):
        if not self.moon: return
        self.world.contactListener = None
        self._clean_particles(True)
        self.world.DestroyBody(self.moon)
        self.moon = None
        for ii, _ in enumerate(self.lander):
            self.world.DestroyBody(self.lander[ii])
            self.lander[ii] = None
            self.world.DestroyBody(self.legs[ii][0])
            self.world.DestroyBody(self.legs[ii][1])

    def generate_terrain_and_landers(self, terrain_polyline=None):
        # self.chunk_x = [W/(CHUNKS-1)*i for i in range(CHUNKS)]
        # self.helipad_x1 = self.chunk_x[CHUNKS//2-1]
        # self.helipad_x2 = self.chunk_x[CHUNKS//2+1]
        # self.helipad_y = HELIPAD_Y
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

        self.moon.color1 = (0.0, 0.0, 0.0)
        self.moon.color2 = (0.0, 0.0, 0.0)

        initial_states = [self.rejection_sample() for _ in range(2)]
        self.drawlist = []
        for ii, initial_state in enumerate(initial_states):
            self.generate_lander(initial_state, ii)
            self.drawlist += [self.lander[ii], self.legs[ii]]

        s, _, _, _ = self.step(0)
        return s

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

    def reset(self, state_in=None, terrain_polyline=None):
        """
        resets the environment accoring to a uniform distribution.
        state_in assumed to be in simulation scale.
        :return: current state as 6d NumPy array of floats
        """
        self._destroy()
        # This returns something in --> observation scale <--.

        s = self.generate_terrain_and_landers(
            terrain_polyline=terrain_polyline)


        # Rewrite internal lander variables in --> simulation scale <--.
        if state_in is None:
            state_in = np.copy(self.obs_scale_to_simulator_scale(s))
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
        self.set_lander_state(state_in[:6], 1)
        self.set_lander_state(state_in[6:], 2)

        # Convert from simulator scale to observation scale.
        s = self.simulator_scale_to_obs_scale(state_in)

        # Return in --> observation scale <--.
        return s

    def parent_step(self, action, key):
        # Action needs to be single action 0-3.
        assert self.action_space.contains(action), "%r (%s) invalid " % (action, type(action))

        # Engines
        tip  = (math.sin(self.lander[key].angle), math.cos(self.lander[key].angle))
        side = (-tip[1], tip[0])

        m_power = 0.0
        if action == 2:
            # Main engine
            m_power = 1.0
            ox = tip[0] * 4/SCALE  # 4 is move a bit downwards, +-2 for randomness
            oy = -tip[1] * 4/SCALE
            impulse_pos = (self.lander[key].position[0] + ox, self.lander[key].position[1] + oy)
            p = self._create_particle(3.5,  # 3.5 is here to make particle speed adequate
                                      impulse_pos[0],
                                      impulse_pos[1],
                                      m_power)  # particles are just a decoration
            p.ApplyLinearImpulse((ox * MAIN_ENGINE_POWER * m_power, oy * MAIN_ENGINE_POWER * m_power),
                                 impulse_pos,
                                 True)
            self.lander[key].ApplyLinearImpulse((-ox * MAIN_ENGINE_POWER * m_power, -oy * MAIN_ENGINE_POWER * m_power),
                                           impulse_pos,
                                           True)

        s_power = 0.0
        if action in [1, 3]:
            # Orientation engines
            direction = action-2
            s_power = 1.0
            ox = side[0] * direction * SIDE_ENGINE_AWAY/SCALE
            oy = side[1] * direction * SIDE_ENGINE_AWAY/SCALE
            impulse_pos = (self.lander[key].position[0] + ox - tip[0] * 17/SCALE,
                           self.lander[key].position[1] + oy + tip[1] * SIDE_ENGINE_HEIGHT/SCALE)
            p = self._create_particle(0.7, impulse_pos[0], impulse_pos[1], s_power)
            p.ApplyLinearImpulse((ox * SIDE_ENGINE_POWER * s_power, oy * SIDE_ENGINE_POWER * s_power),
                                 impulse_pos
                                 , True)
            self.lander[key].ApplyLinearImpulse((-ox * SIDE_ENGINE_POWER * s_power, -oy * SIDE_ENGINE_POWER * s_power),
                                           impulse_pos,
                                           True)

        self.world.Step(1.0/FPS, 6*30, 2*30)

        pos = self.lander[key].position
        vel = self.lander[key].linearVelocity
        state = [
            (pos.x - VIEWPORT_W/SCALE/2) / (VIEWPORT_W/SCALE/2),
            (pos.y - (self.helipad_y+LEG_DOWN/SCALE)) / (VIEWPORT_H/SCALE/2),
            vel.x*(VIEWPORT_W/SCALE/2)/FPS,
            vel.y*(VIEWPORT_H/SCALE/2)/FPS,
            self.lander[key].angle,
            20.0*self.lander[key].angularVelocity/FPS,
            1.0 if self.legs[key][0].ground_contact else 0.0,
            1.0 if self.legs[key][1].ground_contact else 0.0
            ]
        assert len(state) == 8

        reward = 0
        shaping = \
            - 100*np.sqrt(state[0]*state[0] + state[1]*state[1]) \
            - 100*np.sqrt(state[2]*state[2] + state[3]*state[3]) \
            - 100*abs(state[4]) + 10*state[6] + 10*state[7]  # And ten points for legs contact, the idea is if you
                                                             # lose contact again after landing, you get negative reward
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        reward -= m_power*0.30  # less fuel spent is better, about -30 for heuristic landing
        reward -= s_power*0.03

        done = False
        if self.game_over or abs(state[0]) >= 1.0:
            done = True
            reward = -100
        if not self.lander[key].awake:
            done = True
            reward = +100
        return np.array(state, dtype=np.float32), reward, done, {}

    def step(self, action):

        l_x_cur = self.target_margin(self.sim_state)
        g_x_cur = self.safety_margin(self.sim_state)

        actions = [int(action/4), action % 4]
        state_list = []
        reward_list = []
        done_list = []
        info_list = []
        for ii in range(2):
            state_ii, reward_ii, done_ii, info_ii = self.parent_step(
                actions[ii], ii)
            state_list.append(state_ii[:-2])
            reward_list.append(reward_ii)
            done_list.append(done_ii)
            info_list.append(info_ii)
        self.obs_state = np.concatenate(state_list)
        self.sim_state = self.obs_scale_to_simulator_scale(self.obs_state)

        l_x_nxt = self.target_margin(self.sim_state)
        g_x_nxt = self.safety_margin(self.sim_state)

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
        if not np.any(done_list) and self.doneType == 'toEnd':
            outsideTop = (self.sim_state[1] >= self.bounds_simulation[1, 1])
            outsideLeft = (self.sim_state[0] <= self.bounds_simulation[0, 0])
            outsideRight = (self.sim_state[0] >= self.bounds_simulation[0, 1])
            done = outsideTop or outsideLeft or outsideRight
        elif not np.any(done_list):
            done = fail or success
            assert self.doneType == 'TF', 'invalid doneType'

        info = {"g_x": g_x_cur,  "l_x": l_x_cur, "g_x_nxt": g_x_nxt,
                "l_x_nxt": l_x_nxt}
        return np.copy(self.obs_state), cost, done, info

    def _create_particle(self, mass, x, y, ttl):
        p = self.world.CreateDynamicBody(
            position = (x, y),
            angle=0.0,
            fixtures = fixtureDef(
                shape=circleShape(radius=2/SCALE, pos=(0, 0)),
                density=mass,
                friction=0.1,
                categoryBits=0x0100,
                maskBits=0x001,  # collide only with ground
                restitution=0.3)
                )
        p.ttl = ttl
        self.particles.append(p)
        self._clean_particles(False)
        return p

    def _clean_particles(self, all):
        while self.particles and (all or self.particles[0].ttl < 0):
            self.world.DestroyBody(self.particles.pop(0))

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
        (x_a, y_a, x_dot_a, y_dot_a,
         theta_a, theta_dot_a,
         x_d, y_d, x_dot_d, y_dot_d,
         theta_d, theta_dot_d) = copy_state[:chg_dims]
        copy_state[:int(chg_dims/2)] = np.array([
            (x_a - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2),
            (y_a - (HELIPAD_Y + LEG_DOWN/SCALE)) / (VIEWPORT_H / SCALE / 2),
            x_dot_a * (VIEWPORT_W / SCALE / 2) / FPS,
            y_dot_a * (VIEWPORT_H / SCALE / 2) / FPS,
            theta_a,
            20.0*theta_dot_a / FPS], dtype=np.float32)  # theta_dot])
        copy_state[int(chg_dims/2):chg_dims] = np.array([
            (x_d - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2),
            (y_d - (HELIPAD_Y + LEG_DOWN/SCALE)) / (VIEWPORT_H / SCALE / 2),
            x_dot_d * (VIEWPORT_W / SCALE / 2) / FPS,
            y_dot_d * (VIEWPORT_H / SCALE / 2) / FPS,
            theta_d,
            20.0*theta_dot_d / FPS], dtype=np.float32)  # theta_dot])
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
        (x_a, y_a, x_dot_a, y_dot_a, theta_a, theta_dot_a,
         x_d, y_d, x_dot_d, y_dot_d, theta_d, theta_dot_d) = copy_state[:chg_dims]
        copy_state[:int(chg_dims/2)] = np.array([
            (x_a * (VIEWPORT_W / SCALE / 2)) + (VIEWPORT_W / SCALE / 2),
            (y_a * (VIEWPORT_H / SCALE / 2)) + (HELIPAD_Y + LEG_DOWN/SCALE),
            x_dot_a / ((VIEWPORT_W / SCALE / 2) / FPS),
            y_dot_a / ((VIEWPORT_H / SCALE / 2) / FPS),
            theta_a,
            theta_dot_a * FPS / 20.0], dtype=np.float64)  # theta_dot])
        copy_state[int(chg_dims/2):chg_dims] = np.array([
            (x_d * (VIEWPORT_W / SCALE / 2)) + (VIEWPORT_W / SCALE / 2),
            (y_d * (VIEWPORT_H / SCALE / 2)) + (HELIPAD_Y + LEG_DOWN/SCALE),
            x_dot_d / ((VIEWPORT_W / SCALE / 2) / FPS),
            y_dot_d / ((VIEWPORT_H / SCALE / 2) / FPS),
            theta_d,
            theta_dot_d * FPS / 20.0], dtype=np.float64)  # theta_dot])
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
            plt.scatter(traj_x[0], traj_y[0], s=48, c=c)
            plt.plot(traj_x, traj_y, color=c, linewidth=2)

        return results

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
