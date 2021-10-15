"""
Please contact the author(s) of this library if you have any questions.
Authors: Vicenc Rubies Royo   ( vrubies@berkeley.edu )

This module implements the parent class for the lunar lander environments,
e.g., single lunar lander environment.
"""

import numpy as np
import math

import gym
from gym import spaces
import Box2D
from Box2D.b2 import (
    edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef,
    contactListener
)
from gym.utils import seeding
from gym.utils import EzPickle

# NOTE the overrides cause crashes with ray in this file but I would like to
# include them for clarity in the future
# from ray.rllib.utils.annotations import override
import matplotlib.pyplot as plt
import torch
import random
from shapely.geometry import Polygon, Point
from shapely.affinity import affine_transform
from shapely.ops import triangulate


class MultiPlayerContactDetector(contactListener):
  """Implements a contact detector.
  """

  def __init__(self, env):
    contactListener.__init__(self)
    self.env = env

  def BeginContact(self, contact):
    for ii in range(self.env.num_players):
      if (
          self.env.lander[ii] == contact.fixtureA.body
          or self.env.lander[ii] == contact.fixtureB.body
      ):
        self.env.game_over = True
      for i in range(2):
        if self.env.legs[ii][i] in [
            contact.fixtureA.body, contact.fixtureB.body
        ]:
          self.env.legs[ii][i].ground_contact = True

  def EndContact(self, contact):
    for ii in range(self.env.num_players):
      for i in range(2):
        if self.env.legs[ii][i] in [
            contact.fixtureA.body, contact.fixtureB.body
        ]:
          self.env.legs[ii][i].ground_contact = False


class MultiPlayerLunarLanderReachability(gym.Env, EzPickle):
  """
    In the LunarLander environment, the variables self.LANDER_POLY,
    self.LEG_AWAY, self.LEG_DOWN, self.LEG_W, self.LEG_H,
    self.SIDE_ENGINE_HEIGHT, self.SIDE_ENGINE_AWAY, self.VIEWPORT_W and
    self.VIEWPORT_H are measured in pixels.

    The x and y coordinates (and their time derivatives) used for physics
    calculations in the simulator use those values scaled by 1 / self.SCALE.

    The observations sent to the learning algorithm when reset() or step()
    is called use those values scaled by self.SCALE / (2 * self.VIEWPORT_H)
    and self.SCALE / (2 * VIEWPORT_Y) and centered at
    (2 * self.VIEWPORT_W) / self.SCALE and
    self.HELIPAD_Y + self.LEG_DOWN / self.SCALE for x and y respectively
    theta_dot is scaled by 20.0 / self.FPS.

    This makes reading the lunar_lander.py file difficult so I have tried to
    make clear what self.SCALE is being used here by calling them: pixel
    self.SCALE, simulator self.SCALE, and observation self.SCALE.

    TODO(vrubies) Make this into base class. Specific problems inherit from
    here.

    Args:
        gym ([type]): [description]
        EzPickle ([type]): [description]

    Raises:
        NotImplementedError: [description]
        NotImplementedError: [description]

    Returns:
        [type]: [description]
    """

  def __init__(
      self, device=torch.device("cpu"), num_players=1,
      observation_type='default', param_dict={}, rnd_seed=0,
      doneType='toFailureOrSuccess', obstacle_sampling=False
  ):
    """Initializes the environment with arguments.

    Args:
        device (torch.device, optional): CPU or GPU. Defaults to CPU.
        num_players (int, optional): number of lunar landers. Defaults to 1.
        observation_type (string, optional): type of observations available
                to the players.
            Defaults to 'default'.
            Additional options: 'LiDAR'.
        param_dict (dict, optional): dictionary to set simulation params.
            Defaults to empty dict.
        rnd_seed (int, optional): random seed.
            Defaults to 0.
        doneType (string, optional): type of end-of-trajectory condition.
            Defaults to 'toFailureOrSuccess'.
            Additional Options: 'toEnd' (to outer env boundary),
                                'toThreshold' (to specific g thr. value),
                                'toDone' (to collision or done flag true).
    """

    self.param_dict = self._generate_param_dict(param_dict)
    self.initialize_simulator_variables(self.param_dict)
    self.set_costParam(penalty=1, reward=-1)
    # in LunarLander init() calls reset() which calls step() so some
    # variables need to be set up before calling init() to prevent problems
    # from variables not being defined
    self.num_players = num_players
    self.observation_type = observation_type

    self.chunk_x = [self.W / (self.CHUNKS - 1) * i for i in range(self.CHUNKS)]
    self.chunk_y = [self.H / (self.CHUNKS - 1) * i for i in range(self.CHUNKS)]

    # Set random seed.
    self.set_seed(rnd_seed)

    # For torch.
    self.device = device

    # From parent constuctor.
    EzPickle.__init__(self)
    self.seed()
    self.viewer = None
    self.world = Box2D.b2World()
    self.moon = None
    self.lander = {}  # {ii: None for ii in range(self.num_players)}
    self.legs = {}  # {ii: None for ii in range(self.num_players)}
    self.lidar = {}
    self.particles = []
    self.prev_reward = None

    # we don't use the states regarding whether the legs are touching
    # so 6 dimensions total.
    # Observations.
    self.one_player_obs_dim = 6
    self.total_obs_dim = self.one_player_obs_dim * self.num_players
    self.sim_state = np.zeros(self.total_obs_dim)
    self.observation_space = spaces.Box(
        -np.inf, np.inf, shape=(self.total_obs_dim,), dtype=np.float32
    )
    # Actions.
    self.one_player_act_dim = 4
    self.total_act_dim = self.one_player_act_dim**self.num_players
    self.action_space = spaces.Discrete(self.total_act_dim)
    self.bounds_simulation_one_player = np.array([
        [0, self.W], [0, self.H], [-self.vx_bound, self.vx_bound],
        [-self.vy_bound, self.vy_bound], [-self.theta_bound, self.theta_bound],
        [-self.theta_dot_bound, self.theta_dot_bound]
    ])
    self.bounds_simulation = np.concatenate([
        self.bounds_simulation_one_player for _ in range(self.num_players)
    ])
    # Check conversions are ok.
    assert np.all(
        np.abs(
            self.obs_scale_to_simulator_scale(
                self.
                simulator_scale_to_obs_scale(self.bounds_simulation[:, 0])
            ) - self.bounds_simulation[:, 0]
        ) < 1e-5
    )

    # convert to observation self.SCALE so network can be evaluated
    self.bounds_observation = np.copy(self.bounds_simulation)
    self.bounds_observation[:, 0] = self.simulator_scale_to_obs_scale(
        self.bounds_simulation[:, 0].T
    )
    self.bounds_observation[:, 1] = self.simulator_scale_to_obs_scale(
        self.bounds_simulation[:, 1].T
    )

    self.doneType = doneType
    self.sio = obstacle_sampling

    self.reset()

  def _generate_param_dict(self, input_dict):
    """Returns dictionary with common parameters and user-specific parameters.

    Args:
        input_dict (dict): user-specific parameters.
    """
    param_dict = {}

    param_dict["FPS"] = 50
    param_dict["SCALE"] = 30.0
    param_dict["MAIN_ENGINE_POWER"] = 13.0
    param_dict["SIDE_ENGINE_POWER"] = 0.6
    param_dict["LANDER_POLY"] = [(-14, +17), (-17, 0), (-17, -10), (+17, -10),
                                 (+17, 0), (+14, +17)]
    param_dict["LEG_AWAY"] = 20
    param_dict["LEG_DOWN"] = 18
    param_dict["LEG_W"] = 2
    param_dict["LEG_H"] = 8
    param_dict["LEG_SPRING_TORQUE"] = 40
    param_dict["SIDE_ENGINE_HEIGHT"] = 14.0
    param_dict["SIDE_ENGINE_AWAY"] = 12.0
    param_dict["VIEWPORT_W"] = 600
    param_dict["VIEWPORT_H"] = 400
    param_dict["CHUNKS"] = 17
    param_dict["INITIAL_RANDOM"] = 1000.0
    param_dict["LIDAR_RANGE"] = 500

    param_dict["vx_bound"] = 10
    param_dict["vy_bound"] = 10
    param_dict["theta_bound"] = np.radians(90)
    param_dict["theta_dot_bound"] = np.radians(90)

    for key, value in input_dict:
      param_dict[key] = value
    return param_dict

  def initialize_simulator_variables(self, param_dict):
    """Initializes simulator of the lunar lander given parameters.

    Args:
        param_dict (dict): common parameters and user-specific parameters.
    """
    self.FPS = param_dict["FPS"]
    # scale affects how fast-paced the game is, forces should be adjusted
    # as well
    self.SCALE = param_dict["SCALE"]
    self.MAIN_ENGINE_POWER = param_dict["MAIN_ENGINE_POWER"]
    self.SIDE_ENGINE_POWER = param_dict["SIDE_ENGINE_POWER"]
    self.LANDER_POLY = param_dict["LANDER_POLY"]
    self.LEG_AWAY = param_dict["LEG_AWAY"]
    self.LEG_DOWN = param_dict["LEG_DOWN"]
    self.LEG_W = param_dict["LEG_W"]
    self.LEG_H = param_dict["LEG_H"]
    self.LEG_SPRING_TORQUE = param_dict["LEG_SPRING_TORQUE"]
    self.SIDE_ENGINE_HEIGHT = param_dict["SIDE_ENGINE_HEIGHT"]
    self.SIDE_ENGINE_AWAY = param_dict["SIDE_ENGINE_AWAY"]
    self.VIEWPORT_W = param_dict["VIEWPORT_W"]
    self.VIEWPORT_H = param_dict["VIEWPORT_H"]
    self.CHUNKS = param_dict["CHUNKS"]
    self.INITIAL_RANDOM = param_dict["INITIAL_RANDOM"]
    self.LIDAR_RANGE = param_dict["LIDAR_RANGE"] / self.SCALE

    self.W = self.VIEWPORT_W / self.SCALE
    self.H = self.VIEWPORT_H / self.SCALE
    self.HELIPAD_Y = self.H / 2
    # height of lander body in simulator self.SCALE. self.LANDER_POLY has
    # the (x,y) points that define the shape of the lander in pixel
    # self.SCALE
    self.LANDER_POLY_X = np.array(self.LANDER_POLY)[:, 0]
    self.LANDER_POLY_Y = np.array(self.LANDER_POLY)[:, 1]
    self.LANDER_W = (
        np.max(self.LANDER_POLY_X) - np.min(self.LANDER_POLY_X)
    ) / self.SCALE
    self.LANDER_H = (
        np.max(self.LANDER_POLY_Y) - np.min(self.LANDER_POLY_Y)
    ) / self.SCALE
    # distance of edge of legs from center of lander body in simulator
    # self.SCALE
    self.LEG_X_DIST = self.LEG_AWAY / self.SCALE
    self.LEG_Y_DIST = self.LEG_DOWN / self.SCALE
    # radius around lander to check for collisions
    self.LANDER_RADIUS = (
        (self.LANDER_H / 2 + self.LEG_Y_DIST + self.LEG_H / self.SCALE)**2 +
        (self.LANDER_W / 2 + self.LEG_X_DIST + self.LEG_W / self.SCALE)**2
    )**0.5

    # set up state space bounds used in evaluating the q value function
    self.vx_bound = param_dict["vx_bound"]
    self.vy_bound = param_dict["vy_bound"]
    self.theta_bound = param_dict["theta_bound"]
    self.theta_dot_bound = param_dict["theta_dot_bound"]

  # found online at:
  # https://codereview.stackexchange.com/questions/69833/..
  # generate-sample-coordinates-inside-a-polygon
  @staticmethod
  def random_points_in_polygon(polygon, k):
    """Returns list of k points uniformly at random inside the polygon.
    """
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
    """Extends the state to consist of max{ell, g}. Only used for mode='extend'.

    Args:
        states (np.ndarray): (x, y) position of states.

    Returns:
        np.ndarray: extended states.
    """
    new_states = []
    for state in states:
      l_x = self.target_margin(state)
      g_x = self.safety_margin(state)
      new_states.append(np.append(state, max(l_x, g_x)))
    return new_states

  def set_lander_state(self, state, key):
    """Sets the state of the lunar lander.

    Args:
        state (np.ndarray): the position, yaw, linear velocity and angular
            velocity of the agent's body
        key (str): the key for the lunar lander you want to set the state.
    """
    # convention is x, y, x_dot, y_dot, theta, theta_dot.
    # These internal variables are in --> simulator self.SCALE <--
    # changes need to be in np.float64
    self.lander[key].position = np.array([state[0], state[1]],
                                         dtype=np.float64)
    self.lander[key].linearVelocity = np.array([state[2], state[3]],
                                               dtype=np.float64)
    self.lander[key].angle = np.float64(state[4])
    self.lander[key].angularVelocity = np.float64(state[5])

    # after lander position is set, we have to set leg positions to be where
    # new lander position is.
    self.legs[key][0].position = np.array([
        self.lander[key].position.x + self.LEG_AWAY / self.SCALE,
        self.lander[key].position.y
    ], dtype=np.float64)
    self.legs[key][1].position = np.array([
        self.lander[key].position.x - self.LEG_AWAY / self.SCALE,
        self.lander[key].position.y
    ], dtype=np.float64)

  def generate_lander(self, initial_state, key):
    """Generates the lunar lander given an inital state and a key for it.

    Args:
        initial_state (np.ndarray): we only care the x- and y- position.
        key (str): the name of the lunar lander, serving as the key to call it.
    """
    # Generate Landers
    initial_x = initial_state[0]
    initial_y = initial_state[1]
    self.lander[key] = self.world.CreateDynamicBody(
        position=(initial_x, initial_y),
        angle=0.0,
        fixtures=fixtureDef(
            shape=polygonShape(
                vertices=[(x / self.SCALE, y / self.SCALE)
                          for x, y in self.LANDER_POLY]
            ),
            density=5.0,
            friction=0.1,
            categoryBits=0x0010,
            maskBits=0x001,  # collide only with ground
            restitution=0.0
        )  # 0.99 bouncy
    )
    self.lander[key].color1 = (0.5, 0.4, 0.9)
    self.lander[key].color2 = (0.3, 0.3, 0.5)
    self.lander[key].ApplyForceToCenter((
        np.random.uniform(-self.INITIAL_RANDOM, self.INITIAL_RANDOM),
        np.random.uniform(-self.INITIAL_RANDOM, self.INITIAL_RANDOM)
    ), True)

    self.legs[key] = []
    for i in [-1, +1]:
      leg = self.world.CreateDynamicBody(
          position=(initial_x - i * self.LEG_AWAY / self.SCALE, initial_y),
          angle=(i * 0.05), fixtures=fixtureDef(
              shape=polygonShape(
                  box=(self.LEG_W / self.SCALE, self.LEG_H / self.SCALE)
              ), density=1.0, restitution=0.0, categoryBits=0x0020,
              maskBits=0x001
          )
      )
      leg.ground_contact = False
      leg.color1 = (0.5, 0.4, 0.9)
      leg.color2 = (0.3, 0.3, 0.5)
      rjd = revoluteJointDef(
          bodyA=self.lander[key],
          bodyB=leg,
          localAnchorA=(0, 0),
          localAnchorB=(
              i * self.LEG_AWAY / self.SCALE, self.LEG_DOWN / self.SCALE
          ),
          enableMotor=True,
          enableLimit=True,
          maxMotorTorque=self.LEG_SPRING_TORQUE,
          motorSpeed=+0.3 * i  # Low enough not to jump back into sky.
      )
      if i == -1:
        # The most esoteric numbers here, angled legs have freedom to
        # travel within.
        rjd.lowerAngle = +0.9 - 0.5
        rjd.upperAngle = +0.9
      else:
        rjd.lowerAngle = -0.9
        rjd.upperAngle = -0.9 + 0.5
      leg.joint = self.world.CreateJoint(rjd)
      self.legs[key].append(leg)

  def generate_terrain_and_landers(
      self, terrain_polyline=None, sample_inside_obs=False
  ):
    """Generates the terrain of the world and lunar landers in the world.

    Args:
        terrain_polyline (list, optional): height of each chunk.
            Defaults to None.
        sample_inside_obs (bool, optional): consider sampling the state inside
            the obstacles if True. Defaults to False.

    Returns:
        np.ndarray: states of lunar landers.
    """
    # terrain
    if terrain_polyline is None:
      height = np.ones((self.CHUNKS + 1,))
    else:
      height = terrain_polyline
    height[self.CHUNKS // 2 - 3] = self.HELIPAD_Y + 2.5
    height[self.CHUNKS // 2 - 2] = self.HELIPAD_Y
    height[self.CHUNKS // 2 - 1] = self.HELIPAD_Y
    height[self.CHUNKS // 2 + 0] = self.HELIPAD_Y
    height[self.CHUNKS // 2 + 1] = self.HELIPAD_Y
    height[self.CHUNKS // 2 + 2] = self.HELIPAD_Y
    smooth_y = list(height[:-1])

    self.moon = self.world.CreateStaticBody(
        shapes=edgeShape(vertices=[(0, 0), (self.W, 0)])
    )
    self.sky_polys = []
    self.moon_chunk = []
    obstacle_polyline = [(self.chunk_x[0], smooth_y[0])]
    for i in range(self.CHUNKS - 1):
      p1 = (self.chunk_x[i], smooth_y[i])
      p2 = (self.chunk_x[i + 1], smooth_y[i + 1])
      obstacle_polyline.append(p2)
      self.moon.CreateEdgeFixture(vertices=[p1, p2], density=0, friction=0.1)
      self.sky_polys.append([p1, p2, (p2[0], self.H), (p1[0], self.H)])
      self.moon_chunk.append([p1, p2, (p2[0], 0), (p1[0], 0)])
    # Enclose terrain within window.
    obstacle_polyline.append((self.W, self.H))
    obstacle_polyline.append((0, self.H))
    obstacle_polyline.append(obstacle_polyline[0])
    self.obstacle_polyline = Polygon(obstacle_polyline)
    self.paint_obstacle = obstacle_polyline

    self.moon.color1 = (0.6, 0.6, 0.6)
    self.moon.color2 = (0.6, 0.6, 0.6)

    # Instantiate the Landers in random initial states.
    initial_states = [
        self.rejection_sample(sample_inside_obs)
        for _ in range(self.num_players)
    ]
    self.drawlist = []
    for ii, initial_state in enumerate(initial_states):
      self.generate_lander(initial_state, ii)
      self.drawlist += [self.lander[ii]] + self.legs[ii]

    if self.observation_type == "LiDAR":

      class LidarCallback(Box2D.b2.rayCastCallback):

        def ReportFixture(self, fixture, point, normal, fraction):
          if (fixture.filterData.categoryBits & 1) == 0:
            return -1
          self.p2 = point
          self.fraction = fraction
          return fraction

      for ii in range(self.num_players):
        self.lidar[ii] = [LidarCallback() for _ in range(10)]

    s, _, _, _ = self.step(0)
    return s

  def rejection_sample(self, sample_inside_obs=False):
    """Returns the initial states sampled uniformly at random.

    Args:
        sample_inside_obs (bool, optional): consider sampling the state inside
            the obstacles if True. Defaults to False.
    """
    flag_sample = False
    while not flag_sample:
      xy_sample = np.random.uniform(low=[0, 0], high=[self.W, self.H])
      flag_sample = self.obstacle_polyline.contains(
          Point(xy_sample[0], xy_sample[1])
      )
      if sample_inside_obs:
        break
    return xy_sample

  def _destroy(self):
    """Destroys the environment.
    """
    if self.moon is not None:
      self.world.contactListener = None
      self._clean_particles(True)
      self.world.DestroyBody(self.moon)
      self.moon = None
    if len(self.lander) > 0 and any([l is not None for l in self.lander]):
      for ii, _ in enumerate(self.lander):
        self.world.DestroyBody(self.lander[ii])
        self.lander[ii] = None
        self.world.DestroyBody(self.legs[ii][0])
        self.world.DestroyBody(self.legs[ii][1])

  def reset(self, state_in=None, terrain_polyline=None):
    """Resets the environment according to a uniform distribution.

    Args:
        state_in (np.ndarray, optional): assumed to be in simulation
            self.SCALE. Defaults to None.
        terrain_polyline (list, optional): height of each chunk.
            Defaults to None.

    Returns:
        current state as 6d NumPy array of floats
    """
    self._destroy()
    self.world.contactListener_keepref = MultiPlayerContactDetector(self)
    self.world.contactListener = self.world.contactListener_keepref
    self.game_over = False
    self.prev_shaping = None
    # This returns something in --> observation self.SCALE <--.

    s = self.generate_terrain_and_landers(
        terrain_polyline=terrain_polyline, sample_inside_obs=self.sio
    )
    # When 'toThreshold' is enabled we need to remove obstacles.
    if (
        self.doneType == 'toThreshold' or self.doneType == 'toFailureOrSuccess'
    ):
      self.world.contactListener = None
      self._clean_particles(True)
      self.world.DestroyBody(self.moon)
      self.moon = None

    # Rewrite internal lander variables in --> simulation self.SCALE <--.
    if state_in is None:
      state_in = np.copy(self.obs_scale_to_simulator_scale(s))
      state_in[4] = np.random.uniform(
          low=-self.theta_bound, high=self.theta_bound
      )
    else:
      # Ensure that when specifing a state it is within
      # our simulation bounds.
      for ii in range(len(state_in)):
        state_in[ii] = np.float64(
            min(state_in[ii], self.bounds_simulation[ii, 1])
        )
        state_in[ii] = np.float64(
            max(state_in[ii], self.bounds_simulation[ii, 0])
        )

    # Set the states for the landers.
    for ii in range(self.num_players):
      self.set_lander_state(
          state_in[ii * self.one_player_obs_dim:(ii+1)
                   * self.one_player_obs_dim], ii
      )

    # Convert from simulator self.SCALE to observation self.SCALE.
    s = self.simulator_scale_to_obs_scale(state_in)

    # Return in --> observation self.SCALE <--.
    return s

  def parent_step(self, action, key):
    """Evolves the environment one step forward given an action.

    Args:
        action (int): the index of the action in the action set.
        key (str): the name of the lunar lander.

    Returns:
        np.ndarray: next state.
        float: the standard cost used in reinforcement learning.
        bool: True if the episode is terminated.
        dict: empty one.
    """
    # Action needs to be single action 0-3.
    assert self.action_space.contains(
        action
    ), "%r (%s) invalid " % (action, type(action))

    # Engines
    tip = (math.sin(self.lander[key].angle), math.cos(self.lander[key].angle))
    side = (-tip[1], tip[0])

    if self.observation_type == "LiDAR":
      pos = self.lander[key].position
      num_beams = 10
      for ii in range(num_beams):
        self.lidar[key][ii].fraction = 1.0
        self.lidar[key][ii].p1 = pos
        self.lidar[key][ii].p2 = (
            pos[0] + math.sin(2.0 * np.pi * ii / num_beams) * self.LIDAR_RANGE,
            pos[1] + math.cos(2.0 * np.pi * ii / num_beams) * self.LIDAR_RANGE
        )
        self.world.RayCast(
            self.lidar[key][ii], self.lidar[key][ii].p1, self.lidar[key][ii].p2
        )

    m_power = 0.0
    if action == 2:
      # Main engine
      m_power = 1.0
      ox = tip[
          0] * 4 / self.SCALE  # 4 is move a bit downwards, +-2 for randomness
      oy = -tip[1] * 4 / self.SCALE
      impulse_pos = (
          self.lander[key].position[0] + ox, self.lander[key].position[1] + oy
      )
      p = self._create_particle(
          3.5,  # 3.5 is here to make particle speed adequate
          impulse_pos[0],
          impulse_pos[1],
          m_power
      )  # particles are just a decoration
      p.ApplyLinearImpulse((
          ox * self.MAIN_ENGINE_POWER * m_power,
          oy * self.MAIN_ENGINE_POWER * m_power
      ), impulse_pos, True)
      self.lander[key].ApplyLinearImpulse((
          -ox * self.MAIN_ENGINE_POWER * m_power,
          -oy * self.MAIN_ENGINE_POWER * m_power
      ), impulse_pos, True)

    s_power = 0.0
    if action in [1, 3]:
      # Orientation engines
      direction = action - 2
      s_power = 1.0
      ox = side[0] * direction * self.SIDE_ENGINE_AWAY / self.SCALE
      oy = side[1] * direction * self.SIDE_ENGINE_AWAY / self.SCALE
      impulse_pos = (
          self.lander[key].position[0] + ox - tip[0] * 17 / self.SCALE,
          self.lander[key].position[1] + oy
          + tip[1] * self.SIDE_ENGINE_HEIGHT / self.SCALE
      )
      p = self._create_particle(0.7, impulse_pos[0], impulse_pos[1], s_power)
      p.ApplyLinearImpulse((
          ox * self.SIDE_ENGINE_POWER * s_power,
          oy * self.SIDE_ENGINE_POWER * s_power
      ), impulse_pos, True)
      self.lander[key].ApplyLinearImpulse((
          -ox * self.SIDE_ENGINE_POWER * s_power,
          -oy * self.SIDE_ENGINE_POWER * s_power
      ), impulse_pos, True)

    self.world.Step(1.0 / self.FPS, 6 * 30, 2 * 30)

    pos = self.lander[key].position
    vel = self.lander[key].linearVelocity
    state = [(pos.x - self.W / 2) / (self.W / 2),
             (pos.y - (self.HELIPAD_Y + self.LEG_DOWN / self.SCALE)) /
             (self.H / 2), vel.x * (self.W / 2) / self.FPS, vel.y *
             (self.H / 2) / self.FPS, self.lander[key].angle,
             20.0 * self.lander[key].angularVelocity / self.FPS,
             1.0 if self.legs[key][0].ground_contact else 0.0,
             1.0 if self.legs[key][1].ground_contact else 0.0]
    if self.observation_type == "LiDAR":
      state += [l.fraction for l in self.lidar[key]]

    reward = 0
    shaping = \
        - 100*np.sqrt(state[0]*state[0] + state[1]*state[1]) \
        - 100*np.sqrt(state[2]*state[2] + state[3]*state[3]) \
        - 100*abs(state[4]) + 10*state[6] + 10*state[7]
    # And ten points for legs contact, the idea is if you lose contact
    # again after landing, you get negative reward
    if self.prev_shaping is not None:
      reward = shaping - self.prev_shaping
    self.prev_shaping = shaping

    # less fuel spent is better, about -30 for heuristic landing
    reward -= m_power * 0.30
    reward -= s_power * 0.03

    done = False
    if (
        self.game_over or self.legs[key][0].ground_contact
        or self.legs[key][1].ground_contact
    ):
      done = True
      reward = -100
    if not self.lander[key].awake:
      done = True
      reward = +100
    return np.array(state, dtype=np.float32), reward, done, {}

  def step(self, action):
    """Evolves the environment one step forward for each lander.

    Args:
        action (int): the index of the action in the action set.

    Returns:
        np.ndarray: next state.
        float: safety margin value for the next state.
        bool: True if the episode is terminated for any lander.
        dict: dictionary with safety and target margins.
    """
    info = {}

    l_x_cur = self.target_margin(self.sim_state)
    g_x_cur = self.safety_margin(self.sim_state)

    # Fail or Success?
    fail = g_x_cur > 0
    success = l_x_cur <= 0

    # Transform decimal action to individual player actions.
    actions = self.decimal_actions_to_player_actions(action)

    state_list = []
    reward_list = []
    done_list = []
    info_list = []
    for ii in range(self.num_players):
      state_ii, reward_ii, done_ii, info_ii = self.parent_step(actions[ii], ii)
      state_list.append(state_ii[:-2])
      reward_list.append(reward_ii)
      done_list.append(done_ii)
      info_list.append(info_ii)
    self.obs_state = np.concatenate(state_list)
    self.sim_state = self.obs_scale_to_simulator_scale(self.obs_state)

    l_x_nxt = self.target_margin(self.sim_state)
    g_x_nxt = self.safety_margin(self.sim_state)

    done = False
    if self.doneType == 'toFailureOrSuccess' or self.doneType == 'toDone':
      if fail:
        done = True
        info = {
            "g_x": self.penalty,
            "l_x": l_x_cur,
            "g_x_nxt": self.penalty,
            "l_x_nxt": l_x_nxt
        }
      if success:
        done = True
        info = {
            "g_x": g_x_cur,
            "l_x": self.reward,
            "g_x_nxt": g_x_nxt,
            "l_x_nxt": self.reward
        }
      if self.doneType == 'toDone' and np.any(done_list):
        done = True
        info = {
            "g_x": g_x_cur,
            "l_x": l_x_cur,
            "g_x_nxt": g_x_nxt,
            "l_x_nxt": l_x_nxt
        }
    elif self.doneType == 'toThreshold':
      if g_x_cur > self.penalty:  # or success:
        done = True
        info = {
            "g_x": g_x_cur,
            "l_x": l_x_cur,
            "g_x_nxt": g_x_nxt,
            "l_x_nxt": l_x_nxt
        }
    elif self.doneType == 'toEnd':
      # Not implemented.
      pass

    # If done flag has not triggered, just collect normal info.
    if not done:
      info = {
          "g_x": g_x_cur,
          "l_x": l_x_cur,
          "g_x_nxt": g_x_nxt,
          "l_x_nxt": l_x_nxt
      }

    return np.copy(self.obs_state), l_x_nxt, done, info

  def _create_particle(self, mass, x, y, ttl):
    """Generates particles for the thrusters for cosmetic purposes.

    Args:
        mass (float): mass of the particle.
        x (float): x position.
        y (float): y position.
        ttl (float): power.

    Returns:
        Dynamic body representing a thurster particle.
    """
    p = self.world.CreateDynamicBody(
        position=(x, y),
        angle=0.0,
        fixtures=fixtureDef(
            shape=circleShape(radius=2 / self.SCALE, pos=(0, 0)),
            density=mass,
            friction=0.1,
            categoryBits=0x0100,
            maskBits=0x001,  # collide only with ground
            restitution=0.3
        )
    )
    p.ttl = ttl
    self.particles.append(p)
    self._clean_particles(False)
    return p

  def _clean_particles(self, all_):
    """Destroys particles with options.

    Args:
        all_ (bool): If true it destroys all particles.
    """
    while self.particles and (all_ or self.particles[0].ttl < 0):
      self.world.DestroyBody(self.particles.pop(0))

  def decimal_actions_to_player_actions(self, action):
    """
    Transforms decimal actions into an action for each lunar lander.
    Goes from base 10 to base #actions.
    (e.g. 3 actions, 2 landers. Decimal: 0, Players: [0, 0]
                                Decimal: 1, Players: [0, 1]
                                Decimal: 5, Players: [1, 1]

    Args:
        action (int): a decimal action.

    Returns:
        list: actions for each lunar lander.
    """
    base_actions = self.one_player_act_dim
    player_actions = []
    for ii in range(self.num_players - 1, -1, -1):
      for jj in range(base_actions):
        if (action < jj * base_actions**ii):
          jj -= 1
          break
        elif (action == jj * base_actions**ii):
          break
      action -= jj * base_actions**ii
      player_actions.append(jj)
    player_actions.reverse()
    return player_actions

  def target_margin(self, state):
    raise NotImplementedError

  def safety_margin(self, state):
    raise NotImplementedError

  # =========== Methods for conversions (BEGIN).
  def simulator_scale_to_obs_scale_single(self, state):
    """Transforms the state from simulator scle to observation scale and return.

    Args:
        state (np.ndarray): state of a lunar lander in simulator scale.
    """
    copy_state = np.copy(state)
    chg_dims = self.one_player_obs_dim
    x, y, x_dot, y_dot, theta, theta_dot = copy_state[:chg_dims]
    copy_state[:chg_dims] = np.array(
        [(x - self.W / 2) / (self.W / 2),
         (y - (self.HELIPAD_Y + self.LEG_DOWN / self.SCALE)) /
         (self.H / 2), x_dot * (self.W / 2) / self.FPS, y_dot *
         (self.H / 2) / self.FPS, theta, 20.0 * theta_dot / self.FPS],
        dtype=np.float32
    )  # theta_dot])
    return copy_state

  def simulator_scale_to_obs_scale(self, state):
    """Transforms the state from simulator scle to observation scale and return.

    Args:
        state (np.ndarray): states of all lunar landers in simulator scale.
    """
    copy_state = np.copy(state)
    chg_dims = self.one_player_obs_dim
    for ii in range(self.num_players):
      copy_state[ii * chg_dims:(ii+1) * chg_dims] = (
          self.simulator_scale_to_obs_scale_single(
              copy_state[ii * chg_dims:(ii+1) * chg_dims]
          )
      )
    return copy_state

  def obs_scale_to_simulator_scale_single(self, state):
    """Transforms the state from observation scle to simulator scale and return.

    Args:
        state (np.ndarray): state of a lunar lander in observation scale.
    """
    copy_state = np.copy(state)
    chg_dims = self.one_player_obs_dim
    x, y, x_dot, y_dot, theta, theta_dot = copy_state[:chg_dims]
    copy_state[:chg_dims] = np.array(
        [(x * (self.W / 2)) + (self.W / 2), (y * (self.H / 2)) +
         (self.HELIPAD_Y + self.LEG_DOWN / self.SCALE), x_dot /
         ((self.W / 2) / self.FPS), y_dot /
         ((self.H / 2) / self.FPS), theta, theta_dot * self.FPS / 20.0],
        dtype=np.float64
    )  # theta_dot])
    return copy_state

  def obs_scale_to_simulator_scale(self, state):
    """Transforms the state from observation scle to simulator scale and return.

    Args:
        state (np.ndarray): states of all lunar landers in observation scale.
    """
    copy_state = np.copy(state)
    chg_dims = self.one_player_obs_dim
    for ii in range(self.num_players):
      copy_state[ii * chg_dims:(ii+1) * chg_dims] = (
          self.obs_scale_to_simulator_scale_single(
              copy_state[ii * chg_dims:(ii+1) * chg_dims]
          )
      )
    return copy_state

  # =========== Methods for conversions (END).

  def set_seed(self, seed):
    """Sets the random seed.

    Args:
        seed (int): Random seed.
    """
    self.seed_val = seed
    np.random.seed(self.seed_val)

  def set_doneType(self, doneType):
    """Sets the condition to raise `done` flag.

    Args:
        doneType (str): condition to raise `done` flag.
    """
    self.doneType = doneType

  def set_costParam(self, penalty=1, reward=-1):
    """
    Sets the hyper-parameters for the `cost` signal used in training, important
    for standard (Lagrange-type) reinforcement learning.

    Args:
        penalty (float, optional): cost when entering the obstacles or crossing
            the environment boundary. Defaults to 1.0.
        reward (float, optional): cost when reaching the targets.
            Defaults to -1.0.
    """
    self.penalty = penalty
    self.reward = reward

  def render(self, mode='human', plot_landers=True, **kwargs):
    """Returns rendered image.

    Args:
        mode (str, optional): Defaults to 'human'.
        plot_landers (bool, optional): Plot the lunar lander if True.
            Defaults to True.
        kwargs (dict): polyline for the target.
    """
    from gym.envs.classic_control import rendering
    if self.viewer is None:
      self.viewer = rendering.Viewer(self.VIEWPORT_W, self.VIEWPORT_H)
      self.viewer.set_bounds(
          0, self.VIEWPORT_W / self.SCALE, 0, self.VIEWPORT_H / self.SCALE
      )

    for obj in self.particles:
      obj.ttl -= 0.15
      obj.color1 = (
          max(0.2, 0.2 + obj.ttl), max(0.2,
                                       0.5 * obj.ttl), max(0.2, 0.5 * obj.ttl)
      )
      obj.color2 = (
          max(0.2, 0.2 + obj.ttl), max(0.2,
                                       0.5 * obj.ttl), max(0.2, 0.5 * obj.ttl)
      )

    self._clean_particles(False)

    for p in self.sky_polys:
      self.viewer.draw_polygon(p, color=(0, 0, 0))

    if plot_landers:
      if self.observation_type == "LiDAR":
        for ii in range(self.num_players):
          for l in self.lidar[ii]:
            self.viewer.draw_polyline([l.p1, l.p2], color=(1, 0, 0),
                                      linewidth=2)

      for obj in self.particles + self.drawlist:
        for f in obj.fixtures:
          trans = f.body.transform
          if type(f.shape) is circleShape:
            t = rendering.Transform(translation=trans * f.shape.pos)
            self.viewer.draw_circle(f.shape.radius, 20,
                                    color=obj.color1).add_attr(t)
            self.viewer.draw_circle(
                f.shape.radius, 20, color=obj.color2, filled=False, linewidth=2
            ).add_attr(t)
          else:
            path = [trans * v for v in f.shape.vertices]
            self.viewer.draw_polygon(path, color=obj.color1)
            path.append(path[0])
            self.viewer.draw_polyline(path, color=obj.color2, linewidth=2)

    for x in [self.helipad_x1, self.helipad_x2]:
      flagy1 = self.HELIPAD_Y
      flagy2 = flagy1 + 50 / self.SCALE
      self.viewer.draw_polyline([(x, flagy1), (x, flagy2)], color=(0, 0, 0))
      self.viewer.draw_polygon(
          [(x, flagy2), (x, flagy2 - 10 / self.SCALE),
           (x + 25 / self.SCALE, flagy2 - 5 / self.SCALE)],
          color=(0.8, 0.8, 0),
      )

    self.viewer.draw_polyline(
        self.paint_obstacle, color=(1, 0, 0), linewidth=10
    )

    for _, value in kwargs.items():
      self.viewer.draw_polyline(value, color=(0, 1, 0), linewidth=10)

    return self.viewer.render(return_rgb_array=mode == 'rgb_array')

  def close(self):
    """Closes the viewer.
    """
    if self.viewer is not None:
      self.viewer.close()
      self.viewer = None
