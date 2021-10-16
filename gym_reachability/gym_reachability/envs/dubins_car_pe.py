"""
Please contact the author(s) of this library if you have any questions.
Authors: Kai-Chieh Hsu        ( kaichieh@princeton.edu )

This module implements an environment considering a pursuir-evasion game
between two identical Dubins car. This environemnt shows the proposed
reach-avoid reinforcemnt learning can be extended to find an approximate
solution of the two-player zero-sum game.
"""

import gym.spaces
import numpy as np
import gym
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator
import torch
import random

from .dubins_car_dyn import DubinsCarDyn
from .env_utils import plot_circle, rotatePoint

# Local Variables
purple = '#9370DB'
tiffany = '#0abab5'
silver = '#C0C0C0'


class DubinsCarPEEnv(gym.Env):
  """
  A gym environment considering a pursuit-evasion game between two identical
  Dubins cars.
  """

  def __init__(
      self, device, mode='RA', doneType='toEnd', sample_inside_obs=False,
      sample_inside_tar=True, considerPursuerFailure=False
  ):
    """Initializes the environment with given arguments.

    Args:
        device (str): device type (used in PyTorch).
        mode (str, optional): reinforcement learning type. Defaults to 'RA'.
        doneType (str, optional): conditions to raise `done flag in training.
            Defaults to 'toEnd'.
        sample_inside_obs (bool, optional): consider sampling the state inside
            the obstacles if True. Defaults to False.
        sample_inside_tar (bool, optional): consider sampling the state inside
            the target if True. Defaults to True.
        considerPursuerFailure (bool, optional): the game outcome considers
            the pursuer hitting the failure set if True. Defaults to False.
    """
    # Set random seed.
    self.set_seed(0)

    # State bounds.
    self.bounds = np.array([[-1.1, 1.1], [-1.1, 1.1], [0, 2 * np.pi]])
    self.low = self.bounds[:, 0]
    self.high = self.bounds[:, 1]
    self.sample_inside_obs = sample_inside_obs
    self.sample_inside_tar = sample_inside_tar

    # Gym variables.
    self.numActionList = [3, 3]
    self.action_space = gym.spaces.Discrete(9)
    midpoint = (self.low + self.high) / 2.0
    interval = self.high - self.low
    self.observation_space = gym.spaces.Box(
        np.float32(midpoint - interval/2), np.float32(midpoint + interval/2)
    )

    # Constraint set parameters.
    self.evader_constraint_center = np.array([0, 0])
    self.evader_constraint_radius = 1.0
    self.pursuer_constraint_center = np.array([0, 0])
    self.pursuer_constraint_radius = 1.0
    self.capture_range = 0.25

    # Target set parameters.
    self.evader_target_center = np.array([0, 0])
    self.evader_target_radius = 0.5
    self.considerPursuerFailure = considerPursuerFailure

    # Dubins cars parameters.
    self.time_step = 0.05
    self.speed = 0.75  # v
    self.R_turn = self.speed / 3
    self.pursuer = DubinsCarDyn(doneType='toEnd')
    self.evader = DubinsCarDyn(doneType='toEnd')
    self.init_car()

    # Internal state.
    self.mode = mode
    self.state = np.zeros(6)
    self.doneType = doneType

    # Visualization parameters
    self.visual_initial_states = [
        np.array([-0.9, 0., 0., -0.1, -0.3, .75 * np.pi]),
        np.array([-0.9, 0., 0., -0.2, -0.3, .75 * np.pi]),
        np.array([-0.6, 0., np.pi, 0.1, 0., np.pi]),
        np.array([-0.8, -0.4, 0., -0.4, 0.8, 0.])
    ]

    # Cost parameters
    self.targetScaling = 1.
    self.safetyScaling = 1.
    self.penalty = 1.
    self.reward = -1.
    self.costType = 'sparse'
    self.device = device

  def init_car(self):
    """Initalizes the pursuer and evader.
    """
    self.evader.set_bounds(bounds=self.bounds)
    self.evader.set_constraint(
        center=self.evader_constraint_center,
        radius=self.evader_constraint_radius
    )
    self.evader.set_target(
        center=self.evader_target_center, radius=self.evader_target_radius
    )
    self.evader.set_speed(speed=self.speed)
    self.evader.set_time_step(time_step=self.time_step)
    self.evader.set_radius_rotation(R_turn=self.R_turn, verbose=False)

    self.pursuer.set_bounds(bounds=self.bounds)
    self.pursuer.set_constraint(
        center=self.pursuer_constraint_center,
        radius=self.pursuer_constraint_radius
    )
    self.pursuer.set_speed(speed=self.speed)
    self.pursuer.set_time_step(time_step=self.time_step)
    self.pursuer.set_radius_rotation(R_turn=self.R_turn, verbose=False)

  # == Reset Functions ==
  def reset(self, start=None):
    """Resets the state of the environment.

    Args:
        start (np.ndarray, optional): state to reset the environment to.
            If None, pick the state uniformly at random. Defaults to None.

    Returns:
        np.ndarray: The state that the environment has been reset to.
    """
    if start is not None:
      stateEvader = self.evader.reset(start=start[:3])
      statePursuer = self.pursuer.reset(start=start[3:])
    else:
      stateEvader = self.evader.reset(
          sample_inside_obs=self.sample_inside_obs,
          sample_inside_tar=self.sample_inside_tar
      )
      statePursuer = self.pursuer.reset(
          sample_inside_obs=self.sample_inside_obs,
          sample_inside_tar=self.sample_inside_tar
      )
    self.state = np.concatenate((stateEvader, statePursuer), axis=0)
    return np.copy(self.state)

  def sample_random_state(
      self, sample_inside_obs=False, sample_inside_tar=True, theta=None
  ):
    """Picks the state uniformly at random.

    Args:
        sample_inside_obs (bool, optional): consider sampling the state inside
            the obstacles if True. Defaults to False.
        sample_inside_tar (bool, optional): consider sampling the state inside
            the target if True. Defaults to True.
        theta (float, optional): if provided, set the theta to its value.
            Defaults to None.

    Returns:
        np.ndarray: sampled initial state.
    """
    stateEvader = self.evader.sample_random_state(
        sample_inside_obs=sample_inside_obs,
        sample_inside_tar=sample_inside_tar, theta=theta
    )
    statePursuer = self.pursuer.sample_random_state(
        sample_inside_obs=sample_inside_obs,
        sample_inside_tar=sample_inside_tar, theta=theta
    )
    return np.concatenate((stateEvader, statePursuer), axis=0)

  # == Dynamics Functions ==
  def step(self, action):
    """Evolves the environment one step forward under given action.

    Args:
        action (list of ints)): contains the index of action in the evader and
            the pursuer's action set, respectively.

    Returns:
        np.ndarray: next state.
        float: the standard cost used in reinforcement learning.
        bool: True if the episode is terminated.
        dict: consist of target margin and safety margin at the new state.
    """
    state_tmp = np.concatenate((self.evader.state, self.pursuer.state), axis=0)
    distance = np.linalg.norm(self.state - state_tmp)
    assert distance < 1e-8, (
        "There is a mismatch between the env state"
        + "and car state: {:.2e}".format(distance)
    )

    stateEvader, doneEvader = self.evader.step(action[0])
    statePursuer, donePursuer = self.pursuer.step(action[1])

    self.state = np.concatenate((stateEvader, statePursuer), axis=0)
    l_x = self.target_margin(self.state)
    g_x = self.safety_margin(self.state)

    fail = g_x > 0
    success = l_x <= 0

    # cost
    assert self.mode == 'RA', (
        "PE environment doesn't support standard reinforcement learning yet"
    )
    if self.mode == 'RA':
      if fail:
        cost = self.penalty
      elif success:
        cost = self.reward
      else:
        cost = 0.

    # = `done` signal
    if self.doneType == 'toEnd':
      done = doneEvader and donePursuer
    elif self.doneType == 'fail':
      done = fail
    elif self.doneType == 'TF':
      done = fail or success
    else:
      raise ValueError("invalid done type!")

    # = `info`
    if done and self.doneType == 'fail':
      info = {"g_x": self.penalty, "l_x": l_x}
    else:
      info = {"g_x": g_x, "l_x": l_x}
    return np.copy(self.state), cost, done, info

  # == Setting Hyper-Parameter Functions ==
  def set_costParam(
      self, penalty=1.0, reward=-1.0, costType='sparse', targetScaling=1.0,
      safetyScaling=1.0
  ):
    """
    Sets the hyper-parameters for the `cost` signal used in training, important
    for standard (Lagrange-type) reinforcement learning.

    Args:
        penalty (float, optional): cost when entering the obstacles or
            crossing the environment boundary. Defaults to 1.0.
        reward (float, optional): cost when reaching the targets.
            Defaults to -1.0.
        costType (str, optional): providing extra information when in neither
            the failure set nor the target set. Defaults to 'sparse'.
        targetScaling (float, optional): scaling factor of the target margin.
            Defaults to 1.0.
        safetyScaling (float, optional): scaling factor of the safety margin.
            Defaults to 1.0.
    """
    self.penalty = penalty
    self.reward = reward
    self.costType = costType
    self.safetyScaling = safetyScaling
    self.targetScaling = targetScaling

  def set_capture_range(self, capture_range=.1):
    """Sets the caspture radius of the pursuer.

    Args:
        capture_range (float, optional): the radius of the capture range.
            Defaults to .1.
    """
    self.capture_range = capture_range

  def set_seed(self, seed):
    """Sets the seed for `numpy`, `random`, `PyTorch` packages.

    Args:
        seed (int): seed value.
    """
    self.seed_val = seed
    np.random.seed(self.seed_val)
    torch.manual_seed(self.seed_val)
    torch.cuda.manual_seed(self.seed_val)
    torch.cuda.manual_seed_all(self.seed_val)  # if use multi GPU.
    random.seed(self.seed_val)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

  def set_bounds(self, bounds):
    """Sets the boundary and the observation space of the environment.

    Args:
        bounds (np.ndarray): of the shape (n_dim, 2). Each row is [LB, UB].
    """
    self.bounds = bounds

    # Get lower and upper bounds
    self.low = np.array(self.bounds)[:, 0]
    self.high = np.array(self.bounds)[:, 1]

    # Double the range in each state dimension for Gym interface.
    midpoint = (self.low + self.high) / 2.0
    interval = self.high - self.low
    self.observation_space = gym.spaces.Box(
        np.float32(midpoint - interval/2), np.float32(midpoint + interval/2)
    )
    self.evader.set_bounds(bounds)
    self.pursuer.set_bounds(bounds)

  def set_radius_rotation(self, R_turn=.6, verbose=False):
    """
    Sets radius of the car's circular motion. The turning radius influences the
    angular speed and the discrete control set.

    Args:
        R_turn (float, optional): the radius of the car's circular motion.
            Defaults to .6.
        verbose (bool, optional): print messages if True. Defaults to False.
    """
    self.R_turn = R_turn
    self.evader.set_radius_rotation(R_turn=R_turn, verbose=verbose)
    self.pursuer.set_radius_rotation(R_turn=R_turn, verbose=verbose)

  def set_constraint(self, center=np.array([0., 0.]), radius=1., car='evader'):
    """Sets the constraint set (complement of failure set).

    Args:
        center (np.ndarray, optional): center of the constraint set.
            Defaults to np.array([0.,0.]).
        radius (float, optional): radius of the constraint set.
            Defaults to 1.0.
        car (str, optional): which car to set. Defaults to 'evader'.
    """
    if car == 'evader':
      self.evader_constraint_center = center
      self.evader_constraint_radius = radius
      self.evader.set_constraint(center=center, radius=radius)
    elif car == 'pursuer':
      self.pursuer_constraint_center = center
      self.pursuer_constraint_radius = radius
      self.pursuer.set_constraint(center=center, radius=radius)
    elif car == 'both':
      self.evader_constraint_center = center
      self.evader_constraint_radius = radius
      self.evader.set_constraint(center=center, radius=radius)
      self.pursuer_constraint_center = center
      self.pursuer_constraint_radius = radius
      self.pursuer.set_constraint(center=center, radius=radius)

  def set_target(self, center=np.array([0., 0.]), radius=.4, car='evader'):
    """Sets the target set.

    Args:
        center (np.ndarray, optional): center of the target set.
            Defaults to np.array([0.,0.]).
        radius (float, optional): radius of the target set. Defaults to .4.
        car (str, optional): which car to set. Defaults to 'evader'.
    """
    if car == 'evader':
      self.evader_target_center = center
      self.evader_target_radius = radius
      self.evader.set_target(center=center, radius=radius)
    elif car == 'pursuer':
      self.pursuer_target_center = center
      self.pursuer_target_radius = radius
      self.pursuer.set_target(center=center, radius=radius)
    elif car == 'both':
      self.evader_target_center = center
      self.evader_target_radius = radius
      self.evader.set_target(center=center, radius=radius)
      self.pursuer_target_center = center
      self.pursuer_target_radius = radius
      self.pursuer.set_constraint(center=center, radius=radius)

  def set_considerPursuerFailure(self, considerPursuerFailure):
    """Sets the flag whether to consider pursuer's failure.

    Args:
        considerPursuerFailure (bool): the game outcome considers the pursuer
            hitting the failure set if True.
    """
    self.considerPursuerFailure = considerPursuerFailure

  # == Margin Functions ==
  def safety_margin(self, s):
    """
    Computes the margin (e.g. distance) between evader and its failue set and
    the distance between the evader and the pursuer.

    Args:
        s (np.ndarray): the state of the environment.

    Returns:
        float: postivive numbers indicate that the evader is inside its failure
            set or captured by the pursuer (safety violation).
    """
    evader_g_x = self.evader.safety_margin(s[:2])
    dist_evader_pursuer = np.linalg.norm(s[:2] - s[3:5], ord=2)
    capture_g_x = self.capture_range - dist_evader_pursuer
    return max(evader_g_x, capture_g_x)

  def target_margin(self, s):
    """
    Computes the margin (e.g. distance) between the evader and its target set.
    If self.considerPursuerFailure is True, then it also considers the failure
    of the pursuer.

    Args:
        s (np.ndarray): the state.

    Returns:
        float: target margin. Negative numbers indicate that the evader reaches
            the target (or the pursuer fail).
    """
    evader_l_x = self.evader.target_margin(s[:2])
    if self.considerPursuerFailure:
      pursuer_g_x = self.evader.safety_margin(s[3:5])
      return min(evader_l_x, -pursuer_g_x)
    else:
      return evader_l_x

  # == Getting Functions ==
  def get_warmup_examples(
      self, num_warmup_samples=100, theta=None, xPursuer=None, yPursuer=None,
      thetaPursuer=None
  ):
    """Gets the warmup samples to initialize the Q-network.

    Args:
        num_warmup_samples (int, optional): # warmup samples. Defaults to 100.
        theta (float, optional): if provided, set the theta to its value.
            Defaults to None.
        xPursuer (float, optional): if provided, set the x-position of the
            pursuer to its value. Defaults to None.
        yPursuer (float, optional): if provided, set the y-position of the
            pursuer to its value. Defaults to None.
        thetaPursuer (float, optional): if provided, set the theta of the
            pursuer to its value. Defaults to None. Defaults to None.

    Returns:
        np.ndarray: sampled states.
        np.ndarray: the heuristic values, here we used max{ell, g}.
    """
    lowExt = np.tile(self.low, 2)
    highExt = np.tile(self.high, 2)
    states = np.random.default_rng().uniform(
        low=lowExt, high=highExt,
        size=(num_warmup_samples, self.state.shape[0])
    )
    if theta is not None:
      states[:, 2] = theta
    if xPursuer is not None:
      states[:, 3] = xPursuer
    if yPursuer is not None:
      states[:, 4] = yPursuer
    if thetaPursuer is not None:
      states[:, 5] = thetaPursuer

    heuristic_v = np.zeros((num_warmup_samples, self.action_space.n))

    for i in range(num_warmup_samples):
      state = states[i]
      l_x = self.target_margin(state)
      g_x = self.safety_margin(state)
      heuristic_v[i, :] = np.maximum(l_x, g_x)

    return states, heuristic_v

  # 2D-plot based on evader's x and y
  def get_axes(self):
    """Gets the axes bounds and aspect_ratio.

    Returns:
        np.ndarray: axes bounds.
        float: aspect ratio.
    """
    aspect_ratio = ((self.bounds[0, 1] - self.bounds[0, 0]) /
                    (self.bounds[1, 1] - self.bounds[1, 0]))
    axes = np.array([
        self.bounds[0, 0], self.bounds[0, 1], self.bounds[1, 0], self.bounds[1,
                                                                             1]
    ])
    return [axes, aspect_ratio]

  # Fix evader's theta and pursuer's (x, y, theta)
  def get_value(
      self, q_func, theta, xPursuer, yPursuer, thetaPursuer, nx=101, ny=101,
      addBias=False, verbose=False
  ):
    """
    Gets the state values given the Q-network. We fix evader's heading angle to
    theta and pursuer's state to [xPursuer, yPursuer, thetaPursuer].

    Args:
        q_func (object): agent's Q-network.
        theta (float): the heading angle of the evader.
        xPursuer (float): the x-position of the pursuer.
        yPursuer (float): the y-position of the pursuer.
        thetaPursuer (float): the heading angle of the pursuer.
        nx (int, optional): # points in x-axis. Defaults to 101.
        ny (int, optional): # points in y-axis. Defaults to 101.
        addBias (bool, optional): adding bias to the values if True.
            Defaults to False.
        verbose (bool, optional): print if True. Defaults to False.

    Returns:
        np.ndarray: values
    """
    if verbose:
      print(
          "Getting values with evader's theta and pursuer's"
          + "(x, y, theta) equal to {:.1f} and ".format(theta) +
          "({:.1f}, {:.1f}, {:.1f})".format(xPursuer, yPursuer, thetaPursuer)
      )
    v = np.zeros((nx, ny))
    it = np.nditer(v, flags=['multi_index'])
    xs = np.linspace(self.bounds[0, 0], self.bounds[0, 1], nx)
    ys = np.linspace(self.bounds[1, 0], self.bounds[1, 1], ny)
    while not it.finished:
      idx = it.multi_index
      x = xs[idx[0]]
      y = ys[idx[1]]

      state = np.array([x, y, theta, xPursuer, yPursuer, thetaPursuer])
      l_x = self.target_margin(state)
      g_x = self.safety_margin(state)

      # Q(s, a^*)
      state = torch.FloatTensor(state).to(self.device)
      with torch.no_grad():
        state_action_values = q_func(state)
      Q_mtx = state_action_values.reshape(
          self.numActionList[0], self.numActionList[1]
      )
      pursuerValues, _ = Q_mtx.max(dim=-1)
      minmaxValue, _ = pursuerValues.min(dim=-1)
      minmaxValue = minmaxValue.cpu().numpy()

      if addBias:
        v[idx] = minmaxValue + max(l_x, g_x)
      else:
        v[idx] = minmaxValue
      it.iternext()
    return v

  def report(self):
    """Reports the information about the environment, the evader and the pursuer.
    """
    stateDim = self.state.shape[0]
    actionNum = self.action_space.n
    print("Env: mode---{:s}; doneType---{:s}".format(self.mode, self.doneType))
    print(
        "State Dimension: {:d}, ActionSpace Dimension: {:d}".format(
            stateDim, actionNum
        )
    )
    print("Dynamic parameters:")
    print("  EVADER", end='\n    ')
    print(
        "Constraint: {:.1f} ".format(self.evader.constraint_radius)
        + "Target: {:.1f} ".format(self.evader.target_radius)
        + "Turn: {:.2f} ".format(self.evader.R_turn)
        + "Max speed: {:.2f} ".format(self.evader.speed)
        + "Max angular speed: {:.3f}".format(self.evader.max_turning_rate)
    )
    print("  PURSUER", end='\n    ')
    print(
        "Constraint: {:.1f} ".format(self.pursuer.constraint_radius)
        + "Turn: {:.2f} ".format(self.pursuer.R_turn)
        + "Max speed: {:.2f} ".format(self.pursuer.speed)
        + "Max angular speed: {:.3f}".format(self.pursuer.max_turning_rate)
    )
    if self.considerPursuerFailure:
      print("Target set also includes failure set of the pursuer")
    else:
      print("Target set only includes target set of the evader")
    print('Discrete Controls:', self.evader.discrete_controls)
    tmp = 2 * self.evader.R_turn - self.evader.constraint_radius
    if tmp > self.evader.target_radius:
      print("Type II Reach-Avoid Set")
    else:
      print("Type I Reach-Avoid Set")

  # == Trajectory Functions ==
  def simulate_one_trajectory(
      self, q_func, T=10, state=None, theta=None, keepOutOf=False, toEnd=False
  ):
    """Simulates the trajectory given the state or randomly initialized.

    Args:
        q_func (object): agent's Q-network.
        T (int, optional): the maximum length of the trajectory.
            Defaults to 250.
        state (np.ndarray, optional): if provided, set the initial state to
            its value. Defaults to None.
        theta (float, optional): if provided, set the theta to its value.
            Defaults to None.
        keepOutOf (bool, optional): smaple states inside obstacles if False.
            Defaults to False.
        toEnd (bool, optional): simulate the trajectory until the robot
            crosses the boundary if True. Defaults to False.

    Returns:
        np.ndarray: states of the evader, of the shape (length, 3).
        np.ndarray: states of the pursuer, of the shape (length, 3).
        int: the binary reach-avoid outcome.
        float: the minimum reach-avoid value of the trajectory.
        dictionary: extra information, (v_x, g_x, ell_x) along the trajectory.
    """
    # reset
    sample_inside_obs = not keepOutOf
    sample_inside_tar = not keepOutOf
    if state is None:
      stateEvader = self.evader.sample_random_state(
          sample_inside_obs=sample_inside_obs,
          sample_inside_tar=sample_inside_tar, theta=theta
      )
      statePursuer = self.pursuer.sample_random_state(
          sample_inside_obs=sample_inside_obs,
          sample_inside_tar=sample_inside_tar, theta=theta
      )
    else:
      stateEvader = state[:3]
      statePursuer = state[3:]

    trajEvader = []
    trajPursuer = []
    result = -1  # not finished

    valueList = []
    gxList = []
    lxList = []
    for t in range(T):
      trajEvader.append(stateEvader[:3])
      trajPursuer.append(statePursuer[:3])
      state = np.concatenate((stateEvader, statePursuer), axis=0)
      doneEvader = not self.evader.check_within_bounds(stateEvader)
      donePursuer = not self.pursuer.check_within_bounds(statePursuer)

      g_x = self.safety_margin(state)
      l_x = self.target_margin(state)

      # = Rollout Record
      if t == 0:
        maxG = g_x
        current = max(l_x, maxG)
        minV = current
      else:
        maxG = max(maxG, g_x)
        current = max(l_x, maxG)
        minV = min(current, minV)

      valueList.append(minV)
      gxList.append(g_x)
      lxList.append(l_x)

      if toEnd:
        if doneEvader and donePursuer:
          result = 1
          break
      else:
        if g_x > 0:
          result = -1  # failed
          break
        elif l_x <= 0:
          result = 1  # succeeded
          break

      # = Dynamics
      stateTensor = torch.FloatTensor(state).to(self.device)
      with torch.no_grad():
        state_action_values = q_func(stateTensor)
      Q_mtx = state_action_values.reshape(
          self.numActionList[0], self.numActionList[1]
      )
      pursuerValues, colIndices = Q_mtx.max(dim=1)
      _, rowIdx = pursuerValues.min(dim=0)
      colIdx = colIndices[rowIdx]

      # If cars are within the boundary, we update their states according
      # to the controls
      # if not doneEvader:
      uEvader = self.evader.discrete_controls[rowIdx]
      stateEvader = self.evader.integrate_forward(stateEvader, uEvader)
      # if not donePursuer:
      uPursuer = self.pursuer.discrete_controls[colIdx]
      statePursuer = self.pursuer.integrate_forward(statePursuer, uPursuer)

    trajEvader = np.array(trajEvader)
    trajPursuer = np.array(trajPursuer)
    info = {'valueList': valueList, 'gxList': gxList, 'lxList': lxList}
    return trajEvader, trajPursuer, result, minV, info

  def simulate_trajectories(
      self, q_func, T=10, num_rnd_traj=None, states=None, theta=None,
      keepOutOf=False, toEnd=False
  ):
    """
    Simulates the trajectories. If the states are not provided, we pick the
    initial states from the discretized state space.

    Args:
        q_func (object): agent's Q-network.
        T (int, optional): the maximum length of the trajectory.
            Defaults to 250.
        num_rnd_traj (int, optional): #states. Defaults to None.
        theta (float, optional): if provided, set the theta to its value.
            Defaults to None.
        states ([type], optional): if provided, set the initial states to
            its value. Defaults to None.
        keepOutOf (bool, optional): smaple states inside the obstacles or
            not. Defaults to False.
        toEnd (bool, optional): simulate the trajectory until the robot
            crosses the boundary if True. Defaults to False.

    Returns:
        list of np.ndarrays: each element is a tuple consisting of trajectories
            of the evader and pursuer.
        np.ndarray: the binary reach-avoid outcomes.
        np.ndarray: the minimum reach-avoid values of the trajectories.
    """
    assert ((num_rnd_traj is None and states is not None)
            or (num_rnd_traj is not None and states is None)
            or (len(states) == num_rnd_traj))
    trajectories = []

    if states is None:
      results = np.empty(shape=(num_rnd_traj,), dtype=int)
      minVs = np.empty(shape=(num_rnd_traj,), dtype=float)
      for idx in range(num_rnd_traj):
        trajEvader, trajPursuer, result, minV, _ = \
            self.simulate_one_trajectory(
                q_func, T=T, theta=theta, keepOutOf=keepOutOf,
                toEnd=toEnd
            )
        trajectories.append((trajEvader, trajPursuer))
        results[idx] = result
        minVs[idx] = minV
    else:
      results = np.empty(shape=(len(states),), dtype=int)
      minVs = np.empty(shape=(len(states),), dtype=float)
      for idx, state in enumerate(states):
        trajEvader, trajPursuer, result, minV, _ = \
            self.simulate_one_trajectory(
                q_func, T=T, state=state, toEnd=toEnd)
        trajectories.append((trajEvader, trajPursuer))
        results[idx] = result
        minVs[idx] = minV
    return trajectories, results, minVs

  # == Plotting Functions ==
  def render(self):
    pass

  def visualize(
      self, q_func, vmin=-1, vmax=1, nx=101, ny=101, cmap='seismic',
      labels=None, boolPlot=False, addBias=False, theta=0., rndTraj=False,
      num_rnd_traj=10, keepOutOf=False
  ):
    """
    Visulaizes the trained Q-network in terms of state values and trajectories
    rollout.

    Args:
        q_func (object): agent's Q-network.
        vmin (int, optional): vmin in colormap. Defaults to -1.
        vmax (int, optional): vmax in colormap. Defaults to 1.
        nx (int, optional): # points in x-axis. Defaults to 101.
        ny (int, optional): # points in y-axis. Defaults to 101.
        cmap (str, optional): color map. Defaults to 'seismic'.
        labels (list, optional): x- and y- labels. Defaults to None.
        boolPlot (bool, optional): plot the values in binary form.
            Defaults to False.
        addBias (bool, optional): adding bias to the values if True.
            Defaults to False.
        theta (float, optional): if provided, set the theta to its value.
            Defaults to np.pi/2.
        rndTraj (bool, optional): random trajectories if True.
            Defaults toFalse.
        num_rnd_traj (int, optional): #states. Defaults to None.
        keepOutOf (bool, optional): smaple states inside obstacles if False.
            Defaults to False.
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    for i, (ax, state) in enumerate(zip(axes, self.visual_initial_states)):
      state = [state]
      ax.cla()
      if i == 3:
        cbarPlot = True
      else:
        cbarPlot = False

      # == Plot failure / target set ==
      self.plot_target_failure_set(
          ax=ax, xPursuer=state[0][3], yPursuer=state[0][4]
      )

      # == Plot V ==
      self.plot_v_values(
          q_func, ax=ax, fig=fig, theta=state[0][2], xPursuer=state[0][3],
          yPursuer=state[0][4], thetaPursuer=state[0][5], vmin=vmin, vmax=vmax,
          nx=nx, ny=ny, cmap=cmap, boolPlot=boolPlot, cbarPlot=cbarPlot,
          addBias=addBias
      )

      # == Plot Trajectories ==
      if rndTraj:
        self.plot_trajectories(
            q_func, T=200, num_rnd_traj=num_rnd_traj, theta=theta, toEnd=False,
            keepOutOf=keepOutOf, ax=ax, orientation=0
        )
      else:
        self.plot_trajectories(
            q_func, T=200, states=state, toEnd=False, ax=ax, orientation=0
        )

      # == Formatting ==
      self.plot_formatting(ax=ax, labels=labels)
    plt.tight_layout()

  #  2D-plot based on evader's x and y
  def plot_v_values(
      self, q_func, theta=0, xPursuer=.5, yPursuer=.5, thetaPursuer=0, ax=None,
      fig=None, vmin=-1, vmax=1, nx=101, ny=101, cmap='seismic',
      boolPlot=False, cbarPlot=True, addBias=False
  ):
    """Plots state values.

    Args:
        q_func (object): agent's Q-network.
        theta (float, optional): if provided, fix the evader's heading angle to
            its value. Defaults to 0.
        xPursuer (float, optional): if provided, fix the pursuer's x position
            to its value. Defaults to .5.
        yPursuer (float, optional): if provided, fix the pursuer's y position
            to its value. Defaults to .5.
        thetaPursuer (int, optional): if provided, fix the pursuer's heading
            angle to its value. Defaults to 0.
        ax (matplotlib.axes.Axes, optional): Defaults to None.
        fig (matplotlib.figure, optional): Defaults to None.
        vmin (int, optional): vmin in colormap. Defaults to -1.
        vmax (int, optional): vmax in colormap. Defaults to 1.
        nx (int, optional): # points in x-axis. Defaults to 201.
        ny (int, optional): # points in y-axis. Defaults to 201.
        cmap (str, optional): color map. Defaults to 'seismic'.
        boolPlot (bool, optional): plot the values in binary form.
            Defaults to False.
        cbarPlot (bool, optional): plot the color bar if True.
            Defaults to True.
        addBias (bool, optional): adding bias to the values if True.
            Defaults to False.
    """
    axStyle = self.get_axes()

    # == Plot V ==
    if theta is None:
      theta = 2.0 * np.random.uniform() * np.pi
    v = self.get_value(
        q_func, theta, xPursuer, yPursuer, thetaPursuer, nx, ny,
        addBias=addBias
    )

    if boolPlot:
      im = ax.imshow(
          v.T > 0., interpolation='none', extent=axStyle[0], origin="lower",
          cmap=cmap
      )
    else:
      im = ax.imshow(
          v.T, interpolation='none', extent=axStyle[0], origin="lower",
          cmap=cmap, vmin=vmin, vmax=vmax
      )
      if cbarPlot:
        cbar = fig.colorbar(
            im, ax=ax, pad=0.01, fraction=0.05, shrink=.95,
            ticks=[vmin, 0, vmax]
        )
        cbar.ax.set_yticklabels(labels=[vmin, 0, vmax], fontsize=16)

  # Plot trajectories based on x-y location of the evader and the pursuer
  def plot_trajectories(
      self, q_func, T=100, num_rnd_traj=None, states=None, theta=None,
      keepOutOf=False, toEnd=False, ax=None, c=[tiffany,
                                                'y'], lw=2, orientation=0
  ):
    """Plots trajectories given the agent's Q-network.

    Args:
        q_func (object): agent's Q-network.
        T (int, optional): the maximum length of the trajectory.
            Defaults to 100.
        num_rnd_traj (int, optional): #states. Defaults to None.
        states ([type], optional): if provided, set the initial states to its
            value. Defaults to None.
        theta (float, optional): if provided, set the car's heading angle to
            its value. Defaults to None.
        keepOutOf (bool, optional): smaple states inside obstacles if False.
            Defaults to False.
        toEnd (bool, optional): simulate the trajectory until the robot crosses
            the boundary if True. Defaults to False.
        ax (matplotlib.axes.Axes, optional): Defaults to None.
        c (list, optional): colors. Defaults to [tiffany, 'y'].
        lw (int, optional): linewidth. Defaults to 2.
        orientation (int, optional): counter-clockwise angle.
            Defaults to 0.

    Returns:
        np.ndarray: the binary reach-avoid outcomes.
        np.ndarray: the minimum reach-avoid values of the trajectories.
    """
    assert ((num_rnd_traj is None and states is not None)
            or (num_rnd_traj is not None and states is None)
            or (len(states) == num_rnd_traj))

    if states is not None:
      tmpStates = []
      for state in states:
        stateEvaderTilde = rotatePoint(state[:3], orientation)
        statePursuerTilde = rotatePoint(state[3:], orientation)
        tmpStates.append(
            np.concatenate((stateEvaderTilde, statePursuerTilde), axis=0)
        )
      states = tmpStates

    trajectories, results, minVs = self.simulate_trajectories(
        q_func, T=T, num_rnd_traj=num_rnd_traj, states=states, theta=theta,
        keepOutOf=keepOutOf, toEnd=toEnd
    )
    if ax is None:
      ax = plt.gca()
    for traj, result in zip(trajectories, results):
      trajEvader, trajPursuer = traj
      trajEvaderX = trajEvader[:, 0]
      trajEvaderY = trajEvader[:, 1]
      trajPursuerX = trajPursuer[:, 0]
      trajPursuerY = trajPursuer[:, 1]

      ax.scatter(trajEvaderX[0], trajEvaderY[0], s=48, c=c[0], zorder=3)
      ax.plot(trajEvaderX, trajEvaderY, color=c[0], linewidth=lw, zorder=2)
      ax.scatter(trajPursuerX[0], trajPursuerY[0], s=48, c=c[1], zorder=3)
      ax.plot(trajPursuerX, trajPursuerY, color=c[1], linewidth=lw, zorder=2)
      if result == 1:
        ax.scatter(
            trajEvaderX[-1], trajEvaderY[-1], s=60, c=c[0], marker='*',
            zorder=3
        )
      if result == -1:
        ax.scatter(
            trajEvaderX[-1], trajEvaderY[-1], s=60, c=c[0], marker='x',
            zorder=3
        )

    return results, minVs

  # Plot evader's target, constraint and pursuer's capture range
  def plot_target_failure_set(
      self, ax=None, xPursuer=.5, yPursuer=.5, lw=3, showCapture=True, c_c='m',
      c_t='y', zorder=1
  ):
    """Plots evader's target, constraint and pursuer's capture range.

    Args:
        ax (matplotlib.axes.Axes, optional)
        xPursuer (float, optional): if provided, fix the pursuer's x position
            to its value. Defaults to .5.
        yPursuer (float, optional): if provided, fix the pursuer's y position
            to its value. Defaults to .5.
        lw (float, optional): liewidth. Defaults to 3.
        showCapture (bool, optional): show pursuer's capture range if True.
            Defaults to True.
        c_c (str, optional): color of the constraint set boundary.
            Defaults to 'm'.
        c_t (str, optional): color of the target set boundary.
            Defaults to 'y'.
        zorder (int, optional): graph layers order. Defaults to 1.
    """
    plot_circle(
        self.evader.constraint_center, self.evader.constraint_radius, ax,
        c=c_c, lw=lw, zorder=zorder
    )
    plot_circle(
        self.evader.target_center, self.evader.target_radius, ax, c=c_t, lw=lw,
        zorder=zorder
    )
    if showCapture:
      plot_circle(
          np.array([xPursuer, yPursuer]), self.capture_range, ax, c=c_c, lw=lw,
          ls='--', zorder=zorder
      )

  # ! Analytic solutions available?
  def plot_reach_avoid_set(self):
    pass

  def plot_formatting(self, ax=None, labels=None):
    """Formats the visualization.

    Args:
        ax (matplotlib.axes.Axes, optional): ax to plot.
        labels (list, optional): x- and y- labels. Defaults to None.
    """
    axStyle = self.get_axes()
    ax.plot([0., 0.], [axStyle[0][2], axStyle[0][3]], c='k', zorder=0)
    ax.plot([axStyle[0][0], axStyle[0][1]], [0., 0.], c='k', zorder=0)
    # == Formatting ==
    ax.axis(axStyle[0])
    ax.set_aspect(axStyle[1])  # makes equal aspect ratio
    ax.grid(False)
    if labels is not None:
      ax.set_xlabel(labels[0], fontsize=52)
      ax.set_ylabel(labels[1], fontsize=52)

    ax.tick_params(
        axis='both', which='both', bottom=False, top=False, left=False,
        right=False
    )
    ax.xaxis.set_major_locator(LinearLocator(5))
    ax.xaxis.set_major_formatter('{x:.1f}')
    ax.yaxis.set_major_locator(LinearLocator(5))
    ax.yaxis.set_major_formatter('{x:.1f}')
