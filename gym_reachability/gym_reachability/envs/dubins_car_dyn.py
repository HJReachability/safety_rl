"""
Please contact the author(s) of this library if you have any questions.
Authors: Kai-Chieh Hsu        ( kaichieh@princeton.edu )

This module implements the parent class for the Dubins car environments, e.g.,
one car environment and pursuit-evasion game between two Dubins cars.
"""

import numpy as np
from .env_utils import calculate_margin_circle, calculate_margin_rect


class DubinsCarDyn(object):
  """
  This base class implements a Dubins car dynamical system as well as the
  environment with concentric circles. The inner circle is the target set
  boundary, while the outer circle is the boundary of the constraint set.
  """

  def __init__(self, doneType='toEnd'):
    """Initialize the environment with the episode termination criterion.

    Args:
        doneType (str, optional): conditions to raise `done` flag in
            training. Defaults to 'toEnd'.
    """
    # State bounds.
    self.bounds = np.array([[-1.1, 1.1], [-1.1, 1.1], [0, 2 * np.pi]])
    self.low = self.bounds[:, 0]
    self.high = self.bounds[:, 1]

    # Dubins car parameters.
    self.alive = True
    self.time_step = 0.05
    self.speed = 0.5  # v

    # Control parameters.
    self.R_turn = .6
    self.max_turning_rate = self.speed / self.R_turn  # w
    self.discrete_controls = np.array([
        -self.max_turning_rate, 0., self.max_turning_rate
    ])

    # Constraint set parameters.
    self.constraint_center = None
    self.constraint_radius = None

    # Target set parameters.
    self.target_center = None
    self.target_radius = None

    # Internal state.
    self.state = np.zeros(3)
    self.doneType = doneType

    # Set random seed.
    self.seed_val = 0
    np.random.seed(self.seed_val)

    # Cost Params
    self.targetScaling = 1.
    self.safetyScaling = 1.

  def reset(
      self, start=None, theta=None, sample_inside_obs=False,
      sample_inside_tar=True
  ):
    """Reset the state of the environment.

    Args:
        start (np.ndarray, optional): the state to reset the Dubins car to. If
            None, pick the state uniformly at random. Defaults to None.
        theta (float, optional): if provided, set the initial heading angle
            (yaw). Defaults to None.
        sample_inside_obs (bool, optional): consider sampling the state inside
            the obstacles if True. Defaults to False.
        sample_inside_tar (bool, optional): consider sampling the state inside
            the target if True. Defaults to True.

    Returns:
        np.ndarray: the state that Dubins car has been reset to.
    """
    if start is None:
      x_rnd, y_rnd, theta_rnd = self.sample_random_state(
          sample_inside_obs=sample_inside_obs,
          sample_inside_tar=sample_inside_tar, theta=theta
      )
      self.state = np.array([x_rnd, y_rnd, theta_rnd])
    else:
      self.state = start
    return np.copy(self.state)

  def sample_random_state(
      self, sample_inside_obs=False, sample_inside_tar=True, theta=None
  ):
    """Pick the state uniformly at random.

    Args:
        sample_inside_obs (bool, optional): consider sampling the state inside
            the obstacles if True. Defaults to False.
        sample_inside_tar (bool, optional): consider sampling the state inside
            the target if True. Defaults to True.
        theta (float, optional): if provided, set the initial heading angle
            (yaw). Defaults to None.

    Returns:
        np.ndarray: the sampled initial state.
    """
    # random sample `theta`
    if theta is None:
      theta_rnd = 2.0 * np.random.uniform() * np.pi
    else:
      theta_rnd = theta

    # random sample [`x`, `y`]
    flag = True
    while flag:
      rnd_state = np.random.uniform(low=self.low[:2], high=self.high[:2])
      l_x = self.target_margin(rnd_state)
      g_x = self.safety_margin(rnd_state)

      if (not sample_inside_obs) and (g_x > 0):
        flag = True
      elif (not sample_inside_tar) and (l_x <= 0):
        flag = True
      else:
        flag = False
    x_rnd, y_rnd = rnd_state

    return x_rnd, y_rnd, theta_rnd

  # == Dynamics ==
  def step(self, action):
    """Evolve the environment one step forward given an action.

    Args:
        action (int): the index of the action in the action set.

    Returns:
        np.ndarray: next state.
        bool: True if the episode is terminated.
    """
    l_x_cur = self.target_margin(self.state[:2])
    g_x_cur = self.safety_margin(self.state[:2])

    u = self.discrete_controls[action]
    state = self.integrate_forward(self.state, u)
    self.state = state

    # done
    if self.doneType == 'toEnd':
      done = not self.check_within_bounds(self.state)
    else:
      assert self.doneType == 'TF', 'invalid doneType'
      fail = g_x_cur > 0
      success = l_x_cur <= 0
      done = fail or success

    if done:
      self.alive = False

    return np.copy(self.state), done

  def integrate_forward(self, state, u):
    """Integrate the dynamics forward by one step.

    Args:
        state (np.ndarray): (x, y, yaw).
        u (float): the contol input, angular speed.

    Returns:
        np.ndarray: next state.
    """
    x, y, theta = state

    x = x + self.time_step * self.speed * np.cos(theta)
    y = y + self.time_step * self.speed * np.sin(theta)
    theta = np.mod(theta + self.time_step * u, 2 * np.pi)

    state = np.array([x, y, theta])
    return state

  # == Setting Hyper-Parameter Functions ==
  def set_bounds(self, bounds):
    """Set the boundary of the environment.

    Args:
        bounds (np.ndarray): of the shape (n_dim, 2). Each row is [LB, UB].
    """
    self.bounds = bounds

    # Get lower and upper bounds
    self.low = np.array(self.bounds)[:, 0]
    self.high = np.array(self.bounds)[:, 1]

  def set_speed(self, speed=.5):
    """Set speed of the car. The speed influences the angular speed and the
        discrete control set.

    Args:
        speed (float, optional): speed of the car. Defaults to .5.
    """
    self.speed = speed
    self.max_turning_rate = self.speed / self.R_turn  # w
    self.discrete_controls = np.array([
        -self.max_turning_rate, 0., self.max_turning_rate
    ])

  def set_time_step(self, time_step=.05):
    """Set the time step for dynamics integration.

    Args:
        time_step (float, optional): time step used in the integrate_forward.
            Defaults to .05.
    """
    self.time_step = time_step

  def set_radius(self, target_radius=.3, constraint_radius=1., R_turn=.6):
    """Set target_radius, constraint_radius and turning radius.

    Args:
        target_radius (float, optional): the radius of the target set.
            Defaults to .3.
        constraint_radius (float, optional): the radius of the constraint set.
            Defaults to 1.0.
        R_turn (float, optional): the radius of the car's circular motion.
            Defaults to .6.
    """
    self.target_radius = target_radius
    self.constraint_radius = constraint_radius
    self.set_radius_rotation(R_turn=R_turn)

  def set_radius_rotation(self, R_turn=.6, verbose=False):
    """Set radius of the car's circular motion. The turning radius influences
        the angular speed and the discrete control set.

    Args:
        R_turn (float, optional): the radius of the car's circular motion.
            Defaults to .6.
        verbose (bool, optional): print messages if True. Defaults to False.
    """
    self.R_turn = R_turn
    self.max_turning_rate = self.speed / self.R_turn  # w
    self.discrete_controls = np.array([
        -self.max_turning_rate, 0., self.max_turning_rate
    ])
    if verbose:
      print(self.discrete_controls)

  def set_constraint(self, center, radius):
    """Set the constraint set (complement of failure set).

    Args:
        center (np.ndarray, optional): center of the constraint set.
        radius (float, optional): radius of the constraint set.
    """
    self.constraint_center = center
    self.constraint_radius = radius

  def set_target(self, center, radius):
    """Set the target set.

    Args:
        center (np.ndarray, optional): center of the target set.
        radius (float, optional): radius of the target set.
    """
    self.target_center = center
    self.target_radius = radius

  # == Getting Functions ==
  def check_within_bounds(self, state):
    """Check if the agent is still in the environment.

    Args:
        state (np.ndarray): the state of the agent.

    Returns:
        bool: False if the agent is not in the environment.
    """
    for dim, bound in enumerate(self.bounds):
      flagLow = state[dim] < bound[0]
      flagHigh = state[dim] > bound[1]
      if flagLow or flagHigh:
        return False
    return True

  # == Compute Margin ==
  def safety_margin(self, s):
    """Compute the margin (e.g. distance) between the state and the failue set.

    Args:
        s (np.ndarray): the state of the agent.

    Returns:
        float: postivive numbers indicate being inside the failure set (safety
            violation).
    """
    x, y = (self.low + self.high)[:2] / 2.0
    w, h = (self.high - self.low)[:2]
    boundary_margin = calculate_margin_rect(
        s, [x, y, w, h], negativeInside=True
    )
    g_xList = [boundary_margin]

    c_c_exists = (self.constraint_center is not None)
    c_r_exists = (self.constraint_radius is not None)
    if (c_c_exists and c_r_exists):
      g_x = calculate_margin_circle(
          s, [self.constraint_center, self.constraint_radius],
          negativeInside=True
      )
      g_xList.append(g_x)

    safety_margin = np.max(np.array(g_xList))
    return self.safetyScaling * safety_margin

  def target_margin(self, s):
    """Compute the margin (e.g. distance) between the state and the target set.

    Args:
        s (np.ndarray): the state of the agent.

    Returns:
        float: negative numbers indicate reaching the target. If the target set
            is not specified, return None.
    """
    if self.target_center is not None and self.target_radius is not None:
      target_margin = calculate_margin_circle(
          s, [self.target_center, self.target_radius], negativeInside=True
      )
      return self.targetScaling * target_margin
    else:
      return None
