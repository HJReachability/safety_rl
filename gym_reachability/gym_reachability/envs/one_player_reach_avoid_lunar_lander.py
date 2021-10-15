"""
Please contact the author(s) of this library if you have any questions.
Authors: Vicenc Rubies Royo   ( vrubies@berkeley.edu )

This module implements an environment considering lunar lander dynamics. This
environemnt shows reach-avoid reinforcement learning's performance on a
high-dimensional dynamic ststem.
"""

import numpy as np

import gym
from Box2D.b2 import (
    edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef,
    contactListener
)

import matplotlib.pyplot as plt
import torch
from shapely.geometry import Polygon, Point
from gym_reachability.gym_reachability.envs import (
    MultiPlayerLunarLanderReachability
)


class OnePlayerReachAvoidLunarLander(MultiPlayerLunarLanderReachability):

  def __init__(
      self, device=torch.device("cpu"), mode='RA', observation_type='default',
      param_dict={}, rnd_seed=0, doneType='toEnd', obstacle_sampling=False
  ):
    """Initializes the environment with given arguments.

    Args:
        device (str, optional): device type (used in PyTorch).
            Defaults to torch.device("cpu").
        mode (str, optional): reinforcement learning type. Defaults to "RA".
        observation_type (str, optional): [description]. Defaults to 'default'.
        param_dict (dict, optional): dictionary to set simulation params.
            Defaults to {}.
        rnd_seed (int, optional): random seed. Defaults to 0.
        doneType (str, optional): the condition to raise `done flag in
            training. Defaults to 'toEnd'.
        obstacle_sampling (bool, optional): consider sampling the state inside
            the obstacles if True. Defaults to False.
    """

    self.parent_init = False
    super(OnePlayerReachAvoidLunarLander, self).__init__(
        device=device, num_players=1, observation_type=observation_type,
        param_dict=param_dict, rnd_seed=rnd_seed, doneType=doneType,
        obstacle_sampling=obstacle_sampling
    )
    self.parent_init = True

    # safety problem limits in --> simulator self.SCALE <--

    self.helipad_x1 = self.chunk_x[self.CHUNKS // 2 - 1]
    self.helipad_x2 = self.chunk_x[self.CHUNKS // 2 + 1]

    self.hover_min_y_dot = -0.1
    self.hover_max_y_dot = 0.1
    self.hover_min_x_dot = -0.1
    self.hover_max_x_dot = 0.1

    # fastest that lander can be falling when it hits the ground
    self.land_min_v = -1.6

    self.theta_hover_max = np.radians(
        15.0
    )  # most the lander can be tilted when landing
    self.theta_hover_min = np.radians(-15.0)

    self.midpoint_x = self.W / 2
    self.width_x = self.W

    self.midpoint_y = self.H / 2
    self.width_y = self.H

    self.hover_min_x = self.W / (self.CHUNKS - 1) * (self.CHUNKS // 2 - 1)
    self.hover_max_x = self.W / (self.CHUNKS - 1) * (self.CHUNKS // 2 + 1)
    self.hover_min_y = self.HELIPAD_Y  # calc of edges of landing pad based
    self.hover_max_y = self.HELIPAD_Y + 2  # on calc in parent reset()

    # Visualization params
    self.img_data = None
    self.scaling_factor = 3.0
    self.slices_y = np.array([1, 0, -1]) * self.scaling_factor
    self.slices_x = np.array([-1, 0, 1]) * self.scaling_factor
    self.vis_init_flag = True
    self.visual_initial_states = [
        np.array([
            self.midpoint_x + self.width_x / 4,
            self.midpoint_y + self.width_y / 4, 0, 0, 0, 0
        ])
    ]

    self.polygon_target = [(self.helipad_x1, self.HELIPAD_Y),
                           (self.helipad_x2, self.HELIPAD_Y),
                           (self.helipad_x2, self.HELIPAD_Y + 2),
                           (self.helipad_x1, self.HELIPAD_Y + 2),
                           (self.helipad_x1, self.HELIPAD_Y)]
    self.target_xy_polygon = Polygon(self.polygon_target)

    # mode: normal or extend (keep track of ell & g)
    self.mode = mode
    if mode == 'extend':
      self.sim_state = np.zeros(self.total_obs_dim + 1)

    # Visualization params
    self.axes = None
    self.img_data = None
    self.scaling_factor = 3.0
    self.slices_y = np.array([1, 0, -1]) * self.scaling_factor
    self.slices_x = np.array([-1, 0, 1]) * self.scaling_factor
    self.vis_init_flag = True
    self.visual_initial_states = [
        np.array([
            self.midpoint_x + self.width_x / 4,
            self.midpoint_y + self.width_y / 4, 0, 0, 0, 0
        ])
    ]

    if mode == 'extend':
      self.visual_initial_states = self.extend_state(
          self.visual_initial_states
      )

  def reset(self, state_in=None, terrain_polyline=None):
    """Resets the state of the environment.

    Args:
        state_in (np.ndarray, optional): assumed to be in simulation
            self.SCALE. Defaults to None.
        terrain_polyline ([type], optional): [description]. Defaults to None.

    Returns:
        np.ndarray: The state that the environment has been reset to.
    """
    return super().reset(state_in=state_in, terrain_polyline=terrain_polyline)

  def step(self, action):
    """Evolves the environment one step forward given an action.

    Args:
        action (int): the index of the action in the action set.

    Returns:
        np.ndarray: next state.
        float: the standard cost used in reinforcement learning.
        bool: True if the episode is terminated.
        dict: consist of target margin and safety margin at the new state.
    """
    return super().step(action)

  def target_margin(self, state):
    """Computes the margin (e.g. distance) between the state and the target set.

    Args:
        s (np.ndarray): the state of the agent.

    Returns:
        float: negative numbers indicate reaching the target. If the target set
            is not specified, return None.
    """
    if not self.parent_init:
      return 0
    x = state[0]
    y = state[1]
    p = Point(x, y)
    L2_distance = self.target_xy_polygon.exterior.distance(p)
    inside = 2 * self.target_xy_polygon.contains(p) - 1
    return -inside * L2_distance

  def safety_margin(self, state):
    """Computes the margin (e.g. distance) between the state and the failue set.

    Args:
        s (np.ndarray): the state of the agent.

    Returns:
        float: postivive numbers indicate being inside the failure set (safety
            violation).
    """
    if not self.parent_init:
      return 0
    x = state[0]
    y = state[1]
    p = Point(x, y)
    L2_distance = self.obstacle_polyline.exterior.distance(p)
    inside = 2 * self.obstacle_polyline.contains(p) - 1
    return -inside * L2_distance

  def set_seed(self, seed):
    """Sets the random seed.

    Args:
        seed: Random seed.
    """
    self.seed_val = seed
    np.random.seed(self.seed_val)

  def set_doneType(self, doneType):
    """Sets the episode termination criterion.

    Args:
        donetype: (str) doneType.
    """
    self.doneType = doneType

  def simulate_one_trajectory(self, q_func, T=10, state=None, init_q=False):
    """Simulates the trajectory given the state or randomly initialized.

    Args:
        q_func (object): agent's Q-network.
        T (int, optional): maximum length of the trajectory. Defaults to 10.
        state (np.ndarray, optional): if provided, set the initial state to
            its value. Defaults to None.
        init_q (bool, optional): return state values if True.
            Defaults to False.

    Returns:
        np.ndarray: x-position of states in the trajectory.
        np.ndarray: y-position of states in the trajectory.
        int: result.
        np.ndarray (optional): state values.
    """
    if state is None:
      state = self.reset()
    else:
      state = self.reset(state_in=state)
    traj_x = [state[0]]
    traj_y = [state[1]]
    result = 0  # Not finished.
    initial_q = None

    for t in range(T):
      state_sim = self.obs_scale_to_simulator_scale(state)
      s_margin = self.safety_margin(state_sim)
      t_margin = self.target_margin(state_sim)

      state_tensor = torch.FloatTensor(state)
      state_tensor = state_tensor.to(self.device).unsqueeze(0)
      action_index = q_func(state_tensor).min(dim=1)[1].item()
      if initial_q is None:
        initial_q = q_func(state_tensor).min(dim=1)[0].item()

      if s_margin > 0:
        result = -1  # Failed.
        break
      elif t_margin <= 0:
        result = 1  # Succeeded.
        break

      state, _, done, _ = self.step(action_index)
      traj_x.append(state[0])
      traj_y.append(state[1])
      if done:
        result = -1
        break

    # If the Lander get's 'stuck' in a hover position..
    if result == 0:
      result = -1

    if init_q:
      return traj_x, traj_y, result, initial_q
    return traj_x, traj_y, result

  def simulate_trajectories(
      self, q_func, T=10, num_rnd_traj=None, states=None, **kwargs
  ):
    """
    Simulates the trajectories. If the states are not provided, we pick the
    initial states from the discretized state space.

    Args:
        q_func (object): agent's Q-network.
        T (int, optional): maximum length of the trajectory. Defaults to 10.
        num_rnd_traj (int, optional): #trajectories. Defaults to None.
        states (list of np.ndarray, optional): if provided, set the initial
            states to its value. Defaults to None.

    Returns:
        list of np.ndarray: each element is a tuple consisting of x and y
            positions along the trajectory.
        np.ndarray: the binary reach-avoid outcomes.
    """
    assert ((num_rnd_traj is None and states is not None)
            or (num_rnd_traj is not None and states is None)
            or (len(states) == num_rnd_traj))
    trajectories = []

    if states is None:
      results = np.empty(shape=(num_rnd_traj,), dtype=int)
      for idx in range(num_rnd_traj):
        traj_x, traj_y, result = self.simulate_one_trajectory(q_func, T=T)
        trajectories.append((traj_x, traj_y))
        results[idx] = result
    else:
      results = np.empty(shape=(len(states),), dtype=int)
      for idx, state in enumerate(states):
        traj_x, traj_y, result = self.simulate_one_trajectory(
            q_func, T=T, state=state
        )
        trajectories.append((traj_x, traj_y))
        results[idx] = result

    return trajectories, results

  def plot_trajectories(
      self, q_func, T=10, num_rnd_traj=None, states=None, c='w', ax=None
  ):
    """Plots trajectories given the agent's Q-network.

    Args:
        q_func (object): agent's Q-network.
        T (int, optional): maximum length of the trajectory. Defaults to 10.
        num_rnd_traj (int, optional): #states. Defaults to None.
        states (list of np.ndarray, optional): if provided, set the initial
            states to its value. Defaults to None.
        c (str, optional): color. Defaults to 'w'.
        ax (matplotlib.axes.Axes, optional): Defaults to None.

    Returns:
        np.ndarray: the binary reach-avoid outcomes.
    """
    assert ((num_rnd_traj is None and states is not None)
            or (num_rnd_traj is not None and states is None)
            or (len(states) == num_rnd_traj))
    if ax is None:
      ax = plt.gca()
    trajectories, results = self.simulate_trajectories(
        q_func, T=T, num_rnd_traj=num_rnd_traj, states=states
    )
    for traj in trajectories:
      traj_x, traj_y = traj
      ax.scatter(traj_x[0], traj_y[0], s=24, c=c)
      ax.plot(traj_x, traj_y, color=c, linewidth=2)

    return results

  def get_value(
      self, q_func, nx=101, ny=101, x_dot=0., y_dot=0., theta=0., theta_dot=0.,
      addBias=False
  ):
    """Gets the state values given the Q-network.

    Args:
        q_func (object): agent's Q-network.
        theta (float): the heading angle of the car.
        nx (int, optional): # points in x-axis. Defaults to 101.
        ny (int, optional): # points in y-axis. Defaults to 101.
        x_dot (float, optional): the velocity at x direction. Defaults to 0.
        y_dot (float, optional): the velocity at y direction. Defaults to 0.
        theta (float, optional): the yaw of the agent. Defaults to 0.
        theta_dot (float, optional): the angular velocity. Defaults to 0.
        addBias (bool, optional): adding bias to the values or not.
            Defaults to False.

    Returns:
        np.ndarray: values
    """
    v = np.zeros((nx, ny))
    it = np.nditer(v, flags=['multi_index'])
    xs = np.linspace(
        self.bounds_observation[0, 0], self.bounds_observation[0, 1], nx
    )
    ys = np.linspace(
        self.bounds_observation[1, 0], self.bounds_observation[1, 1], ny
    )
    # Convert slice simulation variables to observation scale.
    (_, _, x_dot, y_dot, theta, theta_dot) = \
        self.simulator_scale_to_obs_scale(
            np.array([0, 0, x_dot, y_dot, theta, theta_dot])
    )
    # print("Start value collection on grid...")
    while not it.finished:
      idx = it.multi_index

      x = xs[idx[0]]
      y = ys[idx[1]]
      l_x = self.target_margin(
          self.obs_scale_to_simulator_scale(
              np.array([x, y, x_dot, y_dot, theta, theta_dot])
          )
      )
      g_x = self.safety_margin(
          self.obs_scale_to_simulator_scale(
              np.array([x, y, x_dot, y_dot, theta, theta_dot])
          )
      )

      if self.mode == 'normal' or self.mode == 'RA':
        state = torch.FloatTensor([x, y, x_dot, y_dot, theta,
                                   theta_dot]).to(self.device).unsqueeze(0)
      else:
        z = max([l_x, g_x])
        state = torch.FloatTensor([x, y, x_dot, y_dot, theta, theta_dot,
                                   z]).to(self.device).unsqueeze(0)
      if addBias:
        v[idx] = q_func(state).min(dim=1)[0].item() + max(l_x, g_x)
      else:
        v[idx] = q_func(state).min(dim=1)[0].item()
      # v[idx] = max(g_x, min(l_x, v[idx]))
      it.iternext()
    # print("End value collection on grid.")
    return v, xs, ys

  def get_axes(self):
    """Gets the axes bounds and aspect_ratio.

    Returns:
        np.ndarray: axes bounds.
        float: aspect ratio.
    """
    aspect_ratio = (
        (self.bounds_observation[0, 1] - self.bounds_observation[0, 0]) /
        (self.bounds_observation[1, 1] - self.bounds_observation[1, 0])
    )
    axes = np.array([
        self.bounds_observation[0, 0] - 0.05,
        self.bounds_observation[0, 1] + 0.05,
        self.bounds_observation[1, 0] - 0.15,
        self.bounds_observation[1, 1] + 0.15
    ])
    return [axes, aspect_ratio]

  def imshow_lander(self, extent=None, alpha=0.4, ax=None):
    """Plots the lunar lander.

    Args:
        extent (list, optional): the bounding box in data coordinates that
            the image will fill, os the shape (4, ). Defaults to None.
        alpha (float, optional): opacity coefficient. Defaults to 0.4.
        ax (matplotlib.axes.Axes, optional): Defaults to None.
    """
    if self.img_data is None:
      # todo{vrubies} can we find way to supress gym window?
      img_data = self.render(mode="rgb_array", plot_landers=False)
      self.close()
      self.img_data = img_data[::2, ::3, :]  # Reduce image size.
    if ax is None:
      ax = plt.gca()
    ax.imshow(
        self.img_data, interpolation='none', extent=extent, origin='upper',
        alpha=alpha
    )

  def visualize(
      self, q_func, vmin=-50, vmax=50, nx=91, ny=91, labels=['', ''],
      boolPlot=False, cmap='seismic', addBias=False, trueRAZero=False,
      lvlset=0.
  ):
    """
    Visulaizes the trained Q-network in terms of state values and trajectories
    rollout.

    Args:
        q_func (object): agent's Q-network.
        vmin (int, optional): vmin in colormap. Defaults to -50.
        vmax (int, optional): vmax in colormap. Defaults to 50.
        nx (int, optional): # points in x-axis. Defaults to 101.
        ny (int, optional): # points in y-axis. Defaults to 101.
        labels (list, optional): x- and y- labels. Defaults to None.
        boolPlot (bool, optional): plot the values in binary form.
            Defaults to False.
        cmap (str, optional): color map. Defaults to 'seismic'.
        addBias (bool, optional): adding bias to the values or not.
            Defaults to False.
        trueRAZero (bool, optional): For bool plot, use the rollout outcome if
            True. If False, use the levelset instead. Defaults to False.
        lvlset (float, optional): The levelset threshold for plotting the
            reach-avoid set. Defaults to 0.
    """
    axStyle = self.get_axes()
    numX = len(self.slices_x)
    numY = len(self.slices_y)
    if self.axes is None:
      self.fig, self.axes = plt.subplots(
          numX, numY, figsize=(2 * numY, 2 * numX), sharex=True, sharey=True
      )
    for y_jj, y_dot in enumerate(self.slices_y):
      for x_ii, x_dot in enumerate(self.slices_x):
        ax = self.axes[y_jj][x_ii]
        ax.cla()
        v, xs, ys = self.get_value(
            q_func, nx, ny, x_dot=x_dot, y_dot=y_dot, theta=0, theta_dot=0,
            addBias=addBias
        )

        # == Plot Value Function ==
        if boolPlot:
          if trueRAZero:
            nx1 = nx
            ny1 = ny
            resultMtx = np.empty((nx1, ny1), dtype=int)
            xs = np.linspace(
                self.bounds_simulation[0, 0], self.bounds_simulation[0, 1], nx1
            )
            ys = np.linspace(
                self.bounds_simulation[1, 0], self.bounds_simulation[1, 1], ny1
            )

            it = np.nditer(resultMtx, flags=['multi_index'])
            while not it.finished:
              idx = it.multi_index
              x = xs[idx[0]]
              y = ys[idx[1]]

              state = np.array([x, y, x_dot, y_dot, 0, 0])
              (traj_x, traj_y, result) = \
                  self.simulate_one_trajectory(
                      q_func, T=400, state=state
              )

              resultMtx[idx] = result
              it.iternext()
            ax.imshow(
                resultMtx.T != 1, interpolation='none', extent=axStyle[0],
                origin="lower", cmap=cmap
            )
          else:
            ax.imshow(
                v.T > lvlset, interpolation='none', extent=axStyle[0],
                origin="lower", cmap=cmap
            )
          X, Y = np.meshgrid(xs, ys)
          ax.contour(
              X, Y, v.T, levels=[-0.1], colors=('k',), linestyles=('--',),
              linewidths=(1,)
          )
        else:
          vmin = np.min(v)
          vmax = np.max(v)
          vstar = max(abs(vmin), vmax)
          ax.imshow(
              v.T, interpolation='none', extent=axStyle[0], origin="lower",
              cmap=cmap, vmin=-vstar, vmax=vstar
          )
          X, Y = np.meshgrid(xs, ys)
          ax.contour(
              X, Y, v.T, levels=[-0.1], colors=('k',), linestyles=('--',),
              linewidths=(1,)
          )

        #  == Plot Environment ==
        self.imshow_lander(extent=axStyle[0], alpha=0.4, ax=ax)

        ax.axis(axStyle[0])
        ax.grid(False)
        ax.set_aspect(axStyle[1])  # makes equal aspect ratio
        if labels is not None:
          ax.set_xlabel(labels[0], fontsize=52)
          ax.set_ylabel(labels[1], fontsize=52)

        ax.tick_params(
            axis='both', which='both', bottom=False, top=False, left=False,
            right=False
        )
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        if trueRAZero:
          return
    plt.tight_layout()

  def get_warmup_examples(self, num_warmup_samples=100, s_margin=True):
    """Gets warmup samples to initialize the Q-network.

    Args:
        num_warmup_samples (int, optional): # warmup samples. Defaults to 100.
        s_margin (bool, optional): use safety margin as heuristic values if
            True. If False, use max{ell, g} instead. Defaults to true.

    Returns:
        np.ndarray: sampled states.
        np.ndarray: the heuristic values.
    """
    rv = np.random.uniform(
        low=self.bounds_simulation[:, 0], high=self.bounds_simulation[:, 1],
        size=(num_warmup_samples, self.total_obs_dim)
    )

    heuristic_v = np.zeros((num_warmup_samples, self.action_space.n))
    states = np.zeros((num_warmup_samples, self.observation_space.shape[0]))

    for i in range(num_warmup_samples):
      s = np.array(rv[i, :])
      if s_margin:
        g_x = self.safety_margin(s)
        heuristic_v[i, :] = g_x
        states[i, :] = self.simulator_scale_to_obs_scale(s)
      else:
        l_x = self.target_margin(s)
        g_x = self.safety_margin(s)
        heuristic_v[i, :] = np.maximum(l_x, g_x)
        states[i, :] = self.simulator_scale_to_obs_scale(s)

    return states, heuristic_v

  def confusion_matrix(self, q_func, num_states=50):
    """Gets the confusion matrix using DDQN values and rollout results.

    Args:
        q_func (object): agent's Q-network.
        num_states (int, optional): # initial states to rollout a trajectoy.
            Defaults to 50.

    Returns:
        np.ndarray: confusion matrix.
    """
    confusion_matrix = np.array([[0.0, 0.0], [0.0, 0.0]])
    for ii in range(num_states):
      _, _, result, initial_q = self.simulate_one_trajectory(
          q_func, T=1000, init_q=True
      )
      assert (result == 1) or (result == -1)
      # note that signs are inverted
      if -int(np.sign(initial_q)) == np.sign(result):
        if np.sign(result) == 1:
          # True Positive. (reaches and it predicts so)
          confusion_matrix[0, 0] += 1.0
        elif np.sign(result) == -1:
          # True Negative. (collides and it predicts so)
          confusion_matrix[1, 1] += 1.0
      else:
        if np.sign(result) == 1:
          # False Positive.(reaches target, predicts it will collide)
          confusion_matrix[0, 1] += 1.0
        elif np.sign(result) == -1:
          # False Negative.(collides, predicts it will reach target)
          confusion_matrix[1, 0] += 1.0
    return confusion_matrix / num_states

  def scatter_actions(self, q_func, num_states=50):
    """Gets the action output of the Q-network by sampling states.

    Args:
        q_func (object): agent's Q-network.
        num_states (int, optional): #states to get an action. Defaults to 50.
    """
    lb = self.bounds_simulation[:, 0]
    ub = self.bounds_simulation[:, 1]
    rv = np.random.uniform(
        low=lb, high=ub, size=(num_states, self.total_obs_dim)
    )
    rv[:, 2:] = 0
    for ii in range(num_states):
      s = np.array(rv[ii, :])
      g_x = self.safety_margin(s)
      if g_x < 0:
        obs = self.simulator_scale_to_obs_scale(s)

        state_tensor = torch.FloatTensor(obs)
        state_tensor = state_tensor.to(self.device).unsqueeze(0)
        action_index = q_func(state_tensor).min(dim=1)[1].item()
        # action_index = q_func(state_tensor).sort()[1][0, 1].item()

        if action_index == 0:  # Nothing
          plt.plot(rv[ii, 0], rv[ii, 1], "r*")
        elif action_index == 1:  # Left
          plt.plot(rv[ii, 0], rv[ii, 1], "g*")
        elif action_index == 2:  # Main
          plt.plot(rv[ii, 0], rv[ii, 1], "b*")
        elif action_index == 3:  # Right
          plt.plot(rv[ii, 0], rv[ii, 1], "y*")

  def render(self, mode='human', plot_landers=True):
    """Returns rendered image.

    Args:
        mode (str, optional): Defaults to 'human'.
        plot_landers (bool, optional): Plot the lunar lander if True.
            Defaults to True.
    """
    return super().render(
        mode, plot_landers=plot_landers, target=self.polygon_target
    )
