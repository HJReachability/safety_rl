# Copyright (c) 2020–2021, The Regents of the University of California.
# All rights reserved.
#
# This file is subject to the terms and conditions defined in the LICENSE file
# included in this repository.
#
# Please contact the author(s) of this library if you have any questions.
# Authors: Vicenc Rubies-Royo   ( vrubies@berkeley.edu )

import gym.spaces
import numpy as np
import gym

from utils import nearest_real_grid_point
from utils import visualize_matrix
from utils import q_values_from_q_func
from utils import index_to_state
from utils import v_from_q


class DubinsCarEnv(gym.Env):

    def __init__(self):

        # State bounds.
        self.bounds = np.array([[-2, 2],  # axis_0 = state, axis_1 = bounds.
                                [-2, 2],
                                [-np.pi, np.pi]])
        self.low = self.bounds[:, 0]
        self.high = self.bounds[:, 1]

        # Time step parameter.
        self.time_step = 0.01

        # Dubins car parameters.
        self.speed = 1.0

        # Control parameters.
        # TODO{vrubies: Check proper rates.}
        self.max_turning_rate = 1.0
        self.discrete_controls = np.array([-self.max_turning_rate,
                                           0,
                                           self.max_turning_rate])

        # Constraint set parameters.
        self.inner_radius = 0.25
        self.outer_radius = 1.0

        # Target set parameters.
        self.target_radius = 1.0/16.0
        self.target_center_x = (self.inner_radius + self.outer_radius) / 2.0
        self.target_center_y = 0
        self.target_center = np.array([self.target_center_x,
                                       self.target_center_y])

        # Gym variables.
        self.action_space = gym.spaces.Discrete(3)  # angular_rate = {-1,0,1}
        midpoint = (self.low + self.high)/2.0
        interval = self.high - self.low
        self.observation_space = gym.spaces.Box(midpoint - interval,
                                                midpoint + interval)
        self.viewer = None

        # Discretization.
        self.grid_cells = None

        # Internal state.
        self.state = np.zeros(3)

        self.seed_val = 0

        # Set random seed.
        np.random.seed(self.seed_val)

    def reset(self, start=None):
        """ Reset the state of the environment.

        Args:
            start: Which state to reset the environment to. If None, pick the
                state uniformly at random.

        Returns:
            The state the environment has been reset to.
        """
        angle = (2.0 * np.random.uniform() - 1.0) * np.pi
        dist = np.sqrt(np.random.uniform() *
                       (self.outer_radius**2 - self.inner_radius**2) +
                       self.inner_radius**2)
        x_rnd = dist * np.cos(angle)
        y_rnd = dist * np.sin(angle)
        theta_rnd = np.random.uniform(low=self.low[-1], high=self.high[-1])
        if start is None:
            self.state = np.array([x_rnd, y_rnd, theta_rnd])
        else:
            self.state = start
        return np.copy(self.state)

    def step(self, action):
        """ Evolve the environment one step forward under given input action.

        Args:
            action: Input action.

        Returns:
            Tuple of (next state, signed distance of current state, whether the
            episode is done, info dictionary).
        """
        # The signed distance must be computed before the environment steps
        # forward.
        if self.grid_cells is None:
            l_x = self.target_margin(self.state)
            g_x = self.safety_margin(self.state)
        else:
            nearest_point = nearest_real_grid_point(
                self.grid_cells, self.bounds, self.state)
            l_x = self.target_margin(nearest_point)
            g_x = self.safety_margin(nearest_point)

        # Move dynamics one step forward.
        x, y, theta = self.state
        u = self.discrete_controls[action]

        x = x + self.time_step * self.speed * np.cos(theta)
        y = y + self.time_step * self.speed * np.sin(theta)
        theta = theta + self.time_step * u
        self.state = np.array([x, y, theta])

        # Calculate whether episode is done.
        dist_origin = np.linalg.norm(self.state[:2])
        done = (dist_origin < self.inner_radius or
                dist_origin > self.outer_radius)
        info = {"g_x": g_x}
        return np.copy(self.state), l_x, done, info

    def set_seed(self, seed):
        """ Set the random seed.

        Args:
            seed: Random seed.
        """
        self.seed_val = seed
        np.random.seed(self.seed_val)

    def safety_margin(self, s):
        """ Computes the margin (e.g. distance) between state and failue set.

        Args:
            s: State.

        Returns:
            Margin for the state s.
        """
        dist_to_origin = np.linalg.norm(s[:2])
        outer_dist = self.outer_radius - dist_to_origin
        inner_dist = dist_to_origin - self.inner_radius
        # Note the "-" sign. This ensures x \in K \iff g(x) <= 0.
        safety_margin = -min(outer_dist, inner_dist)
        # if x_in:
        #     return -1 * x_dist
        return safety_margin

    def target_margin(self, s):
        """ Computes the margin (e.g. distance) between state and target set.

        Args:
            s: State.

        Returns:
            Margin for the state s.
        """
        dist_to_target = np.linalg.norm(s[:2] - self.target_center)
        target_margin = dist_to_target - self.target_radius
        # if x_in:
        #     return -1 * x_dist
        return target_margin

    def set_grid_cells(self, grid_cells):
        """ Set number of grid cells.

        Args:
            grid_cells: Number of grid cells as a tuple.
        """
        self.grid_cells = grid_cells

    def set_bounds(self, bounds):
        """ Set state bounds.

        Args:
            bounds: Bounds for the state.
        """
        self.bounds = bounds

        # Get lower and upper bounds
        self.low = np.array(self.bounds)[:, 0]
        self.high = np.array(self.bounds)[:, 1]

        # Double the range in each state dimension for Gym interface.
        midpoint = (self.low + self.high)/2.0
        interval = self.high - self.low
        self.observation_space = gym.spaces.Box(midpoint - interval,
                                                midpoint + interval)

    def render(self, mode='human'):
        pass

    # def ground_truth_comparison(self, q_func):
    #     """ Compares the state-action value function to the ground truth.

    #     The state-action value function is first converted to a state value
    #     function, and then compared to the ground truth analytical solution.

    #     Args:
    #         q_func: State-action value function.

    #     Returns:
    #         Tuple containing number of misclassified safe and misclassified
    #         unsafe states.
    #     """
    #     computed_v = v_from_q(
    #         q_values_from_q_func(q_func, self.grid_cells, self.bounds, 2))
    #     return self.ground_truth_comparison_v(computed_v)

    # def ground_truth_comparison_v(self, computed_v):
    #     """ Compares the state value function to the analytical solution.

    #     The state value function is compared to the ground truth analytical
    #     solution by checking for sign mismatches between state-value pairs.

    #     Args:
    #         computed_v: State value function.

    #     Returns:
    #         Tuple containing number of misclassified safe and misclassified
    #         unsafe states.
    #     """
    #     analytic_v = self.analytic_v()
    #     misclassified_safe = 0
    #     misclassified_unsafe = 0
    #     it = np.nditer(analytic_v, flags=['multi_index'])
    #     while not it.finished:
    #         if analytic_v[it.multi_index] < 0 < computed_v[it.multi_index]:
    #             misclassified_safe += 1
    #         elif computed_v[it.multi_index] < 0 < analytic_v[it.multi_index]:
    #             misclassified_unsafe += 1
    #         it.iternext()
    #     return misclassified_safe, misclassified_unsafe

    # def analytic_safe_set_boundary(self):
    #     """ Computes the safe set boundary based on the analytic solution.

    #     The boundary of the safe set for the double integrator is determined by
    #     two parabolas and two line segments.

    #     Returns:
    #         Set of discrete points describing each parabola. The first and last
    #         two elements of the list describe the set of coordinates for the
    #         first and second parabola respectively.
    #     """
    #     x_low = self.target_low[0]
    #     x_high = self.target_high[0]
    #     u_max = self.control_bounds[1]  # Assumes u_max = -u_min.
    #     x_dot_num_points = self.grid_cells[1]

    #     # Edge of range.
    #     x_dot_high = (((x_high - x_low) * (2 * u_max)) ** 0.5)
    #     x_dot_low = -x_dot_high

    #     # Parabola for x_dot < 0.
    #     def x_dot_negative(x_dot):
    #         return x_low + (x_dot ** 2) / (2 * u_max)

    #     # Parabola for x_dot > 0.
    #     def x_dot_positive(x_dot):
    #         return x_high - (x_dot ** 2) / (2 * u_max)

    #     # Discrete ranges for x_dot.
    #     x_dot_negative_range = np.arange(start=x_dot_low, stop=0,
    #                                      step=-x_dot_low / x_dot_num_points)
    #     x_dot_positive_range = np.arange(start=0, stop=x_dot_high,
    #                                      step=x_dot_high / x_dot_num_points)

    #     # Compute x values.
    #     x_negative_range = x_dot_negative(x_dot_negative_range)
    #     x_positive_range = x_dot_positive(x_dot_positive_range)
    #     return [x_dot_negative_range, x_negative_range,
    #             x_dot_positive_range, x_positive_range]

    # def visualize_analytic_comparison(self, v):
    #     """ Overlays analytic safe set on top of state value function.

    #     Args:
    #         v: State value function.
    #     """
    #     import matplotlib
    #     matplotlib.use("TkAgg")
    #     from matplotlib import pyplot as plt
    #     matplotlib.style.use('ggplot')

    #     # Unpack values from analytic_safe_set_boundary.
    #     analytic_safe_set_boundary = self.analytic_safe_set_boundary()
    #     (x_dot_negative_range,
    #      x_negative_range,
    #      x_dot_positive_range,
    #      x_positive_range) = analytic_safe_set_boundary

    #     # Plot analytic parabolas.
    #     plt.plot(x_dot_positive_range, x_positive_range, color="black")
    #     plt.plot(x_dot_negative_range, x_negative_range, color="black")

    #     # Plot analytic line segments.
    #     plt.plot(x_dot_positive_range, np.ones(len(x_dot_positive_range)),
    #              color="black")
    #     plt.plot(x_dot_negative_range, -1 * np.ones(len(x_dot_negative_range)),
    #              color="black")

    #     # Visualize state value.
    #     visualize_matrix(v, self.get_axes(), no_show=False)

    # def analytic_v(self):
    #     """ Computes the discretized analytic value function.

    #     Returns:
    #         Discretized form of the analytic state value function.
    #     """
    #     x_low = self.target_low[0]
    #     x_high = self.target_high[0]
    #     u_max = self.control_bounds[1]  # Assumes u_max = -u_min.

    #     def analytic_function(x, x_dot):
    #         if x_dot >= 0:
    #             return min(x - x_low,
    #                        x_high - x - x_dot ** 2 / (2 * u_max))
    #         else:
    #             return min(x_high - x,
    #                        x - x_dot ** 2 / (2 * u_max) - x_low)

    #     v = np.zeros(self.grid_cells)
    #     it = np.nditer(v, flags=['multi_index'])
    #     while not it.finished:
    #         x, x_dot = index_to_state(self.grid_cells, self.bounds,
    #                                   it.multi_index)
    #         v[it.multi_index] = analytic_function(x, x_dot)
    #         it.iternext()
    #     return v

    def get_axes(self):
        """ Gets the bounds for the environment.

        Returns:
            List containing a list of bounds for each state coordinate and a
            list for the name of each state coordinate.
        """
        return [np.append(self.bounds[1], self.bounds[0]), ["x dot", "x"]]
