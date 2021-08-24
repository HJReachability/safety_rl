"""
Please contact the author(s) of this library if you have any questions.
Authors: Kai-Chieh Hsu        ( kaichieh@princeton.edu )
"""

import numpy as np
from .env_utils import calculate_margin_circle, calculate_margin_rect


class DubinsCarDyn(object):

    def __init__(self, doneType='toEnd'):
        """
        __init__

        Args:
            doneType (str, optional): conditions to raise `done flag in
                training. Defaults to 'toEnd'.
        """
        # State bounds.
        self.bounds = np.array([[-1.1, 1.1],
                                [-1.1, 1.1],
                                [0, 2 * np.pi]])
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
            -self.max_turning_rate, 0., self.max_turning_rate])

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
        sample_inside_tar=True,
    ):
        """ Reset the state of the environment.

        Args:
            start: Which state to reset the environment to. If None, pick the
                state uniformly at random.

        Returns:
            The state the environment has been reset to.
        """
        if start is None:
            x_rnd, y_rnd, theta_rnd = self.sample_random_state(
                sample_inside_obs=sample_inside_obs,
                sample_inside_tar=sample_inside_tar,
                theta=theta)
            self.state = np.array([x_rnd, y_rnd, theta_rnd])
        else:
            self.state = start
        return np.copy(self.state)

    def sample_random_state(
        self, sample_inside_obs=False, sample_inside_tar=True, theta=None,
    ):
        """
        sample_random_state: pick the state uniformly at random.

        Args:
            sample_inside_obs (bool, optional): sampling initial state inside
                of the obstacles or not. Defaults to False.
            sample_inside_tar (bool, optional): sampling initial state inside
                of the targets or not. Defaults to True.
            theta (float, optional): if provided, set the theta to its value.
                Defaults to None.

        Returns:
            np.ndarray: sampled initial state.
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
        """
        step: Evolve the environment one step forward under given input action.

        Args:
            action (int): the index of action set.

        Returns:
            Tuple of (next state, signed distance of current state, whether the
            episode is done, info dictionary).
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
        """
        integrate_forward: Integrate the dynamics forward by one step.

        Args:
            state (np.ndarray): x, y - position
                                theta: Heading.
            u (float): contol inputs, angular speed.

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
        """
        set_bounds: set the boundary of the environment.

        Args:
            bounds (np.ndarray): of the shape (n_dim, 2). each row is [LB, UB].
        """
        self.bounds = bounds

        # Get lower and upper bounds
        self.low = np.array(self.bounds)[:, 0]
        self.high = np.array(self.bounds)[:, 1]

    def set_speed(self, speed=.5):
        """
        set_speed: set speed of the car. The speed influences the angular speed
            and the discrete control set.

        Args:
            speed (float, optional): spped of the car. Defaults to .5.
        """
        self.speed = speed
        self.max_turning_rate = self.speed / self.R_turn  # w
        self.discrete_controls = np.array([
            -self.max_turning_rate, 0., self.max_turning_rate])

    def set_time_step(self, time_step=.05):
        """
        set_time_step

        Args:
            time_step (float, optional): time step used in the
                integrate_forward. Defaults to .05.
        """
        self.time_step = time_step

    def set_radius(self, target_radius=.3, constraint_radius=1., R_turn=.6):
        """
        set_radius: set target_radius, constraint_radius and turning radius.

        Args:
            target_radius (float, optional): Defaults to .3.
            constraint_radius ([type], optional): Defaults to 1.0.
            R_turn (float, optional): Defaults to .6.
        """
        self.target_radius = target_radius
        self.constraint_radius = constraint_radius
        self.set_radius_rotation(R_turn=R_turn)

    def set_radius_rotation(self, R_turn=.6, verbose=False):
        """
        set_radius_rotation: set turning radius of the car. The turning radius
            influences the angular speed and the discrete control set.

        Args:
            R_turn (float, optional): turning radius. Defaults to .6.
            verbose (bool, optional): print or not. Defaults to False.
        """
        self.R_turn = R_turn
        self.max_turning_rate = self.speed / self.R_turn  # w
        self.discrete_controls = np.array([
            -self.max_turning_rate, 0., self.max_turning_rate])
        if verbose:
            print(self.discrete_controls)

    def set_constraint(self, center, radius):
        """
        set_constraint: set the constraint set (complement of failure set).

        Args:
            center (np.ndarray, optional): center of the constraint set.
            radius (float, optional): radius of the constraint set.
        """
        self.constraint_center = center
        self.constraint_radius = radius

    def set_target(self, center, radius):
        """
        set_target: set the target set.

        Args:
            center (np.ndarray, optional): center of the target set.
            radius (float, optional): radius of the target set.
        """
        self.target_center = center
        self.target_radius = radius

    # == Getting Functions ==
    def check_within_bounds(self, state):
        """
        check_within_bounds: check if the robot is still in the environment.

        Args:
            state (np.ndarray): the state.

        Returns:
            bool: False if not in the environment.
        """
        for dim, bound in enumerate(self.bounds):
            flagLow = state[dim] < bound[0]
            flagHigh = state[dim] > bound[1]
            if flagLow or flagHigh:
                return False
        return True

    # == Compute Margin ==
    def safety_margin(self, s):
        """
        safety_margin: Compute the margin (e.g. distance) between state and
            failue set.

        Args:
            s (np.ndarray): the state.

        Returns:
            float: safetyt margin. Postivive numbers indicate safety violation.
        """
        x, y = (self.low + self.high)[:2] / 2.0
        w, h = (self.high - self.low)[:2]
        boundary_margin = calculate_margin_rect(
            s, [x, y, w, h], negativeInside=True)
        g_xList = [boundary_margin]

        if (
            (self.constraint_center is not None)
            and (self.constraint_radius is not None)
        ):
            g_x = calculate_margin_circle(
                s, [self.constraint_center, self.constraint_radius],
                negativeInside=True)
            g_xList.append(g_x)

        safety_margin = np.max(np.array(g_xList))
        return self.safetyScaling * safety_margin

    def target_margin(self, s):
        """
        target_margin: Compute the margin (e.g. distance) between state and
            target set.

        Args:
            s (np.ndarray): the state.


        Returns:
            float: target margin. Negative numbers indicate reaching the
                target. If the target is not specified, return None.
        """
        if self.target_center is not None and self.target_radius is not None:
            target_margin = calculate_margin_circle(
                s, [self.target_center, self.target_radius],
                negativeInside=True)
            return self.targetScaling * target_margin
        else:
            return None
