# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )

# Modified from `point_mass.py`
# TODO: here the q_func is not a lookup table. Instead, it is a NN.
#  - simulate_trajectories
#  - simulate_one_trajectory
#  - plot_trajectories

import gym.spaces
import numpy as np
import gym
import matplotlib
import matplotlib.pyplot as plt

from utils import visualize_matrix
from utils import index_to_state

import torch

# matplotlib.use("TkAgg")
matplotlib.style.use('ggplot')


class ZermeloKCEnv(gym.Env):

    def __init__(self, device):

        # State bounds.
        self.bounds = np.array([[-1.9, 1.9],  # axis_0 = state, axis_1 = bounds.
                                [-2, 9.25]])
                                
        self.low = self.bounds[:, 0]
        self.high = self.bounds[:, 1]

        # Time step parameter.
        self.time_step = 0.05

        # Dubins car parameters.
        self.upward_speed = 2.0

        # Control parameters.
        self.horizontal_rate = 1
        self.discrete_controls = np.array([-self.horizontal_rate,
                                           0,
                                           self.horizontal_rate])

        # Constraint set parameters.
        # X,Y position and Side Length.
        self.box1_x_y_length = np.array([1.25, 2, 1.5])  # Bottom right.
        self.corners1 = np.array([
                    (self.box1_x_y_length[0] - self.box1_x_y_length[2]/2.0),
                    (self.box1_x_y_length[1] - self.box1_x_y_length[2]/2.0),
                    (self.box1_x_y_length[0] + self.box1_x_y_length[2]/2.0),
                    (self.box1_x_y_length[1] + self.box1_x_y_length[2]/2.0)
                    ])
        self.box2_x_y_length = np.array([-1.25, 2, 1.5])  # Bottom left.
        self.corners2 = np.array([
                    (self.box2_x_y_length[0] - self.box2_x_y_length[2]/2.0),
                    (self.box2_x_y_length[1] - self.box2_x_y_length[2]/2.0),
                    (self.box2_x_y_length[0] + self.box2_x_y_length[2]/2.0),
                    (self.box2_x_y_length[1] + self.box2_x_y_length[2]/2.0)
                    ])
        self.box3_x_y_length = np.array([0, 6, 1.5])  # Top middle.
        self.corners3 = np.array([
                    (self.box3_x_y_length[0] - self.box3_x_y_length[2]/2.0),
                    (self.box3_x_y_length[1] - self.box3_x_y_length[2]/2.0),
                    (self.box3_x_y_length[0] + self.box3_x_y_length[2]/2.0),
                    (self.box3_x_y_length[1] + self.box3_x_y_length[2]/2.0)
                    ])

        # Target set parameters.
        self.box4_x_y_length = np.array([0, 7+1.5, 1.5])  # Top.

        # Gym variables.
        self.action_space = gym.spaces.Discrete(3)  # horizontal_rate = {-1,0,1}
        self.midpoint = (self.low + self.high)/2.0
        self.interval = self.high - self.low
        self.observation_space = gym.spaces.Box(self.midpoint - self.interval/2,
                                                self.midpoint + self.interval/2)
        self.viewer = None

        # Discretization.
        self.grid_cells = None

        # Internal state.
        self.state = np.zeros(3)

        self.seed_val = 0

        # Visualization params
        self.vis_init_flag = True
        (self.x_box1_pos, self.x_box2_pos,
         self.x_box3_pos, self.y_box1_pos,
         self.y_box2_pos, self.y_box3_pos) = self.constraint_set_boundary()
        (self.x_box4_pos, self.y_box4_pos) = self.target_set_boundary()
        self.visual_initial_states = [np.array([0, 0]),
                                      np.array([-1, -1.9]),
                                      np.array([1, -1.9]),
                                      np.array([-1, 4]),
                                      np.array([1, 4])]
        self.scaling = 4.0

        self.device = device # for torch
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
        if start is None:
            self.state = self.sample_random_state()
        else:
            self.state = start
        return np.copy(self.state)

    def sample_random_state(self, keepOutOf=False):
        flag = True
        while flag:
            rnd_state = np.random.uniform(low=self.low,
                                      high=self.high)
            l_x = self.target_margin(rnd_state)
            g_x = self.safety_margin(rnd_state)

            terminal = (g_x > 0) or (l_x <= 0)
            flag = terminal and keepOutOf

        return rnd_state

    def step(self, action):
        """ Evolve the environment one step forward under given input action.

        Args:
            action: Input action.

        Returns:
            Tuple of (next state, signed distance of current state, whether the
            episode is done, info dictionary).
        """

        # Move dynamics one step forward.
        x, y = self.state
        u = self.discrete_controls[action]

        l_x_prev = self.target_margin(self.state)
        g_x_prev = self.safety_margin(self.state)

        x, y = self.integrate_forward(x, y, u)
        self.state = np.array([x, y])

        l_x = self.target_margin(self.state)
        g_x = self.safety_margin(self.state)

        # Calculate whether episode is done.
        done = ((g_x > 0) or (l_x <= 0))
        info = {"g_x": g_x}

        if g_x > 0 or g_x_prev > 0:
            cost = 50
        elif l_x <= 0 or l_x_prev <= 0:
            cost = -20
        else:
            cost = (l_x-l_x_prev) + (g_x-g_x_prev)

        return np.copy(self.state), cost, done, info

    def integrate_forward(self, x, y, u):
        """ Integrate the dynamics forward by one step.

        Args:
            x: Position in x-axis.
            y: Position in y-axis
            theta: Heading.
            u: Contol input.

        Returns:
            State variables (x,y,theta) integrated one step forward in time.
        """
        x = x + self.time_step * u
        y = y + self.time_step * self.upward_speed
        return x, y

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
        box1_safety_margin = -(np.linalg.norm(s - self.box1_x_y_length[:2],
                               ord=np.inf) - self.box1_x_y_length[-1]/2.0)
        box2_safety_margin = -(np.linalg.norm(s - self.box2_x_y_length[:2],
                               ord=np.inf) - self.box2_x_y_length[-1]/2.0)
        box3_safety_margin = -(np.linalg.norm(s - self.box3_x_y_length[:2],
                               ord=np.inf) - self.box3_x_y_length[-1]/2.0)

        vertical_margin = (np.abs(s[1] - (self.low[1] + self.high[1])/2.0)
                           - self.interval[1]/2.0)
        horizontal_margin = np.abs(s[0]) - 2.0
        enclosure_safety_margin = max(horizontal_margin, vertical_margin)

        safety_margin = max(box1_safety_margin,
                            box2_safety_margin,
                            box3_safety_margin,
                            enclosure_safety_margin)

        return self.scaling * safety_margin

    def target_margin(self, s):
        """ Computes the margin (e.g. distance) between state and target set.

        Args:
            s: State.

        Returns:
            Margin for the state s.
        """
        box4_target_margin = (np.linalg.norm(s - self.box4_x_y_length[:2],
                              ord=np.inf) - self.box4_x_y_length[-1]/2.0)

        target_margin = box4_target_margin
        return self.scaling * target_margin

    def set_grid_cells(self, grid_cells):
        """ Set number of grid cells.

        Args:
            grid_cells: Number of grid cells as a tuple.
        """
        self.grid_cells = grid_cells

        # (self.x_opos, self.y_opos, self.x_ipos,
        #  self.y_ipos) = self.constraint_set_boundary()

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

    def set_discretization(self, grid_cells, bounds):
        """ Set number of grid cells and state bounds.

        Args:
            grid_cells: Number of grid cells as a tuple.
            bounds: Bounds for the state.
        """
        self.set_grid_cells(grid_cells)
        self.set_bounds(bounds)

    def render(self, mode='human'):
        pass

    def constraint_set_boundary(self):
        """ Computes the safe set boundary based on the analytic solution.

        The boundary of the safe set for the double integrator is determined by
        two parabolas and two line segments.

        Returns:
            Set of discrete points describing each parabola. The first and last
            two elements of the list describe the set of coordinates for the
            first and second parabola respectively.
        """
        x_box1_pos = np.array([
            self.box1_x_y_length[0] - self.box1_x_y_length[-1]/2.0,
            self.box1_x_y_length[0] - self.box1_x_y_length[-1]/2.0,
            self.box1_x_y_length[0] + self.box1_x_y_length[-1]/2.0,
            self.box1_x_y_length[0] + self.box1_x_y_length[-1]/2.0,
            self.box1_x_y_length[0] - self.box1_x_y_length[-1]/2.0])
        x_box2_pos = np.array([
            self.box2_x_y_length[0] - self.box2_x_y_length[-1]/2.0,
            self.box2_x_y_length[0] - self.box2_x_y_length[-1]/2.0,
            self.box2_x_y_length[0] + self.box2_x_y_length[-1]/2.0,
            self.box2_x_y_length[0] + self.box2_x_y_length[-1]/2.0,
            self.box2_x_y_length[0] - self.box2_x_y_length[-1]/2.0])
        x_box3_pos = np.array([
            self.box3_x_y_length[0] - self.box3_x_y_length[-1]/2.0,
            self.box3_x_y_length[0] - self.box3_x_y_length[-1]/2.0,
            self.box3_x_y_length[0] + self.box3_x_y_length[-1]/2.0,
            self.box3_x_y_length[0] + self.box3_x_y_length[-1]/2.0,
            self.box3_x_y_length[0] - self.box3_x_y_length[-1]/2.0])

        y_box1_pos = np.array([
            self.box1_x_y_length[1] - self.box1_x_y_length[-1]/2.0,
            self.box1_x_y_length[1] + self.box1_x_y_length[-1]/2.0,
            self.box1_x_y_length[1] + self.box1_x_y_length[-1]/2.0,
            self.box1_x_y_length[1] - self.box1_x_y_length[-1]/2.0,
            self.box1_x_y_length[1] - self.box1_x_y_length[-1]/2.0])
        y_box2_pos = np.array([
            self.box2_x_y_length[1] - self.box2_x_y_length[-1]/2.0,
            self.box2_x_y_length[1] + self.box2_x_y_length[-1]/2.0,
            self.box2_x_y_length[1] + self.box2_x_y_length[-1]/2.0,
            self.box2_x_y_length[1] - self.box2_x_y_length[-1]/2.0,
            self.box2_x_y_length[1] - self.box2_x_y_length[-1]/2.0])
        y_box3_pos = np.array([
            self.box3_x_y_length[1] - self.box3_x_y_length[-1]/2.0,
            self.box3_x_y_length[1] + self.box3_x_y_length[-1]/2.0,
            self.box3_x_y_length[1] + self.box3_x_y_length[-1]/2.0,
            self.box3_x_y_length[1] - self.box3_x_y_length[-1]/2.0,
            self.box3_x_y_length[1] - self.box3_x_y_length[-1]/2.0])

        return (x_box1_pos, x_box2_pos, x_box3_pos,
                y_box1_pos, y_box2_pos, y_box3_pos)

    def target_set_boundary(self):
        """ Computes the safe set boundary based on the analytic solution.

        The boundary of the safe set for the double integrator is determined by
        two parabolas and two line segments.

        Returns:
            Set of discrete points describing each parabola. The first and last
            two elements of the list describe the set of coordinates for the
            first and second parabola respectively.
        """
        x_box4_pos = np.array([
            self.box4_x_y_length[0] - self.box4_x_y_length[-1]/2.0,
            self.box4_x_y_length[0] - self.box4_x_y_length[-1]/2.0,
            self.box4_x_y_length[0] + self.box4_x_y_length[-1]/2.0,
            self.box4_x_y_length[0] + self.box4_x_y_length[-1]/2.0,
            self.box4_x_y_length[0] - self.box4_x_y_length[-1]/2.0])

        y_box4_pos = np.array([
            self.box4_x_y_length[1] - self.box4_x_y_length[-1]/2.0,
            self.box4_x_y_length[1] + self.box4_x_y_length[-1]/2.0,
            self.box4_x_y_length[1] + self.box4_x_y_length[-1]/2.0,
            self.box4_x_y_length[1] - self.box4_x_y_length[-1]/2.0,
            self.box4_x_y_length[1] - self.box4_x_y_length[-1]/2.0])

        return (x_box4_pos, y_box4_pos)

    def get_value(self, q_func):
        v = np.zeros(self.grid_cells)
        it = np.nditer(v, flags=['multi_index'])
        while not it.finished:
            idx = it.multi_index
            state = index_to_state(self.grid_cells, self.bounds, idx)
            state = torch.FloatTensor(state, device=self.device).unsqueeze(0)
            v[idx] = q_func(state).min(dim=1)[0].item()
            it.iternext()
        return v

    def visualize_analytic_comparison(  self, q_func, no_show=False, 
                                        vmin=-50, vmax=50,
                                        labels=["x", "y"]):
        """ Overlays analytic safe set on top of state value function.

        Args:
            v: State value function.
        """
        plt.clf()
        v = self.get_value(q_func)
        im = visualize_matrix(v.T, self.get_axes(labels), no_show, vmin=vmin, vmax=vmax)

        # Plot bounadries of constraint set.
        plt.plot(self.x_box1_pos, self.y_box1_pos, color="black")
        plt.plot(self.x_box2_pos, self.y_box2_pos, color="black")
        plt.plot(self.x_box3_pos, self.y_box3_pos, color="black")
        # Plot boundaries of target set.
        plt.plot(self.x_box4_pos, self.y_box4_pos, color="black")

        plt.colorbar(im)

    def simulate_one_trajectory(self, q_func, T=10, state=None):

        if state is None:
            state = self.sample_random_state()
        x, y = state
        traj_x = [x]
        traj_y = [y]

        for t in range(T):
            if self.safety_margin(state) > 0 or self.target_margin(state) < 0:
                break
            state = torch.FloatTensor(state, device=self.device).unsqueeze(0)
            action_index = q_func(state).min(dim=1)[1].item()
            u = self.discrete_controls[action_index]

            x, y = self.integrate_forward(x, y, u)
            state = np.array([x, y])
            traj_x.append(x)
            traj_y.append(y)

        return traj_x, traj_y

    def simulate_trajectories(self, q_func, T=10, num_rnd_traj=None,
                              states=None):

        assert ((num_rnd_traj is None and states is not None) or
                (num_rnd_traj is not None and states is None) or
                (len(states) == num_rnd_traj))
        trajectories = []

        if states is None:
            for _ in range(num_rnd_traj):
                trajectories.append(self.simulate_one_trajectory(q_func, T=T))
        else:
            for state in states:
                trajectories.append(
                    self.simulate_one_trajectory(q_func, T=T, state=state))

        return trajectories

    def plot_trajectories(self, q_func, T=10, num_rnd_traj=None, states=None):

        assert ((num_rnd_traj is None and states is not None) or
                (num_rnd_traj is not None and states is None) or
                (len(states) == num_rnd_traj))
        trajectories = self.simulate_trajectories(q_func, T=T,
                                                  num_rnd_traj=num_rnd_traj,
                                                  states=states)

        for traj in trajectories:
            traj_x, traj_y = traj
            plt.scatter(traj_x[0], traj_y[0], s=32, c='r')
            plt.plot(traj_x, traj_y, color="black")

    def get_axes(self, labels=["x", "y"]):
        """ Gets the bounds for the environment.

        Returns:
            List containing a list of bounds for each state coordinate and a
            list for the name of each state coordinate.
        """
        return [np.append(self.bounds[0], self.bounds[1]), labels]
