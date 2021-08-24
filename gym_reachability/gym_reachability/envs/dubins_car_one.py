"""
Please contact the author(s) of this library if you have any questions.
Authors: Kai-Chieh Hsu        ( kaichieh@princeton.edu )

This environment considers Dubins car dynamics. We construct this
environemnt to show reach-avoid Q-learning's performance on a well-known
reachability analysis benchmark.
"""

import gym.spaces
import numpy as np
import gym
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator
import torch
import random

from .dubins_car_dyn import DubinsCarDyn
from .env_utils import plot_arc, plot_circle


class DubinsCarOneEnv(gym.Env):
    """
    DubinsCarOneEnv: a gym environment considering Dubins car dynamics
    """

    def __init__(
        self,
        device,
        mode="normal",
        doneType="toEnd",
        sample_inside_obs=False,
        sample_inside_tar=True,
    ):
        """
        __init__ [summary]

        Args:
            device (str): device type (used in PyTorch).
            mode (str, optional): RL type. Defaults to 'RA'.
            doneType (str, optional): conditions to raise `done flag in
                training. Defaults to 'toEnd'.
            sample_inside_obs (bool, optional): sampling initial states inside
                of the obstacles or not. Defaults to False.
            sample_inside_tar (bool, optional): sampling initial states inside
                of the targets or not. Defaults to True.
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
        self.action_space = gym.spaces.Discrete(3)
        midpoint = (self.low + self.high) / 2.0
        interval = self.high - self.low
        self.observation_space = gym.spaces.Box(
            np.float32(midpoint - interval/2),
            np.float32(midpoint + interval/2),
        )

        # Constraint set parameters.
        self.constraint_center = np.array([0, 0])
        self.constraint_radius = 1.0

        # Target set parameters.
        self.target_center = np.array([0, 0])
        self.target_radius = 0.3

        # Internal state.
        self.mode = mode
        self.state = np.zeros(3)
        self.doneType = doneType

        # Dubins car parameters.
        self.time_step = 0.05
        self.speed = 0.5  # v
        self.R_turn = 0.6
        self.car = DubinsCarDyn(doneType=doneType)
        self.init_car()

        # Visualization params
        self.visual_initial_states = [
            np.array([0.6 * self.constraint_radius, -0.5, np.pi / 2]),
            np.array([-0.4 * self.constraint_radius, -0.5, np.pi / 2]),
            np.array([-0.95 * self.constraint_radius, 0.0, np.pi / 2]),
            np.array([
                self.R_turn,
                0.95 * (self.constraint_radius - self.R_turn),
                np.pi / 2,
            ]),
        ]
        # Cost Params
        self.targetScaling = 1.0
        self.safetyScaling = 1.0
        self.penalty = 1.0
        self.reward = -1.0
        self.costType = "sparse"
        self.device = device
        self.scaling = 1.0

        print(
            "Env: mode-{:s}; doneType-{:s}; sample_inside_obs-{}".format(
                self.mode, self.doneType, self.sample_inside_obs
            )
        )

    def init_car(self):
        """
        init_car
        """
        self.car.set_bounds(bounds=self.bounds)
        self.car.set_constraint(
            center=self.constraint_center, radius=self.constraint_radius
        )
        self.car.set_target(
            center=self.target_center, radius=self.target_radius
        )
        self.car.set_speed(speed=self.speed)
        self.car.set_time_step(time_step=self.time_step)
        self.car.set_radius_rotation(R_turn=self.R_turn, verbose=False)

    # == Reset Functions ==
    def reset(self, start=None):
        """
        reset: Reset the state of the environment.

        Args:
            start (np.ndarray, optional): state to reset the environment to.
                If None, pick the state uniformly at random. Defaults to None.

        Returns:
            np.ndarray: The state the environment has been reset to.
        """
        self.state = self.car.reset(
            start=start,
            sample_inside_obs=self.sample_inside_obs,
            sample_inside_tar=self.sample_inside_tar,
        )
        return np.copy(self.state)

    def sample_random_state(
        self, sample_inside_obs=False, sample_inside_tar=True, theta=None
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
        state = self.car.sample_random_state(
            sample_inside_obs=sample_inside_obs,
            sample_inside_tar=sample_inside_tar,
            theta=theta,
        )
        return state

    # == Dynamics Functions ==
    def step(self, action):
        """
        step: Evolve the environment one step forward under given input action.

        Args:
            action (int): the index of action set.

        Returns:
            Tuple of (next state, signed distance of current state, whether the
            episode is done, info dictionary).
        """
        distance = np.linalg.norm(self.state - self.car.state)
        assert distance < 1e-8, (
            "There is a mismatch between the env state"
            + "and car state: {:.2e}".format(distance)
        )

        state_nxt, _ = self.car.step(action)
        self.state = state_nxt
        l_x = self.target_margin(self.state[:2])
        g_x = self.safety_margin(self.state[:2])

        fail = g_x > 0
        success = l_x <= 0

        # cost
        if self.mode == "RA":
            if fail:
                cost = self.penalty
            elif success:
                cost = self.reward
            else:
                cost = 0.0
        else:
            if fail:
                cost = self.penalty
            elif success:
                cost = self.reward
            else:
                if self.costType == "dense_ell":
                    cost = l_x
                elif self.costType == "dense_ell_g":
                    cost = l_x + g_x
                elif self.costType == "sparse":
                    cost = 0.0 * self.scaling
                elif self.costType == "max_ell_g":
                    cost = max(l_x, g_x)
                else:
                    cost = 0.0

        # = `done` signal
        if self.doneType == "toEnd":
            done = not self.car.check_within_bounds(self.state)
        elif self.doneType == "fail":
            done = fail
        elif self.doneType == "TF":
            done = fail or success
        else:
            raise ValueError("invalid done type!")

        # = `info`
        if done and self.doneType == "fail":
            info = {"g_x": self.penalty * self.scaling, "l_x": l_x}
        else:
            info = {"g_x": g_x, "l_x": l_x}
        return np.copy(self.state), cost, done, info

    # == Setting Hyper-Parameter Functions ==
    def set_costParam(
        self,
        penalty=1.0,
        reward=-1.0,
        costType="sparse",
        targetScaling=1.0,
        safetyScaling=1.0,
    ):
        """
        set_costParam: set the hyper-parameters for the `cost` signal used in
            training, important for Sum Q-learning.

        Args:
            penalty (float, optional): cost when entering the obstacles or
                crossing the environment boundary. Defaults to 1.0.
            reward (float, optional): cost when reaching the targets.
                Defaults to -1.0.
            costType (str, optional): providing extra information when in
                neither the failure set nor the target set.
                Defaults to 'sparse'.
            targetScaling (float, optional): scaling factor of the target
                margin. Defaults to 1.0.
            safetyScaling (float, optional): scaling factor of the safety
                margin. Defaults to 1.0.
        """
        self.penalty = penalty
        self.reward = reward
        self.costType = costType
        self.safetyScaling = safetyScaling
        self.targetScaling = targetScaling

    def set_seed(self, seed):
        """
        set_seed: set the seed for `numpy`, `random`, `PyTorch` packages.

        Args:
            seed (int): seed value.
        """
        self.seed_val = seed
        np.random.seed(self.seed_val)
        torch.manual_seed(self.seed_val)
        torch.cuda.manual_seed(self.seed_val)
        # if you are using multi-GPU.
        torch.cuda.manual_seed_all(self.seed_val)
        random.seed(self.seed_val)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def set_bounds(self, bounds):
        """
        set_bounds: set the boundary and the observation_space of the
            environment.

        Args:
            bounds (np.ndarray): of the shape (n_dim, 2). each row is [LB, UB].
        """
        self.bounds = bounds

        # Get lower and upper bounds
        self.low = np.array(self.bounds)[:, 0]
        self.high = np.array(self.bounds)[:, 1]

        # Double the range in each state dimension for Gym interface.
        midpoint = (self.low + self.high) / 2.0
        interval = self.high - self.low
        self.observation_space = gym.spaces.Box(
            np.float32(midpoint - interval/2),
            np.float32(midpoint + interval/2),
        )
        self.car.set_bounds(bounds)

    def set_speed(self, speed=0.5):
        """
        set_speed

        Args:
            speed (float, optional): spped of the car. Defaults to .5.
        """
        self.speed = speed
        self.car.set_speed(speed=speed)

    def set_radius(self, target_radius=0.3, constraint_radius=1.0, R_turn=0.6):
        """
        set_radius: set target_radius, constraint_radius and turning radius.

        Args:
            target_radius (float, optional): Defaults to .3.
            constraint_radius ([type], optional): Defaults to 1.0.
            R_turn (float, optional): Defaults to .6.
        """
        self.target_radius = target_radius
        self.constraint_radius = constraint_radius
        self.R_turn = R_turn
        self.car.set_radius(
            target_radius=target_radius,
            constraint_radius=constraint_radius,
            R_turn=R_turn,
        )

    def set_radius_rotation(self, R_turn=0.6, verbose=False):
        """
        set_radius_rotation

        Args:
            R_turn (float, optional): turning radius. Defaults to .6.
            verbose (bool, optional): print or not. Defaults to False.
        """
        self.R_turn = R_turn
        self.car.set_radius_rotation(R_turn=R_turn, verbose=verbose)

    def set_constraint(self, center=np.array([0.0, 0.0]), radius=1.0):
        """
        set_constraint: set the constraint set (complement of failure set).

        Args:
            center (np.ndarray, optional): center of the constraint set.
                Defaults to np.array([0.,0.]).
            radius (float, optional): radius of the constraint set.
                Defaults to 1.0.
        """
        self.constraint_center = center
        self.constraint_radius = radius
        self.car.set_constraint(center=center, radius=radius)

    def set_target(self, center=np.array([0.0, 0.0]), radius=0.4):
        """
        set_target: set the target set.

        Args:
            center (np.ndarray, optional): center of the target set.
                Defaults to np.array([0.,0.]).
            radius (float, optional): radius of the target set. Defaults to .4.
        """
        self.target_center = center
        self.target_radius = radius
        self.car.set_target(center=center, radius=radius)

    # == Margin Functions ==
    def safety_margin(self, s):
        """
        safety_margin: Compute the margin (e.g. distance) between state and
            failue set.

        Args:
            s (np.ndarray): the state.

        Returns:
            float: safetyt margin. Postivive numbers indicate safety violation.
        """
        return self.car.safety_margin(s[:2])

    def target_margin(self, s):
        """
        target_margin: Compute the margin (e.g. distance) between state and
            target set.

        Args:
            s (np.ndarray): the state.

        Returns:
            float: target margin. Negative numbers indicate reaching the
                target.
        """
        return self.car.target_margin(s[:2])

    # == Getting Functions ==
    def get_warmup_examples(self, num_warmup_samples=100):
        """
        get_warmup_examples: Get the warmup samples.

        Args:
            num_warmup_samples (int, optional): # warmup samples.
                Defaults to 100.

        Returns:
            np.ndarray: sampled states.
            np.ndarray: the heuristic values, here we used max{ell, g}.
        """
        rv = np.random.uniform(
            low=self.low, high=self.high, size=(num_warmup_samples, 3)
        )
        x_rnd, y_rnd, theta_rnd = rv[:, 0], rv[:, 1], rv[:, 2]

        heuristic_v = np.zeros((num_warmup_samples, self.action_space.n))
        states = np.zeros(
            (num_warmup_samples, self.observation_space.shape[0])
        )

        for i in range(num_warmup_samples):
            x, y, theta = x_rnd[i], y_rnd[i], theta_rnd[i]
            l_x = self.target_margin(np.array([x, y]))
            g_x = self.safety_margin(np.array([x, y]))
            heuristic_v[i, :] = np.maximum(l_x, g_x)
            states[i, :] = x, y, theta

        return states, heuristic_v

    def get_axes(self):
        """
        get_axes: Get the axes bounds and aspect_ratio.

        Returns:
            np.ndarray: axes bounds.
            float: aspect ratio.
        """
        aspect_ratio = ((self.bounds[0, 1] - self.bounds[0, 0]) /
                        (self.bounds[1, 1] - self.bounds[1, 0]))
        axes = np.array([
            self.bounds[0, 0],
            self.bounds[0, 1],
            self.bounds[1, 0],
            self.bounds[1, 1],
        ])
        return [axes, aspect_ratio]

    def get_value(self, q_func, theta, nx=101, ny=101, addBias=False):
        """
        get_value: get the state values given the Q-network. We fix the heading
            angle of the car to `theta`.

        Args:
            q_func (object): agent's Q-network.
            theta (float): the heading angle of the car.
            nx (int, optional): # points in x-axis. Defaults to 101.
            ny (int, optional): # points in y-axis. Defaults to 101.
            addBias (bool, optional): adding bias to the values or not.
                Defaults to False.

        Returns:
            np.ndarray: values
        """
        v = np.zeros((nx, ny))
        it = np.nditer(v, flags=["multi_index"])
        xs = np.linspace(self.bounds[0, 0], self.bounds[0, 1], nx)
        ys = np.linspace(self.bounds[1, 0], self.bounds[1, 1], ny)
        while not it.finished:
            idx = it.multi_index
            x = xs[idx[0]]
            y = ys[idx[1]]
            l_x = self.target_margin(np.array([x, y]))
            g_x = self.safety_margin(np.array([x, y]))

            if self.mode == "normal" or self.mode == "RA":
                state = (
                    torch.FloatTensor([x, y,
                                       theta]).to(self.device).unsqueeze(0)
                )
            else:
                z = max([l_x, g_x])
                state = (
                    torch.FloatTensor([x, y, theta,
                                       z]).to(self.device).unsqueeze(0)
                )
            if addBias:
                v[idx] = q_func(state).min(dim=1)[0].item() + max(l_x, g_x)
            else:
                v[idx] = q_func(state).min(dim=1)[0].item()
            it.iternext()
        return v

    # == Trajectory Functions ==
    def simulate_one_trajectory(
        self,
        q_func,
        T=10,
        state=None,
        theta=None,
        sample_inside_obs=True,
        sample_inside_tar=True,
        toEnd=False,
    ):
        """
        simulate_one_trajectory: simulate the trajectory given the state or
            randomly initialized.

        Args:
            q_func (object): agent's Q-network.
            T (int, optional): the maximum length of the trajectory. Defaults
                to 250.
            state (np.ndarray, optional): if provided, set the initial state to
                its value. Defaults to None.
            theta ([type], optional): if provided, set the theta to its value.
                Defaults to None.
            sample_inside_obs (bool, optional): sampling initial states inside
                of the obstacles or not. Defaults to True.
            sample_inside_tar (bool, optional): sampling initial states inside
                of the targets or not. Defaults to True.
            toEnd (bool, optional): simulate the trajectory until the robot
                crosses the boundary or not. Defaults to False.

        Returns:
            np.ndarray: states of the trajectory, of the shape (length, 3).
            int: result.
            float: the minimum reach-avoid value of the trajectory.
            dictionary: extra information, (v_x, g_x, ell_x) along the traj.
        """
        # reset
        if state is None:
            state = self.car.sample_random_state(
                sample_inside_obs=sample_inside_obs,
                sample_inside_tar=sample_inside_tar,
                theta=theta,
            )
        traj = []
        result = 0  # not finished
        valueList = []
        gxList = []
        lxList = []
        for t in range(T):
            traj.append(state)

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
                done = not self.car.check_within_bounds(state)
                if done:
                    result = 1
                    break
            else:
                if g_x > 0:
                    result = -1  # failed
                    break
                elif l_x <= 0:
                    result = 1  # succeeded
                    break

            q_func.eval()
            state_tensor = (
                torch.FloatTensor(state).to(self.device).unsqueeze(0)
            )
            action_index = q_func(state_tensor).min(dim=1)[1].item()
            u = self.car.discrete_controls[action_index]

            state = self.car.integrate_forward(state, u)
        traj = np.array(traj)
        info = {"valueList": valueList, "gxList": gxList, "lxList": lxList}
        return traj, result, minV, info

    def simulate_trajectories(
        self, q_func, T=10, num_rnd_traj=None, states=None, toEnd=False
    ):
        """
        simulate_trajectories: simulate the trajectories. If the states are not
            provided, we pick the initial states from the discretized state
            space.

        Args:
            q_func (object): agent's Q-network.
            T (int, optional): the maximum length of the trajectory. Defaults
                to 250.
            num_rnd_traj (int, optional): #states. Defaults to None.
            states ([type], optional): if provided, set the initial states to
                its value. Defaults to None.
            toEnd (bool, optional): simulate the trajectory until the robot
                crosses the boundary or not. Defaults to False.

        Returns:
            list of np.ndarray: each element is a tuple consisting of x and y
                positions along the trajectory.
            np.ndarray: the binary reach-avoid outcomes.
            np.ndarray: the minimum reach-avoid values of the trajectories.
        """
        assert ((num_rnd_traj is None and states is not None)
                or (num_rnd_traj is not None and states is None)
                or (len(states) == num_rnd_traj))
        trajectories = []

        if states is None:
            nx = 41
            ny = nx
            xs = np.linspace(self.bounds[0, 0], self.bounds[0, 1], nx)
            ys = np.linspace(self.bounds[1, 0], self.bounds[1, 1], ny)
            results = np.empty((nx, ny), dtype=int)
            minVs = np.empty((nx, ny), dtype=float)

            it = np.nditer(results, flags=["multi_index"])
            print()
            while not it.finished:
                idx = it.multi_index
                print(idx, end="\r")
                x = xs[idx[0]]
                y = ys[idx[1]]
                state = np.array([x, y, 0.0])
                traj, result, minV, _ = self.simulate_one_trajectory(
                    q_func, T=T, state=state, toEnd=toEnd
                )
                trajectories.append((traj))
                results[idx] = result
                minVs[idx] = minV
                it.iternext()
            results = results.reshape(-1)
            minVs = minVs.reshape(-1)

        else:
            results = np.empty(shape=(len(states),), dtype=int)
            minVs = np.empty(shape=(len(states),), dtype=float)
            for idx, state in enumerate(states):
                traj, result, minV, _ = self.simulate_one_trajectory(
                    q_func, T=T, state=state, toEnd=toEnd
                )
                trajectories.append(traj)
                results[idx] = result
                minVs[idx] = minV

        return trajectories, results, minVs

    # == Plotting Functions ==
    def render(self):
        pass

    def visualize(
        self,
        q_func,
        vmin=-1,
        vmax=1,
        nx=101,
        ny=101,
        cmap="seismic",
        labels=None,
        boolPlot=False,
        addBias=False,
        theta=np.pi / 2,
        rndTraj=False,
        num_rnd_traj=10,
    ):
        """
        visualize

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
            addBias (bool, optional): adding bias to the values or not.
                Defaults to False.
            theta (float, optional): if provided, set the theta to its value.
                Defaults to np.pi/2.
            rndTraj (bool, optional): random trajectories or not. Defaults to
                False.
            num_rnd_traj (int, optional): #states. Defaults to None.
        """
        thetaList = [np.pi / 6, np.pi / 3, np.pi / 2]
        fig = plt.figure(figsize=(12, 4))
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)
        axList = [ax1, ax2, ax3]

        for i, (ax, theta) in enumerate(zip(axList, thetaList)):
            # for i, (ax, theta) in enumerate(zip(self.axes, thetaList)):
            ax.cla()
            if i == len(thetaList) - 1:
                cbarPlot = True
            else:
                cbarPlot = False

            # == Plot failure / target set ==
            self.plot_target_failure_set(ax)

            # == Plot reach-avoid set ==
            self.plot_reach_avoid_set(ax, orientation=theta)

            # == Plot V ==
            self.plot_v_values(
                q_func,
                ax=ax,
                fig=fig,
                theta=theta,
                vmin=vmin,
                vmax=vmax,
                nx=nx,
                ny=ny,
                cmap=cmap,
                boolPlot=boolPlot,
                cbarPlot=cbarPlot,
                addBias=addBias,
            )
            # == Formatting ==
            self.plot_formatting(ax=ax, labels=labels)

            # == Plot Trajectories ==
            if rndTraj:
                self.plot_trajectories(
                    q_func,
                    T=200,
                    num_rnd_traj=num_rnd_traj,
                    theta=theta,
                    toEnd=False,
                    ax=ax,
                    c="y",
                    lw=2,
                    orientation=0,
                )
            else:
                # `visual_initial_states` are specified for theta = pi/2. Thus,
                # we need to use "orientation = theta-pi/2"
                self.plot_trajectories(
                    q_func,
                    T=200,
                    states=self.visual_initial_states,
                    toEnd=False,
                    ax=ax,
                    c="y",
                    lw=2,
                    orientation=theta - np.pi / 2,
                )

            ax.set_xlabel(
                r"$\theta={:.0f}^\circ$".format(theta * 180 / np.pi),
                fontsize=28,
            )

        plt.tight_layout()

    def plot_v_values(
        self,
        q_func,
        theta=np.pi / 2,
        ax=None,
        fig=None,
        vmin=-1,
        vmax=1,
        nx=201,
        ny=201,
        cmap="seismic",
        boolPlot=False,
        cbarPlot=True,
        addBias=False,
    ):
        """
        plot_v_values: plot state values.

        Args:
            q_func (object): agent's Q-network.
            theta (float, optional): if provided, fix the car's heading angle
                to its value. Defaults to np.pi/2.
            ax (matplotlib.axes.Axes, optional): Defaults to None.
            fig (matplotlib.figure, optional): Defaults to None.
            vmin (int, optional): vmin in colormap. Defaults to -1.
            vmax (int, optional): vmax in colormap. Defaults to 1.
            nx (int, optional): # points in x-axis. Defaults to 201.
            ny (int, optional): # points in y-axis. Defaults to 201.
            cmap (str, optional): color map. Defaults to 'seismic'.
            boolPlot (bool, optional): plot the values in binary form.
                Defaults to False.
            cbarPlot (bool, optional): plot the color bar or not. Defaults to
                True.
            addBias (bool, optional): adding bias to the values or not.
                Defaults to False.
        """
        axStyle = self.get_axes()
        ax.plot([0.0, 0.0], [axStyle[0][2], axStyle[0][3]], c="k")
        ax.plot([axStyle[0][0], axStyle[0][1]], [0.0, 0.0], c="k")

        # == Plot V ==
        if theta is None:
            theta = 2.0 * np.random.uniform() * np.pi
        v = self.get_value(q_func, theta, nx, ny, addBias=addBias)

        if boolPlot:
            im = ax.imshow(
                v.T > 0.0,
                interpolation="none",
                extent=axStyle[0],
                origin="lower",
                cmap=cmap,
                zorder=-1,
            )
        else:
            im = ax.imshow(
                v.T,
                interpolation="none",
                extent=axStyle[0],
                origin="lower",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                zorder=-1,
            )
            if cbarPlot:
                cbar = fig.colorbar(
                    im,
                    ax=ax,
                    pad=0.01,
                    fraction=0.05,
                    shrink=0.95,
                    ticks=[vmin, 0, vmax],
                )
                cbar.ax.set_yticklabels(labels=[vmin, 0, vmax], fontsize=16)

    def plot_trajectories(
        self,
        q_func,
        T=100,
        num_rnd_traj=None,
        states=None,
        theta=None,
        toEnd=False,
        ax=None,
        c="y",
        lw=1.5,
        orientation=0,
        zorder=2,
    ):
        """
        plot_trajectories: plot trajectories given the agent's Q-network.

        Args:
            q_func (object): agent's Q-network.
            T (int, optional): the maximum length of the trajectory.
                Defaults to 100.
            num_rnd_traj (int, optional): #states. Defaults to None.
            states ([type], optional): if provided, set the initial states to
                its value. Defaults to None.
            theta (float, optional): if provided, set the car's heading angle
                to its value. Defaults to None.
            toEnd (bool, optional): simulate the trajectory until the robot
                crosses the boundary or not. Defaults to False.
            ax (matplotlib.axes.Axes, optional): Defaults to None.
            c (str, optional): color. Defaults to 'y'.
            lw (float, optional): linewidth. Defaults to 1.5.
            orientation (float, optional): counter-clockwise angle. Defaults
                to 0.
            zorder (int, optional): graph layers order. Defaults to 2.

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
                x, y, theta = state
                xtilde = x * np.cos(orientation) - y * np.sin(orientation)
                ytilde = y * np.cos(orientation) + x * np.sin(orientation)
                thetatilde = theta + orientation
                tmpStates.append(np.array([xtilde, ytilde, thetatilde]))
            states = tmpStates

        trajectories, results, minVs = self.simulate_trajectories(
            q_func, T=T, num_rnd_traj=num_rnd_traj, states=states, toEnd=toEnd
        )
        if ax is None:
            ax = plt.gca()
        for traj in trajectories:
            traj_x = traj[:, 0]
            traj_y = traj[:, 1]
            ax.scatter(traj_x[0], traj_y[0], s=48, c=c, zorder=zorder)
            ax.plot(traj_x, traj_y, color=c, linewidth=lw, zorder=zorder)

        return results, minVs

    def plot_target_failure_set(
        self, ax=None, c_c="m", c_t="y", lw=3, zorder=0
    ):
        """
        plot_target_failure_set: plot the target and the failure set.

        Args:
            ax (matplotlib.axes.Axes, optional)
            c_c (str, optional): color of the constraint set boundary.
                Defaults to 'm'.
            c_t (str, optional): color of the target set boundary.
                Defaults to 'y'.
            lw (float, optional): liewidth. Defaults to 3.
            zorder (int, optional): graph layers order. Defaults to 0.
        """
        plot_circle(
            self.constraint_center,
            self.constraint_radius,
            ax,
            c=c_c,
            lw=lw,
            zorder=zorder,
        )
        plot_circle(
            self.target_center,
            self.target_radius,
            ax,
            c=c_t,
            lw=lw,
            zorder=zorder,
        )

    def plot_reach_avoid_set(
        self, ax=None, c="g", lw=3, orientation=0, zorder=1
    ):
        """
        plot_reach_avoid_set: plot the analytic reach-avoid set.

        Args:
            ax (matplotlib.axes.Axes, optional)
            c (str, optional): color of the rach-avoid set boundary. Defaults
                to 'g'.
            lw (int, optional): liewidth. Defaults to 3.
            orientation (float, optional): counter-clockwise angle. Defaults
                to 0.
            zorder (int, optional): graph layers order. Defaults to 1.
        """
        r = self.target_radius
        R = self.constraint_radius
        R_turn = self.R_turn
        if r >= 2*R_turn - R:
            # plot arc
            tmpY = (r**2 - R**2 + 2*R_turn*R) / (2*R_turn)
            tmpX = np.sqrt(r**2 - tmpY**2)
            tmpTheta = np.arcsin(tmpX / (R-R_turn))
            # two sides
            plot_arc(
                (0.0, R_turn),
                R - R_turn,
                (tmpTheta - np.pi / 2, np.pi / 2),
                ax,
                c=c,
                lw=lw,
                orientation=orientation,
                zorder=zorder,
            )
            plot_arc(
                (0.0, -R_turn),
                R - R_turn,
                (-np.pi / 2, np.pi / 2 - tmpTheta),
                ax,
                c=c,
                lw=lw,
                orientation=orientation,
                zorder=zorder,
            )
            # middle
            tmpPhi = np.arcsin(tmpX / r)
            plot_arc(
                (0.0, 0),
                r,
                (tmpPhi - np.pi / 2, np.pi / 2 - tmpPhi),
                ax,
                c=c,
                lw=lw,
                orientation=orientation,
                zorder=zorder,
            )
            # outer boundary
            plot_arc(
                (0.0, 0),
                R,
                (np.pi / 2, 3 * np.pi / 2),
                ax,
                c=c,
                lw=lw,
                orientation=orientation,
                zorder=zorder,
            )
        else:
            # two sides
            tmpY = (R**2 + 2*R_turn*r - r**2) / (2*R_turn)
            tmpX = np.sqrt(R**2 - tmpY**2)
            tmpTheta = np.arcsin(tmpX / (R_turn-r))
            tmpTheta2 = np.arcsin(tmpX / R)
            plot_arc(
                (0.0, R_turn),
                R_turn - r,
                (np.pi / 2 + tmpTheta, 3 * np.pi / 2),
                ax,
                c=c,
                lw=lw,
                orientation=orientation,
                zorder=zorder,
            )
            plot_arc(
                (0.0, -R_turn),
                R_turn - r,
                (np.pi / 2, 3 * np.pi / 2 - tmpTheta),
                ax,
                c=c,
                lw=lw,
                orientation=orientation,
                zorder=zorder,
            )
            # middle
            plot_arc(
                (0.0, 0),
                r,
                (np.pi / 2, -np.pi / 2),
                ax,
                c=c,
                lw=lw,
                orientation=orientation,
                zorder=zorder,
            )
            # outer boundary
            plot_arc(
                (0.0, 0),
                R,
                (np.pi / 2 + tmpTheta2, 3 * np.pi / 2 - tmpTheta2),
                ax,
                c=c,
                lw=lw,
                orientation=orientation,
                zorder=zorder,
            )

    def plot_formatting(self, ax=None, labels=None):
        """
        plot_formatting: formatting the visualization

        Args:
            ax (matplotlib.axes.Axes, optional)
            labels (list, optional): x- and y- labels. Defaults to None.
        """
        axStyle = self.get_axes()
        # == Formatting ==
        ax.axis(axStyle[0])
        ax.set_aspect(axStyle[1])  # makes equal aspect ratio
        ax.grid(False)
        if labels is not None:
            ax.set_xlabel(labels[0], fontsize=52)
            ax.set_ylabel(labels[1], fontsize=52)

        ax.tick_params(
            axis="both",
            which="both",
            bottom=False,
            top=False,
            left=False,
            right=False,
        )
        ax.xaxis.set_major_locator(LinearLocator(5))
        ax.xaxis.set_major_formatter("{x:.1f}")
        ax.yaxis.set_major_locator(LinearLocator(5))
        ax.yaxis.set_major_formatter("{x:.1f}")
