# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )

# Modified from `zermelo_kc.py`
# TODO: construct an environment to showcase Reach-Avoid RL > Lagrange RL
#  - `target/safety margin` (done)
#  - `target/constraint _set_boundary` (done)
#  - `step`, `integrate_forward` (done)
#  - `simulate_one_trajectory` (done)
#  - `visualize_analytic_comparison` (done)
#  - `plot_reach_avoid_set` (done)
#  - bounds, upward_speed, discrete_controls, box_x_y_length, visual_initial_states (done)


import gym.spaces
import numpy as np
import gym
import matplotlib.pyplot as plt
import torch
import random

class ZermeloShowEnv(gym.Env):

    def __init__(self, device, mode='RA', doneType='toEnd', thickness=.1,
        sample_inside_obs=False, easy=False):

        # State Bounds.
        self.bounds = np.array([[-3., 3.],
                                [ 0., 6.]])
        self.low  = self.bounds[:, 0]
        self.high = self.bounds[:, 1]
        self.sample_inside_obs = sample_inside_obs

        # Time-step Parameters.
        self.time_step = 0.05

        # Control Parameters.
        self.upward_speed = .5
        self.horizontal_rate = 1.
        self.discrete_controls = np.array([
            [-self.horizontal_rate, self.upward_speed],
            [0, self.upward_speed],
            [self.horizontal_rate, self.upward_speed] ])

        # Constraint Set Parameters.
        # [X-position, Y-position, width, height]
        if easy:
            self.constraint_x_y_w_h = np.array([
                [0., 1.5, 4., 0.5]])
        else:
            self.constraint_x_y_w_h = np.array([
                [0., 1.5, 4., 0.5],
                [0., 4., 4., thickness] ])

        # Target Set Parameters.
        self.target_x_y_w_h = np.array([[0., 5.5, 1., 1.]])

        # Gym variables.
        self.action_space = gym.spaces.Discrete(3)  # {left, up, right}
        self.midpoint = (self.low + self.high)/2.0
        self.interval = self.high - self.low
        self.observation_space = gym.spaces.Box(
            np.float32(self.midpoint - self.interval/2),
            np.float32(self.midpoint + self.interval/2))
        self.viewer = None

        # Set random seed.
        self.seed_val = 0
        self.set_seed(self.seed_val)

        # Cost Parameters
        self.penalty = 1.
        self.reward = -1.
        self.costType = 'sparse'
        self.scaling = 1.

        # mode: normal or extend (keep track of ell & g)
        self.mode = mode
        if mode == 'extend':
            self.state = np.zeros(3)
        else:
            self.state = np.zeros(2)
        self.doneType = doneType

        # Visualization Parameters
        self.constraint_set_boundary = self.get_constraint_set_boundary()
        self.target_set_boundary     = self.get_target_set_boundary()
        self.visual_initial_states = [
                np.array([  0.,  0.]),
                np.array([ -1.,  0.]),
                np.array([  1.,  0.]),
                np.array([ -2.5, 0.]),
                np.array([  2.5, 0.]),
                np.array([ -1.,  2.5]),
                np.array([  1.,  2.5])]
        if mode == 'extend':
            self.visual_initial_states = \
                self.extend_state(self.visual_initial_states)

        print("Env: mode-{:s}; doneType-{:s}; sample_inside_obs-{}".format(
            self.mode, self.doneType, self.sample_inside_obs))

        # for torch
        self.device = device 


    def extend_state(self, states):
        new_states = []
        for state in states:
            l_x = self.target_margin(state)
            g_x = self.safety_margin(state)
            new_states.append(np.append(state, max(l_x, g_x)))
        return new_states


    def reset(self, start=None):
        """ Reset the state of the environment.

        Args:
            start: Which state to reset the environment to. If None, pick the
                state uniformly at random.

        Returns:
            The state the environment has been reset to.
        """
        if start is None:
            self.state = self.sample_random_state(
                sample_inside_obs=self.sample_inside_obs)
        else:
            self.state = start
        return np.copy(self.state)


    def sample_random_state(self, sample_inside_obs=False):
        inside_obs = True
        # Repeat sampling until outside obstacle if needed.
        while inside_obs:
            xy_sample = np.random.uniform(low=self.low, high=self.high)
            g_x = self.safety_margin(xy_sample)
            inside_obs = (g_x > 0)
            if sample_inside_obs:
                break

        return xy_sample


#== Dynamics ==
    def step(self, action):
        """ Evolve the environment one step forward under given input action.

        Args:
            action: Input action.

        Returns:
            Tuple of (next state, signed distance of current state, whether the
            episode is done, info dictionary).
        """

        u = self.discrete_controls[action]
        state, [l_x, g_x] = self.integrate_forward(self.state, u)
        self.state = state

        fail = g_x > 0
        success = l_x <= 0

        #= `cost` signal
        # cost = 0.
        if self.mode == 'RA':
            if fail:
                cost = self.penalty
            elif success:
                cost = self.reward
            else:
                cost = 0.
        else:
            if fail:
                cost = self.penalty
            elif success:
                cost = self.reward
            else:
                if self.costType == 'dense_ell':
                    cost = l_x
                elif self.costType == 'dense_ell_g':
                    cost = l_x + g_x
                elif self.costType == 'sparse':
                    cost = 0. * self.scaling
                elif self.costType == 'max_ell_g':
                    cost = max(l_x, g_x)
                else:
                    cost = 0.

        #= `done` signal
        # done = fail
        if self.doneType == 'toEnd':
            done = self.check_within_env(self.state)
        elif self.doneType == 'fail':
            done = fail
        elif self.doneType == 'TF':
            done = fail or success
        else:
            raise ValueError("invalid doneType")

        #= `info`
        if done and self.doneType == 'fail':
            info = {"g_x": self.penalty, "l_x": l_x}
        else:
            info = {"g_x": g_x, "l_x": l_x}
        return np.copy(self.state), cost, done, info


    def integrate_forward(self, state, u):
        """ Integrate the dynamics forward by one step.

        Args:
            state:  x, y - position
                    [z]  - optional, extra state dimension capturing 
                            reach-avoid outcome so far)
            u: Contol input, consisting of v_x and v_y.

        Returns:
            State variables (x, y, [z])  integrated one step forward in time.
        """
        if self.mode == 'extend':
            x, y, z = state
        else:
            x, y = state

        # one step forward
        x = x + self.time_step * u[0]
        y = y + self.time_step * u[1]

        l_x = self.target_margin(np.array([x, y]))
        g_x = self.safety_margin(np.array([x, y]))

        if self.mode == 'extend':
            z = min(z, max(l_x, g_x) )
            state = np.array([x, y, z])
        else:
            state = np.array([x, y])

        info = np.array([l_x, g_x])
        
        return state, info


#== Getting Margin ==
    def _calculate_margin(self, s, x_y_w_h, negativeInside=True):
        x, y, w, h = x_y_w_h
        delta_x = np.abs(s[0] - x)
        delta_y = np.abs(s[1] - y)
        margin = max(delta_y - h/2, delta_x - w/2)

        if negativeInside:
            return margin
        else:
            return - margin


    def safety_margin(self, s):
        """ Computes the margin (e.g. distance) between state and failue set.

        Args:
            s: State.

        Returns:
            Margin for the state s.
        """
        g_x_list = []

        # constraint_set_safety_margin
        for _, constraint_set in enumerate(self.constraint_x_y_w_h):
            g_x = self._calculate_margin(s, constraint_set, negativeInside=False)
            g_x_list.append(g_x)

        # enclosure_safety_margin
        boundary_x_y_w_h = np.append(self.midpoint, self.interval)
        g_x = self._calculate_margin(s, boundary_x_y_w_h, negativeInside=True)
        g_x_list.append(g_x)

        safety_margin = np.max(np.array(g_x_list))

        return self.scaling * safety_margin


    def target_margin(self, s):
        """ Computes the margin (e.g. distance) between state and target set.

        Args:
            s: State.

        Returns:
            Margin for the state s.
        """
        l_x_list = []

        # target_set_safety_margin
        for idx, target_set in enumerate(self.target_x_y_w_h):
            l_x = self._calculate_margin(s, target_set, negativeInside=True)
            l_x_list.append(l_x)

        target_margin = np.max(np.array(l_x_list))

        return self.scaling * target_margin


#== Setting Hyper-Parameters ==
    def set_costParam(self, penalty=1, reward=-1, costType='sparse', scaling=1):
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
        torch.manual_seed(self.seed_val)
        torch.cuda.manual_seed(self.seed_val)
        torch.cuda.manual_seed_all(self.seed_val)  # if you are using multi-GPU.
        random.seed(self.seed_val)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


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
        self.observation_space = gym.spaces.Box(
            np.float32(midpoint - interval/2),
            np.float32(midpoint + interval/2))


    def set_doneType(self, doneType):
        self.doneType = doneType


    def set_sample_type(self, sample_inside_obs=False, verbose=False):
        self.sample_inside_obs = sample_inside_obs
        if verbose:
            print("sample_inside_obs-{}".format(self.sample_inside_obs))


#== Getting Information ==
    def check_within_env(self, state):
        outsideTop   = (state[1] >= self.bounds[1,1])
        outsideLeft  = (state[0] <= self.bounds[0,0])
        outsideRight = (state[0] >= self.bounds[0,1])
        return outsideTop or outsideLeft or outsideRight


    def get_constraint_set_boundary(self):
        """ Computes the safe set boundary based on the analytic solution.

        The boundary of the safe set for the double integrator is determined by
        two parabolas and two line segments.

        Returns:
            Set of discrete points describing each parabola. The first and last
            two elements of the list describe the set of coordinates for the
            first and second parabola respectively.
        """
        num_constarint_set = self.constraint_x_y_w_h.shape[0]
        constraint_set_boundary = np.zeros((num_constarint_set, 5, 2))

        for idx, constraint_set in enumerate(self.constraint_x_y_w_h):
            x, y, w, h = constraint_set
            x_l  = x - w/2.0
            x_h = x + w/2.0
            y_l  = y - h/2.0
            y_h = y + h/2.0
            constraint_set_boundary[idx, :, 0] = [x_l, x_l, x_h, x_h, x_l]
            constraint_set_boundary[idx, :, 1] = [y_l, y_h, y_h, y_l, y_l]

        return constraint_set_boundary


    def get_target_set_boundary(self):
        """ Computes the safe set boundary based on the analytic solution.

        The boundary of the safe set for the double integrator is determined by
        two parabolas and two line segments.

        Returns:
            Set of discrete points describing each parabola. The first and last
            two elements of the list describe the set of coordinates for the
            first and second parabola respectively.
        """
        num_target_set = self.target_x_y_w_h.shape[0]
        target_set_boundary = np.zeros((num_target_set, 5, 2))

        for idx, target_set in enumerate(self.target_x_y_w_h):
            x, y, w, h = target_set
            x_l  = x - w/2.0
            x_h = x + w/2.0
            y_l  = y - h/2.0
            y_h = y + h/2.0
            target_set_boundary[idx, :, 0] = [x_l, x_l, x_h, x_h, x_l]
            target_set_boundary[idx, :, 1] = [y_l, y_h, y_h, y_l, y_l]

        return target_set_boundary


    def get_value(self, q_func, nx=41, ny=121, addBias=False):
        v = np.zeros((nx, ny))
        it = np.nditer(v, flags=['multi_index'])
        xs = np.linspace(self.bounds[0,0], self.bounds[0,1], nx)
        ys = np.linspace(self.bounds[1,0], self.bounds[1,1], ny)
        while not it.finished:
            idx = it.multi_index

            x = xs[idx[0]]
            y = ys[idx[1]]
            l_x = self.target_margin(np.array([x, y]))
            g_x = self.safety_margin(np.array([x, y]))

            if self.mode == 'normal' or self.mode == 'RA':
                state = torch.FloatTensor([x, y]).to(self.device).unsqueeze(0)
            else:
                z = max([l_x, g_x])
                state = torch.FloatTensor([x, y, z]).to(self.device).unsqueeze(0)

            if addBias:
                v[idx] = q_func(state).min(dim=1)[0].item() + max(l_x, g_x)
            else:
                v[idx] = q_func(state).min(dim=1)[0].item()
            it.iternext()
        return xs, ys, v


    def get_axes(self):
        """ Gets the bounds for the environment.

        Returns:
            List containing a list of bounds for each state coordinate and a
        """
        x_span = self.bounds[0,1] - self.bounds[0,0]
        y_span = self.bounds[1,1] - self.bounds[1,0]
        aspect_ratio = x_span / y_span
        axes = np.array([
            self.bounds[0,0]-.05,
            self.bounds[0,1]+.05,
            self.bounds[1,0]-.05,
            self.bounds[1,1]+.05])
        return [axes, aspect_ratio]


    def get_warmup_examples(self, num_warmup_samples=100):
        x_min, x_max = self.bounds[0,:]
        y_min, y_max = self.bounds[1,:]

        xs = np.random.uniform(x_min, x_max, num_warmup_samples)
        ys = np.random.uniform(y_min, y_max, num_warmup_samples)
        heuristic_v = np.zeros((num_warmup_samples, self.action_space.n))
        states = np.zeros((num_warmup_samples, self.observation_space.shape[0]))

        for i in range(num_warmup_samples):
            x, y = xs[i], ys[i]
            l_x = self.target_margin(np.array([x, y]))
            g_x = self.safety_margin(np.array([x, y]))
            heuristic_v[i,:] = np.maximum(l_x, g_x)
            states[i, :] = x, y

        return states, heuristic_v


    def simulate_one_trajectory(self, q_func, T=250, state=None,
        keepOutOf=False, toEnd=False):

        if state is None:
            state = self.sample_random_state(sample_inside_obs=keepOutOf)
        x, y = state[:2]
        traj_x = [x]
        traj_y = [y]
        result = 0 # not finished

        for _ in range(T):
            if toEnd:
                outsideTop   = (state[1] > self.bounds[1,1])
                outsideLeft  = (state[0] < self.bounds[0,0])
                outsideRight = (state[0] > self.bounds[0,1])
                done = outsideTop or outsideLeft or outsideRight
                if done:
                    result = 1
                    break
            else:
                if self.safety_margin(state[:2]) > 0:
                    result = -1 # failed
                    break
                elif self.target_margin(state[:2]) <= 0:
                    result = 1 # succeeded
                    break

            state_tensor = torch.FloatTensor(state).to(self.device).unsqueeze(0)
            action_index = q_func(state_tensor).min(dim=1)[1].item()
            u = self.discrete_controls[action_index]

            state, _ = self.integrate_forward(state, u)
            traj_x.append(state[0])
            traj_y.append(state[1])

        return traj_x, traj_y, result


    def simulate_trajectories(self, q_func, T=250,
        num_rnd_traj=None, states=None, keepOutOf=False, toEnd=False):

        assert ((num_rnd_traj is None and states is not None) or
                (num_rnd_traj is not None and states is None) or
                (len(states) == num_rnd_traj))
        trajectories = []

        if states is None:
            results = np.empty(shape=(num_rnd_traj,), dtype=int)
            for idx in range(num_rnd_traj):
                traj_x, traj_y, result = self.simulate_one_trajectory(
                    q_func, T=T, keepOutOf=keepOutOf, toEnd=toEnd)
                trajectories.append((traj_x, traj_y))
                results[idx] = result
        else:
            results = np.empty(shape=(len(states),), dtype=int)
            for idx, state in enumerate(states):
                traj_x, traj_y, result = self.simulate_one_trajectory(
                    q_func, T=T, state=state, toEnd=toEnd)
                trajectories.append((traj_x, traj_y))
                results[idx] = result

        return trajectories, results


#== Visualizing ==
    def visualize(  self, q_func,
                    vmin=-1, vmax=1, nx=201, ny=201,
                    labels=None, boolPlot=False, addBias=False,
                    cmap='seismic',
                    plotFigure=False, storeFigure=True):

        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        cbarPlot = True

        #== Plot failure / target set ==
        self.plot_target_failure_set(ax)

        #== Plot reach-avoid set ==
        self.plot_reach_avoid_set(ax)

        #== Plot V ==
        self.plot_v_values(q_func, ax=ax, fig=fig,
            vmin=vmin, vmax=vmax, nx=nx, ny=ny, cmap=cmap,
            boolPlot=boolPlot, cbarPlot=cbarPlot, addBias=addBias)
        
        #== Plot Trajectories ==
        self.plot_trajectories(q_func, states=self.visual_initial_states,
            toEnd=False, ax=ax)

        #== Formatting ==
        self.plot_formatting(ax=ax, labels=labels)


    def plot_v_values(self, q_func, ax=None, fig=None,
                        vmin=-1, vmax=1, nx=201, ny=201, cmap='seismic',
                        boolPlot=False, cbarPlot=True, addBias=False):
        axStyle = self.get_axes()

        #== Plot V ==
        _, _, v = self.get_value(q_func, nx, ny, addBias=addBias)

        if boolPlot:
            im = ax.imshow(v.T>0., interpolation='none', extent=axStyle[0],
                origin="lower", cmap=cmap)
        else:
            im = ax.imshow(v.T, interpolation='none', extent=axStyle[0],
                origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
            if cbarPlot:
                cbar = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.05,
                    shrink=.95, ticks=[vmin, 0, vmax])
                cbar.ax.set_yticklabels(labels=[vmin, 0, vmax], fontsize=24)


    def plot_target_failure_set(self, ax=None, c_c='m', c_t='y', lw=1.5,
        zorder=1):
        # Plot bounadries of constraint set.
        for one_boundary in self.constraint_set_boundary:
            ax.plot(one_boundary[:, 0], one_boundary[:, 1], color=c_c, lw=lw,
                zorder=zorder)

        # Plot boundaries of target set.
        for one_boundary in self.target_set_boundary:
            ax.plot(one_boundary[:, 0], one_boundary[:, 1], color=c_t, lw=lw,
                zorder=zorder)


    def plot_trajectories(self, q_func, T=250, num_rnd_traj=None, states=None, 
        keepOutOf=False, toEnd=False, ax=None, c='k', lw=2, zorder=2):

        assert ((num_rnd_traj is None and states is not None) or
                (num_rnd_traj is not None and states is None) or
                (len(states) == num_rnd_traj))

        trajectories, results = self.simulate_trajectories(q_func, T=T, 
            num_rnd_traj=num_rnd_traj, states=states, keepOutOf=keepOutOf,
            toEnd=toEnd)

        for traj in trajectories:
            traj_x, traj_y = traj
            ax.scatter(traj_x[0], traj_y[0], s=48, c=c, zorder=zorder)
            ax.plot(traj_x, traj_y, color=c, linewidth=lw, zorder=zorder)

        return results


    def plot_reach_avoid_set(self, ax, c='g', lw=3, zorder=1):
        slope = self.upward_speed / self.horizontal_rate

        def get_line(slope, end_point, x_limit, ns=100):
            x_end, y_end = end_point
            b = y_end - slope * x_end

            xs = np.linspace(x_limit, x_end, ns)
            ys = xs * slope + b
            return xs, ys

        # unsafe set
        for _, constraint_set in enumerate(self.constraint_x_y_w_h):
            x, y, w, h = constraint_set
            x1 = x - w/2.0
            x2 = x + w/2.0
            y_min = y - h/2.0
            xs, ys = get_line(-slope, end_point=[x1, y_min], x_limit=x)
            ax.plot(xs, ys, color=c, linewidth=lw, zorder=zorder)
            xs, ys = get_line( slope, end_point=[x2, y_min], x_limit=x)
            ax.plot(xs, ys, color=c, linewidth=lw, zorder=zorder)

        # border unsafe set
        x, y, w, h = self.target_x_y_w_h[0]
        x1 = x - w/2.0
        x2 = x + w/2.0
        y_max = y + h/2.0
        xs, ys = get_line( slope, end_point=[x1, y_max], x_limit=self.low[0])
        ax.plot(xs, ys, color=c, linewidth=lw, zorder=zorder)
        xs, ys = get_line(-slope, end_point=[x2, y_max], x_limit=self.high[0])
        ax.plot(xs, ys, color=c, linewidth=lw, zorder=zorder)


    def plot_formatting(self, ax=None, labels=None):
        axStyle = self.get_axes()
        #== Formatting ==
        ax.axis(axStyle[0])
        ax.set_aspect(axStyle[1])  # makes equal aspect ratio
        ax.grid(False)
        if labels is not None:
            ax.set_xlabel(labels[0], fontsize=52)
            ax.set_ylabel(labels[1], fontsize=52)

        ax.tick_params( axis='both', which='both',
                        bottom=False, top=False,
                        left=False, right=False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])