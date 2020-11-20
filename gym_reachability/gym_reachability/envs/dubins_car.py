# Copyright (c) 2020–2021, The Regents of the University of California.
# All rights reserved.
#
# This file is subject to the terms and conditions defined in the LICENSE file
# included in this repository.
#
# Please contact the author(s) of this library if you have any questions.
# Authors: Vicenc Rubies-Royo   ( vrubies@berkeley.edu )
#          Kai-Chieh Hsu        ( kaichieh@princeton.edu )

import gym.spaces
import numpy as np
import gym
import matplotlib
import matplotlib.pyplot as plt
import torch


class DubinsCarEnv(gym.Env):

    def __init__(self, device, mode='normal', doneType='toEnd'):

        # State bounds.
        self.bounds = np.array([[-1.1, 1.1],  # axis_0 = state, axis_1 = bounds.
                                [-1.1, 1.1],
                                [0, 2*np.pi]])
        self.low = self.bounds[:, 0]
        self.high = self.bounds[:, 1]

        # Time step parameter.
        self.time_step = 0.05

        # Dubins car parameters.
        self.speed = 0.5 # v

        # Control parameters.
        self.R_turn = .6
        self.max_turning_rate = self.speed / self.R_turn # w
        self.discrete_controls = np.array([ -self.max_turning_rate,
                                            0.,
                                            self.max_turning_rate])

        # Constraint set parameters.
        self.constraint_center = np.array([0, 0])
        self.inner_radius = .3
        self.outer_radius = 1.0

        # Target set parameters.
        # self.target_center_x = (self.inner_radius + self.outer_radius) / 2.0
        # self.target_center_y = 0
        # self.target_center = np.array([self.target_center_x,
        #                                self.target_center_y])
        self.target_center = np.array([0, 0])

        # Gym variables.
        self.action_space = gym.spaces.Discrete(self.discrete_controls.shape[0])
        midpoint = (self.low + self.high)/2.0
        interval = self.high - self.low
        self.observation_space = gym.spaces.Box(np.float32(midpoint - interval/2),
                                                np.float32(midpoint + interval/2))
        self.viewer = None

        # Internal state.
        self.mode = mode
        self.state = np.zeros(3)
        self.doneType = doneType

        # Set random seed.
        self.seed_val = 0
        np.random.seed(self.seed_val)

        # Visualization params
        self.visual_initial_states = [  np.array([ .6*self.outer_radius,  -.5, np.pi/2]),
                                        np.array([ -.4*self.outer_radius, -.5, np.pi/2]),
                                        np.array([ -0.95*self.outer_radius, 0., np.pi/2]),
                                        np.array([ self.R_turn, 0.95*(self.outer_radius-self.R_turn), np.pi/2]),
                                     ]
        # Cost Params
        self.targetScaling = 1.
        self.safetyScaling = 1.
        self.penalty = 1.
        self.reward = -1.
        self.costType = 'sparse'
        self.device = device

        print("Env: mode---{:s}; doneType---{:s}".format(mode, doneType))


    def reset(self, start=None):
        """ Reset the state of the environment.

        Args:
            start: Which state to reset the environment to. If None, pick the
                state uniformly at random.

        Returns:
            The state the environment has been reset to.
        """
        if start is None:
            x_rnd, y_rnd, theta_rnd = self.sample_random_state()
            self.state = np.array([x_rnd, y_rnd, theta_rnd])
        else:
            self.state = start
        return np.copy(self.state)


    def sample_random_state(self, keepOutOf=False, theta=None):

        if keepOutOf:
            angle = 2.0 * np.random.uniform() * np.pi     # the position angle
            dist = np.random.uniform(low=self.outer_radius, high=self.inner_radius)
            x_rnd = dist * np.cos(angle)
            y_rnd = dist * np.sin(angle)
            if theta is None:
                theta_rnd = 2.0 * np.random.uniform() * np.pi
            else:
                theta_rnd = theta
        else:
            x_rnd, y_rnd, theta_rnd = np.random.uniform(low=self.low,
                                                        high=self.high)
        
        return x_rnd, y_rnd, theta_rnd


    def step(self, action):
        """ Evolve the environment one step forward under given input action.

        Args:
            action: Input action.

        Returns:
            Tuple of (next state, signed distance of current state, whether the
            episode is done, info dictionary).
        """
        # The signed distance must be computed before the environment steps forward.
        x, y, theta = self.state

        l_x_cur = self.target_margin(self.state[:2])
        g_x_cur = self.safety_margin(self.state[:2])

        u = self.discrete_controls[action]
        state, [l_x_nxt, g_x_nxt] = self.integrate_forward(self.state, u)
        self.state = state

        # cost
        if self.mode == 'RA':
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
                elif self.costType == 'sparse':
                    cost = 0. * self.scaling
                else:
                    cost = 0.
        # done
        if self.doneType == 'toEnd':
            outsideTop    = (self.state[1] > self.bounds[1,1])
            outsideBottom = (self.state[1] < self.bounds[1,0])
            outsideLeft   = (self.state[0] < self.bounds[0,0])
            outsideRight  = (self.state[0] > self.bounds[0,1])
            done = outsideTop or outsideLeft or outsideRight or outsideBottom
        else:
            done = fail or success
            assert self.doneType == 'TF', 'invalid doneType'

        info = {"g_x": g_x_cur, "l_x": l_x_cur, "g_x_nxt": g_x_nxt, "l_x_nxt": l_x_nxt}    
        return np.copy(self.state), cost, done, info


    def integrate_forward(self, state, u):
        """ Integrate the dynamics forward by one step.

        Args:
            x: Position in x-axis.
            y: Position in y-axis
            theta: Heading.
            u: Contol input.

        Returns:
            State variables (x,y,theta) integrated one step forward in time.
        """
        x, y, theta = state

        x = x + self.time_step * self.speed * np.cos(theta)
        y = y + self.time_step * self.speed * np.sin(theta)
        theta = np.mod(theta + self.time_step * u, 2*np.pi)
        
        l_x = self.target_margin(np.array([x, y]))
        g_x = self.safety_margin(np.array([x, y]))

        state = np.array([x, y, theta])
        info = np.array([l_x, g_x])
        
        return state, info


    def set_costParam(self, penalty=1, reward=-1, costType='normal', targetScaling=1., safetyScaling=1.):
        self.penalty = penalty
        self.reward = reward
        self.costType = costType
        self.safetyScaling = safetyScaling
        self.targetScaling = targetScaling


    def set_radius(self, inner_radius=.3, outer_radius=1., R_turn=.6):
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.R_turn = R_turn
        self.max_turning_rate = self.speed / self.R_turn # w
        self.discrete_controls = np.array([ -self.max_turning_rate,
                                            0.,
                                            self.max_turning_rate])
        print(self.discrete_controls)


    def set_seed(self, seed):
        """ Set the random seed.

        Args:
            seed: Random seed.
        """
        self.seed_val = seed
        np.random.seed(self.seed_val)


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
        self.observation_space = gym.spaces.Box(np.float32(midpoint - interval/2),
                                                np.float32(midpoint + interval/2))


    def safety_margin(self, s):
        """ Computes the margin (e.g. distance) between state and failue set.

        Args:
            s: State.

        Returns:
            Margin for the state s.
        """
        dist_to_constraint_center = np.linalg.norm(s[:2] - self.constraint_center)
        # dist. to the boundary = | dist_to_constraint_center - radius |
        outer_dist = dist_to_constraint_center - self.outer_radius # This ensures x \in K \iff g(x) <= 0.
        # safety_margin = maxmin?(outer_dist, inner_dist)
        safety_margin = outer_dist
        # if safety_margin > 0:
        #     safety_margin = 10
        # if x_in:
        #     return -1 * x_dist
        return self.safetyScaling * safety_margin


    def target_margin(self, s):
        """ Computes the margin (e.g. distance) between state and target set.

        Args:
            s: State.

        Returns:
            Margin for the state s.
        """
        dist_to_target = np.linalg.norm(s[:2] - self.target_center)
        target_margin = dist_to_target - self.inner_radius
        # if x_in:
        #     return -1 * x_dist
        return self.targetScaling * target_margin


    def render(self, mode='human'):
        pass


    def get_warmup_examples(self, num_warmup_samples=100):

        rv = np.random.uniform( low=self.low,
                                high=self.high,
                                size=(num_warmup_samples,3))
        x_rnd, y_rnd, theta_rnd = rv[:,0], rv[:,1], rv[:,2]

        heuristic_v = np.zeros((num_warmup_samples, self.action_space.n))
        states = np.zeros((num_warmup_samples, self.observation_space.shape[0]))

        for i in range(num_warmup_samples):
            x, y, theta = x_rnd[i], y_rnd[i], theta_rnd[i]
            l_x = self.target_margin(np.array([x, y]))
            g_x = self.safety_margin(np.array([x, y]))
            heuristic_v[i,:] = np.maximum(l_x, g_x)
            states[i, :] = x, y, theta

        return states, heuristic_v


    def get_axes(self):
        """ Gets the bounds for the environment.

        Returns:
            List containing a list of bounds for each state coordinate and a
        """
        aspect_ratio = (self.bounds[0,1]-self.bounds[0,0])/(self.bounds[1,1]-self.bounds[1,0])
        axes = np.array([self.bounds[0,0], self.bounds[0,1], self.bounds[1,0], self.bounds[1,1]])
        return [axes, aspect_ratio]


    def get_value(self, q_func, theta, nx=101, ny=101, addBias=False):
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
                state = torch.FloatTensor([x, y, theta], device=self.device).unsqueeze(0)
            else:
                z = max([l_x, g_x])
                state = torch.FloatTensor([x, y, theta, z], device=self.device).unsqueeze(0)
            if addBias:
                v[idx] = q_func(state).min(dim=1)[0].item() + max(l_x, g_x)
            else:
                v[idx] = q_func(state).min(dim=1)[0].item()
            it.iternext()
        return v


    def simulate_one_trajectory(self, q_func, T=10, state=None, theta=None,
                                keepOutOf=False, toEnd=False):

        if state is None:
            state = self.sample_random_state(keepOutOf=keepOutOf, theta=theta)
        x, y = state[:2]
        traj_x = [x]
        traj_y = [y]
        result = 0 # not finished

        for t in range(T):
            if toEnd:
                outsideTop    = (self.state[1] > self.bounds[1,1])
                outsideBottom = (self.state[1] < self.bounds[1,0])
                outsideLeft   = (self.state[0] < self.bounds[0,0])
                outsideRight  = (self.state[0] > self.bounds[0,1])
                done = outsideTop or outsideLeft or outsideRight or outsideBottom
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

            state_tensor = torch.FloatTensor(state, device=self.device).unsqueeze(0)
            action_index = q_func(state_tensor).min(dim=1)[1].item()
            u = self.discrete_controls[action_index]

            state, _ = self.integrate_forward(state, u)
            traj_x.append(state[0])
            traj_y.append(state[1])

        return traj_x, traj_y, result


    def simulate_trajectories(self, q_func, T=10,
                              num_rnd_traj=None, states=None, theta=None,
                              keepOutOf=False, toEnd=False):

        assert ((num_rnd_traj is None and states is not None) or
                (num_rnd_traj is not None and states is None) or
                (len(states) == num_rnd_traj))
        trajectories = []

        if states is None:
            results = np.empty(shape=(num_rnd_traj,), dtype=int)
            for idx in range(num_rnd_traj):
                traj_x, traj_y, result = self.simulate_one_trajectory(q_func, T=T, theta=theta, 
                                                                      keepOutOf=keepOutOf, toEnd=toEnd)
                trajectories.append((traj_x, traj_y))
                results[idx] = result
        else:
            results = np.empty(shape=(len(states),), dtype=int)
            for idx, state in enumerate(states):
                traj_x, traj_y, result = self.simulate_one_trajectory(q_func, T=T, state=state, toEnd=toEnd)
                trajectories.append((traj_x, traj_y))
                results[idx] = result

        return trajectories, results


    def visualize_analytic_comparison(self, q_func, no_show=False,
                                      vmin=-1, vmax=1, nx=101, ny=101, cmap='coolwarm',
                                      labels=None, boolPlot=False, theta=np.pi/2):
        """ Overlays analytic safe set on top of state value function.

        Args:
            q_func: NN or Tabular-Q
        """
        fig = plt.figure(figsize=(12,4))
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)
        axStyle = self.get_axes()
        axList = [ax1, ax2, ax3]
        thetaList = [np.pi/6, np.pi/3, np.pi/2]

        for i, (ax, theta) in enumerate(zip(axList, thetaList)):
            if i == len(thetaList)-1:
                cbarPlot=True
            else: 
                cbarPlot=False

            #== Plot failure / target set ==
            self.plot_target_failure_set(ax)

            #== Plot reach-avoid set ==
            self.plot_reach_avoid_set(ax, orientation=theta)

            #== Plot V ==
            self.plot_v_values(q_func, ax=ax, fig=fig, theta=theta,
                                vmin=vmin, vmax=vmax, nx=nx, ny=ny, cmap=cmap,
                                boolPlot=boolPlot, cbarPlot=cbarPlot)
            #== Formatting ==
            self.plot_formatting(ax=ax, labels=labels)

            #== Plot Trajectories ==
            self.plot_trajectories(q_func, T=200, states=self.visual_initial_states, toEnd=False, 
                                   ax=ax, c='y', lw=2, orientation=theta-np.pi/2)

            ax.set_xlabel(r'$\theta={:.0f}^\circ$'.format(theta*180/np.pi), fontsize=28)

        plt.tight_layout()
        if not no_show:
            plt.show()


    def plot_formatting(self, ax=None, labels=None):
        axStyle = self.get_axes()
        #== Formatting ==
        ax.axis(axStyle[0])
        ax.set_aspect(axStyle[1])  # makes equal aspect ratio
        ax.grid(False)
        if labels is not None:
            ax.set_xlabel(labels[0], fontsize=52)
            ax.set_ylabel(labels[1], fontsize=52)

        ax.tick_params( axis='both', which='both',  # both x and y axes, both major and minor ticks are affected
                        bottom=False, top=False,    # ticks along the top and bottom edges are off
                        left=False, right=False)    # ticks along the left and right edges are off
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        #ax.set_title(r"$\theta$={:.1f}".format(theta * 180 / np.pi), fontsize=24)


    def plot_v_values(self, q_func, theta=np.pi/2, ax=None, fig=None,
                        vmin=-1, vmax=1, nx=201, ny=201, cmap='seismic',
                        boolPlot=False, cbarPlot=True):
        axStyle = self.get_axes()
        ax.plot([0., 0.], [axStyle[0][2], axStyle[0][3]], c='k')
        ax.plot([axStyle[0][0], axStyle[0][1]], [0., 0.], c='k')

        #== Plot V ==
        if theta == None:
            theta = 2.0 * np.random.uniform() * np.pi
        v = self.get_value(q_func, theta, nx, ny)

        if boolPlot:
            im = ax.imshow(v.T>0., interpolation='none', extent=axStyle[0], origin="lower",
                       cmap=cmap)
        else:
            im = ax.imshow(v.T, interpolation='none', extent=axStyle[0], origin="lower",
                       cmap=cmap, vmin=vmin, vmax=vmax)
            if cbarPlot:
                cbar = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.05, shrink=.95, ticks=[vmin, 0, vmax])
                cbar.ax.set_yticklabels(labels=[vmin, 0, vmax], fontsize=24)


    def plot_trajectories(self, q_func, T=10, num_rnd_traj=None, states=None, theta=None,
                          keepOutOf=False, toEnd=False, ax=None, c='y', lw=1.5, orientation=0):

        assert ((num_rnd_traj is None and states is not None) or
                (num_rnd_traj is not None and states is None) or
                (len(states) == num_rnd_traj))

        if states != None:
            tmpStates = []
            for state in states:
                x, y, theta = state
                xtilde = x*np.cos(orientation) - y*np.sin(orientation)
                ytilde = y*np.cos(orientation) + x*np.sin(orientation)
                thetatilde = theta+orientation
                tmpStates.append(np.array([xtilde, ytilde, thetatilde]))
            states = tmpStates
                
        trajectories, results = self.simulate_trajectories(q_func, T=T, num_rnd_traj=num_rnd_traj, 
                                                           states=states, theta=theta, 
                                                           keepOutOf=keepOutOf, toEnd=toEnd)
        if ax == None:
            ax = plt.gca()
        for traj in trajectories:
            traj_x, traj_y = traj
            ax.scatter(traj_x[0], traj_y[0], s=48, c=c)
            ax.plot(traj_x, traj_y, color=c,  linewidth=lw)

        return results


    def plot_reach_avoid_set(self, ax, c='g', lw=3, orientation=0):
        r = self.inner_radius
        R = self.outer_radius
        R_turn = self.R_turn
        if r >=  2*R_turn - R:
            # plot arc
            tmpY = (r**2 - R**2 + 2*R_turn*R) / (2*R_turn)
            tmpX = np.sqrt(r**2 - tmpY**2)
            tmpTheta = np.arcsin(tmpX / (R-R_turn))
            # two sides
            self.plot_arc((0.,  R_turn), R-R_turn, (tmpTheta-np.pi/2, np.pi/2),  ax, c=c, lw=lw, orientation=orientation)
            self.plot_arc((0., -R_turn), R-R_turn, (-np.pi/2, np.pi/2-tmpTheta), ax, c=c, lw=lw, orientation=orientation)
            # middle
            tmpPhi = np.arcsin(tmpX/r)
            self.plot_arc((0., 0), r, (tmpPhi - np.pi/2, np.pi/2-tmpPhi), ax, c=c, lw=lw, orientation=orientation)
            # outer boundary
            self.plot_arc((0., 0), R, (np.pi/2, 3*np.pi/2), ax, c=c, lw=lw, orientation=orientation)
        else:
            # two sides
            tmpY = (R**2 + 2*R_turn*r - r**2) / (2*R_turn)
            tmpX = np.sqrt(R**2 - tmpY**2)
            tmpTheta = np.arcsin( tmpX / (R_turn-r))
            self.plot_arc((0.,  R_turn), R_turn-r, (np.pi/2+tmpTheta, 3*np.pi/2), ax, c=c, lw=lw, orientation=orientation)
            self.plot_arc((0., -R_turn), R_turn-r, (np.pi/2, 3*np.pi/2-tmpTheta), ax, c=c, lw=lw, orientation=orientation)
            # middle
            self.plot_arc((0., 0), r, (np.pi/2, -np.pi/2), ax, c=c, lw=lw, orientation=orientation)
            # outer boundary
            self.plot_arc((0., 0), R, (np.pi/2, 3*np.pi/2), ax, c=c, lw=lw, orientation=orientation)


    def plot_target_failure_set(self, ax):
        self.plot_circle(self.constraint_center, self.outer_radius, ax, c='k', lw=3)
        self.plot_circle(self.target_center,     self.inner_radius, ax, c='m', lw=3)

    
    def plot_arc(self, p, r, thetaParam, ax, c='b', lw=1.5, orientation=0):
        x, y = p
        thetaInit, thetaFinal = thetaParam
        
        xtilde = x*np.cos(orientation) - y*np.sin(orientation)
        ytilde = y*np.cos(orientation) + x*np.sin(orientation)
        
        theta = np.linspace(thetaInit+orientation, thetaFinal+orientation, 100)
        xs = xtilde + r * np.cos(theta)
        ys = ytilde + r * np.sin(theta)
        
        ax.plot(xs, ys, c=c, lw=lw)

    
    def plot_circle(self, center, r, ax, c='b', lw=1.5, orientation=0, scatter=False):
        x, y = center
        xtilde = x*np.cos(orientation) - y*np.sin(orientation)
        ytilde = y*np.cos(orientation) + x*np.sin(orientation)
        
        theta = np.linspace(0, 2*np.pi, 200)
        xs = xtilde + r * np.cos(theta)
        ys = ytilde + r * np.sin(theta)
        ax.plot(xs, ys, c=c, lw=lw)
        if scatter:
            ax.scatter(xtilde+r, ytilde, c=c, s=80)
            ax.scatter(xtilde-r, ytilde, c=c, s=80)
            print(xtilde+r, ytilde, xtilde-r, ytilde)
 
