# Copyright (c) 2020–2021, The Regents of the University of California.
# All rights reserved.
#
# This file is subject to the terms and conditions defined in the LICENSE file
# included in this repository.
#
# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu        ( kaichieh@princeton.edu )

import gym.spaces
import numpy as np
import gym
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator
import torch
import random

from .dubins_car_dyn_cont import DubinsCarDynCont
from .env_utils import plot_arc, plot_circle



class DubinsCarOneContEnv(gym.Env):
    def __init__(self, device, mode='normal', doneType='toEnd',
        sample_inside_obs=False, sample_inside_tar=True, seed=0):
        # State bounds.
        self.bounds = np.array([[-1.1, 1.1],
                                [-1.1, 1.1],
                                [0, 2*np.pi]])
        self.low = self.bounds[:, 0]
        self.high = self.bounds[:, 1]
        self.sample_inside_obs = sample_inside_obs
        self.sample_inside_tar = sample_inside_tar

        # Gym variables.
        midpoint = (self.low + self.high)/2.0
        interval = self.high - self.low
        self.observation_space = gym.spaces.Box(np.float32(midpoint - interval/2),
                                                np.float32(midpoint + interval/2))

        # Constraint set parameters.
        self.constraint_center = np.array([0, 0])
        self.constraint_radius = 1.0

        # Target set parameters.
        self.target_center = np.array([0, 0])
        self.target_radius = .3

        # Internal state.
        self.mode = mode
        self.state = np.zeros(3)
        self.doneType = doneType

        # Dubins car parameters.
        self.time_step = 0.05
        self.speed = 0.5 # v
        self.R_turn = .6
        self.car = DubinsCarDynCont(doneType=doneType)
        self.init_car()

        # Set random seed.
        self.set_seed(seed)

        # Visualization params 
        self.visual_initial_states =[   np.array([ .6*self.constraint_radius,  -.5, np.pi/2]),
                                        np.array([ -.4*self.constraint_radius, -.5, np.pi/2]),
                                        np.array([ -0.95*self.constraint_radius, 0., np.pi/2]),
                                        np.array([ self.R_turn, 0.95*(self.constraint_radius-self.R_turn), np.pi/2]),
                                    ]
        # Cost Params
        self.targetScaling = 1.
        self.safetyScaling = 1.
        self.penalty = 1.
        self.reward = -1.
        self.costType = 'sparse'
        self.device = device

        print("Env: mode-{:s}; doneType-{:s}".format(self.mode, self.doneType))
        print("Sample Type: inside obs-{}; inside tar-{}".format(
            self.sample_inside_obs, self.sample_inside_tar))


    def init_car(self):
        self.car.set_bounds(bounds=self.bounds)
        self.car.set_constraint(center=self.constraint_center, radius=self.constraint_radius)
        self.car.set_target(center=self.target_center, radius=self.target_radius)
        self.car.set_speed(speed=self.speed)
        self.car.set_time_step(time_step=self.time_step)
        self.car.set_radius_rotation(R_turn=self.R_turn, verbose=False)
        self.action_space = self.car.action_space


#== Reset Functions ==
    def reset(self, start=None):
        """ Reset the state of the environment.

        Args:
            start: Which state to reset the environment to. If None, pick the
                state uniformly at random.

        Returns:
            The state the environment has been reset to.
        """
        self.state = self.car.reset(start=start, sample_inside_obs=self.sample_inside_obs,
            sample_inside_tar=self.sample_inside_tar)
        return np.copy(self.state)


    def sample_random_state(self, sample_inside_obs=False, sample_inside_tar=True, theta=None):
        state = self.car.sample_random_state(sample_inside_obs=sample_inside_obs,
            sample_inside_tar=sample_inside_tar, theta=theta)
        return state


#== Dynamics Functions ==
    def step(self, action):
        """ Evolve the environment one step forward under given input action.

        Args:
            action: Input action.

        Returns:
            Tuple of (next state, signed distance of current state, whether the
            episode is done, info dictionary).
        """
        distance = np.linalg.norm(self.state-self.car.state)
        if distance >= 1e-8:
            raise "There is a mismatch between the env state and car state: {:.2e}".format(distance)

        if not np.isscalar(action):
            action = action[0]

        state_nxt = self.car.step(action)
        self.state = state_nxt
        l_x = self.target_margin(self.state[:2])
        g_x = self.safety_margin(self.state[:2])

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
            done = not self.car.check_within_bounds(self.state)
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


#== Setting Hyper-Parameter Functions ==
    def set_costParam(self, penalty=1, reward=-1, costType='normal', targetScaling=1., safetyScaling=1.):
        self.penalty = penalty
        self.reward = reward
        self.costType = costType
        self.safetyScaling = safetyScaling
        self.targetScaling = targetScaling


    def set_radius(self, target_radius=.3, constraint_radius=1., R_turn=.6):
        self.target_radius = target_radius
        self.constraint_radius = constraint_radius
        self.R_turn = R_turn
        self.car.set_radius(target_radius=target_radius,
                            constraint_radius=constraint_radius,
                            R_turn=R_turn)
        self.action_space = self.car.action_space


    def set_constraint(self, center=np.array([0.,0.]), radius=1.):
        self.constraint_center = center
        self.constraint_radius = radius
        self.car.set_constraint(center=center, radius=radius)


    def set_target(self, center=np.array([0.,0.]), radius=.4):
        self.target_center = center
        self.target_radius = radius
        self.car.set_target(center=center, radius=radius)


    def set_radius_rotation(self, R_turn=.6, verbose=False):
        self.R_turn = R_turn
        self.car.set_radius_rotation(R_turn=R_turn, verbose=verbose)
        self.action_space = self.car.action_space


    def set_speed(self, speed=.5):
        self.speed = speed
        self.car.set_speed(speed=speed)
        self.action_space = self.car.action_space


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
        self.car.set_seed(seed)
        self.action_space = self.car.action_space


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
        self.car.set_bounds(bounds)


    def set_sample_type(self, sample_inside_obs=True, sample_inside_tar=True,
        verbose=False):
        self.sample_inside_obs = sample_inside_obs
        self.sample_inside_tar = sample_inside_tar
        if verbose:
            print("Sample Type: inside obs-{}; inside tar-{}".format(
                self.sample_inside_obs, self.sample_inside_tar))


#== Margin Functions ==
    def safety_margin(self, s):
        """ Computes the margin (e.g. distance) between state and failue set.

        Args:
            s: State.

        Returns:
            Margin for the state s.
        """
        return self.car.safety_margin(s[:2])


    def target_margin(self, s):
        """ Computes the margin (e.g. distance) between state and target set.

        Args:
            s: State.

        Returns:
            Margin for the state s.
        """
        return self.car.target_margin(s[:2])


#== Getting Functions ==
    def get_warmup_examples(self, num_warmup_samples=100):
        # rv = np.random.uniform( low=self.low,
        #                         high=self.high,
        #                         size=(num_warmup_samples,3))
        # x_rnd, y_rnd, theta_rnd = rv[:,0], rv[:,1], rv[:,2]

        heuristic_v = np.zeros((num_warmup_samples, 1))
        states = np.zeros((num_warmup_samples, self.observation_space.shape[0]))

        for i in range(num_warmup_samples):
            x, y, theta = self.car.sample_random_state(
                sample_inside_obs=self.sample_inside_obs,
                sample_inside_tar=self.sample_inside_tar)
            # x, y, theta = x_rnd[i], y_rnd[i], theta_rnd[i]
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


    def get_value(self, q_func, policy, theta, nx=101, ny=101, addBias=False):
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
                state = torch.FloatTensor([x, y, theta]).to(self.device)
            else:
                z = max([l_x, g_x])
                state = torch.FloatTensor([x, y, theta, z]).to(self.device)
            with torch.no_grad():
                action = policy(state)

            xx = torch.cat([state, action]).to(self.device)
            if addBias:
                v[idx] = q_func(xx).item() + max(l_x, g_x)
            else:
                v[idx] = q_func(xx).item()

            it.iternext()
        return v


#== Trajectory Functions ==
    def simulate_one_trajectory(self, policy, T=10, state=None, theta=None,
            sample_inside_obs=True, sample_inside_tar=True, toEnd=False):
        # reset
        if state is None:
            state = self.car.sample_random_state(sample_inside_obs=sample_inside_obs,
                sample_inside_tar=sample_inside_tar, theta=theta)
        traj = []
        result = 0 # not finished
        valueList = []
        gxList = []
        lxList = []
        for t in range(T):
            traj.append(state)

            g_x = self.safety_margin(state)
            l_x = self.target_margin(state)

            #= Rollout Record
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
                    result = -1 # failed
                    break
                elif l_x <= 0:
                    result = 1 # succeeded
                    break

            state_tensor = torch.FloatTensor(state).to(self.device)
            u = policy(state_tensor).detach().cpu().numpy()[0]
            state = self.car.integrate_forward(state, u)
        traj = np.array(traj)
        info = {'valueList':valueList, 'gxList':gxList, 'lxList':lxList}
        return traj, result, minV, info


    def simulate_trajectories(  self, policy, T=10,
        num_rnd_traj=None, states=None, theta=None,
        keepOutOf=False, toEnd=False):

        sample_inside_obs = not keepOutOf
        sample_inside_tar = not keepOutOf

        assert ((num_rnd_traj is None and states is not None) or
                (num_rnd_traj is not None and states is None) or
                (len(states) == num_rnd_traj))
        trajectories = []

        if states is None:
            nx=41
            ny=nx
            xs = np.linspace(self.bounds[0,0], self.bounds[0,1], nx)
            ys = np.linspace(self.bounds[1,0], self.bounds[1,1], ny)
            results  = np.empty((nx, ny), dtype=int)
            minVs  = np.empty((nx, ny), dtype=float)

            it = np.nditer(results, flags=['multi_index'])
            print()
            while not it.finished:
                idx = it.multi_index
                print(idx, end='\r')
                x = xs[idx[0]]
                y = ys[idx[1]]
                state = np.array([x, y, 0.])
                traj, result, minV, _ = self.simulate_one_trajectory(policy,
                    T=T, theta=theta, sample_inside_obs=sample_inside_obs,
                    sample_inside_tar=sample_inside_tar, toEnd=toEnd)
                trajectories.append((traj))
                results[idx] = result
                minVs[idx] = minV
                it.iternext()
            results = results.reshape(-1)
            minVs = minVs.reshape(-1)

            # results = np.empty(shape=(num_rnd_traj,), dtype=int)
            # minVs = np.empty(shape=(num_rnd_traj,), dtype=float)
            # for idx in range(num_rnd_traj):
            #     traj, result, minV, _ = self.simulate_one_trajectory(policy,
            #         T=T, theta=theta, sample_inside_obs=sample_inside_obs,
            #         sample_inside_tar=sample_inside_tar, toEnd=toEnd)
            #     trajectories.append(traj)
            #     results[idx] = result
            #     minVs[idx] = minV
        else:
            results = np.empty(shape=(len(states),), dtype=int)
            minVs = np.empty(shape=(len(states),), dtype=float)
            for idx, state in enumerate(states):
                traj, result, minV, _ = self.simulate_one_trajectory(policy,
                    T=T, state=state, toEnd=toEnd)
                trajectories.append(traj)
                results[idx] = result
                minVs[idx] = minV

        return trajectories, results, minVs


#== Plotting Functions ==
    def visualize(  self, q_func, policy,
        vmin=-1, vmax=1, nx=101, ny=101, cmap='seismic',
        labels=None, boolPlot=False, addBias=False, theta=np.pi/2,
        rndTraj=False, num_rnd_traj=10,
        sample_inside_obs=True, sample_inside_tar=True):
        """ Overlays analytic safe set on top of state value function.

        Args:
            q_func: NN or Tabular-Q
        """
        axStyle = self.get_axes()
        thetaList = [np.pi/6, np.pi/3, np.pi/2]
        fig = plt.figure(figsize=(12,4))
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)  
        axList = [ax1, ax2, ax3]

        for i, (ax, theta) in enumerate(zip(axList, thetaList)):
        # for i, (ax, theta) in enumerate(zip(self.axes, thetaList)):
            ax.cla()
            if i == len(thetaList)-1:
                cbarPlot=True
            else: 
                cbarPlot=False

            #== Plot failure / target set ==
            self.plot_target_failure_set(ax)

            #== Plot reach-avoid set ==
            self.plot_reach_avoid_set(ax, orientation=theta)

            #== Plot V ==
            self.plot_v_values( q_func, policy, ax=ax, fig=fig, theta=theta,
                vmin=vmin, vmax=vmax, nx=nx, ny=ny, cmap=cmap,
                boolPlot=boolPlot, cbarPlot=cbarPlot, addBias=addBias)
            #== Formatting ==
            self.plot_formatting(ax=ax, labels=labels)

            #== Plot Trajectories ==
            if rndTraj:
                self.plot_trajectories(policy, T=200, num_rnd_traj=num_rnd_traj,
                    theta=theta, toEnd=False,
                    sample_inside_obs=sample_inside_obs,
                    sample_inside_tar=sample_inside_tar,
                    ax=ax, c='k', lw=2, orientation=0)
            else:
                # `visual_initial_states` are specified for theta = pi/2. Thus,
                # we need to use "orientation = theta-pi/2"
                self.plot_trajectories(policy, T=200,
                    states=self.visual_initial_states, toEnd=False, 
                    ax=ax, c='k', lw=2, orientation=theta-np.pi/2)
            theta = theta*180/np.pi
            ax.set_xlabel(r'$\theta={:.0f}^\circ$'.format(theta), fontsize=28)

        plt.tight_layout()


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
        ax.xaxis.set_major_locator(LinearLocator(5))
        ax.xaxis.set_major_formatter('{x:.1f}')
        ax.yaxis.set_major_locator(LinearLocator(5))
        ax.yaxis.set_major_formatter('{x:.1f}')


    def plot_v_values(  self, q_func, policy, theta=np.pi/2, ax=None, fig=None,
                        vmin=-1, vmax=1, nx=201, ny=201, cmap='seismic',
                        boolPlot=False, cbarPlot=True, addBias=False):
        axStyle = self.get_axes()
        ax.plot([0., 0.], [axStyle[0][2], axStyle[0][3]], c='k')
        ax.plot([axStyle[0][0], axStyle[0][1]], [0., 0.], c='k')

        #== Plot V ==
        if theta == None:
            theta = 2.0 * np.random.uniform() * np.pi
        v = self.get_value(q_func, policy, theta, nx, ny, addBias=addBias)

        if boolPlot:
            im = ax.imshow(v.T>0., interpolation='none', extent=axStyle[0],
                origin="lower", cmap=cmap, zorder=-1)
        else:
            im = ax.imshow( v.T, interpolation='none', extent=axStyle[0], origin="lower",
                            cmap=cmap, vmin=vmin, vmax=vmax, zorder=-1)
            if cbarPlot:
                cbar = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.05, shrink=.95, ticks=[vmin, 0, vmax])
                cbar.ax.set_yticklabels(labels=[vmin, 0, vmax], fontsize=16)


    def plot_trajectories(  self, policy, T=10, num_rnd_traj=None, states=None,
        theta=None, sample_inside_obs=True, sample_inside_tar=True, toEnd=False,
        ax=None, c='k', lw=2, orientation=0, zorder=2):

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

        trajectories, results, minVs = self.simulate_trajectories(policy,
            T=T, num_rnd_traj=num_rnd_traj, states=states, theta=theta, 
            keepOutOf=False, toEnd=toEnd)
        if ax == None:
            ax = plt.gca()
        for traj in trajectories:
            traj_x = traj[:,0]
            traj_y = traj[:,1]
            ax.scatter(traj_x[0], traj_y[0], s=48, c=c, zorder=zorder)
            ax.plot(traj_x, traj_y, color=c,  linewidth=lw, zorder=zorder)

        return results, minVs


    def plot_reach_avoid_set(self, ax, c='g', lw=3, orientation=0, zorder=1):
        r = self.target_radius
        R = self.constraint_radius
        R_turn = self.R_turn
        if r >=  2*R_turn - R:
            # plot arc
            tmpY = (r**2 - R**2 + 2*R_turn*R) / (2*R_turn)
            tmpX = np.sqrt(r**2 - tmpY**2)
            tmpTheta = np.arcsin(tmpX / (R-R_turn))
            # two sides
            plot_arc((0.,  R_turn), R-R_turn, (tmpTheta-np.pi/2, np.pi/2),
                ax, c=c, lw=lw, orientation=orientation, zorder=zorder)
            plot_arc((0., -R_turn), R-R_turn, (-np.pi/2, np.pi/2-tmpTheta),
                ax, c=c, lw=lw, orientation=orientation, zorder=zorder)
            # middle
            tmpPhi = np.arcsin(tmpX/r)
            plot_arc((0., 0), r, (tmpPhi - np.pi/2, np.pi/2-tmpPhi), ax,
                c=c, lw=lw, orientation=orientation, zorder=zorder)
            # outer boundary
            plot_arc((0., 0), R, (np.pi/2, 3*np.pi/2), ax, c=c, lw=lw,
                orientation=orientation, zorder=zorder)
        else:
            # two sides
            tmpY = (R**2 + 2*R_turn*r - r**2) / (2*R_turn)
            tmpX = np.sqrt(R**2 - tmpY**2)
            tmpTheta = np.arcsin( tmpX / (R_turn-r))
            plot_arc((0.,  R_turn), R_turn-r, (np.pi/2+tmpTheta, 3*np.pi/2),
                ax, c=c, lw=lw, orientation=orientation, zorder=zorder)
            plot_arc((0., -R_turn), R_turn-r, (np.pi/2, 3*np.pi/2-tmpTheta),
                ax, c=c, lw=lw, orientation=orientation, zorder=zorder)
            # middle
            plot_arc((0., 0), r, (np.pi/2, -np.pi/2), ax, c=c, lw=lw,
                orientation=orientation, zorder=zorder)
            # outer boundary
            plot_arc((0., 0), R, (np.pi/2, 3*np.pi/2), ax, c=c, lw=lw,
                orientation=orientation, zorder=zorder)


    def plot_target_failure_set(self, ax, c_c='m', c_t='y', lw=3, zorder=0):
        plot_circle(self.constraint_center, self.constraint_radius, ax,
            c=c_c, lw=lw, zorder=zorder)
        plot_circle(self.target_center, self.target_radius, ax, c=c_t,
            lw=lw, zorder=zorder)