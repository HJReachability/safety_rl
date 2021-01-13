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
import matplotlib
import matplotlib.pyplot as plt
import torch

from .dubins_car_dyn import DubinsCarDyn


class DubinsCarPEEnv(gym.Env):
    def __init__(self, device, mode='normal', doneType='toEnd'):
        # Set random seed.
        self.seed_val = 0
        np.random.seed(self.seed_val)

        # State bounds.
        self.bounds = np.array([[-1.1, 1.1],
                                [-1.1, 1.1],
                                [0, 2*np.pi]])
        self.low = self.bounds[:, 0]
        self.high = self.bounds[:, 1]

        # Gym variables.
        self.numActionList = [3, 3]
        self.action_space = gym.spaces.Discrete(6)
        midpoint = (self.low + self.high)/2.0
        interval = self.high - self.low
        self.observation_space = gym.spaces.Box(np.float32(midpoint - interval/2),
                                                np.float32(midpoint + interval/2))

        # Constraint set parameters.
        self.evader_constraint_center = np.array([0, 0])
        self.evader_constraint_radius = 1.0
        self.pursuer_constraint_center = np.array([0, 0])
        self.pursuer_constraint_radius = 1.0
        self.capture_range = 0.1

        # Target set parameters.
        self.evader_target_center = np.array([0, 0])
        self.evader_target_radius = 0.3

        # Dubins cars parameters.
        self.time_step = 0.05
        self.speed = 0.5 # v
        self.R_turn = 0.6
        self.pursuer = DubinsCarDyn(doneType=doneType)
        self.evader = DubinsCarDyn(doneType=doneType)
        self.init_car()

        # Internal state.
        self.mode = mode
        self.state = np.zeros(6)
        self.doneType = doneType

        # Visualization params 
        # self.visual_initial_states =[   np.array([ .6*self.constraint_radius,  -.5, np.pi/2]),
        #                                 np.array([ -.4*self.constraint_radius, -.5, np.pi/2]),
        #                                 np.array([ -0.95*self.constraint_radius, 0., np.pi/2]),
        #                                 np.array([ self.R_turn, 0.95*(self.constraint_radius-self.R_turn), np.pi/2]),
        #                             ]
        
        # Cost Params
        self.targetScaling = 1.
        self.safetyScaling = 1.
        self.penalty = 1.
        self.reward = -1.
        self.costType = 'sparse'
        self.device = device

        print("Env: mode---{:s}; doneType---{:s}".format(mode, doneType))


    def init_car(self):
        self.evader.set_seed(seed=self.seed_val)
        self.evader.set_bounds(bounds=self.bounds)
        self.evader.set_constraint(center=self.evader_constraint_center, radius=self.evader_constraint_radius)
        self.evader.set_target(center=self.evader_target_center, radius=self.evader_target_radius)
        self.evader.set_speed(speed=self.speed)
        self.evader.set_time_step(time_step=self.time_step)
        self.evader.set_radius_rotation(R_turn=self.R_turn, verbose=False)

        self.pursuer.set_seed(seed=self.seed_val)
        self.pursuer.set_bounds(bounds=self.bounds)
        self.pursuer.set_constraint(center=self.pursuer_constraint_center, radius=self.pursuer_constraint_radius)
        self.pursuer.set_speed(speed=self.speed)
        self.pursuer.set_time_step(time_step=self.time_step)
        self.pursuer.set_radius_rotation(R_turn=self.R_turn, verbose=False)


#== Reset Functions ==
    def reset(self, start=None):
        """ Reset the state of the environment.

        Args:
            start: Which state to reset the environment to. If None, pick the
                state uniformly at random.

        Returns:
            The state the environment has been reset to.
        """
        stateEvader = self.evader.reset(start=start[:3])
        statePursuer = self.pursuer.reset(start=start[3:])
        self.state = np.concatenate((stateEvader, statePursuer), axis=0)
        return np.copy(self.state)


    def sample_random_state(self, keepOutOf=False, theta=None):
        stateEvader = self.evader.sample_random_state(keepOutOf=keepOutOf, theta=theta)
        statePursuer = self.pursuer.sample_random_state(keepOutOf=keepOutOf, theta=theta)
        return np.concatenate((stateEvader, statePursuer), axis=0)


#== Dynamics Functions ==
    def step(self, action):
        """ Evolve the environment one step forward under given input action.

        Args:
            action: a vector consist of action indices for the evader and pursuer.

        Returns:
            Tuple of (next state, signed distance of current state, whether the
            episode is done, info dictionary).
        """
        state_tmp = np.concatenate((self.evader.state, self.pursuer.state), axis=0)
        distance = np.linalg.norm(self.state-state_tmp)
        if distance >= 1e-8:
            raise "There is a mismatch between the env state and car state: {:.2e}".format(distance)

        l_x_cur = self.target_margin(self.state)
        g_x_cur = self.safety_margin(self.state)

        stateEvader,  doneEvader = self.evader.step(action[0])
        statePursuer, donePursuer = self.pursuer.step(action[0])

        self.state = np.concatenate((stateEvader, statePursuer), axis=0)
        l_x_nxt = self.target_margin(self.state)
        g_x_nxt = self.safety_margin(self.state)
        info = {"g_x": g_x_cur, "l_x": l_x_cur, "g_x_nxt": g_x_nxt, "l_x_nxt": l_x_nxt} 

        # cost
        assert self.mode == 'RA', "PE environment doesn't support conventional RL yet"
        if self.mode == 'RA':
            fail = g_x_cur > 0
            success = l_x_cur <= 0
            if fail:
                cost = self.penalty
            elif success:
                cost = self.reward
            else:
                cost = 0.

        done = doneEvader and donePursuer

        return np.copy(self.state), cost, done, info


#== Setting Hyper-Parameter Functions ==
    def set_costParam(self, penalty=1, reward=-1, costType='normal', targetScaling=1., safetyScaling=1.):
        self.penalty = penalty
        self.reward = reward
        self.costType = costType
        self.safetyScaling = safetyScaling
        self.targetScaling = targetScaling


    def set_capture_range(self, capture_range=.1):
        self.capture_range = capture_range


    def set_constraint(self, center=np.array([0.,0.]), radius=1., car='evader'):
        if car == 'evader':
            self.evader_constraint_center = center
            self.evader_constraint_radius = radius
            self.evader.set_constraint(center=center, radius=radius)
        elif car=='pursuer':
            self.pursuer_constraint_center = center
            self.pursuer_constraint_radius = radius
            self.pursuer.set_constraint(center=center, radius=radius)
        elif car=='both':
            self.evader_constraint_center = center
            self.evader_constraint_radius = radius
            self.evader.set_constraint(center=center, radius=radius)
            self.pursuer_constraint_center = center
            self.pursuer_constraint_radius = radius
            self.pursuer.set_constraint(center=center, radius=radius)


    def set_target(self, center=np.array([0.,0.]), radius=.4, car='evader'):
        if car == 'evader':
            self.evader_target_center = center
            self.evader_target_radius = radius
            self.evader.set_target(center=center, radius=radius)
        elif car=='pursuer':
            self.pursuer_target_center = center
            self.pursuer_target_radius = radius
            self.pursuer.set_target(center=center, radius=radius)
        elif car=='both':
            self.evader_target_center = center
            self.evader_target_radius = radius
            self.evader.set_target(center=center, radius=radius)
            self.pursuer_target_center = center
            self.pursuer_target_radius = radius
            self.pursuer.set_constraint(center=center, radius=radius)


    def set_radius_rotation(self, R_turn=.6, verbose=False):
        self.R_turn = R_turn
        self.evader.set_radius_rotation(R_turn=R_turn, verbose=verbose)
        self.pursuer.set_radius_rotation(R_turn=R_turn, verbose=verbose)


    def set_seed(self, seed):
        """ Set the random seed.

        Args:
            seed: Random seed.
        """
        self.seed_val = seed
        np.random.seed(self.seed_val)
        self.evader.set_seed(seed)
        self.pursuer.set_seed(seed)


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
        self.evader.set_bounds(bounds)
        self.pursuer.set_bounds(bounds)


#== Margin Functions ==
    def safety_margin(self, s):
        evader_g_x = self.evader.safety_margin(s[:2])
        dist_evader_pursuer = np.linalg.norm(s[:2]-s[3:5], ord=2)
        capture_g_x = self.capture_range - dist_evader_pursuer
        print(evader_g_x, dist_evader_pursuer, capture_g_x)
        return max(evader_g_x, capture_g_x)


    def target_margin(self, s):
        evader_l_x = self.evader.target_margin(s[:2])
        return evader_l_x


#== Getting Functions ==
    def get_warmup_examples(self, num_warmup_samples=100):
        lowExt = np.repeat(self.low, 2)
        highExt = np.repeat(self.high, 2)
        states = np.random.uniform( low=lowExt,
                                    high=highExt,
                                    size=(num_warmup_samples, self.state.shape[0]))

        heuristic_v = np.zeros((num_warmup_samples, self.action_space.n))
        states = np.zeros((num_warmup_samples, self.state.shape[0]))

        for i in range(num_warmup_samples):
            state = states[i]
            l_x = self.target_margin(state)
            g_x = self.safety_margin(state)
            heuristic_v[i,:] = np.maximum(l_x, g_x)

        return states, heuristic_v


    # ? 2D-plot based on x-y locations
    def get_axes(self):
        """ Gets the bounds for the environment.

        Returns:
            List containing a list of bounds for each state coordinate and a
        """
        aspect_ratio = (self.bounds[0,1]-self.bounds[0,0])/(self.bounds[1,1]-self.bounds[1,0])
        axes = np.array([self.bounds[0,0], self.bounds[0,1], self.bounds[1,0], self.bounds[1,1]])
        return [axes, aspect_ratio]


    # TODO: how to visualize
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


#== Trajectory Functions ==
    def simulate_one_trajectory(self, q_func, T=10, state=None, theta=None,
                                keepOutOf=False, toEnd=False):
        # reset
        if state is None:
            stateEvader = self.evader.sample_random_state(keepOutOf=keepOutOf, theta=theta)
            statePursuer = self.pursuer.sample_random_state(keepOutOf=keepOutOf, theta=theta)
        else:
            stateEvader  = state[:3]
            statePursuer = state[3:]

        trajEvader = [stateEvader[:2]]
        trajPursuer = [statePursuer[:2]]
        result = 0 # not finished

        for t in range(T):
            state = np.concatenate((stateEvader, statePursuer), axis=0)
            doneEvader = not self.evader.check_within_bounds(stateEvader)
            donePursuer = not self.pursuer.check_within_bounds(statePursuer)
            if toEnd:
                if doneEvader and donePursuer:
                    result = 1
                    break
            else:
                if self.safety_margin(state) > 0:
                    result = -1 # failed
                    break
                elif self.target_margin(state) <= 0:
                    result = 1 # succeeded
                    break
                    
            stateTensor = torch.FloatTensor(state, device=self.device)
            state_action_values = q_func(stateTensor)
            Q_mtx = state_action_values.detach().reshape(self.numActionList[0], self.numActionList[1])
            pursuerValues, colIndices = Q_mtx.max(dim=1)
            minmaxValue, rowIdx = pursuerValues.min(dim=0)
            colIdx = colIndices[rowIdx]

            # If cars are within the boundary, we update their states according to the controls
            if not doneEvader:
                uEvader = self.evader.discrete_controls[rowIdx]
                stateEvader = self.evader.integrate_forward(stateEvader, uEvader)
            if not donePursuer:
                uPursuer = self.pursuer.discrete_controls[colIdx]
                statePursuer = self.pursuer.integrate_forward(statePursuer, uPursuer)

            trajEvader.append(stateEvader[:2])
            trajPursuer.append(statePursuer[:2])

        trajEvader = np.array(trajEvader)
        trajPursuer = np.array(trajPursuer)
        return trajEvader, trajPursuer, result


    def simulate_trajectories(  self, q_func, T=10,
                                num_rnd_traj=None, states=None, theta=None,
                                keepOutOf=False, toEnd=False):

        assert ((num_rnd_traj is None and states is not None) or
                (num_rnd_traj is not None and states is None) or
                (len(states) == num_rnd_traj))
        trajectories = []

        if states is None:
            results = np.empty(shape=(num_rnd_traj,), dtype=int)
            for idx in range(num_rnd_traj):
                trajEvader, trajPursuer, result = self.simulate_one_trajectory(
                    q_func, T=T, theta=theta, keepOutOf=keepOutOf, toEnd=toEnd)
                trajectories.append((trajEvader, trajPursuer))
                results[idx] = result
        else:
            results = np.empty(shape=(len(states),), dtype=int)
            for idx, state in enumerate(states):
                trajEvader, trajPursuer, result = self.simulate_one_trajectory(
                    q_func, T=T, state=state, toEnd=toEnd)
                trajectories.append((trajEvader, trajPursuer))
                results[idx] = result

        return trajectories, results


#== Plotting Functions ==
    # TODO: how to visualize
    def visualize(  self, q_func, no_show=False,
                    vmin=-1, vmax=1, nx=101, ny=101, cmap='coolwarm',
                    labels=None, boolPlot=False, addBias=False, theta=np.pi/2,
                    rndTraj=False, num_rnd_traj=10, keepOutOf=False):
        """ Overlays analytic safe set on top of state value function.

        Args:
            q_func: NN or Tabular-Q
        """
        axStyle = self.get_axes()
        thetaList = [np.pi/6, np.pi/3, np.pi/2]
        # numX = 1
        # numY = 3
        # if self.axes is None:
        #     self.fig, self.axes = plt.subplots(
        #         numX, numY, figsize=(4*numY, 4*numX), sharex=True, sharey=True)
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
            self.plot_v_values( q_func, ax=ax, fig=fig, theta=theta,
                                vmin=vmin, vmax=vmax, nx=nx, ny=ny, cmap=cmap,
                                boolPlot=boolPlot, cbarPlot=cbarPlot, addBias=addBias)
            #== Formatting ==
            self.plot_formatting(ax=ax, labels=labels)

            #== Plot Trajectories ==
            if rndTraj:
                self.plot_trajectories( q_func, T=200, num_rnd_traj=num_rnd_traj, theta=theta,
                                        toEnd=False, keepOutOf=keepOutOf,
                                        ax=ax, c='y', lw=2, orientation=0)
            else:
                # `visual_initial_states` are specified for theta = pi/2. Thus,
                # we need to use "orientation = theta-pi/2"
                self.plot_trajectories( q_func, T=200, states=self.visual_initial_states, toEnd=False, 
                                        ax=ax, c='y', lw=2, orientation=theta-np.pi/2)

            ax.set_xlabel(r'$\theta={:.0f}^\circ$'.format(theta*180/np.pi), fontsize=28)

        plt.tight_layout()
        plt.show()


    # TODO: how to visualize
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
        # ax.set_xticklabels([])
        # ax.set_yticklabels([])
        #ax.set_title(r"$\theta$={:.1f}".format(theta * 180 / np.pi), fontsize=24)


    # TODO: how to visualize
    def plot_v_values(  self, q_func, theta=np.pi/2, ax=None, fig=None,
                        vmin=-1, vmax=1, nx=201, ny=201, cmap='seismic',
                        boolPlot=False, cbarPlot=True, addBias=False):
        axStyle = self.get_axes()
        ax.plot([0., 0.], [axStyle[0][2], axStyle[0][3]], c='k')
        ax.plot([axStyle[0][0], axStyle[0][1]], [0., 0.], c='k')

        #== Plot V ==
        if theta == None:
            theta = 2.0 * np.random.uniform() * np.pi
        v = self.get_value(q_func, theta, nx, ny, addBias=addBias)

        if boolPlot:
            im = ax.imshow(v.T>0., interpolation='none', extent=axStyle[0], origin="lower", cmap=cmap)
        else:
            im = ax.imshow( v.T, interpolation='none', extent=axStyle[0], origin="lower",
                            cmap=cmap, vmin=vmin, vmax=vmax)
            if cbarPlot:
                cbar = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.05, shrink=.95, ticks=[vmin, 0, vmax])
                cbar.ax.set_yticklabels(labels=[vmin, 0, vmax], fontsize=24)


    # ? Plot trajectories based on x-y location of the evader and the pursuer
    def plot_trajectories(  self, q_func, T=10, num_rnd_traj=None, states=None, theta=None,
                            keepOutOf=False, toEnd=False, ax=None, c=['y', 'w'], lw=1.5, orientation=0):

        assert ((num_rnd_traj is None and states is not None) or
                (num_rnd_traj is not None and states is None) or
                (len(states) == num_rnd_traj))

        if states != None:
            tmpStates = []
            for state in states:
                stateEvaderTilde = rotatePoint(state[:3], orientation)
                statePursuerTilde = rotatePoint(state[3:], orientation)
                tmpStates.append(np.concatenate((stateEvaderTilde, statePursuerTilde), axis=0))
            states = tmpStates

        trajectories, results = self.simulate_trajectories( q_func, T=T, num_rnd_traj=num_rnd_traj, 
                                                            states=states, theta=theta, 
                                                            keepOutOf=keepOutOf, toEnd=toEnd)
        if ax == None:
            ax = plt.gca()
        for traj in trajectories:
            trajEvader, trajPursuer = traj
            trajEvaderX = trajEvader[:,0]
            trajEvaderY = trajEvader[:,1]
            trajPursuerX = trajPursuer[:,0]
            trajPursuerY = trajPursuer[:,1]

            ax.scatter(trajEvaderX[0], trajEvaderY[0], s=48, c=c[0])
            ax.plot(trajEvaderX, trajEvaderY, color=c[0],  linewidth=lw)
            ax.scatter(trajPursuerX[0], trajPursuerY[0], s=48, c=c[1])
            ax.plot(trajPursuerX, trajPursuerY, color=c[1],  linewidth=lw)

        return results


    # TODO: how to visualize
    def plot_reach_avoid_set(self, ax, c='g', lw=3, orientation=0):
        r = self.target_radius
        R = self.constraint_radius
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


    # TODO: how to visualize
    def plot_target_failure_set(self, ax):
        self.plot_circle(self.constraint_center, self.constraint_radius, ax, c='k', lw=3)
        self.plot_circle(self.target_center,     self.target_radius, ax, c='m', lw=3)


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


    def rotatePoint(state, orientation):
        x, y, theta = state
        xtilde = x*np.cos(orientation) - y*np.sin(orientation)
        ytilde = y*np.cos(orientation) + x*np.sin(orientation)
        thetatilde = theta+orientation

        return np.array([xtilde, ytilde, thetatilde])
