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

# Local Variables
purple  = '#9370DB'
tiffany = '#0abab5'
silver = '#C0C0C0'

# Local Functions
def plot_arc(p, r, thetaParam, ax, c='b', lw=1.5, orientation=0):
    x, y = p
    thetaInit, thetaFinal = thetaParam

    xtilde = x*np.cos(orientation) - y*np.sin(orientation)
    ytilde = y*np.cos(orientation) + x*np.sin(orientation)

    theta = np.linspace(thetaInit+orientation, thetaFinal+orientation, 100)
    xs = xtilde + r * np.cos(theta)
    ys = ytilde + r * np.sin(theta)

    ax.plot(xs, ys, c=c, lw=lw)


def plot_circle(center, r, ax, c='b', lw=1.5, ls='-', orientation=0, scatter=False, zorder=0):
    x, y = center
    xtilde = x*np.cos(orientation) - y*np.sin(orientation)
    ytilde = y*np.cos(orientation) + x*np.sin(orientation)

    theta = np.linspace(0, 2*np.pi, 200)
    xs = xtilde + r * np.cos(theta)
    ys = ytilde + r * np.sin(theta)
    ax.plot(xs, ys, c=c, lw=lw, linestyle=ls, zorder=zorder)
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
        self.action_space = gym.spaces.Discrete(9)
        midpoint = (self.low + self.high)/2.0
        interval = self.high - self.low
        self.observation_space = gym.spaces.Box(np.float32(midpoint - interval/2),
                                                np.float32(midpoint + interval/2))

        # Constraint set parameters.
        self.evader_constraint_center = np.array([0, 0])
        self.evader_constraint_radius = 1.0
        self.pursuer_constraint_center = np.array([0, 0])
        self.pursuer_constraint_radius = 1.0
        self.capture_range = 0.25

        # Target set parameters.
        self.evader_target_center = np.array([0, 0])
        self.evader_target_radius = 0.5

        # Dubins cars parameters.
        self.time_step = 0.05
        self.speed = 0.75 # v
        self.R_turn = self.speed / 3
        self.pursuer = DubinsCarDyn(doneType=doneType)
        self.evader = DubinsCarDyn(doneType=doneType)
        self.init_car()

        # Internal state.
        self.mode = mode
        self.state = np.zeros(6)
        self.doneType = doneType

        # Visualization params
        self.evader_initial_states =[   np.array([ -.1, -self.R_turn, 0.]),
                                        np.array([ -0.95*self.evader_constraint_radius, 0., 0.]) ]

        self.pursuer_initial_states = [ np.array([ 0., .5, 0.]),
                                        np.array([ .5,  0, 1.5*np.pi]),
                                        np.array([ -.5, .5, 1.5*np.pi]),
                                        np.array([ -.1, -.5, .75*np.pi]) ]
        
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
        if start is not None:
            stateEvader = self.evader.reset(start=start[:3])
            statePursuer = self.pursuer.reset(start=start[3:])
        else:
            stateEvader = self.evader.reset()
            statePursuer = self.pursuer.reset()
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
        return max(evader_g_x, capture_g_x)


    def target_margin(self, s):
        evader_l_x = self.evader.target_margin(s[:2])
        return evader_l_x


#== Getting Functions ==
    def get_warmup_examples(self, num_warmup_samples=100, 
        theta=None, xPursuer=None, yPursuer=None, thetaPursuer=None):
        lowExt = np.tile(self.low, 2)
        highExt = np.tile(self.high, 2)
        states = np.random.default_rng().uniform(   low=lowExt,
                                                    high=highExt,
                                                    size=(num_warmup_samples, self.state.shape[0]))
        if theta is not None:
            states[:, 2] = theta
        if xPursuer is not None:
            states[:, 3] = xPursuer
        if yPursuer is not None:
            states[:, 4] = yPursuer
        if thetaPursuer is not None:
            states[:, 5] = thetaPursuer

        heuristic_v = np.zeros((num_warmup_samples, self.action_space.n))

        for i in range(num_warmup_samples):
            state = states[i]
            l_x = self.target_margin(state)
            g_x = self.safety_margin(state)
            heuristic_v[i,:] = np.maximum(l_x, g_x)

        return states, heuristic_v


    # ? 2D-plot based on evader's x and y
    def get_axes(self):
        """ Gets the bounds for the environment.

        Returns:
            List containing a list of bounds for each state coordinate and a
        """
        aspect_ratio = (self.bounds[0,1]-self.bounds[0,0])/(self.bounds[1,1]-self.bounds[1,0])
        axes = np.array([self.bounds[0,0], self.bounds[0,1], self.bounds[1,0], self.bounds[1,1]])
        return [axes, aspect_ratio]


    # ? Fix evader's theta and pursuer's (x, y, theta)
    def get_value(self, q_func, theta, xPursuer, yPursuer, thetaPursuer,
            nx=101, ny=101, addBias=False, verbose=False):

        if verbose:
            print("Getting values with evader's theta and pursuer's (x, y, theta) equal to", end=' ')
            print("{:.1f} and ({:.1f}, {:.1f}, {:.1f})".format(theta, xPursuer, yPursuer, thetaPursuer))
        v = np.zeros((nx, ny))
        it = np.nditer(v, flags=['multi_index'])
        xs = np.linspace(self.bounds[0,0], self.bounds[0,1], nx)
        ys = np.linspace(self.bounds[1,0], self.bounds[1,1], ny)
        while not it.finished:
            idx = it.multi_index
            x = xs[idx[0]]
            y = ys[idx[1]]

            state = np.array([x, y, theta, xPursuer, yPursuer, thetaPursuer])
            l_x = self.target_margin(state)
            g_x = self.safety_margin(state)

            # Q(s, a^*)
            state = torch.FloatTensor(state).to(self.device)
            with torch.no_grad():
                state_action_values = q_func(state)
            Q_mtx = state_action_values.reshape(self.numActionList[0], self.numActionList[1])
            pursuerValues, _ = Q_mtx.max(dim=-1)
            minmaxValue, _ = pursuerValues.min(dim=-1)
            minmaxValue = minmaxValue.cpu().numpy()

            if addBias:
                v[idx] = minmaxValue + max(l_x, g_x)
            else:
                v[idx] = minmaxValue
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

        trajEvader = [stateEvader[:3]]
        trajPursuer = [statePursuer[:3]]
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
                    
            stateTensor = torch.FloatTensor(state).to(self.device)
            with torch.no_grad():
                state_action_values = q_func(stateTensor)
            Q_mtx = state_action_values.reshape(self.numActionList[0], self.numActionList[1])
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

            trajEvader.append(stateEvader[:3])
            trajPursuer.append(statePursuer[:3])

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
    # ? Check all plotting functions
    def visualize(  self, q_func,
                    vmin=-1, vmax=1, nx=101, ny=101, cmap='seismic',
                    labels=None, boolPlot=False, addBias=False, theta=0.,
                    rndTraj=False, num_rnd_traj=10, keepOutOf=False):

        axStyle = self.get_axes()
        fig, axes = plt.subplots(1,4, figsize=(16, 4))
        init_states = []
        for i, stateEvader in enumerate(self.evader_initial_states):
            for j in range(2):
                statePursuer = self.pursuer_initial_states[2*i+j]
                init_states.append(np.concatenate((stateEvader, statePursuer), axis=0))

        for i, (ax, state) in enumerate(zip(axes, init_states)):
            state=[state]
            ax.cla()
            if i == len(init_states)-1:
                cbarPlot=True
            else: 
                cbarPlot=False

            #== Formatting ==
            self.plot_formatting(ax=ax, labels=labels)

            #== Plot failure / target set ==
            self.plot_target_failure_set(ax)

            #== Plot V ==
            self.plot_v_values( q_func, ax=ax, fig=fig, theta=theta,
                                vmin=vmin, vmax=vmax, nx=nx, ny=ny, cmap=cmap,
                                boolPlot=boolPlot, cbarPlot=cbarPlot, addBias=addBias)

            #== Plot Trajectories ==
            if rndTraj:
                self.plot_trajectories( q_func, T=200, num_rnd_traj=num_rnd_traj, theta=theta,
                                        toEnd=False, keepOutOf=keepOutOf,
                                        ax=ax, orientation=0)
            else:
                self.plot_trajectories( q_func, T=200, states=state, toEnd=False, 
                                        ax=ax, orientation=0)
        plt.tight_layout()


    # ? 2D-plot based on evader's x and y
    def plot_formatting(self, ax=None, labels=None):
        axStyle = self.get_axes()
        ax.plot([0., 0.], [axStyle[0][2], axStyle[0][3]], c='k', zorder=0)
        ax.plot([axStyle[0][0], axStyle[0][1]], [0., 0.], c='k', zorder=0)
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


    # ? Check get_values, 2D-plot based on evader's x and y
    def plot_v_values(  self, q_func, theta=0, xPursuer=.5, yPursuer=.5, thetaPursuer=0,
                        ax=None, fig=None,
                        vmin=-1, vmax=1, nx=101, ny=101, cmap='coolwarm',
                        boolPlot=False, cbarPlot=True, addBias=False):
        axStyle = self.get_axes()

        #== Plot V ==
        if theta == None:
            theta = 2.0 * np.random.uniform() * np.pi
        v = self.get_value(q_func, theta, xPursuer, yPursuer, thetaPursuer, nx, ny, addBias=addBias)

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
                            keepOutOf=False, toEnd=False, ax=None, c=[tiffany, 'y'], lw=2, orientation=0):

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
        for traj, result in zip(trajectories, results):
            trajEvader, trajPursuer = traj
            trajEvaderX = trajEvader[:,0]
            trajEvaderY = trajEvader[:,1]
            trajPursuerX = trajPursuer[:,0]
            trajPursuerY = trajPursuer[:,1]

            ax.scatter(trajEvaderX[0], trajEvaderY[0], s=48, c=c[0], zorder=3)
            ax.plot(trajEvaderX, trajEvaderY, color=c[0],  linewidth=lw, zorder=2)
            ax.scatter(trajPursuerX[0], trajPursuerY[0], s=48, c=c[1], zorder=3)
            ax.plot(trajPursuerX, trajPursuerY, color=c[1],  linewidth=lw, zorder=2)
            if result == 1:
                ax.scatter(trajEvaderX[-1], trajEvaderY[-1], s=60, c=c[0], marker='*', zorder=3)
            if result == -1:
                ax.scatter(trajEvaderX[-1], trajEvaderY[-1], s=60, c=c[0], marker='x', zorder=3)

        return results


    # ! Analytic solutions available?
    def plot_reach_avoid_set(self, ax, c='g', lw=3, orientation=0):
        pass


    # ? Plot evader's target, constraint and pursuer's capture range
    def plot_target_failure_set(self, ax, xPursuer=.5, yPursuer=.5):
        plot_circle(self.evader.constraint_center, self.evader.constraint_radius,
            ax, c='k', lw=3, zorder=1)
        plot_circle(self.evader.target_center,     self.evader.target_radius,
            ax, c='m', lw=3, zorder=1)
        plot_circle(np.array([xPursuer, yPursuer]), self.capture_range,
            ax, c='k', lw=3, ls='--', zorder=1)