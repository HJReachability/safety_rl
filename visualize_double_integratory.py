import ray
import mdr_dqn as dqn
import os
import json
import sys
import gym
import numpy as np
import pickle
from ray.tune.registry import register_env
from MDR_RL.utils import visualize_matrix, save
import glob
from tensorflow.train import latest_checkpoint

# setup
source_dir = "/Users/Neil/reach_RL/MDR_RL/Q_learning/train_double_integrator_Jul7/TrainDQN_double_integrator-v0_0_2019-07-07_16-30-44580qi7l5"
experiment_directories = glob.glob(source_dir)
num_points = [500, 500]  # grid resolution

generate_data = True
visualize_data = True

# generate value functions
if generate_data:
	def env_creator(env_config):
		from MDR_RL import gym_reachability
		return gym.make("double_integrator-v0")

	from MDR_RL.gym_reachability import gym_reachability
	#register_env("double_integrator-v0", env_creator)
	env = env_creator(None)  # need environment to specify state space and action space size
	ray.init()

	for experiment in experiment_directories:
		param_file = os.path.join(experiment, "params.json")  # load config from json file
		checkpoint_dir = latest_checkpoint(experiment)  # gets latest checkpoint

		with open(param_file) as f:
			config = json.load(f)
		dqn.DQNAgent._allow_unknown_configs = True
		agent = dqn.DQNAgent(config=config, env="double_integrator-v0")

		# TODO temporary will need to find a better solution if this works
		agent.restore(os.path.join(experiment, "checkpoint_200/checkpoint_200/checkpoint-200"))

		bins = []
		for i in range(len(env.bounds)):
			a = env.bounds[i][0]
			b = env.bounds[i][1]
			bins.append(np.arange(start=a, stop=b, step=(b - a) / num_points[i]))

		data = []
		for x_dot in bins[1]:
			for x in bins[0]:  # TODO check which ordering of loops is correct
				data.append(np.array([x, x_dot]))
		Q_values = agent.q_values(data, batched=True)
		V = np.max(Q_values, axis=1)
		V = V.reshape(num_points)
		save(V, None, directory=experiment, name="value_function", date_dir=False)
		visualize_matrix(V, vmax=None, vmin=None)

if visualize_data:
	for experiment in experiment_directories:
		try:
			with open(os.path.join(experiment, 'value_function.pickle'), 'rb') as handle:
				data = pickle.load(handle)
			V = data['V']
			visualize_matrix(V)
			visualize_matrix(np.sign(V))
		except:
			pass
