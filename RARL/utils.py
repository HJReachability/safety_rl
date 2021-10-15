"""
Please contact the author(s) of this library if you have any questions.
Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )

This module provides some utils functions for reinforcement learning
algortihms and some general save and load functions.
"""

import os
import glob
import pickle
import torch


def soft_update(target, source, tau):
  """Uses soft_update method to update the target network.

  Args:
      target (toch.nn.Module): target network in double deep Q-network.
      source (toch.nn.Module): Q-network in double deep Q-network.
      tau (float): the ratio of the weights in the target.
  """
  for target_param, param in zip(target.parameters(), source.parameters()):
    target_param.data.copy_(target_param.data * (1.0-tau) + param.data * tau)


def save_model(model, step, logs_path, types, MAX_MODEL):
  """Saves the weights of the model.

  Args:
      model (toch.nn.Module): the model to be saved.
      step (int): the number of updates so far.
      logs_path (str): the path to save the weights.
      types (str): the decorater of the file name.
      MAX_MODEL (int): the maximum number of models to be saved.
  """
  start = len(types) + 1
  os.makedirs(logs_path, exist_ok=True)
  model_list = glob.glob(os.path.join(logs_path, "*.pth"))
  if len(model_list) > MAX_MODEL - 1:
    min_step = min([int(li.split("/")[-1][start:-4]) for li in model_list])
    os.remove(os.path.join(logs_path, "{}-{}.pth".format(types, min_step)))
  logs_path = os.path.join(logs_path, "{}-{}.pth".format(types, step))
  torch.save(model.state_dict(), logs_path)
  print("  => Save {} after [{}] updates".format(logs_path, step))


def save_obj(obj, filename):
  """Saves the object into a pickle file.

  Args:
      obj (object): the object to be saved.
      filename (str): the path to save the object.
  """
  with open(filename + ".pkl", "wb") as f:
    pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(filename):
  """Loads the object and return the object.

  Args:
      filename (str): the path to save the object.
  """
  with open(filename + ".pkl", "rb") as f:
    return pickle.load(f)
