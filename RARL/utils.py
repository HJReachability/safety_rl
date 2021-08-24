"""
Please contact the author(s) of this library if you have any questions.
Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )
"""

import os
import glob
import pickle
import torch


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0-tau) + param.data * tau
        )


def save_model(model, step, logs_path, types, MAX_MODEL):
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
    with open(filename + ".pkl", "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(filename):
    with open(filename + ".pkl", "rb") as f:
        return pickle.load(f)
