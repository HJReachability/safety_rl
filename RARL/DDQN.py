"""
Please contact the author(s) of this library if you have any questions.
Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )
"""

import torch
import torch.optim as optim

from collections import namedtuple
import os
import pickle

from .model import StepLRMargin, StepResetLR
from .ReplayMemory import ReplayMemory
from .utils import soft_update, save_model


Transition = namedtuple("Transition", ["s", "a", "r", "s_", "info"])


class DDQN:

    def __init__(self, CONFIG):
        """
        __init__

        Args:
            CONFIG (object): configuration.
        """
        self.CONFIG = CONFIG
        self.saved = False
        self.memory = ReplayMemory(CONFIG.MEMORY_CAPACITY)

        # == PARAM ==
        # Exploration
        self.EpsilonScheduler = StepResetLR(
            initValue=CONFIG.EPSILON,
            period=CONFIG.EPS_PERIOD,
            resetPeriod=CONFIG.EPS_RESET_PERIOD,
            decay=CONFIG.EPS_DECAY,
            endValue=CONFIG.EPS_END,
        )
        self.EPSILON = self.EpsilonScheduler.get_variable()
        # Learning Rate
        self.LR_C = CONFIG.LR_C
        self.LR_C_PERIOD = CONFIG.LR_C_PERIOD
        self.LR_C_DECAY = CONFIG.LR_C_DECAY
        self.LR_C_END = CONFIG.LR_C_END
        # NN: batch size, maximal number of NNs stored
        self.BATCH_SIZE = CONFIG.BATCH_SIZE
        self.MAX_MODEL = CONFIG.MAX_MODEL
        self.device = CONFIG.DEVICE
        # Discount Factor
        self.GammaScheduler = StepLRMargin(
            initValue=CONFIG.GAMMA,
            period=CONFIG.GAMMA_PERIOD,
            decay=CONFIG.GAMMA_DECAY,
            endValue=CONFIG.GAMMA_END,
            goalValue=1.0,
        )
        self.GAMMA = self.GammaScheduler.get_variable()
        # Target Network Update
        self.double = CONFIG.DOUBLE
        self.TAU = CONFIG.TAU
        self.HARD_UPDATE = CONFIG.HARD_UPDATE  # int, update period
        self.SOFT_UPDATE = CONFIG.SOFT_UPDATE  # bool

    def build_network(self):
        raise NotImplementedError

    def build_optimizer(self):
        self.optimizer = torch.optim.AdamW(
            self.Q_network.parameters(), lr=self.LR_C, weight_decay=1e-3
        )
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=self.LR_C_PERIOD, gamma=self.LR_C_DECAY
        )
        self.max_grad_norm = 1
        self.cntUpdate = 0

    def update(self):
        raise NotImplementedError

    def initBuffer(self, env):
        raise NotImplementedError

    def initQ(self):
        raise NotImplementedError

    def learn(self):
        raise NotImplementedError

    def update_target_network(self):
        if self.SOFT_UPDATE:
            # Soft Replace
            soft_update(self.target_network, self.Q_network, self.TAU)
        elif self.cntUpdate % self.HARD_UPDATE == 0:
            # Hard Replace
            self.target_network.load_state_dict(self.Q_network.state_dict())

    def updateHyperParam(self):
        lr = self.optimizer.state_dict()["param_groups"][0]["lr"]
        if (lr <= self.LR_C_END):
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.LR_C_END
        else:
            self.scheduler.step()

        self.EpsilonScheduler.step()
        self.EPSILON = self.EpsilonScheduler.get_variable()
        self.GammaScheduler.step()
        self.GAMMA = self.GammaScheduler.get_variable()

    def select_action(self):
        raise NotImplementedError

    def store_transition(self, *args):
        self.memory.update(Transition(*args))

    def save(self, step, logs_path):
        save_model(self.Q_network, step, logs_path, "Q", self.MAX_MODEL)
        if not self.saved:
            config_path = os.path.join(logs_path, "CONFIG.pkl")
            pickle.dump(self.CONFIG, open(config_path, "wb"))
            self.saved = True

    def restore(self, step, logs_path, verbose=True):
        logs_path = os.path.join(logs_path, "model", "Q-{}.pth".format(step))
        self.Q_network.load_state_dict(
            torch.load(logs_path, map_location=self.device)
        )
        self.target_network.load_state_dict(
            torch.load(logs_path, map_location=self.device)
        )
        if verbose:
            print("  => Restore {}".format(logs_path))

    def unpack_batch(self, batch):
        """
        unpack_batch: decompose batch into different variables.

        Args:
            batch (object): Transition of batch-arrays.

        Returns:
            tuple of torch.Tensor.
        """
        # `non_final_mask` is used for environments that have next state to be
        # None
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.s_)), dtype=torch.bool
        ).to(self.device)
        non_final_state_nxt = torch.FloatTensor([
            s for s in batch.s_ if s is not None
        ]).to(self.device)
        state = torch.FloatTensor(batch.s).to(self.device)
        action = torch.LongTensor(batch.a).to(self.device).view(-1, 1)
        reward = torch.FloatTensor(batch.r).to(self.device)

        g_x = torch.FloatTensor([info["g_x"] for info in batch.info])
        g_x = g_x.to(self.device).view(-1)

        l_x = torch.FloatTensor([info["l_x"] for info in batch.info])
        l_x = l_x.to(self.device).view(-1)

        return (
            non_final_mask, non_final_state_nxt, state, action, reward, g_x,
            l_x
        )
