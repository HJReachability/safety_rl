"""
Please contact the author(s) of this library if you have any questions.
Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )

This module implements the parent class and the Transition object for deep
q-netowrk based reinforcement learning algorithms.

This file is based on Adam Paszke's implementation of Deep Q-network,
available at:

https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
"""
import torch
import torch.optim as optim
import abc

from collections import namedtuple
import os
import pickle

from .model import StepLRMargin, StepResetLR
from .ReplayMemory import ReplayMemory
from .utils import soft_update, save_model

Transition = namedtuple("Transition", ["s", "a", "r", "s_", "info"])


class DDQN(abc.ABC):
  """
  The parent class for DDQNSingle and DDQNPursuitEvasion. It implements the
  basic utils functions and defines abstract functions to be implemented in
  the child class.
  """

  def __init__(self, CONFIG):
    """init DDQN with configuration file.

    Args:
        CONFIG (object): a class object containing all the hyper-parameters
            for the learning algorithm, such as exploration-exploitation
            tradeoff, learning rate, discount factor, neural network
            architecture, update methods.
    """
    self.CONFIG = CONFIG
    self.saved = False
    self.memory = ReplayMemory(CONFIG.MEMORY_CAPACITY)

    # == PARAM ==
    # Exploration-exploitation tradeoff.
    # Decaying periodically and reset to the initial value at lower
    # frequency
    self.EpsilonScheduler = StepResetLR(
        initValue=CONFIG.EPSILON,
        period=CONFIG.EPS_PERIOD,
        resetPeriod=CONFIG.EPS_RESET_PERIOD,
        decay=CONFIG.EPS_DECAY,
        endValue=CONFIG.EPS_END,
    )
    self.EPSILON = self.EpsilonScheduler.get_variable()

    # Learning rate of updating the Q-network
    self.LR_C = CONFIG.LR_C
    self.LR_C_PERIOD = CONFIG.LR_C_PERIOD
    self.LR_C_DECAY = CONFIG.LR_C_DECAY
    self.LR_C_END = CONFIG.LR_C_END

    # Neural network related : batch size, maximal number of NNs stored
    self.BATCH_SIZE = CONFIG.BATCH_SIZE
    self.MAX_MODEL = CONFIG.MAX_MODEL
    self.device = CONFIG.DEVICE

    # Discount factor: anneal to one
    self.GammaScheduler = StepLRMargin(
        initValue=CONFIG.GAMMA,
        period=CONFIG.GAMMA_PERIOD,
        decay=CONFIG.GAMMA_DECAY,
        endValue=CONFIG.GAMMA_END,
        goalValue=1.0,
    )
    self.GAMMA = self.GammaScheduler.get_variable()

    # Target network update: also check `update_target_network()`
    self.double_network = CONFIG.DOUBLE  # bool: double DQN if True
    self.SOFT_UPDATE = CONFIG.SOFT_UPDATE  # bool, use soft_update if True
    self.TAU = CONFIG.TAU  # float, soft-update coefficient
    self.HARD_UPDATE = CONFIG.HARD_UPDATE  # int, period for hard_update

  @abc.abstractmethod
  def build_network(self):
    """
    Should be implemented in child class. And, within this function, you
    should implement self.Q_network (define the neural network
    architecture).
    """
    raise NotImplementedError

  def build_optimizer(self):
    """
    Build optimizer for the Q_network and construct a scheduler for
    learning rate and reset counter for updates.
    """
    self.optimizer = torch.optim.AdamW(
        self.Q_network.parameters(), lr=self.LR_C, weight_decay=1e-3
    )
    self.scheduler = optim.lr_scheduler.StepLR(
        self.optimizer, step_size=self.LR_C_PERIOD, gamma=self.LR_C_DECAY
    )
    self.max_grad_norm = 1
    self.cntUpdate = 0

  @abc.abstractmethod
  def update(self):
    """
    Should be implemented in child class. Implement how to update the
    Q_network.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def initBuffer(self, env):
    """
    Should be implemented in child class. Implement how to initialize the
    replay buffer (memory).
    """
    raise NotImplementedError

  @abc.abstractmethod
  def initQ(self):
    """
    Should be implemented in child class. Implement how to initialize the
    Q_network.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def learn(self):
    """
    Should be implemented in child class. Implement the learning algorithm,
    it may call initBuffer, initQ and update.
    """
    raise NotImplementedError

  def update_target_network(self):
    """
    Update the target network periodically.
    """
    if self.SOFT_UPDATE:
      # Soft Replace: update the target_network right after every
      # gradient update of the Q-network by
      # target = TAU * Q_network + (1-TAU) * target
      soft_update(self.target_network, self.Q_network, self.TAU)
    elif self.cntUpdate % self.HARD_UPDATE == 0:
      # Hard Replace: copy the Q-network into the target nework every
      # HARD_UPDATE updates
      self.target_network.load_state_dict(self.Q_network.state_dict())

  def updateHyperParam(self):
    """
    Update the hypewr-parameters, such as learning rate, discount factor
    (GAMMA) and exploration-exploitation tradeoff (EPSILON)
    """
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

  @abc.abstractmethod
  def select_action(self):
    """
    Should be implemented in child class. Implement action selection.
    """
    raise NotImplementedError

  def store_transition(self, *args):
    """
    Store the transition into the replay buffer (memory).
    """
    self.memory.update(Transition(*args))

  def save(self, step, logs_path):
    """
    Save the model weights and save the configuration file in first call.

    Args:
        step (int): the number of updates so far.
        logs_path (str): the folder path to save the model.
    """
    save_model(self.Q_network, step, logs_path, "Q", self.MAX_MODEL)
    if not self.saved:
      config_path = os.path.join(logs_path, "CONFIG.pkl")
      pickle.dump(self.CONFIG, open(config_path, "wb"))
      self.saved = True

  def restore(self, step, logs_path, verbose=True):
    """
    Restore the model weights from the given model path.

    Args:
        step (int): the number of updates of the model.
        logs_path (str): he folder path of the model.
        verbose (bool, optional): print messages if True. Defaults to True.
    """
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
    Decompose the batch into different variables.

    Args:
        batch (object): Transition of batch-arrays.

    Returns:
        A tuple of torch.Tensor objects, extracted from the elements in the
            batch and processed for update().
    """
    # `non_final_mask` is used for environments that have next state to be
    # None.
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
        non_final_mask, non_final_state_nxt, state, action, reward, g_x, l_x
    )
