# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )

import torch
import torch.nn as nn
from torch.distributions import Normal
import sys

class Sin(nn.Module):
    """
    Sin: Wraps element-wise `sin` activation as a nn.Module.

    Shape:
        - Input: `(N, *)` where `*` means, any number of additional dimensions
        - Output: `(N, *)`, same shape as the input

    Examples:
        >>> m = Sin()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """
    def __init__(self):
        super().__init__() # init the base class

    def forward(self, input):
        return torch.sin(input) # simply apply already implemented sin


class model(nn.Module):
    """
    model: Constructs a fully-connected neural network with flexible depth, width
        and activation function choices.
    """
    def __init__(self, dimList, actType='Tanh', verbose=False):
        """
        __init__: Initalizes.

        Args:
            dimList (int List): the dimension of each layer.
            actType (str, optional): the type of activation function. Defaults to 'Tanh'.
                Currently supports 'Sin', 'Tanh' and 'ReLU'.
            verbose (bool, optional): print info or not. Defaults to False.
        """
        super(model, self).__init__()

        # Construct module list: if use `Python List`, the modules are not added to
        # computation graph. Instead, we should use `nn.ModuleList()`.
        self.moduleList = nn.ModuleList()
        numLayer = len(dimList)-1
        for idx in range(numLayer):
            i_dim = dimList[idx]
            o_dim = dimList[idx+1]

            self.moduleList.append(nn.Linear(in_features=i_dim, out_features=o_dim))
            if idx == numLayer-1: # final linear layer, no act.
                pass
            else:
                if actType == 'Sin':
                    self.moduleList.append(Sin())
                elif actType == 'Tanh':
                    self.moduleList.append(nn.Tanh())
                elif actType == 'ReLU':
                    self.moduleList.append(nn.ReLU())
                else:
                    raise ValueError('Activation type ({:s}) is not included!'.format(actType))
                # self.moduleList.append(nn.Dropout(p=.5))
        if verbose:
            print(self.moduleList)

        # Initalizes the weight
        self._initialize_weights()


    def forward(self, x):
        for m in self.moduleList:
            x = m(x)
        return x


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.1)
                m.bias.data.zero_()


# TODO == Twinned Q-Network ==
class TwinnedQNetwork(nn.Module):
    def __init__(self, dimList, actType='Tanh', device='cpu'):
        super(TwinnedQNetwork, self).__init__()

        self.Q1 = model(dimList, actType, verbose=True)
        self.Q2 = model(dimList, actType, verbose=False)

        if device == torch.device('cuda'):
            self.Q1.cuda()
            self.Q2.cuda()
        self.device=device

    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=1).to(self.device)
        q1 = self.Q1(x)
        q2 = self.Q2(x)
        return q1, q2


# TODO == Policy (Actor) Model ==
class GaussianPolicy(nn.Module):
    LOG_STD_MAX = 1
    LOG_STD_MIN = -8
    eps = 1e-8

    def __init__(self, dimList, actType='Tanh', device='cpu', action_space=None):
        super(GaussianPolicy, self).__init__()
        self.device = device
        self.mean = model(dimList, actType, verbose=True).to(device)
        self.log_std = model(dimList, actType, verbose=True).to(device)

        # Action Scale and Bias
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.).to(device)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.).to(device)


    def forward(self, state):
        stateTensor = state.to(self.device)
        mean = self.mean(stateTensor)
        log_std = self.log_std(stateTensor)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std


    def sample(self, state):
        stateTensor = state.to(self.device)
        mean, log_std = self.forward(stateTensor)
        std = log_std.exp()
        normalRV = Normal(mean, std)

        x = normalRV.rsample()  # reparameterization trick (mean + std * N(0,1))
        y = torch.tanh(x)   # constrain the output to be within [-1, 1]

        action = y * self.action_scale + self.action_bias
        log_prob = normalRV.log_prob(x)

        # Get the correct probability: x -> y, y = c tanh(x) + b
        # followed by: p(y) = p(x) x |det(dy/dx)|^-1
        log_prob -= torch.log(self.action_scale * (1 - y.pow(2)) + eps)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


class DeterministicPolicy(nn.Module):
    def __init__(self, dimList, actType='Tanh', device='cpu', action_space=None):
        super(DeterministicPolicy, self).__init__()
        self.device = device
        self.mean = model(dimList, actType, verbose=True).to(device)
        self.noise = Normal(0., 0.1)
        self.noiseClamp = 0.25

        # action rescaling
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.).to(device)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.).to(device)


    def forward(self, state):
        stateTensor = state.to(self.device)
        mean = self.mean(stateTensor)
        return mean


    def sample(self, state):
        stateTensor = state.to(self.device)
        mean = self.forward(stateTensor)
        noise = self.noise.sample().to(self.device)
        noise = noise.clamp(-self.noiseClamp, self.noiseClamp)
        action = mean + noise
        action = action.clamp(-1., 1.)
        action = action * self.action_scale + self.action_bias
        mean = mean * self.action_scale + self.action_bias
        return action, torch.tensor(0.), mean


#== Scheduler ==
class _scheduler(object):
    def __init__(self, last_epoch=-1, verbose=False):
        self.cnt = last_epoch
        self.verbose = verbose
        self.variable = None
        self.step()

    def step(self):
        self.cnt += 1
        value = self.get_value()
        self.variable = value

    def get_value(self):
        raise NotImplementedError

    def get_variable(self):
        return self.variable


class StepLR(_scheduler):
    def __init__(self, initValue, period, decay=0.1, endValue=None, last_epoch=-1, verbose=False):
        self.initValue = initValue
        self.period = period
        self.decay = decay
        self.endValue = endValue
        super(StepLR, self).__init__(last_epoch, verbose)

    def get_value(self):
        if self.cnt == -1:
            return self.initValue

        numDecay = int(self.cnt/self.period)
        tmpValue =  self.initValue * (self.decay ** numDecay)
        if self.endValue is not None and tmpValue <= self.endValue:
            return self.endValue
        return tmpValue


class StepLRMargin(_scheduler):
    def __init__(self, initValue, period, goalValue, decay=0.1, endValue=None, last_epoch=-1, verbose=False):
        self.initValue = initValue
        self.period = period
        self.decay = decay
        self.endValue = endValue
        self.goalValue = goalValue
        super(StepLRMargin, self).__init__(last_epoch, verbose)

    def get_value(self):
        if self.cnt == -1:
            return self.initValue

        numDecay = int(self.cnt/self.period)
        tmpValue =  self.goalValue - (self.goalValue-self.initValue) * (self.decay ** numDecay)
        if self.endValue is not None and tmpValue >= self.endValue:
            return self.endValue
        return tmpValue


class StepResetLR(_scheduler):
    def __init__(self, initValue, period, resetPeriod, decay=0.1, endValue=None,
        last_epoch=-1, verbose=False):
        self.initValue = initValue
        self.period = period
        self.decay = decay
        self.endValue = endValue
        self.resetPeriod = resetPeriod
        super(StepResetLR, self).__init__(last_epoch, verbose)

    def get_value(self):
        if self.cnt == -1:
            return self.initValue

        numDecay = int(self.cnt/self.period)
        tmpValue =  self.initValue * (self.decay ** numDecay)
        if self.endValue is not None and tmpValue <= self.endValue:
            return self.endValue
        return tmpValue

    def step(self):
        self.cnt += 1
        value = self.get_value()
        self.variable = value
        if (self.cnt+1) % self.resetPeriod == 0:
            self.cnt = -1