# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )

import torch
import torch.nn as nn
from torch.distributions import Normal, Uniform
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
    def __init__(self, dimList, actType='Tanh', output_activation=nn.Identity, verbose=False):
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
                self.moduleList.append(output_activation())
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
        # self._initialize_weights()


    def forward(self, x):
        for m in self.moduleList:
            x = m(x)
        return x


    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Linear):
    #             m.weight.data.normal_(0, 0.1)
    #             m.bias.data.zero_()


# TODO == Twinned Q-Network ==
class TwinnedQNetwork(nn.Module):
    def __init__(self, dimList, actType='Tanh', device='cpu'):
        super(TwinnedQNetwork, self).__init__()

        self.Q1 = model(dimList, actType, verbose=True).to(device)
        self.Q2 = model(dimList, actType, verbose=False).to(device)

        if device == torch.device('cuda'):
            self.Q1.cuda()
            self.Q2.cuda()
        self.device = device

    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=-1).to(self.device)
        q1 = self.Q1(x)
        q2 = self.Q2(x)
        return q1, q2


# TODO == Policy (Actor) Model ==
class GaussianPolicy(nn.Module):

    def __init__(self, dimList, actionSpace, actType='Tanh', device='cpu'):
        super(GaussianPolicy, self).__init__()
        self.device = device
        self.mean = model(dimList, actType, verbose=True).to(device)
        self.log_std = model(dimList, actType, verbose=True).to(device)

        self.actionSpace = actionSpace
        self.a_max = self.actionSpace.high[0]
        self.a_min = self.actionSpace.low[0]  
        self.scale = (self.a_max - self.a_min) / 2.0
        self.bias = (self.a_max + self.a_min) / 2.0

        self.LOG_STD_MAX = -1
        self.LOG_STD_MIN = -10
        self.log_scale = (self.LOG_STD_MAX - self.LOG_STD_MIN) / 2.0
        self.log_bias = (self.LOG_STD_MAX + self.LOG_STD_MIN) / 2.0
        self.eps = 1e-8
        # Action Scale and Bias
        # if actionSpace is None:
        #     self.actionScale = torch.tensor(1.)
        #     self.actionBias = torch.tensor(0.)
        # else:
        #     self.actionScale = torch.FloatTensor(
        #         (actionSpace.high - actionSpace.low) / 2.).to(device)
        #     self.actionBias = torch.FloatTensor(
        #         (actionSpace.high + actionSpace.low) / 2.).to(device)


    def forward(self, state):
        stateTensor = state.to(self.device)
        mean = self.mean(stateTensor)
        log_std = self.log_std(stateTensor)
        # log_std = torch.clamp(log_std, min=self.LOG_STD_MIN, max=self.LOG_STD_MAX)
        log_std = torch.tanh(log_std) * self.log_scale + self.log_bias
        return mean, log_std


    def sample(self, state, deterministic=False):
        stateTensor = state.to(self.device)
        mean, log_std = self.forward(stateTensor)
        if deterministic:
            return torch.tanh(mean) * self.scale + self.bias
        std = torch.exp(log_std)
        normalRV = Normal(mean, std)

        x = normalRV.rsample()  # reparameterization trick (mean + std * N(0,1))
        y = torch.tanh(x)   # constrain the output to be within [-1, 1]

        action = y * self.scale + self.bias
        log_prob = normalRV.log_prob(x)

        # Get the correct probability: x -> a, a = c * y + b, y = tanh x
        # followed by: p(a) = p(x) x |det(da/dx)|^-1
        # log p(a) = log p(x) - log |det(da/dx)|
        # log |det(da/dx)| = sum log (d a_i / d x_i)
        # d a_i / d x_i = c * ( 1 - y_i^2 )
        # TODO(vrubies): Understand this!
        log_prob -= torch.log(self.scale * (1 - y.pow(2)) + self.eps).sum(-1, keepdim=True)
        # log_prob -= (2*(np.log(2) - x - F.softplus(-2*x)))
        # if log_prob.dim() > 1:
        #     log_prob = log_prob.sum(1, keepdim=True)
        # else:
        #     log_prob = log_prob.sum()
        # mean = torch.tanh(mean) * self.scale + self.bias
        return action, log_prob

class DeterministicPolicy(nn.Module):
    def __init__(self, dimList, actionSpace, actType='Tanh', device='cpu',
        noiseStd=0.1, noiseClamp=0.5):
        super(DeterministicPolicy, self).__init__()
        self.device = device
        self.mean = model(dimList, actType, output_activation=nn.Tanh, verbose=True).to(device)
        self.noise = Normal(0., noiseStd)
        self.noiseClamp = noiseClamp
        self.actionSpace = actionSpace
        self.noiseStd = noiseStd

        self.a_max = self.actionSpace.high[0]
        self.a_min = self.actionSpace.low[0]  
        self.scale = (self.a_max - self.a_min) / 2.0
        self.bias = (self.a_max + self.a_min) / 2.0
        # action rescaling
        # if actionSpace is None:
        #     self.actionScale = 1.
        #     self.actionBias = 0.
        # else:
        #     self.actionScale = torch.FloatTensor(
        #         (actionSpace.high - actionSpace.low) / 2.).to(device)
        #     self.actionBias = torch.FloatTensor(
        #         (actionSpace.high + actionSpace.low) / 2.).to(device)


    def forward(self, state):
        stateTensor = state.to(self.device)
        mean = self.mean(stateTensor)
        return mean * self.scale + self.bias


    def sample(self, state):
        stateTensor = state.to(self.device)
        mean = self.forward(stateTensor)
        noise = self.noise.sample().to(self.device)
        noise_clipped = torch.randn_like(mean) * self.noiseStd
        noise_clipped = torch.clamp(noise_clipped, -self.noiseClamp, self.noiseClamp)

        # Action.
        action = mean + noise
        action = torch.clamp(action, self.a_min, self.a_max)

        # Target action.
        action_target = mean + noise_clipped
        action_target = torch.clamp(action_target, self.a_min, self.a_max)

        return action.numpy(), action_target


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