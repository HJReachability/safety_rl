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
        if verbose:
            print(self.moduleList)

    def forward(self, x):
        for m in self.moduleList:
            x = m(x)
        return x


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