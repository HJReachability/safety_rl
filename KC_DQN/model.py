# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )

import torch
import torch.nn as nn             

class modelTanhTwo(nn.Module):
    def __init__(self, state_num, action_num):
        super(modelTanhTwo, self).__init__()
        # nn.ReLU() nn.Tanh()
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=state_num, out_features=100),
            nn.Tanh())
        self.final = nn.Linear(100, action_num)
        self._initialize_weights()
        print("Using two-layer NN architecture with Tanh act.")

    def forward(self, x):
        out1 = self.fc1(x)
        a = self.final(out1)
        return a

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 1)
                m.bias.data.zero_()


class modelTanhThree(nn.Module):
    def __init__(self, state_num, action_num):
        super(modelTanhThree, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=state_num, out_features=100),
            nn.Tanh())
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=100, out_features=100),
            nn.Tanh())
        self.final = nn.Linear(100, action_num)
        self._initialize_weights()
        print("Using three-layer NN architecture with Tanh act.")

    def forward(self, x):
        out1 = self.fc1(x)
        out2 = self.fc2(out1)
        a = self.final(out2)
        return a

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 1)
                m.bias.data.zero_()


class modelSinTwo(nn.Module):
    def __init__(self, state_num, action_num):
        super(modelSinTwo, self).__init__()
        self.fc1 = nn.Linear(in_features=state_num, out_features=100)
        self.final = nn.Linear(100, action_num)
        self._initialize_weights()
        print("Using two-layer NN architecture with Sin act.")

    def forward(self, x):
        out1 = self.fc1(x)
        out1 = torch.sin(out1)
        a = self.final(out1)
        return a

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 1)
                m.bias.data.zero_()


class modelSinThree(nn.Module):
    def __init__(self, state_num, action_num):
        super(modelSinThree, self).__init__()
        self.fc1 = nn.Linear(in_features=state_num, out_features=100)
        self.fc2 = nn.Linear(in_features=100, out_features=100)
        self.final = nn.Linear(100, action_num)
        self._initialize_weights()
        print("Using three-layer NN architecture with Sin act.")

    def forward(self, x):
        out1 = self.fc1(x)
        out1 = torch.sin(out1)
        out2 = self.fc2(out1)
        out2 = torch.sin(out2)
        a = self.final(out2)
        return a

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 1)
                m.bias.data.zero_()


class modelSinFour(nn.Module):
    def __init__(self, state_num, action_num):
        super(modelSinFour, self).__init__()
        self.fc1 = nn.Linear(in_features=state_num, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=512)
        self.final = nn.Linear(512, action_num)
        self._initialize_weights()
        print("Using four-layer NN architecture with Sin act.")

    def forward(self, x):
        out1 = self.fc1(x)
        out1 = torch.sin(out1)
        out2 = self.fc2(out1)
        out2 = torch.sin(out2)
        out3 = self.fc2(out2)
        out3 = torch.sin(out3)
        a = self.final(out3)
        return a

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 1)
                m.bias.data.zero_()