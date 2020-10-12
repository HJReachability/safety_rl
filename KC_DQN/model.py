# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )

import torch
import torch.nn as nn             

class model(nn.Module):
    def __init__(self, state_num, action_num):
        super(model, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=state_num, out_features=100),
            nn.Tanh())
        self.fc2 = nn.Linear(100, action_num)
        self._initialize_weights()

    def forward(self, x):
        out1 = self.fc1(x)
        a = self.fc2(out1)
        return a
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 1)
                #m.weight.data.zero_()
                m.bias.data.zero_()
