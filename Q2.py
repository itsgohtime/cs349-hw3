import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class Q2_Net(nn.Module):

    def __init__(self):
        super(Q2_Net, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(int(28 * 28 / 2), 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10))

    def softmax(self, x):
        exp_sum = 0
        for i in range(len(x)):
            exp_sum += torch.exp(x[i])

        for i in range(len(x)):
            out = torch.exp(x[i])/exp_sum
            x[i] = out
        return x
    
    def forward(self, x):
        x = self.layers(x)
        return x
    