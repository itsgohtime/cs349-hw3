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
        self.softmax = nn.Softmax(dim=0)


    def forward(self, x):
        x = self.layers(x)
        return x
    