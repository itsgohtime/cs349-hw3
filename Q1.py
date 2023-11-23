import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class Q1_Net(nn.Module):

    def __init__(self, use_bias):
        super(Q1_Net, self).__init__()
        self.lin1 = nn.Linear(3, 2, bias=use_bias)
        self.sig = nn.Sigmoid()
        self.lin2 = nn.Linear(2, 3, bias=use_bias)
        self.soft = self.softmax


    def forward(self, x):
        x[0] = x[0]/100
        x[1] = x[1]/200
        x[2] = x[2]/100
        x = self.lin1(x)
        x = self.sig(x)
        x = self.lin2(x)
        return x
    
    def softmax(self, x):
        exp_sum = 0
        for i in range(len(x)):
            exp_sum += torch.exp(x[i])

        for i in range(len(x)):
            out = torch.exp(x[i])/exp_sum
            x[i] = out
        return x