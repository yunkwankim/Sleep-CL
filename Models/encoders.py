
from typing import Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from models.utils import *


# ############### CNN ####################
class CNNEncoder(nn.Module):
    """
    Modified from  https://github.com/emadeldeen24/AdaTime/blob/adatime_v2/models/models.py
    """
    def __init__(self, input_channels, feature_dims, norm='BN', dropout=0, hidden_dims=64, depth=3):
        super(CNNEncoder, self).__init__()

        self.feature_dims = feature_dims
        self.stacks = nn.ModuleList()
        self.in_channels = [2**(i-1) * hidden_dims if i != 0 else input_channels for i in range(depth)]

        for i in range(depth):
            if i == 0:
                stack = CNNEncoder.convStack(self.in_channels[i], hidden_dims, norm, dropout)
            elif i == depth-1:
                stack = CNNEncoder.convStack(self.in_channels[i], feature_dims, norm, dropout)
            else:
                stack = CNNEncoder.convStack(self.in_channels[i], 2 * self.in_channels[i], norm, dropout)
            self.stacks.append(stack)

    @staticmethod
    def convStack(in_channels, out_channel, norm, dropout):
        NormLayer = get_norm_layer(norm)
        stack = nn.Sequential(
            nn.Conv1d(in_channels, out_channel, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            NormLayer(out_channel),  # Input: (N, C, L)
            nn.MaxPool1d(kernel_size=2, stride=2),  # After each stack, the length will be halved.
            nn.Dropout(dropout),
        )

        return stack

    def forward(self,  x, pooling=True):
        # Input x is (N, L_in, C_in), need to transform it to (N, C_in, L_in)
        # x = x.transpose(1, 2)

        for stack in self.stacks:
            x = stack(x)

        if pooling:
            return x.mean(dim=-1)  # global average pooling, (N, C_out)
        else:
            return x
