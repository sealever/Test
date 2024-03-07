import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch
from torchvision import datasets
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch.optim as optim
import warnings


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(19, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 3)
        )

    def forward(self, x):
        return self.model(x)


class RegL1loss(nn.Module):
    def __init__(self, lamda, params):
        super(RegL1loss, self).__init__()
        self.lamda = lamda
        self.params = list(params)
        self.n = len(self.params)

    def forward(self):
        ls = 0
        for i in self.params:
            ls += torch.sum(torch.abs(i))
        return self.lamda * ls / self.n


if __name__ == '__main__':
    net = Network()
    print(next(net.named_parameters()))
    loss = nn.MSELoss()
    # _loss = loss()
    reg_l1_loss = RegL1loss(lamda=0.1, params=net.parameters())
    l1 = reg_l1_loss()
    # l1_loss = _loss + l1
    print(l1)
    # print(l1_loss)
    opt = optim.SGD(params=net.parameters(), lr=0.05, weight_decay=0.5)  # weight_decay给定的是L2惩罚项的系数
