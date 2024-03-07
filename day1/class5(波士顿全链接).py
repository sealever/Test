import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn as nn
import torch
from torchvision import datasets
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch.optim as optim
import warnings

warnings.filterwarnings('ignore')


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        # 第一层全连接
        self.w1 = nn.Parameter(torch.empty(13, 5))
        self.b1 = nn.Parameter(torch.empty(5))

        # 第二层全连接
        self.w2 = nn.Parameter(torch.empty(5, 3))
        self.b2 = nn.Parameter(torch.empty(3))

        # 第三层全连接
        self.w3 = nn.Parameter(torch.empty(3, 1))
        self.b3 = nn.Parameter(torch.empty(1))

        # 参数初始化
        nn.init.kaiming_uniform_(self.w1)
        nn.init.kaiming_uniform_(self.w2)
        nn.init.kaiming_uniform_(self.w3)
        nn.init.constant_(self.b1, 0.1)
        nn.init.constant_(self.b2, 0.1)
        nn.init.constant_(self.b3, 0.1)

    def forward(self, x):
        # 第一层隐层
        x = x @ self.w1 + self.b1
        x = torch.sigmoid(x)
        # 第二层隐层
        x = x @ self.w2 + self.b2
        # 第三层隐层
        x = x @ self.w3 + self.b3
        return x


def training():
    path = 'G:/AI-study/class/datas/boston_housing.data'
    data = pd.read_csv(path, header=None, sep='\s+')
    X = data.iloc[:, :-1].values.astype('float32')
    Y = data.iloc[:, -1].values.reshape(-1, 1).astype('float32')
    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.1, random_state=1)
    x_scaler = StandardScaler()
    y_scaler = MinMaxScaler()
    train_x = x_scaler.fit_transform(train_x)
    test_x = x_scaler.fit_transform(test_x)
    train_y = y_scaler.fit_transform(train_y)
    test_y = y_scaler.fit_transform(test_y)
    print(f'训练数据shape：{train_x.shape} - {train_y.shape}')
    print(f'测试数据shape：{test_x.shape} - {test_y.shape}')

    # 模型对象的构建
    net = Network()
    loss = nn.MSELoss()
    opt = optim.SGD(params=net.parameters(), lr=0.01)

    # 模型训练
    net.train()
    epoch = 100
    size = 50
    batch = len(train_x) // size
    for i in range(epoch):
        rnd = np.random.permutation(len(train_x))
        for n in range(batch):
            # 前向过程
            id = rnd[batch * size:(batch + 1) * size]
            _x = torch.from_numpy(train_x[id])
            _y = torch.from_numpy(train_y[id])
            pre_y = net(_x)
            _loss = loss(pre_y, _y)

            # 反向过程
            opt.zero_grad()
            _loss.backward()
            opt.step()
            print(f'epoch:{i}, batch:{n}, loss:{_loss}')

    net.eval()
    with torch.no_grad():
        pre_test_y = net(torch.from_numpy(test_x))
        pre_test_loss = loss(pre_test_y, torch.from_numpy(test_y))
        print(np.hstack([pre_test_y.detach().numpy(), test_y]))
        print(pre_test_loss.item())

    path = '../output/net01.pkl'
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    torch.save({
        'net': net,
        'epoch': epoch,
        'lr': 0.01,
        'opt': opt,
    }, path)
    obj = torch.load(path, map_location='cpu')
    print(obj)


def t1():
    net = Network()
    _x = torch.rand(8, 13)
    _r = net(_x)
    print(_r)


if __name__ == '__main__':
    # training()
    t1()
