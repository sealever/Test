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
import importlib
import distutils.version

warnings.filterwarnings('ignore')


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=13, out_features=6),
            nn.Sigmoid(),
            nn.Linear(in_features=6, out_features=3),
            nn.Sigmoid(),
            nn.Linear(in_features=3, out_features=1)
        )

    def forward(self, x):
        return self.model(x)


class NumpyDataset(Dataset):
    def __init__(self, x, y):
        super(NumpyDataset, self).__init__()
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


def fetch_dataloader(batch_size):
    # 加载数据 + 特征工程
    path = 'G:/AI-study/class/datas/boston_housing.data'
    data = pd.read_csv(path, header=None, sep='\s+')
    X = data.iloc[:, :-1].values.astype('float32')
    Y = data.iloc[:, -1].values.reshape(-1, 1).astype('float32')
    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2, random_state=1)
    x_scaler = StandardScaler()
    train_x = x_scaler.fit_transform(train_x)
    test_x = x_scaler.fit_transform(test_x)
    print(f'训练数据shape：{train_x.shape} - {train_y.shape}')
    print(f'测试数据shape：{test_x.shape} - {test_y.shape}')

    # 构建Dataset 对象
    train_dataset = NumpyDataset(x=train_x, y=train_y)
    test_dataset = NumpyDataset(x=test_x, y=test_y)

    # 构建数据遍历器
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=None
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=None
    )
    return train_dataloader, test_dataloader, test_x, test_y


def save_model(path, net, epoch, train_batch, test_batch):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    torch.save({
        'net': net,
        'epoch': epoch,
        'train_batch': train_batch,
        'test_batch': test_batch
    }, path)


def training(restore_path=None):
    root_dir = '../output/02'
    total_epoch = 1000
    # 数据加载
    train_dataloader, test_dataloader, test_x, test_y = fetch_dataloader(16)
    # 模型对象的构建
    net = Network()
    loss = nn.MSELoss()
    opt = optim.SGD(params=net.parameters(), lr=0.01)
    # 模型恢复
    # restore_path = f'{root_dir}/net_{i}.pkl' # 加载第i批次的模型数据
    if(restore_path is not None) and os.path.exists(restore_path):
        original_net = torch.load(restore_path, map_location='cpu')
        net.load_state_dict(state_dict=original_net['net'].state_dict())
        train_batch = original_net['train_batch']
        test_batch = original_net['test_batch']
        start_epoch = original_net['epoch'] + 1
        total_epoch = total_epoch + start_epoch
    else:
        train_batch = 0
        test_batch = 0
        start_epoch = 0
    # 2.1. 运行状态可视化 -- 一般写在网络结构定义之后
    # 解决tensorflow框架来实现可视化，首先安装tensorflow框架：pip install tensorflow
    # 命令行执行如下命令: tensorboard --logdir + 路径
    writer = SummaryWriter(log_dir='../output/01/summary01')
    writer.add_graph(net, input_to_model=torch.rand(1, 13))

    # 模型训练
    for i in range(start_epoch, total_epoch):
        net.train()
        train_loss = []
        for _x, _y in train_dataloader:
            # 前向过程
            pre_y = net(_x)
            _loss = loss(pre_y, _y)

            # 反向过程
            opt.zero_grad()
            _loss.backward()
            opt.step()
            train_batch += 1
            print(f'train-epoch:{i}, batch:{train_batch}, loss:{_loss}')
            writer.add_scalar('train_batch_loss', _loss.item(), global_step=train_batch)
            train_loss.append(_loss.item())

        net.eval()
        test_loss = []
        with torch.no_grad():
            for _x, _y in test_dataloader:
                pre_y = net(_x)
                _loss = loss(pre_y, _y)
                test_batch += 1
                print(f'test-epoch:{i}, batch:{test_batch}, loss:{_loss}')
                writer.add_scalar('test_batch_loss', _loss.item(), global_step=test_batch)
                test_loss.append(_loss.item())

        # 可视化
        writer.add_histogram('w1', net.model[0].weight, global_step=i)
        writer.add_histogram('b1', net.model[0].bias, global_step=i)
        writer.add_histogram('w3', net.model[4].weight, global_step=i)
        writer.add_histogram('b3', net.model[4].weight, global_step=i)
        # writer.add_scalar('loss', {'train': np.mean(np.array(train_loss)), 'test': np.mean(np.array(test_loss))}, global_step=i)

        # if i % 100 == 0:
        #     save_model(
        #         f'{root_dir}/net_{i}.pkl',
        #         net, i, train_batch, test_batch
        #     )

    writer.close()

    net.eval()
    with torch.no_grad():
        pre_test_y = net(torch.from_numpy(test_x))
        pre_test_loss = loss(pre_test_y, torch.from_numpy(test_y))
        print(np.hstack([pre_test_y.detach().numpy(), test_y]))
        print(pre_test_loss.item())

    save_model(
        f'{root_dir}/net_200.pkl',
        net, 200, train_batch, test_batch
        )


def t1():
    net = Network()
    _x = torch.rand(8, 13)
    _r = net(_x)
    print(_r)


if __name__ == '__main__':
    training()
    # t1()
