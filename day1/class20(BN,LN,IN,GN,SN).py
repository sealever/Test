from pathlib import Path

import torch
import numpy as np
import torch.nn as nn
from typing import Optional

from torch import Tensor


class BN(nn.Module):
    # BN:按每个通道（C）对样本数据进行归一化操作
    def __init__(self, num_class, momentum=0.1, eps=1e-8):
        super(BN, self).__init__()
        self.momentum = momentum
        self.eps = eps
        # register_buffer: 将属性当成parameter进行处理，唯一的区别就是不参与反向传播的梯度求解
        self.register_buffer('running_mean', torch.zeros(1, num_class, 1, 1))
        self.register_buffer('running_var', torch.zeros(1, num_class, 1, 1))
        self.running_var: Optional[Tensor]
        self.running_mean: Optional[Tensor]
        self.gama = nn.Parameter(torch.ones([1, num_class, 1, 1]))
        self.beta = nn.Parameter(torch.zeros([1, num_class, 1, 1]))

    def forward(self, x):
        if self.training:
            _mean = torch.mean(x, dim=(0, 2, 3), keepdim=True)  # [1, c, 1, 1]
            _var = torch.var(x, dim=(0, 2, 3), keepdim=True)  # [1,c,1,1]
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * _mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * _var
        else:
            _mean = self.running_mean
            _var = self.running_var
        z = (x - _mean) / torch.sqrt(_var + self.eps) * self.gama + self.beta
        return z


class LN(nn.Module):
    # LN:按每个批次（N）对样本数据进行归一化操作
    def __init__(self, num_class, eps=1e-8):
        super(LN, self).__init__()
        self.eps = eps
        self.gama = nn.Parameter(torch.ones(1, num_class, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_class, 1, 1))

    def forward(self, x):
        _mean = torch.mean(x, dim=(1, 2, 3), keepdim=True)  # [N, 1, 1, 1]
        _var = torch.var(x, dim=(1, 2, 3), keepdim=True)  # [N, 1, 1, 1]
        z = (x - _mean) / torch.sqrt(_var + self.eps) * self.gama + self.beta
        return z


class IN(nn.Module):
    # IN:按每个features map（HxW）对样本数据进行归一化操作
    def __init__(self, num_class, eps=1e-8):
        super(IN, self).__init__()
        self.eps = eps
        self.gama = nn.Parameter(torch.ones(1, num_class, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_class, 1, 1))

    def forward(self, x):
        _mean = torch.mean(x, dim=(2, 3), keepdim=True)  # [N, C, 1, 1]
        _var = torch.var(x, dim=(2, 3), keepdim=True)  # [N, C, 1, 1]
        z = (x - _mean) / torch.sqrt(_var + self.eps) * self.gama + self.beta
        return z


class GN(nn.Module):
    # GN:通道分组后按组对样本数据进行归一化操作
    def __init__(self, num_class, groups, eps=1e-8):
        super(GN, self).__init__()
        assert num_class % groups == 0, "要求特征数必须整除"
        self.groups = groups
        self._groups = num_class // groups
        self.eps = eps
        self.gama = nn.Parameter(torch.ones(1, num_class, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_class, 1, 1))

    def forward(self, x):
        n, c, h, w = x.shape
        x = x.viem(n, self.groups, self._groups, h, w)
        _mean = torch.mean(x, dim=(2, 3, 4), keepdim=True)  # [N, groups, 1, 1, 1]
        _var = torch.var(x, dim=(2, 3, 4), keepdim=True)  # [N, groups, 1, 1, 1]
        x = (x - _mean) / torch.sqrt(_var + self.eps)
        x = x.view(n, c, h, w)
        z = x * self.gama + self.beta
        return z


class SN(nn.Module):
    # SN:自适配归一化，SN = a * bn + b * in + c * Ln (a, b, c为权重系数)
    def __init__(self, num_class, momentum=0.1, eps=1e-8):
        super(SN, self).__init__()
        self.momentum = momentum
        self.eps = eps
        # register_buffer: 将属性当成parameter进行处理，唯一的区别就是不参与反向传播的梯度求解
        self.register_buffer('running_mean', torch.zeros(1, num_class, 1, 1))
        self.register_buffer('running_var', torch.zeros(1, num_class, 1, 1))
        self.running_var: Optional[Tensor]
        self.running_mean: Optional[Tensor]
        self.gama = nn.Parameter(torch.ones([1, num_class, 1, 1]))
        self.beta = nn.Parameter(torch.zeros([1, num_class, 1, 1]))

        self. w = nn.Parameter(torch.ones([3]))

    def get_bn(self, x):
        if self.training:
            bn_mean = torch.mean(x, dim=(0, 2, 3), keepdim=True)  # [1, c, 1, 1]
            bn_var = torch.var(x, dim=(0, 2, 3), keepdim=True)  # [1,c,1,1]
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * bn_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * bn_var
        else:
            bn_mean = self.running_mean
            bn_var = self.running_var
        return bn_mean, bn_var

    def get_ln(self, x):
        ln_mean = torch.mean(x, dim=(1, 2, 3), keepdim=True)
        ln_var = torch.var(x, dim=(1, 2, 3), keepdim=True)
        return ln_mean, ln_var

    def get_in(self, x):
        in_mean = torch.mean(x, dim=(2,3), keepdim=True)
        in_var = torch.mean(x, dim=(2,3), keepdim=True)
        return in_mean, in_var

    def forward(self, x):
        bn_mean, bn_var = self.get_bn(x)
        ln_mean, ln_var = self.get_ln(x)
        in_mean, in_var = self.get_in(x)
        # 权重
        w = torch.softmax(self.w, dim=0)
        bn_w, ln_w, in_w = w[0], w[1], w[2]
        # 合并
        _mean = bn_w * bn_mean + ln_w * ln_mean + in_w * in_mean
        _var = bn_w * bn_var + ln_w * ln_var + in_w * in_var
        z = (x - _mean) / torch.sqrt(_var + self.eps) * self.gama + self.beta
        return z



if __name__ == '__main__':
    torch.manual_seed(8)
    path_dir = Path("../output/models")
    path_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = SN(num_class=12)
    net.to(device)
    # 训练过程
    net.train()
    x = [torch.randn(8, 12, 32, 32).to(device) for _ in range(10)]
    for i in x:
        net(i)
    print(net.running_mean.view(-1))
    print(net.running_var.view(-1))
    # 推理过程
    net.eval()
    r = net(x[0])
    print(r.shape)

    bn = net.cpu()
    # 模型保存（）
    torch.save(net, str(path_dir / 'sn_pkl'))
    # state_dict : 获取当前模块的所有参数(Parameter + register_buffer)
    torch.save(net.state_dict(), str(path_dir / 'sn_paras.pkl'))
    # pt结构保存
    traced_script_module = torch.jit.trace(bn.eval(), x[0].cpu())
    traced_script_module.save('../output/models/sn.pt')
    # 模型恢复
    sn_model = torch.load(str(path_dir / 'sn_pkl'), map_location='cpu')
    sn_params = torch.load(str(path_dir / 'sn_paras.pkl'), map_location='cpu')
    print(sn_params)

