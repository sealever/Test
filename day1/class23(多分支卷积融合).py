import torch
import torch.nn as nn
import torch.nn.functional as F


def t0():
    x = torch.rand(4, 9, 24, 24)

    # 分支结构
    conv1x1 = nn.Conv2d(9, 9, kernel_size=(1, 1), stride=(1, 1), padding=0)
    conv3x3 = nn.Conv2d(9, 9, kernel_size=(3, 3), stride=(1, 1), padding=1)
    r1 = conv1x1(x) + conv3x3(x)
    print(r1.shape)

    # 分支合并后的单链路结构
    conv1x1_weight = F.pad(conv1x1.weight.clone(), [1, 1, 1, 1])  # [OC,IC,1,1] -> [OC,IC,3,3]
    conv1x1_bias = conv1x1.bias.clone()
    conv3x3_weight = conv3x3.weight.clone()
    conv3x3_bias = conv3x3.bias.clone()
    conv = nn.Conv2d(9, 9, kernel_size=(3, 3), stride=(1, 1), padding=1).requires_grad_(False)
    conv.weight.copy_(conv3x3_weight + conv1x1_weight)
    conv.bias.copy_(conv3x3_bias + conv1x1_bias)
    r2 = conv(x)
    print(r2.shape)

    r = torch.abs(r1 - r2)
    print(torch.max(r))

if __name__ == '__main__':
    t0()
