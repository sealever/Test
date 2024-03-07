import torch
import torch.nn as nn
from torch import Tensor


class WSConv2d(nn.Conv2d):
    # LN + conv2d
    def __init__(self, in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',  # TODO: refine this type
                 device=None,
                 dtype=None,
                 eps=1e-8):
        super(WSConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation,
                                       groups, bias, padding_mode, device, dtype)
        self.eps = eps
        self.gama = nn.Parameter(torch.ones([out_channels, 1, 1, 1]))
        self.beta = nn.Parameter(torch.zeros([out_channels, 1, 1, 1]))

    def forward(self, x):
        # 参数标准化
        w = self.weight
        w_mean = torch.mean(w, dim=(1, 2, 3), keepdim=True)  # [OC,1,1,1]
        w_var = torch.var(w, dim=(1, 2, 3), keepdim=True)  # [OC,1,1,1]
        w = (w - w_mean) / torch.sqrt(w_var + self.eps)
        w = w * self.gama + self.beta
        self.weight.data = w
        return super(WSConv2d, self).forward(x)


if __name__ == '__main__':
    conv = WSConv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1)
    x = torch.rand(4, 3, 28, 28)
    r = conv(x)
    print(r.shape)
