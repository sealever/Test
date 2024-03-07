import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from PIL import Image
from torchvision import transforms


class DropBlock2D(nn.Module):
    def __init__(self, p: float = 0.1, block_size: int = 7, inplace: bool = False):
        super(DropBlock2D, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("DropBlock probability has to be between 0 and 1, "
                             "but got {}".format(p))
        if block_size < 1:
            raise ValueError("DropBlock block size必须大于0.")
        if block_size % 2 != 1:
            raise ValueError("当前代码实现的并不是特别完善，要求drop的区域大小必须是奇数")
        self.p = p
        self.inplace = inplace
        self.block_size = block_size

    def forward(self, input: Tensor) -> Tensor:
        if not self.training:
            return input

        N, C, H, W = input.size()
        mask_h = H - self.block_size + 1
        mask_w = W - self.block_size + 1
        gamma = (self.p * H * W) / ((self.block_size ** 2) * mask_h * mask_w)
        mask_shape = (N, C, mask_h, mask_w)
        # bernoulli:伯努利数据产生器，取值只有两种：0或者1；底层每个点会产生一个随机数，随机数小于等于gamma的，对应位置就是1；否则就是0
        mask1 = torch.bernoulli(torch.full(mask_shape, gamma, device=input.device))
        mask2 = F.pad(mask1, [self.block_size // 2] * 4, value=0)  # 当前0表示保留，1表示删除
        mask3 = F.max_pool2d(mask2, (self.block_size, self.block_size), (1, 1), self.block_size // 2)
        mask = 1 - mask3
        normalize_scale = mask.numel() / (10e-6 + mask.sum())

        if self.inplace:
            input.mul_(mask * normalize_scale)
        else:
            input = input * mask * normalize_scale
        return input


def t1():
    feature = torch.rand((2, 3, 10, 10))
    # p: 多大的可能性进行数据删除操作 --> p == drop_prob == 1-keep_prob
    dropout = nn.Dropout(p=0.4)
    dropout_feature = dropout(feature)

    print(dropout_feature)
    dropblock = DropBlock2D(p=0.4)
    dropblock_feature = dropblock(feature)
    print(dropblock_feature)


def t2():
    img = Image.open(r'G:\AI-study\class\datas\MNIST\tietu.png').convert('L')
    ts1 = transforms.ToTensor()
    ts2 = transforms.ToPILImage()
    img = ts1(img)[None]  # [C,H,W] -> [1,C,H,W]
    print(img.shape)

    p = 0.2
    dropout = nn.Dropout(p=p)
    dropblock = DropBlock2D(p=p, block_size=31)

    img0 = ts2(img[0])
    img1 = ts2(dropout(img)[0])
    img2 = ts2(dropblock(img)[0])

    img0.show('img0')
    img1.show('img1')
    img2.show('img2')

if __name__ == '__main__':
    t2()