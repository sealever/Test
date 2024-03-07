import torch
import torch.nn as nn
from torch.onnx import TrainingMode
import torch.nn.functional as F
import warnings

warnings.filterwarnings('ignore')

class VggBlock(nn.Module):
    def __init__(self, in_channel, out_channel, n, change=False):
        super(VggBlock, self).__init__()
        layer = []
        for i in range(n):
            if change and i == n - 1:
                kernel_size = (1, 1)
                padding = 0
            else:
                kernel_size = (3, 3)
                padding = 1
            conv = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=(1, 1), padding=padding),
                nn.ReLU()
            )
            in_channel = out_channel
            layer.append(conv)
        layer.append(nn.MaxPool2d(2, 2))
        self.block = nn.Sequential(*layer)

    def forward(self, x):
        return self.block(x)


class AdaptiveAvgPool2dModule(nn.Module):
    def __init__(self, output_size):
        super(AdaptiveAvgPool2dModule, self).__init__()
        self.k = output_size

    def forward(self, x):
        k = self.k
        n, c, h, w = x.shape
        hk = int(h / k)
        if h % k != 0:
            hk += 1
        wk = int(w / k)
        if w % k != 0:
            wk += 1
        ph = hk * k - h  # 需填充大小
        pw = wk * k - h  # 需填充大小
        x = F.pad(x, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2))
        x = x.reshape(n, c, k, hk, k, wk)
        x = torch.permute(x, dims=(0, 1, 2, 4, 3, 5))
        x = torch.mean(x, dim=(4, 5))
        return x


class VggNet(nn.Module):
    def __init__(self, features, num, input_channel):
        super(VggNet, self).__init__()
        self.features = features
        self.num = num
        self.pooling = AdaptiveAvgPool2dModule(output_size=7)
        self.classify = nn.Sequential(
            nn.Linear(7 * 7 * input_channel, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, num)
        )

    def forward(self, x):
        z = self.features(x)
        z = self.pooling(z)
        z = z.flatten(1)
        return self.classify(z)


class Vgg16Net(nn.Module):
    def __init__(self, num):
        super(Vgg16Net, self).__init__()

        features = nn.Sequential(
            VggBlock(3, 64, 2),
            VggBlock(64, 128, 2),
            VggBlock(128, 256, 3),
            VggBlock(256, 512, 3),
            VggBlock(512, 512, 3),
        )
        self.vgg16 = VggNet(
            features=features,
            num=num,
            input_channel=512
        )

    def forward(self, x):
        return self.vgg16(x)


class Vgg19Net(nn.Module):
    def __init__(self, num):
        super(Vgg19Net, self).__init__()
        features = nn.Sequential(
            VggBlock(3, 64, 2),
            VggBlock(64, 128, 2),
            VggBlock(128, 256, 4),
            VggBlock(256, 512, 4),
            VggBlock(512, 512, 4),
        )
        self.vgg19 = VggNet(
            features=features,
            num=num,
            input_channel=512
        )

    def forward(self, x):
        return self.vgg19(x)


class Vgg16CNet(nn.Module):
    def __init__(self, num):
        super(Vgg16CNet, self).__init__()
        features = nn.Sequential(
            VggBlock(3, 64, 2),
            VggBlock(64, 128, 2),
            VggBlock(128, 256, 3, change=True),
            VggBlock(256, 512, 3, change=True),
            VggBlock(512, 512, 3, change=True),
        )
        self.vgg16c = VggNet(
            features=features,
            num=num,
            input_channel=512
        )

    def forward(self, x):
        return self.vgg16c(x)


class VggLabel(nn.Module):
    def __init__(self, vgg_model):
        super(VggLabel, self).__init__()
        self.vgg = vgg_model
        self.id_name = {
            0: '🐱',
            1: '🐕',
            2: '🐖',
            3: '🐏'
        }

    def forward(self, x):
        scores = self.vgg(x)
        pre = torch.argmax(scores, dim=1)
        pre = pre.detach().numpy()
        res = []
        for i in pre:
            res.append(self.id_name[i])
        return res


if __name__ == '__main__':
    vgg16 = Vgg16Net(4)
    # vgg19 = Vgg19Net(4)
    # vgg16c = Vgg16CNet(4)
    vgg_label = VggLabel(vgg16)
    # print(vgg_label)
    example = torch.rand(6, 3, 244, 244)
    m = vgg_label(example)
    print(m)

    # torch.onnx.export(
    #     model=vgg16.eval().cpu(),
    #     args=example,
    #     f='../output/04/models/vgg16_dynamic.onnx',
    #     training=TrainingMode.EVAL,
    #     do_constant_folding=True,
    #     input_names=['images'],
    #     output_names=['scores'],
    #     opset_version=12,
    #     dynamic_axes={
    #         'images': {
    #             0: 'n',
    #             2: 'h',
    #             3: 'w'
    #         },
    #         'scores': {
    #             0: 'n'
    #         }
    #     }
    # )
    # traced_script_module = torch.jit.trace(vgg16, example)
    # traced_script_module.save('../output/04/models/vgg16.pt')
