import torch
import torch.nn as nn
from torch.onnx import TrainingMode
import torch.nn.functional as F


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        """
        全局平均池化
        :param x: [N,C,H,W]
        :return: [N,C,1,1]
        """
        return torch.mean(x, dim=(2, 3), keepdim=True)


class SeModule(nn.Module):
    def __init__(self, in_channels, r=16):
        super(SeModule, self).__init__()
        self.avg = GlobalAvgPool2d()
        self.fc1 = nn.Conv2d(in_channels, in_channels // r, kernel_size=(1, 1), stride=(1, 1))
        self.fc2 = nn.Conv2d(in_channels // r, in_channels, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        alpha = self.avg(x)
        alpha = F.relu(self.fc1(alpha))
        alpha = torch.sigmoid(self.fc2(alpha))
        return x * alpha


class SeModuleV2(nn.Module):
    def __init__(self, in_channels, r=16):
        super(SeModuleV2, self).__init__()
        self.fc1 = nn.Conv2d(in_channels, in_channels // r, kernel_size=(1, 1), stride=(1, 1))
        self.fc2 = nn.Conv2d(in_channels // r, in_channels, kernel_size=(1, 1), stride=(1, 1))
        self.pool = nn.MaxPool2d(5, 1, padding=2)
        self.bias = nn.Parameter(torch.zeros([1, in_channels, 1, 1]))

    def forward(self, x):
        alpha = F.relu(self.fc1(x))
        alpha = torch.sigmoid(self.fc2(alpha))
        alpha = self.pool(-1 * alpha) * -1
        alpha = F.relu(alpha * F.tanh(self.bias)) / 2
        return x * alpha


class BasicConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, size, stride, padding):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, size, stride, padding)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))


class Inception(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Inception, self).__init__()
        """
        :param in_channels: 输入通道数目, eg: 192
        :param out_channels: 各个分支的输出通道数目， eg: [[64], [96,128], [16,32], [32]]
        """
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channels, out_channels[0][0], 1, 1, 0)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, out_channels[1][0], 1, 1, 0),
            BasicConv2d(out_channels[1][0], out_channels[1][1], 3, 1, 1)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, out_channels[2][0], 1, 1, 0),
            BasicConv2d(out_channels[2][0], out_channels[2][1], 5, 1, 2)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, 1, padding=1),
            BasicConv2d(in_channels, out_channels[3][0], 1, 1, 0)
        )

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        x = torch.concat([x1, x2, x3, x4], dim=1)
        return x


class SeInception(nn.Module):
    def __init__(self, in_channels, out_channels, r=16):
        super(SeInception, self).__init__()
        self.block = Inception(in_channels, out_channels)
        se_in_channel = int(sum([i[-1] for i in out_channels]))
        self.se_block = SeModule(se_in_channel, r)

    def forward(self, x):
        x = self.block(x)
        x = self.se_block(x)
        return x

class SeInceptionV2(nn.Module):
    def __init__(self, in_channels, out_channels, r=16):
        super(SeInceptionV2, self).__init__()
        self.block = Inception(in_channels, out_channels)
        se_in_channel = int(sum([i[-1] for i in out_channels]))
        self.se_block = SeModuleV2(se_in_channel, r)

    def forward(self, x):
        x = self.block(x)
        x = self.se_block(x)
        return  x

class GoogleNet(nn.Module):
    def __init__(self, num_class, add_stage=False):
        super(GoogleNet, self).__init__()
        _Inception = SeInceptionV2
        self.step1 = nn.Sequential(
            BasicConv2d(3, 64, 7, 2, 3),
            nn.MaxPool2d(3, 2, 1),
            BasicConv2d(64, 64, 1, 1, 0),
            BasicConv2d(64, 192, 3, 1, 1),
            nn.MaxPool2d(3, 2, 1),
            _Inception(192, [[64], [96, 128], [16, 32], [32]]),
            _Inception(256, [[128], [128, 192], [32, 96], [64]]),
            nn.MaxPool2d(3, 2, 1),
            _Inception(480, [[192], [96, 208], [16, 48], [64]])
        )
        self.step2 = nn.Sequential(
            _Inception(512, [[160], [112, 224], [24, 64], [64]]),
            _Inception(512, [[128], [128, 256], [24, 64], [64]]),
            _Inception(512, [[112], [144, 288], [32, 64], [64]])
        )
        self.step3 = nn.Sequential(
            _Inception(528, [[256], [160, 320], [32, 128], [128]]),  # inception4e
            nn.MaxPool2d(3, 2, padding=1),
            _Inception(832, [[256], [160, 320], [32, 128], [128]]),  # inception5a
            _Inception(832, [[384], [192, 384], [48, 128], [128]]),  # inception5b
            GlobalAvgPool2d()
        )
        self.classify = nn.Conv2d(1024, num_class, kernel_size=(1, 1), stride=(1, 1), padding=0)
        if add_stage:
            self.stage1 = nn.Sequential(
                nn.MaxPool2d(5, 3, padding=0),
                nn.Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1), padding=0),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(output_size=(2, 2)),
                nn.Flatten(1),
                nn.Linear(4096, 2048),
                nn.Dropout(p=0.4),
                nn.ReLU(),
                nn.Linear(2048, num_class)
            )
            self.stage2 = nn.Sequential(
                nn.MaxPool2d(5, 3, padding=0),
                nn.Conv2d(528, 1024, kernel_size=(1, 1), stride=(1, 1), padding=0),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(output_size=(2, 2)),
                nn.Flatten(1),
                nn.Linear(4096, 2048),
                nn.Dropout(p=0.4),
                nn.ReLU(),
                nn.Linear(2048, num_class)
            )
        else:
            self.stage1 = None
            self.stage2 = None

    def forward(self, x):
        z1 = self.step1(x)
        z2 = self.step2(z1)
        z3 = self.step3(z2)
        # 三个决策分支的输出
        # scores3 = self.classify(z3)[:, :, 0, 0]  # [N,1024,1,1] -> [N,num_classes,1,1] -> [N,num_classes]
        scores3 = torch.squeeze(self.classify(z3))  # [N,1024,1,1] -> [N,num_classes,1,1] -> [N,num_classes]
        if self.stage1 is not None:
            scores1 = self.stage1(z1)
            scores2 = self.stage2(z2)
            return scores1, scores2, scores3
        else:
            return scores3


def t1():
    net = GoogleNet(num_class=4, add_stage=True)
    loss_fn = nn.CrossEntropyLoss()
    x = torch.randn(2, 3, 224, 224)
    y = torch.tensor([0, 3], dtype=torch.long)
    r1, r2, r3 = net(x)
    loss1 = loss_fn(r1, y)
    loss2 = loss_fn(r2, y)
    loss3 = loss_fn(r3, y)
    loss = loss1 + loss2 + loss3
    print(r1)
    print(r2)
    print(r3)
    print(r3.shape)
    print(loss)

    # NOTE: 这里只是为了让大家可视化看结构，实际情况中，如果转换为pt或者onnx的时候，记住一定需要将aux分支删除
    traced_script_module = torch.jit.trace(net.eval(), x)
    traced_script_module.save('../output/models/googlenet_SeNetV2_aux.pt')

    # 模型持久化
    torch.save(net, '../output/models/googlenet_SeNetV2.pkl')


def t2():
    # 参数加载
    net1 = torch.load('../output/models/googlenet_SeNetV2.pkl', map_location='cpu')

    net2 = GoogleNet(num_class=4, add_stage=False)
    # missing_keys: 表示net2中有部分参数没有恢复
    # unexpected_keys: 表示net2中没有这部分参数，但是入参的字典中传入了该参数
    # net1.state_dict(): 返回的是一个dict，key是参数的名称字符串，value是参数tensor对象
    missing_keys, unexpected_keys = net2.load_state_dict(net1.state_dict(), strict=False)
    if len(missing_keys) > 0:
        raise ValueError(f"网络有部分参数没有恢复:{missing_keys}")
    print(unexpected_keys)
    x = torch.randn(2, 3, 224, 224)  # 模拟的输入图像原始数据
    traced_script_module = torch.jit.trace(net2.eval(), x)
    traced_script_module.save('../output/models/googlenet_SeNetV2.pt')

    torch.onnx.export(
        model=net2.eval().cpu(),  # 给定模型对象
        args=x,  # 给定模型forward的输出参数
        f='../output/models/googlenet_SeNet_dynamicV2.onnx',  # 输出文件名称
        training=TrainingMode.EVAL,  # 训练还是eval阶段
        do_constant_folding=True,
        input_names=['images'],  # 给定输入的tensor名称列表
        output_names=['scores'],  # 给定输出的tensor名称列表
        opset_version=12,
        dynamic_axes={
            'images': {
                0: 'n',
                2: 'h',
                3: 'w'
            },
            'scores': {
                0: 'n'
            }
        }  # 给定是否是动态结构
    )


if __name__ == '__main__':
    t1()
    # t2()
