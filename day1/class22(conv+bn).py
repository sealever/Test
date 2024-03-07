from pathlib import Path

import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
from torch.onnx import TrainingMode


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class Network(nn.Module):
    def __init__(self, num_class):
        super(Network, self).__init__()
        self.features = nn.Sequential(
            Conv(3, 64, 3, 1, 1),
            Conv(64, 128, 3, 2, 1),  # 下采样
            Conv(128, 128, 3, 1, 1),
            Conv(128, 256, 3, 2, 1),  # 下采样
            Conv(256, 256, 3, 1, 1),
            nn.AdaptiveMaxPool2d((4, 4))
        )

        self.classify = nn.Sequential(
            nn.Linear(in_features=256 * 4 * 4, out_features=256),
            nn.ReLU(),
            nn.Linear(256, num_class)
        )

    def forward(self, x):
        z = self.features(x)
        z = z.flatten(1)  # [N,?1,?2,?3,..] --> [N,?]
        z = self.classify(z)
        return z


def t0():
    net = Network(10)
    loss_fn = nn.CrossEntropyLoss()
    train_opt = optim.SGD(net.parameters(), lr=0.0001)

    n = 10
    xs = [torch.rand(8, 3, 28, 28) for _ in range(n)]
    ys = [torch.randint(10, size=(8,)) for _ in range(n)]
    for epoch in range(2):
        for i in range(n):
            _x = xs[i]
            _y = ys[i]
            # 前向过程
            loss = loss_fn(net(_x), _y)
            # 反向过程
            train_opt.zero_grad()
            loss.backward()
            train_opt.step()
            print(f"epoch:{epoch}, batch:{i}, loss:{loss.item():.3f}")
    # 模型保存
    path_dir = Path("../output/models")
    path_dir.mkdir(parents=True, exist_ok=True)
    torch.save(net.eval(), str(path_dir / 'conv+bn.pkl'))


def export(model_dir, model_path=None, name="conv+bn"):
    """
   NOTE: 可以通过netron（https://netron.app/）来查看网络结构
   将训练好的模型转换成可以支持多平台部署的结构，常用的结构：
   pt: Torch框架跨语言部署的结构
   onnx: 一种比较通用的深度学习模型框架结构
   tensorRT: 先转换成onnx，然后再进行转换使用TensorRT进行GPU加速
   openvino: 先转换为onnx，然后再进行转换使用OpenVINO进行GPU加速
   :param model_path:
   :return:
   """
    model_dir = Path(model_dir)
    # 模型恢复
    if model_path is None:
        model_path = model_dir / 'conv+bn.pkl'
    net = torch.load(model_path, map_location='cpu')
    net.eval().cpu()

    # 模型转换为pt结构
    example = torch.rand(1, 3, 32, 32)
    _model = torch.jit.trace(net, example)
    _model.save(model_dir / f"{name}.pt")

    # 转换为onnx结构
    torch.onnx.export(
        model=net,
        args=example,
        f=model_dir / f"{name}.pt",
        training=TrainingMode.EVAL,
        input_names=['images'],
        output_names=['scores'],
        opset_version=12,
        dynamic_axes={
            'images': {
                0: 'batch'
            },
            'scores': {
                0: 'batch'
            }
        }
    )


def ConvBn(conv: nn.Conv2d, bn: nn.BatchNorm2d):
    """
   conv和bn的结合
   conv里面有两个参数: weight:[OC,IC,kh,kw]、bias:[OC,]
   bn里面有四个参数:running_mean[OC,]、running_var[OC,]、weight:[OC,]、bias:[OC,]
   NOTE: 基于运行过程的理解，可以将所有[OC,]格式的参数认为新shape为:[OC,1,1,1]
   bn(conv(x))
   -->
       ((x*conv.weight + conv.bias) - bn.running_mean) / torch.sqrt(bn.running_var + bn.eps) * bn.weight + bn.bias
   -->
       x*conv.weight*bn.weight/torch.sqrt(bn.running_var + bn.eps) +
       (conv.bias - bn.running_mean)*bn.weight/torch.sqrt(bn.running_var + bn.eps) + bn.bias
   -->
       新的卷积w: conv.weight*bn.weight/torch.sqrt(bn.running_var + bn.eps)
       新的卷积bias:(conv.bias - bn.running_mean)*bn.weight/torch.sqrt(bn.running_var + bn.eps) + bn.bias
   -->
       x*new_conv.weight + new_conv.bias
   :param conv:
   :param bn:
   :return:
   """
    fuseconv = nn.Conv2d(
        in_channels=conv.in_channels,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        groups=conv.groups,
        bias=True
    ).requires_grad_(False).to(conv.weight.device)
    # 合并weight
    w_bn = bn.weight.div(torch.sqrt(bn.eps + bn.running_var))  # 构建主对角线有数值，其它位置没有的操作 [OC]
    w_bn_conv = w_bn[:, None, None, None]  # [OC, 1, 1, 1]
    new_weight = conv.weight.clone() * w_bn_conv
    fuseconv.weight.copy_(new_weight)
    # 合并bias
    conv_bias = torch.zeros(conv.out_channels, device=conv.weight.device) if conv.bias is None else conv.bias.clone()
    new_bias = (conv_bias - bn.running_mean) * w_bn + bn.bias
    fuseconv.bias.copy_(new_bias)  # 赋值
    return fuseconv


def fuse_modules(model_dir, name="new_model"):
    model_dir = Path(model_dir)
    # 模型恢复
    net = torch.load(model_dir / 'conv+bn.pkl', map_location='cpu')
    net.eval().cpu()

    # 遍历所有模块
    for m in net.modules():
        if type(m) is Conv:
            # 进行模块合并（conv + bn）
            m.conv = ConvBn(m.conv, m.bn)
            delattr(m, 'bn')  # 删除m对象中的bn这个属性
            m.forward = m.forward_fuse  # 方法的复制
        pass
    torch.save(net.cpu(), str(model_dir / f"{name}.pkl"))

    # 转换导出
    export(
        model_dir=model_dir,
        model_path=str(model_dir / f"{name}.pkl"),
        name=name
    )


def tt_fuse(model_dir):
    model_dir = Path(model_dir)
    net1 = torch.jit.load(str(model_dir / 'conv+bn.pt'), map_location='cpu')
    net1.eval().cpu()
    net2 = torch.jit.load(str(model_dir / 'new_model.pt'), map_location='cpu')
    net2.eval().cpu()
    x = torch.rand(4, 3, 28, 28)

    r1 = torch.argmax(net1(x), dim=1)
    r2 = torch.argmax(net2(x), dim=1)
    print(r1 - r2)


def fuse_with_torch(model_dir):
    model_dir = Path(model_dir)
    # 模型恢复
    net = torch.load(model_dir / 'conv+bn.pkl', map_location='cpu')
    net.eval().cpu()
    print(net)
    # 直接调用torch的量化接口
    # modules_to_fuse: 给定的就是网络结构中的属性名称， 可以参数net.state_dict()的key来给定
    fuse_m = torch.quantization.fuse_modules(
        model=net,
        modules_to_fuse=[
            ['features.0.conv', 'features.0.bn', 'features.0.act'],
            ['features.1.conv', 'features.1.bn', 'features.1.act'],
            ['features.2.conv', 'features.2.bn', 'features.2.act'],
            ['features.3.conv', 'features.3.bn', 'features.3.act'],
            ['features.4.conv', 'features.4.bn', 'features.4.act']
        ]
    )
    print(fuse_m)
    x = torch.rand(4, 3, 28, 28)
    r1 = net(x)
    r2 = fuse_m(x)
    print(r1 - r2)
    torch.save(fuse_m.cpu(), str(model_dir / f"fuse_model.pkl"))

    # 转换导出
    export(
        model_dir=model_dir,
        model_path=str(model_dir / f'fuse_model.pkl'),
        name="fuse_model"
    )


if __name__ == '__main__':
    # t0()
    # export(model_dir='../output/models')
    fuse_modules(model_dir='../output/models')
    # tt_fuse(model_dir='../output/models')
    # fuse_with_torch(model_dir='../output/models')
