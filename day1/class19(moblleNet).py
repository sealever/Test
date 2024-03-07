import time
from pathlib import Path
from PIL import Image
import torch
from torchvision import models, transforms
from thop import profile
import torch.nn as nn


def t0():
    path_dir = Path('../output/models/')
    path_dir.mkdir(parents=True, exist_ok=True)
    x = torch.randn(4, 3, 224, 224)
    net = models.mobilenet_v2(pretrained=True)
    print("=" * 100)
    # print(net)
    _model = torch.jit.trace(net.eval(), x)
    _model.save(str(path_dir / 'mobilenet_v2.pt'))
    net1 = models.mobilenet_v3_small(pretrained=True)
    print(net1)
    traced_script_module = torch.jit.trace(net1.eval(), x)
    traced_script_module.save(str(path_dir / 'mobilenet_v3_small.pt'))


def t1(model):
    model.eval().cpu()
    tfs = transforms.ToTensor()
    img_path = {
        '小狗': r'..\datas\images\小狗.png',
        '小狗2': r'..\datas\images\小狗2.png',
        '小猫': r'..\datas\images\小猫.jpg',
        '小猫2': r'..\datas\images\小猫2.jpg',
        '飞机': r'..\datas\images\飞机.jpg',
        '飞机2': r'..\datas\images\飞机2.jpg',
    }
    for name in img_path.keys():
        print('=' * 100)
        img = Image.open(img_path[name]).convert('RGB')
        img = tfs(img)
        img = img[None]

        scores = model(img)
        proba = torch.softmax(scores, dim=1)
        top5 = torch.topk(proba, k=5, dim=1)
        print(name)
        print(top5)


def calc_flops(net, inputs):
    print(type(net))
    if isinstance(inputs, list):
        inputs = tuple(inputs, )
    elif not isinstance(inputs, tuple):
        inputs = (inputs,)
    flops, params = profile(net, inputs, custom_ops={})
    print(f"总的浮点计算量[FLOPs]:{flops}")
    print(f"总的参数量:{params}")

    net.eval()
    with torch.no_grad():
        start_time = time.time()
        net(*inputs)
        end_time = time.time()
        print(f"耗时:{end_time - start_time}")


def t2():
    print("=" * 100)
    calc_flops(net=nn.Sequential(nn.Linear(3, 5)),
               inputs=torch.randn(1, 3))
    _x = torch.randn(1, 3, 224, 224)
    print("=" * 100)
    model = models.vgg16_bn(pretrained=False)
    calc_flops(model, _x)
    model = models.resnet101(pretrained=False)
    calc_flops(model, _x)
    print("=" * 100)
    model = models.densenet121(pretrained=False)
    calc_flops(model, _x)
    print("=" * 100)
    model = models.mobilenet_v2(pretrained=False)
    calc_flops(model, _x)
    print("=" * 100)
    model = models.mobilenet_v3_small(pretrained=False)
    calc_flops(model, _x)
    print("=" * 100)
    model = models.mobilenet_v3_large(pretrained=False)
    calc_flops(model, _x)


if __name__ == '__main__':
    # t0()
    # t1(models.mobilenet_v2(pretrained=True))
    t2()
