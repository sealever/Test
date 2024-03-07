import os
from pathlib import Path
from typing import Union, List

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
import torchvision
from torchvision import models, transforms


class VggHook(object):
    def __init__(self, vgg, indexes: Union[int, List[int]] = 44):
        if isinstance(indexes, int):
            indexes = list(range(indexes))
        self.images = {}
        self.hooks = []
        for idx in indexes:
            # 注册一个钩子: 在nn.Module对象上增加一个属性值
            hook = vgg.features[idx].register_forward_hook(self._build_hook(idx))
            self.hooks.append(hook)

    def _build_hook(self, idx):
        def hook(module, module_input, module_output):
            # NOTE: module_input是一个tuple，实际上就是module的forward的入参
            self.images[idx] = module_output  # 将当前模块的输出保存到images字典中

        return hook

    def remove(self):
        for hook in self.hooks:
            hook.remove()


if __name__ == '__main__':
    # https://blog.csdn.net/winycg/article/details/101722445
    # pretrained: 当参数为True的时候，表示从网络上下载vgg16的模型参数,并使用下载的模型参数进行模型初始化(原始模型训练是基于ImageNet数据集)
    # NOTE: 默认下载地址:C:\Users\HP/.cache\torch\hub\checkpoints
    vgg = models.vgg16_bn(pretrained=True)
    # vgg_hooks = VggHook(vgg, indexes=[0, 1, 2, 3, 4, 5])
    vgg_hooks = VggHook(vgg)
    vgg.cpu().eval()
    print(vgg)

    tfs = transforms.ToTensor()
    resize = transforms.Resize(size=(50, 60))

    images_paths = {
        '小狗': r'..\datas\images\小狗.png',
        '小狗2': r'..\datas\images\小狗2.png',
        '小猫': r'..\datas\images\小猫.jpg',
        '小猫2': r'..\datas\images\小猫2.jpg',
        '飞机': r'..\datas\images\飞机.jpg',
        '飞机2': r'..\datas\images\飞机2.jpg',
    }

    # 模型预测
    # img = Image.open(images_paths['小狗']).convert('RGB')  # 加载图像，并将图像转换为RGB
    # # img.show()
    # img = tfs(img)  # Image --> Tensor [3,H,W]
    # print(type(img))
    # print(img.shape)
    # img = img[None]  # [3,H,W] --> [1,3,H,W]
    # for i in range(1):
    #     scores = vgg(img)  # [1, 1000]
    #     print(scores.shape)
    #     pre_indexes = torch.argmax(scores, dim=1)
    #     print(pre_indexes)
    #     proba = torch.softmax(scores, dim=1)  # 求解概率值
    #     top5 = torch.topk(proba, k=5, dim=1)  # 获取置信度最大的前五个
    #     print(top5)
    #     print(top5.indices)

    output_dir = Path('../output/vgg/features/')
    for name in images_paths.keys():
        print("--" * 100)
        # 模型预测
        img = Image.open(images_paths[name]).convert('RGB')
        img = tfs(img)
        img = img[None]

        scores = vgg(img)
        proba = torch.softmax(scores, dim=1)
        top5 = torch.topk(proba, k=5, dim=1)
        print(name)
        print(top5)

        # 各个阶段的可视化输出
        _output_dir = output_dir / name
        _output_dir.mkdir(parents=True, exist_ok=True)  # 创建文件夹
        for i in vgg_hooks.images.keys():
            features = vgg_hooks.images[i]  # [1,C,H,W]
            # [1,C,H,W] -> [C,H,W] -> [C,1,H,W]
            n, c, h, w = features.shape
            for m in range(n):
                imgs = features[m:m+1]
                imgs = torch.permute(imgs, dims=(1, 0, 2, 3))
                imgs = resize(imgs)
                # 保存图像
                torchvision.utils.save_image(
                    imgs,
                    _output_dir / f'{m}_{i}.png',
                    nrow=8,
                    padding=5,
                    pad_value=128)

        vgg_hooks.remove()  # 删除hooks

