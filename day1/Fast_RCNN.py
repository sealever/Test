import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models


class FastRCNN(nn.Module):
    def __init__(self):
        super(FastRCNN, self).__init__()
        vgg = models.vgg16_bn(pretrained=True)
        self.features = vgg.features  # 迁移
        del self.features[43]  # 删除最后一层池化

        self.features2 = nn.Sequential(
            nn.AdaptiveMaxPool2d((4, 4)),
            nn.Flatten(1),
            nn.Linear(512 * 4 * 4, 16),  # 这部分参数的的迁移恢复是基于原始的FC6参数作了SVD矩阵分解后的
            nn.Linear(16, 4096),
            nn.ReLU(),
            nn.Linear(4096, 32),
            nn.Linear(32, 4096),
            nn.ReLU()
        )
        self.classify_header = nn.Linear(4096, 21)
        self.reg_header = nn.Linear(4096, 4)

    def forward(self, images, roi_list):
        """
        前向过程
        :param images: tensor对象 [2,3,H,W]
        :param roi_list: list列表, list[list[tuple4]], 列表里面是每个图像对应的候选区域坐标信息
            [
                [
                    (lx,ly,rx,ry),
                    ....
                ], # 第一个图像对应的候选区域坐标信息 --> 已经做了坐标映射转换的
                []
            ]
        :return:
        """
        features = self.features(images)  # [2,c,h,w] -> [2,512,nh,nw]
        roi_classify_sources = []
        roi_reg_sources = []
        for image_idx, image_roi in enumerate(roi_list):
            img_feature = features[image_idx:image_idx + 1]  # [1,512,nw,nw]
            for lx, ly, rx, ry in image_roi:
                roi_feature = img_feature[:, :, ly:ry, lx:rx]  # 提取候选区域
                roi_feature = self.features2(roi_feature)  # [1,4096]
                # 分类
                roi_classify = self.classify_header(roi_feature)  # [1,21]
                roi_classify_sources.append(roi_classify)
                # 回归
                roi_reg = self.reg_header(roi_feature)
                roi_reg_sources.append(roi_reg)
        roi_classify_sources = torch.concat(roi_classify_sources, dim=0)  # [?, 21]
        roi_reg_sources = torch.concat(roi_reg_sources, dim=0)  # [?,4]
        return roi_classify_sources, roi_reg_sources


if __name__ == '__main__':
    net = FastRCNN()
    cla_loss_fn = nn.CrossEntropyLoss()
    reg_loss_fn = nn.SmoothL1Loss(reduction='none')
    opt = optim.SGD(net.parameters(), lr=0.0001)
    print(net.features)

    images = torch.rand(2, 3, 224, 224)
    # 模拟的这里就是N=2，R=4
    # 自定义候选框
    roi_list = [
        [
            (1, 0, 5, 2),
            (2, 3, 8, 9),
            (3, 4, 10, 12),
            (5, 9, 12, 13)
        ],
        [
            (4, 0, 6, 2),
            (2, 2, 8, 6),
            (3, 4, 7, 12),
            (5, 3, 12, 13)
        ]
    ]
    # 自定义真实标签
    roi_targets_labels = torch.tensor([0, 0, 0, 2, 0, 5, 0, 0])
    # 自定义真实边框
    roi_targets_regs = torch.tensor([
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.3, -0.2, 0.8, 1.2],
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.5, 0.9, 2.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ])

    r1, r2 = net(images, roi_list)
    print(r1.shape)
    print(r2.shape)
    cla_loss = cla_loss_fn(r1, roi_targets_labels)
    reg_loss = torch.mean(
            torch.sum(reg_loss_fn(r2, roi_targets_regs), dim=1) * (roi_targets_labels >= 1).to(roi_targets_regs.dtype))
    print(cla_loss)
    print(reg_loss)
    loss = cla_loss + 0.5 * reg_loss

    opt.zero_grad()
    loss.backward()
    opt.step()

