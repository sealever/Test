import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.rpn import RegionProposalNetwork, RPNHead


class RFCN(nn.Module):
    def __init__(self, num_classes, k=3):
        super(RFCN, self).__init__()
        resnet = models.resnet101(pretrained=True)
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            nn.Conv2d(2048, 1024, (1,1), stride=(1,1)),
            nn.ReLU()
        )
        # 下列rpn的代码直接从faster rcnn中copy的
        anchor_sizes = ((32, 64, 128),)
        aspect_ratios = ((0.5, 1.0, 2.0),)
        rpn_anchor_generator = AnchorGenerator(
            anchor_sizes,  # 每个feature map上的anchor box对应的尺度大小(原始图像的尺度)
            aspect_ratios  # 每个feature map上的anchor box的比例大小(高宽比)
        )
        rpn_head = RPNHead(
            1024, rpn_anchor_generator.num_anchors_per_location()[0]
        )
        rpn_nms_thresh = 0.7
        rpn_fg_iou_thresh = 0.7
        rpn_bg_iou_thresh = 0.3
        rpn_batch_size_per_image = 256
        rpn_positive_fraction = 0.5
        rpn_score_thresh = 0.0
        rpn_pre_nms_top_n_train = 2000
        rpn_pre_nms_top_n_test = 1000
        rpn_post_nms_top_n_train = 2000
        rpn_post_nms_top_n_test = 1000
        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)
        self.rpn = RegionProposalNetwork(
            rpn_anchor_generator, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh,
            score_thresh=rpn_score_thresh)
        # 判断feature map的每个像素点属于各个类别各个位置的置信度 --> 某个卷积核的功能就是判断feature map上的每个像素点是否属于某个类别、某个部位的特征
        self.cls_header = nn.Conv2d(1024, (num_classes+1)*k*k, (1,1), stride=(1,1))
        # 从feature map的每个像素点进行判断，如果这个像素点是某个位置，那么认为边框应该偏移的tx\ty\tw\th是多少
        self.reg_header = nn.Conv2d(1024, 4*k*k, (1,1), stride=(1,1))

        self.k = k
        self.num_classes = num_classes

    def forward(self, images:ImageList, targets=None):
        #  1. 基础特征提取
        features = self.backbone(images.tensors)  # [N,3,H,W] -> [N,1024,h,w]
        stride = 32
        # 2. rpn网络: proposals:(x1,y1,x2,y2) -> (x,y,w,h)
        proposals, proposal_losses = self.rpn(images, {'0': features}, targets)

        # 3. 分类判断
        cla_scores = self.cls_header(features)  # [N,1024,h,w] -> [N,(c+1)*k*k,h,w]
        roi_classify_sources = []  # 保存的是rpn网络提出的每个候选框对应的类型预测置信度
        for i in range(len(proposals)):
            npy_proposal = proposals[i].detach().cpu().numpy() / stride
            npy_proposal[:, :2] = np.floor(npy_proposal[:,:2])  # 左上角坐标向下取整
            npy_proposal[:, 2:] = np.ceil(npy_proposal[:, 2:])  # 左上角坐标向上取整， 使所选区域偏大
            roi_sources_per_image = []
            for x1, y1, x2, y2 in npy_proposal:
                roi_scores = cla_scores[i:i+1, :, int(y1):int(y2), int(x1): int(x2)]  # [1,(c+1)*k*k,roi_h,roi_w]
                roi_scores = F.adaptive_avg_pool2d(roi_scores, (self.k, self.k))  # [1,(c+1)*k*k,3,3]
                roi_scores = roi_scores.reshape(-1, self.num_classes +1, self.k*self.k, self.k*self.k)
                roi_scores = roi_scores.reshape(-1, self.num_classes + 1, self.k * self.k * self.k * self.k)
                roi_scores = roi_scores[:, :, ::self.k * self.k]
                roi_scores = torch.sum(roi_scores, dim=-1)  # [1,c+1]
                roi_sources_per_image.append(roi_scores)
            roi_sources_per_image = torch.concat(roi_sources_per_image, dim=0)
            roi_classify_sources.append(roi_sources_per_image)

        # 4. 回归判断
        reg_scores = self.reg_header(features)  # [N,1024,h,w] -> [N,4*k*k,h,w]
        roi_reg_sources = []  # 保存的是rpn网络提出的每个候选框对应的边框回归系数
        for i in range(len(proposals)):
            npy_proposal = proposals[i].detach().cpu().numpy() / stride
            npy_proposal[:, :2] = np.floor(npy_proposal[:, :2])
            npy_proposal[:, 2:] = np.ceil(npy_proposal[:, 2:])
            roi_reg_sources_per_image = []
            for x1, y1, x2, y2 in npy_proposal:
                roi_reg_scores = reg_scores[i:i + 1, :, int(y1):int(y2), int(x1):int(x2)]  # [1,4*k*k,roi_h,roi_w]
                roi_reg_scores = F.adaptive_avg_pool2d(roi_reg_scores, (self.k, self.k))  # [1,4*k*k,3,3]
                roi_reg_scores = roi_reg_scores.reshape(-1, 4, self.k * self.k, self.k, self.k)
                roi_reg_scores = roi_reg_scores.reshape(-1, 4, self.k * self.k * self.k * self.k)
                roi_reg_scores = roi_reg_scores[:, :, ::self.k * self.k]
                roi_reg_scores = torch.sum(roi_reg_scores, dim=-1)  # [1,4]  -> tx\ty\tw\th
                roi_reg_sources_per_image.append(roi_reg_scores)
            roi_reg_sources_per_image = torch.concat(roi_reg_sources_per_image, dim=0)
            roi_reg_sources.append(roi_reg_sources_per_image)

        # NOTE: 基于候选框proposals和真实边框targets，选择出正样本和负样本 --> 得到的就是每个候选框对应的类别标签y_proposals-->list(Tensor:[M,num_classes+1])
        # y_proposals + roi_classify_sources --> 计算分类损失
        # y_proposals + proposals + targets + roi_reg_sources --> 计算回归损失
        return roi_classify_sources, roi_reg_sources


if __name__ == '__main__':
    net = RFCN(num_classes=4)  # 猫、狗、人、马
    print(net)

    images = ImageList(
        tensors=torch.rand(2, 3, 800, 800),
        image_sizes=[(800, 800), (450, 380)]  # 两张图像对应的实际大小
    )

    net.eval()
    net(images)

