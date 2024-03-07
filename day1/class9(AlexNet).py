import torch
import torch.nn as nn
import warnings

warnings.filterwarnings('ignore')


class AlexNet(nn.Module):
    def __init__(self, s1, s2):
        super(AlexNet, self).__init__()
        self.s1 = s1
        self.s2 = s2
        self.feature11 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=48, kernel_size=(11, 11), stride=(4, 4), padding=2),
            nn.ReLU(),
            nn.LocalResponseNorm(10),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(in_channels=48, out_channels=128, kernel_size=(11, 11), stride=(1, 1), padding='same'),
            nn.ReLU(),
            nn.LocalResponseNorm(10),
            nn.MaxPool2d(3, 2)
        ).to(self.s1)
        self.feature12 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=48, kernel_size=(11, 11), stride=(4, 4), padding=2),
            nn.ReLU(),
            nn.LocalResponseNorm(10),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(in_channels=48, out_channels=128, kernel_size=(11, 11), stride=(1, 1), padding='same'),
            nn.ReLU(),
            nn.LocalResponseNorm(10),
            nn.MaxPool2d(3, 2)
        ).to(self.s2)
        self.feature21 = nn.Sequential(
            nn.Conv2d(256, 192, kernel_size=(3, 3), stride=(1, 1), padding='same'),
            nn.ReLU(),
            nn.Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding='same'),
            nn.ReLU(),
            nn.Conv2d(192, 128, kernel_size=(3, 3), stride=(1, 1), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        ).to(self.s1)
        self.feature21 = nn.Sequential(
            nn.Conv2d(256, 192, kernel_size=(3, 3), stride=(1, 1), padding='same'),
            nn.ReLU(),
            nn.Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding='same'),
            nn.ReLU(),
            nn.Conv2d(192, 128, kernel_size=(3, 3), stride=(1, 1), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2)
        ).to(self.s2)
        self.classify = nn.Sequential(
            nn.Linear(6 * 6 * 128 * 2, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1000)
        ).to(self.s1)

    def forward(self, x):
        x1 = x.to(self.s1)
        x2 = x.to(self.s2)

        oz1 = self.feature11(x1)
        oz2 = self.feature12(x2)

        z1 = torch.concat([oz1, oz2.to(self.s1)], dim=1)
        z2 = torch.concat([oz1.to(self.s2), oz2], dim=1)

        z1 = self.feature21(z1)
        z2 = self.feature21(z2)

        z = torch.concat([z1, z2.to(self.s1)], dim=1)
        z = z.view(-1, 6 * 6 * 128 * 2)

        z = self.classify(z)

        return z


if __name__ == '__main__':
    s1 = torch.device("cpu")
    s2 = torch.device("cpu")
    net = AlexNet(s1, s2)
    img = torch.randn(2, 3, 224, 224)
    scores = net(img)
    print(scores)
    probs = torch.softmax(scores, dim=1)  # 求解概率值
    print(probs)

    # 参考pytorch中的默认实现
    from torchvision import models

    net = models.alexnet()
    print(net)
