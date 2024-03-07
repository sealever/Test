import torch
import torch.nn as nn


class ZFNet(nn.Module):
    def __init__(self):
        super(ZFNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=(7, 7), stride=(2, 2), padding=1),
            nn.ReLU(),
            nn.LocalResponseNorm(30),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(5, 5), stride=(2, 2)),
            nn.ReLU(),
            nn.LocalResponseNorm(50),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3, 3), stride=(1, 1), padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3, 3), stride=(1, 1), padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding='same'),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
            nn.AdaptiveMaxPool2d(output_size=(6, 6)),
        )

        self.classify = nn.Sequential(
            nn.Linear(6 * 6 * 256, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1000)
        )

    def forward(self, x):
        z = self.features(x)
        z = z.view(-1, 256 * 6 * 6)
        z = self.classify(z)
        return z


if __name__ == '__main__':
    net = ZFNet()
    img = torch.randn(2, 3, 224, 224)
    scores = net(img)
    print(scores)
    probs = torch.softmax(scores, dim=1)
    print(probs)
