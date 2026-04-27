import torch
import torch.nn as nn
import torch.nn.functional as F


class IndependentDilatedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilations=(1, 2, 4)):
        super().__init__()
        self.dilations = dilations

        # Each dilation rate gets its own independent set of learnable weights
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      padding=d * (kernel_size // 2), dilation=d)
            for d in dilations
        ])

        # Per-output-channel mixing weights over dilations (same as TestCNN)
        self.alpha = nn.Parameter(torch.zeros(out_channels, len(dilations)))

        # 1x1 projection to match channels on the skip path when in != out
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        outputs = [conv(x) for conv in self.convs]

        stacked = torch.stack(outputs, dim=2)          # [B, C, num_dilations, H, W]
        weights = F.softmax(self.alpha, dim=1)          # softmax over dilations, per channel
        weights = weights.view(1, -1, len(self.dilations), 1, 1)
        return (stacked * weights).sum(dim=2) + self.shortcut(x)


class IndependentCNN(nn.Module):
    def __init__(self, in_channels, num_classes, img_size=(28, 28)):
        super().__init__()

        self.layer1 = IndependentDilatedConv(in_channels, 8, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(8)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer2 = IndependentDilatedConv(8, 16, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(16)

        # self.layer3 = IndependentDilatedConv(16, 16, kernel_size=3)
        # self.bn3 = nn.BatchNorm2d(16)

        fc_input_size = 16 * (img_size[0] // 4) * (img_size[1] // 4)
        self.fc1 = nn.Linear(fc_input_size, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.layer1(x)))
        x = self.maxpool(x)
        x = F.relu(self.bn2(self.layer2(x)))
        x = self.maxpool(x)
        # x = F.relu(self.bn3(self.layer3(x)))
        # x = self.maxpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x
