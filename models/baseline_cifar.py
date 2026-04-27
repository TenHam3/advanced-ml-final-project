import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class BaselineCIFAR(nn.Module):
    """
    ResNet-18 adapted for CIFAR-10 (32x32 RGB).

    The stem uses a 3x3 conv with stride 1 and no initial MaxPool — the standard
    CIFAR adaptation of ResNet-18. The original 7x7+MaxPool stem would reduce
    32x32 to 8x8 before any residual blocks, discarding most spatial information.

    Spatial progression (32x32 input):
        stem    -> 32x32 x64
        layer1  -> 32x32 x64   (stride 1)
        layer2  -> 16x16 x128  (stride 2)
        layer3  ->  8x8  x256  (stride 2)
        layer4  ->  4x4  x512  (stride 2)
        GAP     ->  1x1  x512
        FC      ->  num_classes
    """

    def __init__(self, in_channels=3, num_classes=10, img_size=(32, 32)):
        super().__init__()

        # CIFAR stem: 3x3 conv, stride 1, no maxpool
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.layer1 = self._make_layer(64,  64,  stride=1)
        self.layer2 = self._make_layer(64,  128, stride=2)
        self.layer3 = self._make_layer(128, 256, stride=2)
        self.layer4 = self._make_layer(256, 512, stride=2)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc  = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, stride):
        return nn.Sequential(
            BasicBlock(in_channels, out_channels, stride=stride),
            BasicBlock(out_channels, out_channels, stride=1),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x)
        x = x.flatten(1)
        return self.fc(x)
