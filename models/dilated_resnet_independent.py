import torch
import torch.nn as nn
import torch.nn.functional as F


class _IndependentDilatedConv(nn.Module):
    """
    Independent-weight multi-dilation conv — the conv operation from
    IndependentDilatedConv without the internal skip connection. Intended to be
    used inside BasicBlock, which provides its own residual shortcut.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilations=(1, 2, 4)):
        super().__init__()
        self.dilations = dilations
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      padding=d * (kernel_size // 2), dilation=d, bias=False)
            for d in dilations
        ])
        self.alpha = nn.Parameter(torch.zeros(out_channels, len(dilations)))

    def forward(self, x):
        outputs = [conv(x) for conv in self.convs]
        stacked = torch.stack(outputs, dim=2)           # [B, C, D, H, W]
        weights = F.softmax(self.alpha, dim=1).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        return (stacked * weights).sum(dim=2)           # [B, C, H, W]


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilations=(1, 2, 4)):
        super().__init__()
        self.conv1 = _IndependentDilatedConv(in_channels, out_channels, kernel_size=3, dilations=dilations)
        self.bn1   = nn.BatchNorm2d(out_channels)
        # AvgPool handles spatial downsampling — dilated convs require same-padding
        # and cannot embed a stride directly
        self.pool  = nn.AvgPool2d(stride, stride) if stride > 1 else nn.Identity()
        self.conv2 = _IndependentDilatedConv(out_channels, out_channels, kernel_size=3, dilations=dilations)
        self.bn2   = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.pool(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class DilatedResNetIndependent(nn.Module):
    """
    ResNet-18 with every 3x3 conv replaced by an IndependentDilatedConv,
    where each dilation rate has its own independent set of learned weights.
    Architecture is otherwise identical to baseline_cifar.py / baseline_stl.py.

    Stem selection:
        img_size <= 32  ->  CIFAR style (3x3 conv, stride 1, no MaxPool)
        img_size >  32  ->  ImageNet style (7x7 conv, stride 2 + MaxPool)
    """

    def __init__(self, in_channels=3, num_classes=10, img_size=(32, 32), dilations=(1, 2, 4)):
        super().__init__()

        if img_size[0] <= 32:
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )
        else:
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )

        self.layer1 = self._make_layer(64,  64,  stride=1, dilations=dilations)
        self.layer2 = self._make_layer(64,  128, stride=2, dilations=dilations)
        self.layer3 = self._make_layer(128, 256, stride=2, dilations=dilations)
        self.layer4 = self._make_layer(256, 512, stride=2, dilations=dilations)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc  = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, stride, dilations):
        return nn.Sequential(
            BasicBlock(in_channels, out_channels, stride=stride, dilations=dilations),
            BasicBlock(out_channels, out_channels, stride=1,      dilations=dilations),
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
