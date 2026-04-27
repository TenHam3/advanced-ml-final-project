import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiDilatedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilations=(1, 2, 4)):
        super().__init__()
        self.dilations = dilations
        self.kernel_size = kernel_size

        # Single conv whose weight/bias are shared across all dilation rates
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

        # Per-output-channel mixing weights over dilations (shape: out_channels x num_dilations)
        self.alpha = nn.Parameter(torch.zeros(out_channels, len(dilations)))

        # 1x1 projection to match channels on the skip path when in != out
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        outputs = []
        for d in self.dilations:
            out = F.conv2d(
                x,
                self.conv.weight,
                self.conv.bias,
                padding=d * (self.kernel_size // 2),
                dilation=d,
            )
            outputs.append(out)

        stacked = torch.stack(outputs, dim=2)          # [B, C, num_dilations, H, W]
        weights = F.softmax(self.alpha, dim=1)          # softmax over dilations, per channel
        weights = weights.view(1, -1, len(self.dilations), 1, 1)
        return (stacked * weights).sum(dim=2) + self.shortcut(x)


class TestCNN(nn.Module):
    def __init__(self, in_channels, num_classes, img_size=(28, 28)):
        super().__init__()

        self.layer1 = MultiDilatedConv(in_channels=in_channels, out_channels=8, kernel_size=3)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(8) # num_features matches output channels
        
        # Fully connected layer — size depends on image dimensions after 2x MaxPool (each halves H and W)
        self.layer2 = MultiDilatedConv(in_channels=8, out_channels=16, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(16)

        # self.layer3 = MultiDilatedConv(in_channels=16, out_channels=16, kernel_size=3)
        # self.bn3 = nn.BatchNorm2d(16)

        # self.meanpool = nn.AvgPool2d(kernel_size=2, stride=2)
        # self.lpnormpool = nn.LPPool2d(norm_type=2, kernel_size=2, stride=2)

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
