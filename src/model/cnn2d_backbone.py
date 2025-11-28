# src/models/cnn_backbone.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ECALayer(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        t = int(abs((math.log2(channels) + b) / gamma))
        k = t if t % 2 == 1 else t + 1
        k = max(1, k)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k,
                              padding=(k - 1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, H, W)
        y = self.avg_pool(x)                 # (B, C, 1, 1)
        y = y.squeeze(-1).transpose(-1, -2)  # (B, 1, C)
        y = self.conv(y)                     # (B, 1, C)
        y = self.sigmoid(y)                  # (B, 1, C)
        y = y.transpose(-1, -2).unsqueeze(-1)  # (B, C, 1, 1)
        return x * y.expand_as(x)


class CNNBackbone(nn.Module):
    """
    2-block CNN backbone: (B,1,28,28) → (B,128,7,7)
    with optional ECA after each block.
    """
    def __init__(self, in_channels=1, c1=64, c2=128, use_eca=False):
        super().__init__()
        self.use_eca = use_eca

        self.conv1 = nn.Conv2d(in_channels, c1, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(c1)
        self.pool1 = nn.MaxPool2d(2, 2)   # 28→14

        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(c2)
        self.pool2 = nn.MaxPool2d(2, 2)   # 14→7

        if use_eca:
            self.eca1 = ECALayer(c1)
            self.eca2 = ECALayer(c2)
        else:
            self.eca1 = None
            self.eca2 = None

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        if self.eca1 is not None:
            x = self.eca1(x)

        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        if self.eca2 is not None:
            x = self.eca2(x)

        # (B, 128, 7, 7)
        return x