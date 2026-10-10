"""Các block cơ bản dùng trong ResNet."""

import torch
from torch import Tensor, nn


class BasicBlock(nn.Module):
    """Hai convolution 3x3 và một shortcut."""
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        channels: int,
        stride: int = 1
    ) -> None:

        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        if stride != 1 or in_channels != channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, channels, 1, stride, bias=False),
                nn.BatchNorm2d(channels),
            )
        else:
            self.shortcut = nn.Identity()
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x: Tensor) -> Tensor:
        shortcut = self.shortcut(x)
        output = self.conv2(self.conv1(x))
        return self.relu(output + shortcut)


class Bottleneck(nn.Module):
    """Block 1x1 - 3x3 - 1x1 dùng cho ResNet sâu hơn."""
    expansion = 4


    def __init__(
        self,
        in_channels: int,
        channels: int,
        stride: int = 1
    ) -> None:

        super().__init__()
        out_channels = channels * self.expansion
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x: Tensor) -> Tensor:
        shortcut = self.shortcut(x)
        output = self.conv3(self.conv2(self.conv1(x)))
        return self.relu(output + shortcut)
