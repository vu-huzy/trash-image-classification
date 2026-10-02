"""Improved residual blocks for ResNet-18 and ResNet-50."""

import torch
from torch import Tensor, nn


class BasicBlock(nn.Module):

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        channels: int,
        stride: int = 1
    ) -> None:

        super().__init__()


        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )


        self.conv2 = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(channels)
        )


        if stride != 1 or in_channels != channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(channels)
            )
        else:
            self.shortcut = nn.Identity()


        self.relu = nn.ReLU(inplace=True)



    def forward(self, x: Tensor) -> Tensor:

        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.conv2(out)

        out += residual

        return self.relu(out)



class Bottleneck(nn.Module):

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
            nn.Conv2d(
                in_channels,
                channels,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )


        self.conv2 = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )


        self.conv3 = nn.Sequential(
            nn.Conv2d(
                channels,
                out_channels,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels)
        )


        if stride != 1 or in_channels != out_channels:

            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

        else:

            self.shortcut = nn.Identity()


        self.relu = nn.ReLU(inplace=True)



    def forward(self, x: Tensor) -> Tensor:

        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        out += residual

        return self.relu(out)