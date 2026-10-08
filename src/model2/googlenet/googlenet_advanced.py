"""Deeper GoogLeNet-style classifier with normalized Inception blocks."""

import torch
from torch import Tensor, nn


class NormalizedInceptionBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        one_by_one: int,
        three_reduce: int,
        three_channels: int,
        five_reduce: int,
        five_channels: int,
        pool_channels: int,
    ) -> None:
        super().__init__()
        self.branch1 = self._conv(in_channels, one_by_one, 1)
        self.branch3 = nn.Sequential(
            self._conv(in_channels, three_reduce, 1),
            self._conv(three_reduce, three_channels, 3, padding=1),
        )
        self.branch5 = nn.Sequential(
            self._conv(in_channels, five_reduce, 1),
            self._conv(five_reduce, five_channels, 3, padding=1),
            self._conv(five_channels, five_channels, 3, padding=1),
        )
        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            self._conv(in_channels, pool_channels, 1),
        )

    @staticmethod
    def _conv(
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return torch.cat(
            (self.branch1(x), self.branch3(x), self.branch5(x), self.branch_pool(x)),
            dim=1,
        )


class GoogLeNetAdvanced(nn.Module):
    def __init__(self, num_classes: int = 9) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 96, kernel_size=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
        )
        self.inception1 = NormalizedInceptionBlock(96, 32, 32, 48, 8, 16, 16)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception2 = NormalizedInceptionBlock(112, 48, 32, 64, 8, 16, 16)
        self.inception3 = NormalizedInceptionBlock(144, 48, 48, 64, 12, 24, 24)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception4 = NormalizedInceptionBlock(160, 64, 48, 80, 12, 24, 24)
        self.inception5 = NormalizedInceptionBlock(192, 64, 64, 96, 16, 32, 32)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(224, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.inception1(x)
        x = self.pool1(x)
        x = self.inception2(x)
        x = self.inception3(x)
        x = self.pool2(x)
        x = self.inception4(x)
        x = self.inception5(x)
        return self.classifier(x)
