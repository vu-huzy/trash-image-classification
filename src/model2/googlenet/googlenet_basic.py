"""Compact GoogLeNet-style classifier with Inception-v1 blocks."""

import torch
from torch import Tensor, nn


class InceptionBlock(nn.Module):
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
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, one_by_one, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, three_reduce, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(three_reduce, three_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.branch5 = nn.Sequential(
            nn.Conv2d(in_channels, five_reduce, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(five_reduce, five_channels, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
        )
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_channels, kernel_size=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return torch.cat(
            (self.branch1(x), self.branch3(x), self.branch5(x), self.branch_pool(x)),
            dim=1,
        )


class GoogLeNetBasic(nn.Module):
    def __init__(self, num_classes: int = 9) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.inception1 = InceptionBlock(64, 32, 32, 48, 8, 16, 16)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception2 = InceptionBlock(112, 48, 32, 64, 8, 16, 16)
        self.inception3 = InceptionBlock(144, 48, 48, 64, 12, 24, 24)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception4 = InceptionBlock(160, 64, 48, 80, 12, 24, 24)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(192, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.inception1(x)
        x = self.pool1(x)
        x = self.inception2(x)
        x = self.inception3(x)
        x = self.pool2(x)
        x = self.inception4(x)
        return self.classifier(x)
