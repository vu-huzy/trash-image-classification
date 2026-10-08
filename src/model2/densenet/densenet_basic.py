"""DenseNet classifier with concatenated dense layers."""

import torch
from torch import Tensor, nn


class DenseLayer(nn.Module):
    def __init__(self, in_channels: int, growth_rate: int) -> None:
        super().__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layer(x)


class DenseBlock(nn.Module):
    def __init__(self, in_channels: int, layers: int, growth_rate: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            DenseLayer(in_channels + index * growth_rate, growth_rate)
            for index in range(layers)
        )
        self.out_channels = in_channels + layers * growth_rate

    def forward(self, x: Tensor) -> Tensor:
        features = [x]
        for layer in self.layers:
            features.append(layer(torch.cat(features, dim=1)))
        return torch.cat(features, dim=1)


class Transition(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )


class DenseNetBasic(nn.Module):
    def __init__(self, num_classes: int = 9) -> None:
        super().__init__()
        growth_rate = 16
        channels = 32
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        blocks = []
        for block_index, layer_count in enumerate((4, 6, 8)):
            block = DenseBlock(channels, layer_count, growth_rate)
            blocks.append(block)
            channels = block.out_channels
            if block_index < 2:
                transitioned_channels = channels // 2
                blocks.append(Transition(channels, transitioned_channels))
                channels = transitioned_channels
        self.features = nn.Sequential(*blocks, nn.BatchNorm2d(channels), nn.ReLU(inplace=True))
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.features(self.stem(x)))
