"""DenseNet-BC classifier using bottleneck layers and compression transitions."""

import torch
from torch import Tensor, nn


class BottleneckDenseLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        growth_rate: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        bottleneck_channels = growth_rate * 4

        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels,
                bottleneck_channels,
                kernel_size=1,
                bias=False,
            ),

            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                bottleneck_channels,
                growth_rate,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.Dropout2d(dropout)
            if dropout > 0
            else nn.Identity(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layer(x)


class DenseBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_layers: int,
        growth_rate: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList()

        for index in range(num_layers):
            layer_in_channels = (
                in_channels
                + index * growth_rate
            )

            self.layers.append(
                BottleneckDenseLayer(
                    layer_in_channels,
                    growth_rate,
                    dropout,
                )
            )

        self.out_channels = (
            in_channels
            + num_layers * growth_rate
        )

    def forward(self, x: Tensor) -> Tensor:
        features = [x]

        for layer in self.layers:
            concatenated = torch.cat(
                features,
                dim=1,
            )

            new_features = layer(
                concatenated
            )

            features.append(
                new_features
            )

        return torch.cat(
            features,
            dim=1,
        )


class Transition(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            ),

            nn.AvgPool2d(
                kernel_size=2,
                stride=2,
            ),
        )


class DenseNetAdvanced(nn.Module):
    def __init__(
        self,
        num_classes: int = 9,
    ) -> None:
        super().__init__()

        growth_rate = 16
        compression = 0.5
        dropout = 0.1

        channels = 48

        self.stem = nn.Sequential(
            nn.Conv2d(
                3,
                channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(
                kernel_size=3,
                stride=2,
                padding=1,
            ),
        )

        blocks = []

        block_layers = (
            6,
            12,
            16,
        )

        for block_index, num_layers in enumerate(
            block_layers
        ):
            block = DenseBlock(
                in_channels=channels,
                num_layers=num_layers,
                growth_rate=growth_rate,
                dropout=dropout,
            )

            blocks.append(
                block
            )

            channels = block.out_channels

            if block_index < len(block_layers) - 1:
                compressed_channels = int(
                    channels * compression
                )

                blocks.append(
                    Transition(
                        channels,
                        compressed_channels,
                    )
                )

                channels = compressed_channels

        blocks.extend(
            [
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            ]
        )

        self.features = nn.Sequential(
            *blocks
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(
                channels,
                num_classes,
            ),
        )

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        x = self.stem(x)
        x = self.features(x)
        x = self.classifier(x)

        return x