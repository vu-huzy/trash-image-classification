"""Regularized, deeper Network-in-Network classifier."""

from torch import Tensor, nn


class RegularizedMLPConvBlock(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class NINAdvanced(nn.Module):
    def __init__(self, num_classes: int = 9) -> None:
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            RegularizedMLPConvBlock(
                3,
                96,
                kernel_size=5,
                padding=2,
            ),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),
            nn.Dropout2d(p=0.15),

            # Block 2
            RegularizedMLPConvBlock(
                96,
                192,
                kernel_size=3,
                padding=1,
            ),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),
            nn.Dropout2d(p=0.20),

            # Block 3
            RegularizedMLPConvBlock(
                192,
                256,
                kernel_size=3,
                padding=1,
            ),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),
            nn.Dropout2d(p=0.25),

            # Block 4
            RegularizedMLPConvBlock(
                256,
                384,
                kernel_size=3,
                padding=1,
            ),
            nn.Dropout2d(p=0.30),

            # Class prediction map
            nn.Conv2d(
                384,
                num_classes,
                kernel_size=1,
            ),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = x.flatten(1)
        return x