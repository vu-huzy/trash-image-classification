"""Basic Network-in-Network classifier using MLPConv blocks."""

from torch import Tensor, nn


class MLPConvBlock(nn.Sequential):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, padding: int
    ) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.ReLU(inplace=True),
        )


class NINBasic(nn.Module):
    def __init__(self, num_classes: int = 9) -> None:
        super().__init__()
        self.features = nn.Sequential(
            MLPConvBlock(3, 64, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            MLPConvBlock(64, 128, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            MLPConvBlock(128, 192, kernel_size=3, padding=1),
            nn.Dropout2d(0.2),
            nn.Conv2d(192, num_classes, kernel_size=1),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: Tensor) -> Tensor:
        return self.pool(self.features(x)).flatten(1)
