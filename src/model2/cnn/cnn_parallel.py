"""CNN with three parallel 3x3 convolution branches."""

from torch import Tensor, nn
import torch


class ResidualConvBlock(nn.Module):
    """Residual convolution stage used after branch fusion."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=2, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.relu(self.main(x) + self.shortcut(x))


class CNNParallel(nn.Module):
    def __init__(self, num_classes: int = 9) -> None:
        super().__init__()

        # Branch 1: học feature cơ bản
        self.branch1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # Branch 2: sâu hơn
        self.branch2 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # Branch 3: dùng dilation để bắt đặc trưng rộng hơn
        self.branch3 = nn.Sequential(
            nn.Conv2d(
                3,
                32,
                kernel_size=3,
                padding=2,
                dilation=2
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # Sau concat: 32+32+32 = 96 channels
        # Match the three downsampling stages of CNNKernel3Sequential.
        self.features = nn.Sequential(
            ResidualConvBlock(96, 64),
            ResidualConvBlock(64, 128),
            ResidualConvBlock(128, 192),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(192, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )


    def forward(self, x: Tensor) -> Tensor:

        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x = torch.cat([x1, x2, x3], dim=1)

        x = self.features(x)

        return self.classifier(x)
