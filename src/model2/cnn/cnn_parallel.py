"""CNN with three parallel 3x3 convolution branches."""

from torch import Tensor, nn
import torch


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
        self.fusion = nn.Sequential(
            nn.Conv2d(96, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.pool = nn.MaxPool2d(2)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )


    def forward(self, x: Tensor) -> Tensor:

        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x = torch.cat([x1, x2, x3], dim=1)

        x = self.fusion(x)
        x = self.pool(x)

        return self.classifier(x)