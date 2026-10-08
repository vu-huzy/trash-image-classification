"""CNN with three parallel 3x3 convolution branches."""

import torch
from torch import Tensor, nn


class CNNParallel(nn.Module):
    def __init__(self, num_classes: int = 9) -> None:
        super().__init__()

        # Stem: học đặc trưng cơ bản trước khi chia nhánh
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        # Branch 1: đặc trưng cục bộ
        self.branch1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Branch 2: học sâu hơn
        self.branch2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Branch 3: dilation giúp nhìn vùng rộng hơn
        self.branch3 = nn.Sequential(
            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=2,
                dilation=2,
                bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # 64 + 64 + 64 = 192 channels
        self.fusion = nn.Sequential(
            nn.Conv2d(192, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # Học feature cấp cao
        self.features = nn.Sequential(
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),

            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(256, num_classes)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)

        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x = torch.cat([x1, x2, x3], dim=1)

        x = self.fusion(x)
        x = self.features(x)

        return self.classifier(x)