"""VGG-16 classifier with a parallel multi-scale input stem."""

import torch
from torch import Tensor, nn


class VGG16Parallel(nn.Module):
    def __init__(self, num_classes: int = 9) -> None:
        super().__init__()

        # Branch 1: đặc trưng cục bộ 3x3
        self.branch1 = nn.Sequential(
            nn.Conv2d(
                3,
                32,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Branch 2: receptive field rộng hơn bằng dilation
        self.branch2 = nn.Sequential(
            nn.Conv2d(
                3,
                32,
                kernel_size=3,
                padding=2,
                dilation=2,
                bias=False,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Branch 3: kernel 5x5
        self.branch3 = nn.Sequential(
            nn.Conv2d(
                3,
                32,
                kernel_size=5,
                padding=2,
                bias=False,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Sau concat: 32 + 32 + 32 = 96 channels
        self.fusion = nn.Sequential(
            nn.Conv2d(
                96,
                64,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(
                64,
                64,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                64,
                64,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),

            # Block 2
            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                128,
                128,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),

            # Block 3
            nn.Conv2d(
                128,
                256,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                256,
                256,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                256,
                256,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),

            # Block 4
            nn.Conv2d(
                256,
                512,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                512,
                512,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                512,
                512,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),

            # Block 5
            nn.Conv2d(
                512,
                512,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                512,
                512,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                512,
                512,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=0.4),
            nn.Linear(
                512,
                num_classes,
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x = torch.cat(
            [x1, x2, x3],
            dim=1,
        )

        x = self.fusion(x)
        x = self.features(x)

        return self.classifier(x)


VGG16Advanced = VGG16Parallel