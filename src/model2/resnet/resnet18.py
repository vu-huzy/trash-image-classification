"""Improved ResNet-18 with eight basic residual blocks."""

from torch import Tensor, nn
from torchvision.models import ResNet18_Weights, resnet18 as torchvision_resnet18

from .resnet_blocks import BasicBlock


class ResNet18(nn.Module):

    def __init__(self, num_classes: int = 9, pretrained: bool = False) -> None:
        super().__init__()

        if pretrained:
            self.backbone = torchvision_resnet18(weights=ResNet18_Weights.DEFAULT)
            self.backbone.fc = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes),
            )
            return

        self.in_channels = 64


        # Stem cải tiến cho ảnh nhỏ
        self.stem = nn.Sequential(

            nn.Conv2d(
                3,
                64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(inplace=True)
        )


        self.layer1 = self._make_layer(
            64, 2, 1
        )

        self.layer2 = self._make_layer(
            128, 2, 2
        )

        self.layer3 = self._make_layer(
            256, 2, 2
        )

        self.layer4 = self._make_layer(
            512, 2, 2
        )


        self.pool = nn.AdaptiveAvgPool2d(1)


        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Dropout(0.5),

            nn.Linear(
                512,
                128
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(0.5),

            nn.Linear(
                128,
                num_classes
            )
        )


    def _make_layer(
        self,
        channels: int,
        count: int,
        stride: int
    ) -> nn.Sequential:

        blocks = []

        blocks.append(
            BasicBlock(
                self.in_channels,
                channels,
                stride
            )
        )

        self.in_channels = channels


        for _ in range(count - 1):

            blocks.append(
                BasicBlock(
                    channels,
                    channels
                )
            )


        return nn.Sequential(*blocks)



    def forward(self, x: Tensor) -> Tensor:

        if hasattr(self, "backbone"):
            return self.backbone(x)

        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.pool(x)

        return self.classifier(x)
