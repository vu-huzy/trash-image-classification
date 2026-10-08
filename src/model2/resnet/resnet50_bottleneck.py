"""Improved ResNet-50 with standard bottleneck stages."""

from torch import Tensor, nn

from resnet.resnet_blocks import Bottleneck


class ResNet50Bottleneck(nn.Module):
    def __init__(self, num_classes: int = 9) -> None:
        super().__init__()

        self.in_channels = 64

        self.stem = nn.Sequential(
            nn.Conv2d(
                3,
                64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # ResNet-50: 3, 4, 6, 3 bottleneck blocks
        self.layer1 = self._make_layer(
            channels=64,
            count=3,
            stride=1,
        )

        self.layer2 = self._make_layer(
            channels=128,
            count=4,
            stride=2,
        )

        self.layer3 = self._make_layer(
            channels=256,
            count=6,
            stride=2,
        )

        self.layer4 = self._make_layer(
            channels=512,
            count=3,
            stride=2,
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.4),
            nn.Linear(
                512 * Bottleneck.expansion,
                num_classes,
            ),
        )

    def _make_layer(
        self,
        channels: int,
        count: int,
        stride: int,
    ) -> nn.Sequential:
        blocks = [
            Bottleneck(
                self.in_channels,
                channels,
                stride,
            )
        ]

        self.in_channels = (
            channels * Bottleneck.expansion
        )

        for _ in range(count - 1):
            blocks.append(
                Bottleneck(
                    self.in_channels,
                    channels,
                    stride=1,
                )
            )

        return nn.Sequential(*blocks)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.pool(x)

        return self.classifier(x)