"""Improved VGG-16 style sequential convolutional feature extractor."""

from torch import Tensor, nn


class VGG16Sequential(nn.Module):

    def __init__(self, num_classes: int = 9) -> None:
        super().__init__()

        self.features = self._features()

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),

            nn.Linear(512, 256),
            nn.ReLU(inplace=True),

            nn.Dropout(0.5),

            nn.Linear(256, num_classes)
        )


    @staticmethod
    def _features() -> nn.Sequential:

        layers: list[nn.Module] = []

        in_channels = 3

        cfg = (
            (64, 2),
            (128, 2),
            (256, 3),
            (512, 3),
            (512, 3)
        )


        for width, count in cfg:

            for _ in range(count):

                layers.extend([
                    nn.Conv2d(
                        in_channels,
                        width,
                        kernel_size=3,
                        padding=1
                    ),

                    nn.BatchNorm2d(width),

                    nn.ReLU(inplace=True)
                ])

                in_channels = width


            layers.append(
                nn.MaxPool2d(
                    kernel_size=2,
                    stride=2
                )
            )


        return nn.Sequential(*layers)



    def forward(self, x: Tensor) -> Tensor:

        x = self.features(x)

        return self.classifier(x)