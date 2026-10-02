"""Improved VGG16 with three parallel convolution branches."""

import torch
from torch import Tensor, nn


class VGG16Parallel(nn.Module):

    def __init__(self, num_classes: int = 9) -> None:
        super().__init__()


        # 3 parallel branches học các đặc trưng khác nhau
        self.branch1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )


        # Sau concat: 64*3 = 192 channels
        self.features = nn.Sequential(

            # Block 1
            nn.Conv2d(192,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),


            # Block 2
            nn.Conv2d(64,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),


            # Block 3
            nn.Conv2d(128,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),


            # Block 4
            nn.Conv2d(256,512,3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512,512,3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),
        )


        self.classifier = nn.Sequential(

            nn.AdaptiveAvgPool2d(1),

            nn.Flatten(),

            nn.Linear(512,256),
            nn.ReLU(inplace=True),

            nn.Dropout(0.5),

            nn.Linear(256,num_classes)
        )


    def forward(self,x:Tensor)->Tensor:

        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x = torch.cat([x1,x2,x3],dim=1)

        x = self.features(x)

        return self.classifier(x)