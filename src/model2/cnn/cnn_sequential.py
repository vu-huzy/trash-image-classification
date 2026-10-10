from torch import Tensor, nn


class CNNSequential(nn.Module):
    #Mô hình CNN cơ bản gồm 3 block conv

    def __init__(self, num_classes: int = 9) -> None:
        # 1. Khởi tạo module cha
        super().__init__()

        # 2. Các block trích xuất đặc trưng
        self.features = nn.Sequential(
            # Block 1: 3 -> 48 channels
            nn.Conv2d(3, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2: 48 -> 96 channels
            nn.Conv2d(48, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: 96 -> 192 channels
            nn.Conv2d(96, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # 3. Adaptive pooling và classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(192, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        # 4. Lan truyền xuôi
        x = self.features(x)
        return self.classifier(x)
