"""M1-02: CNN 2 blocks + BatchNorm + Dense head.

Workflow tensor (B là batch size, C=9 lớp):
    cache/preprocess                         [B, 3, 224, 224]
    Conv(3->32), BN, ReLU, MaxPool(2)        [B, 32, 112, 112]
    Conv(32->64), BN, ReLU, MaxPool(2)       [B, 64, 56, 56]
    AdaptiveAvgPool(4,4)                     [B, 64, 4, 4]
    Flatten                                  [B, 1_024]
    Linear(1_024->128), ReLU, Dropout(0.2)   [B, 128]
    Linear(128->C)                           [B, 9] logits
"""

import torch.nn as nn

from model1.blocks import conv_bn_relu_pool
from model1.workflow import run_model_workflow


MODEL_ID = "M1-02"
ARCHITECTURE = "CNN: 2 x [Conv-BN-ReLU-MaxPool] -> AdaptiveAvgPool(4,4) -> Dense 128 -> 9"


class M1_02_CNN(nn.Module):
    """CNN hai block. Đây là baseline convolutional có BatchNorm."""

    def __init__(self, num_classes: int, dropout: float):
        super().__init__()
        self.features = nn.Sequential(
            conv_bn_relu_pool(3, 32),  # [B, 3, 224, 224] -> [B, 32, 112, 112]
            conv_bn_relu_pool(32, 64),  # -> [B, 64, 56, 56]
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),  # -> [B, 64, 4, 4]
            nn.Flatten(),  # -> [B, 1024]
            nn.Linear(64 * 4 * 4, 128),  # -> [B, 128]
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),  # -> [B, 9] logits
        )

    def forward(self, images):
        return self.classifier(self.features(images))


def build_model(config: dict, num_classes: int) -> M1_02_CNN:
    return M1_02_CNN(num_classes, float(config["dropout"]))


def run(config: dict, args, device, class_names: list[str]) -> None:
    run_model_workflow(
        config,
        args,
        device,
        class_names,
        build_model,
        is_cnn=True,
        dropout_description=str(config["dropout"]),
    )
