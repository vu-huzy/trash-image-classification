"""M1-03: CNN 3 blocks + BatchNorm + Dense head.

M1-03 chỉ thêm block thứ ba vào M1-02; Dense head được giữ nguyên để phép
so sánh cô lập tác động của depth.

Workflow tensor:
    [B, 3, 224, 224] -> Block 1             [B, 32, 112, 112]
    -> Block 2                              [B, 64, 56, 56]
    -> Block 3: Conv(64->128), BN, Pool     [B, 128, 28, 28]
    -> AdaptiveAvgPool(4,4)                 [B, 128, 4, 4]
    -> Flatten                              [B, 2_048]
    -> Dense 128, ReLU, Dropout(0.2)        [B, 128]
    -> classifier                           [B, 9] logits
"""

import torch.nn as nn

from model1.blocks import conv_bn_relu_pool
from model1.workflow import run_model_workflow


MODEL_ID = "M1-03"
ARCHITECTURE = "CNN: 3 x [Conv-BN-ReLU-MaxPool] -> AdaptiveAvgPool(4,4) -> Dense 128 -> 9"


class M1_03_CNN(nn.Module):
    """CNN ba block với Dense head, model Simple CNN tốt nhất đã chạy."""

    def __init__(self, num_classes: int, dropout: float):
        super().__init__()
        self.features = nn.Sequential(
            conv_bn_relu_pool(3, 32),  # [B, 3, 224, 224] -> [B, 32, 112, 112]
            conv_bn_relu_pool(32, 64),  # -> [B, 64, 56, 56]
            conv_bn_relu_pool(64, 128),  # -> [B, 128, 28, 28]
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),  # -> [B, 128, 4, 4]
            nn.Flatten(),  # -> [B, 2048]
            nn.Linear(128 * 4 * 4, 128),  # -> [B, 128]
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),  # -> [B, 9] logits
        )

    def forward(self, images):
        return self.classifier(self.features(images))


def build_model(config: dict, num_classes: int) -> M1_03_CNN:
    return M1_03_CNN(num_classes, float(config["dropout"]))


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
