"""M1-04: CNN 3 blocks + Global Average Pooling (GAP).

Backbone giống hệt M1-03. Khác biệt duy nhất nằm ở head: GAP lấy trung bình
28x28 vị trí của từng channel nên [B,128,28,28] thành [B,128]. Không có Dense
hidden layer và không có Dropout; ``dropout`` trong JSON được giữ để config có
cùng schema với M1-02/M1-03 nhưng không được dùng bởi kiến trúc GAP.

Workflow tensor:
    [B, 3, 224, 224] -> 3 Conv-BN-ReLU-Pool blocks  [B, 128, 28, 28]
    -> AdaptiveAvgPool(1,1)                         [B, 128, 1, 1]
    -> Flatten                                      [B, 128]
    -> Linear(128->9)                               [B, 9] logits
"""

import torch.nn as nn

from model1.blocks import conv_bn_relu_pool
from model1.workflow import run_model_workflow


MODEL_ID = "M1-04"
ARCHITECTURE = "CNN: 3 x [Conv-BN-ReLU-MaxPool] -> Global Average Pooling -> 9"


class M1_04_CNN_GAP(nn.Module):
    """CNN ba block với GAP head để so sánh riêng Dense head và GAP."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            conv_bn_relu_pool(3, 32),  # [B, 3, 224, 224] -> [B, 32, 112, 112]
            conv_bn_relu_pool(32, 64),  # -> [B, 64, 56, 56]
            conv_bn_relu_pool(64, 128),  # -> [B, 128, 28, 28]
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # -> [B, 128, 1, 1]
            nn.Flatten(),  # -> [B, 128]
            nn.Linear(128, num_classes),  # -> [B, 9] logits
        )

    def forward(self, images):
        return self.classifier(self.features(images))


def build_model(config: dict, num_classes: int) -> M1_04_CNN_GAP:
    del config  # GAP head không dùng dropout/config hyperparameter riêng.
    return M1_04_CNN_GAP(num_classes)


def run(config: dict, args, device, class_names: list[str]) -> None:
    run_model_workflow(
        config,
        args,
        device,
        class_names,
        build_model,
        is_cnn=True,
        dropout_description="none (GAP head)",
    )
