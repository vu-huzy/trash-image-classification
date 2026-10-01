"""M1-01: MLP baseline — cố ý không dùng convolution.

Workflow tensor (B là batch size, C=9 lớp):
    cache [B, 3, 224, 224] uint8
    -> resize/normalize chung trong common.data [B, 3, 128, 128] float32
    -> Flatten                              [B, 49_152]
    -> Linear(49_152, 512), ReLU, Dropout   [B, 512]
    -> Linear(512, 128), ReLU, Dropout      [B, 128]
    -> Linear(128, C)                       [B, 9] logits
"""

import torch.nn as nn

from model1.workflow import run_model_workflow


MODEL_ID = "M1-01"
ARCHITECTURE = "MLP: Flatten -> 512 -> 128 -> 9"


class M1_01_MLP(nn.Module):
    """Dense baseline với input 128x128 và hai dropout 0.2 từ config."""

    def __init__(self, num_classes: int, dropout_first: float, dropout_second: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),  # [B, 3, 128, 128] -> [B, 49_152]
            nn.Linear(3 * 128 * 128, 512),  # -> [B, 512]
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_first),
            nn.Linear(512, 128),  # -> [B, 128]
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_second),
            nn.Linear(128, num_classes),  # -> [B, 9] logits
        )

    def forward(self, images):
        return self.network(images)


def build_model(config: dict, num_classes: int) -> M1_01_MLP:
    """Khởi tạo M1-01 từ config, giữ architecture tách biệt với train workflow."""
    second_dropout = float(config["dropout"])
    first_dropout = float(config.get("dropout_first", second_dropout))
    return M1_01_MLP(num_classes, first_dropout, second_dropout)


def run(config: dict, args, device, class_names: list[str]) -> None:
    """Entry point riêng của M1-01, được ``train.py`` gọi qua registry."""
    dropout = f"{config.get('dropout_first', config['dropout'])}/{config['dropout']}"
    run_model_workflow(
        config, args, device, class_names, build_model, is_cnn=False, dropout_description=dropout
    )
