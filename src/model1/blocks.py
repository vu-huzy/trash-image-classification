"""Các block nhỏ được dùng lặp lại bởi những model CNN trong Model 1.

Mỗi Model M1-02/M1-03/M1-04 vẫn tự ghép các block của mình trong file riêng.
File này chỉ tránh phải sao chép lại đúng cùng một định nghĩa Conv-BN-ReLU-Pool.
"""

import torch.nn as nn


def conv_bn_relu_pool(in_channels: int, out_channels: int) -> nn.Sequential:
    """Conv 3x3 (giữ H,W) -> BatchNorm -> ReLU -> MaxPool 2x2 (giảm H,W một nửa)."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2),
    )
