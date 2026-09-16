"""The classification head trained on top of a pretrained backbone.

All four variants (3A-3D) share this exact head, so any difference in their
scores comes from the backbone strategy - frozen, LoRA-adapted or fully
fine-tuned - and not from a different classifier.
"""

from __future__ import annotations

from torch import nn


class MLPHead(nn.Sequential):
    """Two-layer MLP: features -> hidden -> num_classes.

    BatchNorm + ReLU + Dropout between the two linear layers keeps the head
    stable when it is the only thing being trained (variants 3A/3B), where the
    incoming features are fixed and the head has to do all of the work.
    """

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        hidden_dim: int = 512,
        dropout: float = 0.3,
    ) -> None:
        super().__init__(
            nn.Linear(feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes


class LinearHead(nn.Linear):
    """Plain linear probe - kept for reference / ablation, not used by 3A-3D."""

    def __init__(self, feature_dim: int, num_classes: int) -> None:
        super().__init__(feature_dim, num_classes)
