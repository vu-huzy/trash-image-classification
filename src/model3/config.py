"""Experiment registry for model3 - transfer learning with pretrained backbones.

Four variants are compared:

    3A  frozen MobileNetV2       -> train only the MLP head (cheapest baseline)
    3B  frozen strong backbones  -> train only the MLP head (backbone comparison)
    3C  LoRA on the 3B winner    -> train MLP head + low-rank adapters
    3D  full fine-tune           -> train every weight of the 3B winner

Every knob that differs between variants lives here, so the experiment table can
be read at a glance without digging through the training code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


MODEL3_DIR = Path(__file__).resolve().parent
PROJECT_DIR = MODEL3_DIR.parents[1]
DATA_DIR = PROJECT_DIR / "data" / "VN_trash_classification_preprocessing"
PRETRAINED_DIR = MODEL3_DIR / "pretrained"
CHECKPOINT_DIR = MODEL3_DIR / "checkpoints"
RESULTS_DIR = MODEL3_DIR / "results"

# torchvision downloads pretrained weights to $TORCH_HOME/hub/checkpoints.
# Point it inside model3/pretrained so the weights stay with the experiments
# instead of landing in the user-wide cache. Must happen before any
# torchvision.models call, hence at import time of this module.
PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TORCH_HOME", str(PRETRAINED_DIR))

IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

RANDOM_SEED = 42
NUM_WORKERS = 4

# Backbone used by 3A. Kept as a constant because 3A is defined as "the light one".
LIGHT_BACKBONE = "mobilenet_v2"

# Candidate backbones for 3B. The winner (highest validation accuracy) is reused
# by 3C and 3D so the three strategies are compared on identical features.
STRONG_BACKBONES = ("resnet50", "efficientnet_b0", "vit_b_16")


@dataclass
class ExperimentConfig:
    """Everything needed to build, train and evaluate one variant."""

    name: str  # run directory name, e.g. "3b_frozen_resnet50"
    variant: str  # "3A" | "3B" | "3C" | "3D"
    backbone: str  # key into backbones.BACKBONES
    strategy: str  # "frozen" | "lora" | "full_finetune"
    description: str

    # MLP classification head (identical across variants for a fair comparison).
    head_hidden_dim: int = 512
    head_dropout: float = 0.3

    # Optimisation
    batch_size: int = 64
    epochs: int = 30
    patience: int = 6  # early stopping on validation accuracy
    head_lr: float = 1e-3
    backbone_lr: float = 0.0  # 0 => backbone has no trainable weights
    weight_decay: float = 1e-4
    grad_clip: float = 0.0  # 0 => disabled

    # LoRA (only read when strategy == "lora")
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05

    @property
    def checkpoint_path(self) -> Path:
        return CHECKPOINT_DIR / f"{self.name}.pt"

    @property
    def run_dir(self) -> Path:
        return RESULTS_DIR / self.name


def experiment_3a() -> ExperimentConfig:
    """3A - freeze a light backbone entirely, train only the MLP head."""
    return ExperimentConfig(
        name=f"3a_frozen_{LIGHT_BACKBONE}",
        variant="3A",
        backbone=LIGHT_BACKBONE,
        strategy="frozen",
        description="Freeze all of MobileNetV2, train only the MLP head",
        batch_size=64,
        head_lr=1e-3,
    )


def experiments_3b() -> list[ExperimentConfig]:
    """3B - freeze each stronger backbone, train only the MLP head."""
    return [
        ExperimentConfig(
            name=f"3b_frozen_{backbone}",
            variant="3B",
            backbone=backbone,
            strategy="frozen",
            description=f"Freeze all of {backbone}, train only the MLP head",
            batch_size=64,
            head_lr=1e-3,
        )
        for backbone in STRONG_BACKBONES
    ]


def experiment_3c(backbone: str) -> ExperimentConfig:
    """3C - keep the backbone frozen but insert trainable LoRA adapters."""
    return ExperimentConfig(
        name=f"3c_lora_{backbone}",
        variant="3C",
        backbone=backbone,
        strategy="lora",
        description=f"Frozen {backbone} + LoRA adapters, train adapters + MLP head",
        batch_size=32,
        # LoRA adapters start at zero contribution, so they tolerate a head-sized LR.
        head_lr=1e-3,
        backbone_lr=1e-3,
        lora_rank=8,
        lora_alpha=16.0,
        lora_dropout=0.05,
    )


def experiment_3d(backbone: str) -> ExperimentConfig:
    """3D - unfreeze everything and fine-tune the whole network."""
    # Transformers need a markedly smaller LR than CNNs to avoid wrecking the
    # pretrained features in the first few steps.
    backbone_lr = 1e-5 if backbone.startswith("vit") else 1e-4
    return ExperimentConfig(
        name=f"3d_full_finetune_{backbone}",
        variant="3D",
        backbone=backbone,
        strategy="full_finetune",
        description=f"Fine-tune every weight of {backbone} together with the MLP head",
        batch_size=32,
        head_lr=1e-3,
        backbone_lr=backbone_lr,
        weight_decay=0.01,
        grad_clip=1.0,
        patience=5,
    )


def build_experiment(spec: str, backbone: str | None = None) -> ExperimentConfig:
    """Resolve a CLI experiment name such as "3a", "3b:resnet50", "3c", "3d"."""
    key, _, inline_backbone = spec.partition(":")
    key = key.strip().lower()
    backbone = backbone or (inline_backbone.strip() or None)

    if key == "3a":
        return experiment_3a()
    if key == "3b":
        if backbone is None:
            raise ValueError(f"3B needs a backbone, one of {STRONG_BACKBONES}")
        return next(cfg for cfg in experiments_3b() if cfg.backbone == backbone)
    if key == "3c":
        return experiment_3c(backbone or STRONG_BACKBONES[0])
    if key == "3d":
        return experiment_3d(backbone or STRONG_BACKBONES[0])
    raise ValueError(f"Unknown experiment '{spec}'. Expected one of 3a, 3b:<backbone>, 3c, 3d.")
