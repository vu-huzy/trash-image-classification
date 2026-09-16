"""Assembles backbone + MLP head according to the variant's strategy.

This is the one file to read if you want to see how 3A-3D actually differ:

    strategy = "frozen"         backbone weights frozen, run under no_grad,
                                kept in eval() mode -> only the MLP head learns
    strategy = "lora"           backbone weights frozen but LoRA adapters are
                                injected, so gradients flow *through* the
                                backbone into the adapters + MLP head
    strategy = "full_finetune"  nothing frozen, every weight learns
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from backbones import BackboneSpec, count_parameters, create_backbone, freeze_module, unfreeze_module
from config import ExperimentConfig
from heads import MLPHead
from lora import count_lora_parameters, inject_lora


FROZEN = "frozen"
LORA = "lora"
FULL_FINETUNE = "full_finetune"
STRATEGIES = (FROZEN, LORA, FULL_FINETUNE)


@dataclass
class ModelInfo:
    """What the builder did - recorded verbatim into each run's metrics.json."""

    backbone: str
    backbone_label: str
    strategy: str
    feature_dim: int
    head: str
    total_params: int
    trainable_params: int
    lora_params: int = 0
    lora_modules: list[str] = field(default_factory=list)

    @property
    def trainable_fraction(self) -> float:
        return self.trainable_params / self.total_params if self.total_params else 0.0


class TransferModel(nn.Module):
    """Pretrained feature extractor followed by the trainable MLP head."""

    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
        *,
        backbone_no_grad: bool,
        backbone_eval_mode: bool,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head
        # Frozen backbones need no graph at all - skipping it is a large speedup.
        self.backbone_no_grad = backbone_no_grad
        # Frozen / LoRA backbones must not update BatchNorm running statistics,
        # otherwise the "frozen" features silently drift between epochs.
        self.backbone_eval_mode = backbone_eval_mode

    def train(self, mode: bool = True) -> "TransferModel":
        super().train(mode)
        if self.backbone_eval_mode:
            self.backbone.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.backbone_no_grad:
            with torch.no_grad():
                features = self.backbone(images)
        else:
            features = self.backbone(images)
        return self.head(features)


def build_model(config: ExperimentConfig, num_classes: int) -> tuple[TransferModel, ModelInfo]:
    """Build the model for one experiment and report what was frozen/adapted."""
    if config.strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{config.strategy}'. Expected one of {STRATEGIES}.")

    backbone, feature_dim, spec = create_backbone(config.backbone)
    head = MLPHead(
        feature_dim=feature_dim,
        num_classes=num_classes,
        hidden_dim=config.head_hidden_dim,
        dropout=config.head_dropout,
    )

    lora_modules: list[str] = []

    if config.strategy == FROZEN:
        freeze_module(backbone)
        backbone_no_grad = True
        backbone_eval_mode = True

    elif config.strategy == LORA:
        freeze_module(backbone)
        lora_modules = inject_lora(
            backbone,
            target_prefixes=spec.lora_target_prefixes,
            rank=config.lora_rank,
            alpha=config.lora_alpha,
            dropout=config.lora_dropout,
        )
        if not lora_modules:
            raise RuntimeError(
                f"No LoRA adapters were injected into {config.backbone}; "
                f"target prefixes {spec.lora_target_prefixes} matched nothing."
            )
        # Gradients must reach the adapters, so the backbone forward runs with a graph.
        backbone_no_grad = False
        backbone_eval_mode = True

    else:  # FULL_FINETUNE
        unfreeze_module(backbone)
        backbone_no_grad = False
        backbone_eval_mode = False

    model = TransferModel(
        backbone,
        head,
        backbone_no_grad=backbone_no_grad,
        backbone_eval_mode=backbone_eval_mode,
    )
    trainable, total = count_parameters(model)
    info = ModelInfo(
        backbone=config.backbone,
        backbone_label=spec.label,
        strategy=config.strategy,
        feature_dim=feature_dim,
        head=f"MLP({feature_dim}->{config.head_hidden_dim}->{num_classes}, dropout={config.head_dropout})",
        total_params=total,
        trainable_params=trainable,
        lora_params=count_lora_parameters(model) if config.strategy == LORA else 0,
        lora_modules=lora_modules,
    )
    return model, info


def build_optimizer(model: TransferModel, config: ExperimentConfig) -> torch.optim.Optimizer:
    """One parameter group for the head, one for whatever the backbone exposes.

    The backbone group only exists for LoRA (adapter weights) and full
    fine-tuning (all weights), and always uses the smaller `backbone_lr`.
    """
    head_params = [p for p in model.head.parameters() if p.requires_grad]
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]

    param_groups: list[dict] = [
        {"params": head_params, "lr": config.head_lr, "name": "head"},
    ]
    if backbone_params:
        param_groups.append(
            {"params": backbone_params, "lr": config.backbone_lr, "name": "backbone"}
        )

    return torch.optim.AdamW(param_groups, lr=config.head_lr, weight_decay=config.weight_decay)
