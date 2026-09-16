"""Pretrained ImageNet backbones, stripped down to feature extractors.

Each backbone is loaded from torchvision with its ImageNet weights (downloaded
once into model3/pretrained), then its own classifier is replaced by
nn.Identity() so the module returns a plain feature vector. The classification
head we actually train lives in heads.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torchvision import models as tv_models

from config import IMAGE_SIZE


@dataclass(frozen=True)
class BackboneSpec:
    """How to turn one torchvision model into a feature extractor."""

    torchvision_name: str
    weights_name: str  # entry of the model's weight enum
    classifier_attr: str  # attribute replaced by nn.Identity()
    expected_feature_dim: int  # asserted against a real forward pass
    lora_target_prefixes: tuple[str, ...]  # where LoRA may be injected (see lora.py)
    label: str


BACKBONES: dict[str, BackboneSpec] = {
    "mobilenet_v2": BackboneSpec(
        torchvision_name="mobilenet_v2",
        weights_name="IMAGENET1K_V1",
        classifier_attr="classifier",
        expected_feature_dim=1280,
        lora_target_prefixes=("features.14", "features.15", "features.16", "features.17"),
        label="MobileNetV2",
    ),
    "resnet50": BackboneSpec(
        torchvision_name="resnet50",
        weights_name="IMAGENET1K_V2",  # stronger recipe than V1 (~80.8% top-1)
        classifier_attr="fc",
        expected_feature_dim=2048,
        lora_target_prefixes=("layer3", "layer4"),
        label="ResNet50",
    ),
    "efficientnet_b0": BackboneSpec(
        torchvision_name="efficientnet_b0",
        weights_name="IMAGENET1K_V1",
        classifier_attr="classifier",
        expected_feature_dim=1280,
        lora_target_prefixes=("features.5", "features.6", "features.7"),
        label="EfficientNet-B0",
    ),
    "vit_b_16": BackboneSpec(
        torchvision_name="vit_b_16",
        weights_name="IMAGENET1K_V1",
        classifier_attr="heads",
        expected_feature_dim=768,
        lora_target_prefixes=("encoder.layers",),
        label="ViT-B/16",
    ),
}


def create_backbone(name: str) -> tuple[nn.Module, int, BackboneSpec]:
    """Load a pretrained backbone as a feature extractor.

    Returns the module, its output feature dimension and its spec. The feature
    dimension is measured with a real forward pass rather than trusted blindly,
    so a torchvision change cannot silently produce a mismatched head.
    """
    if name not in BACKBONES:
        raise ValueError(f"Unknown backbone '{name}'. Available: {sorted(BACKBONES)}")

    spec = BACKBONES[name]
    weights_enum = tv_models.get_model_weights(spec.torchvision_name)
    weights = weights_enum[spec.weights_name]
    backbone = tv_models.get_model(spec.torchvision_name, weights=weights)

    # Drop the ImageNet classifier - we only want the features it was fed.
    setattr(backbone, spec.classifier_attr, nn.Identity())

    feature_dim = _measure_feature_dim(backbone)
    if feature_dim != spec.expected_feature_dim:
        raise RuntimeError(
            f"{name}: expected {spec.expected_feature_dim}-d features but measured {feature_dim}"
        )
    return backbone, feature_dim, spec


def _measure_feature_dim(backbone: nn.Module) -> int:
    """Run one dummy image through the backbone to read its output width."""
    was_training = backbone.training
    backbone.eval()
    with torch.no_grad():
        features = backbone(torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE))
    backbone.train(was_training)

    if features.ndim != 2:
        raise RuntimeError(f"Expected 2-d backbone output, got shape {tuple(features.shape)}")
    return int(features.shape[1])


def freeze_module(module: nn.Module) -> None:
    """Make every weight in a module non-trainable."""
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def unfreeze_module(module: nn.Module) -> None:
    """Make every weight in a module trainable."""
    for parameter in module.parameters():
        parameter.requires_grad_(True)


def count_parameters(module: nn.Module) -> tuple[int, int]:
    """Return (trainable, total) parameter counts."""
    total = sum(parameter.numel() for parameter in module.parameters())
    trainable = sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
    return trainable, total
