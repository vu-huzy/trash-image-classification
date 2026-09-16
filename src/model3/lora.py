"""LoRA (Low-Rank Adaptation) implemented from scratch - used by variant 3C.

Idea: keep the pretrained weight W0 frozen and learn a low-rank correction

    y = W0 @ x  +  (alpha / r) * B @ A @ x        with A: r x in, B: out x r

Because r is tiny (8 here), B @ A holds a few hundred thousand parameters
instead of the tens of millions in W0. B is initialised to zero, so the adapted
model starts out numerically identical to the frozen pretrained one and only
drifts towards the trash-classification domain as training progresses.

Two adapter flavours are provided:
    LoRALinear  - wraps nn.Linear  (transformer blocks)
    LoRAConv2d  - wraps nn.Conv2d  (CNN stages)
"""

from __future__ import annotations

import math

import torch
from torch import nn


class LoRALinear(nn.Module):
    """nn.Linear with a frozen base weight plus a trainable low-rank update."""

    def __init__(
        self,
        base: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

        self.rank = max(1, min(rank, base.in_features, base.out_features))
        self.scaling = alpha / self.rank
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.lora_A = nn.Parameter(torch.empty(self.rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, self.rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # lora_B stays zero => the adapter contributes nothing at step 0.

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        update = self.lora_dropout(inputs) @ self.lora_A.t() @ self.lora_B.t()
        return self.base(inputs) + self.scaling * update

    def extra_repr(self) -> str:
        return f"rank={self.rank}, scaling={self.scaling:.3f}"


class LoRAConv2d(nn.Module):
    """nn.Conv2d with a frozen base kernel plus a trainable low-rank update.

    The update is factored into two convolutions: a rank-sized one that keeps
    the base geometry (kernel size / stride / padding / dilation), followed by a
    1x1 convolution that projects back up to the output channels.
    """

    def __init__(
        self,
        base: nn.Conv2d,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

        in_channels = base.in_channels
        out_channels = base.out_channels
        self.rank = max(1, min(rank, in_channels, out_channels))
        self.scaling = alpha / self.rank
        self.lora_dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        self.lora_down = nn.Conv2d(
            in_channels,
            self.rank,
            kernel_size=base.kernel_size,
            stride=base.stride,
            padding=base.padding,
            dilation=base.dilation,
            bias=False,
        )
        self.lora_up = nn.Conv2d(self.rank, out_channels, kernel_size=1, bias=False)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        update = self.lora_up(self.lora_down(self.lora_dropout(inputs)))
        return self.base(inputs) + self.scaling * update

    def extra_repr(self) -> str:
        return f"rank={self.rank}, scaling={self.scaling:.3f}"


def inject_lora(
    model: nn.Module,
    target_prefixes: tuple[str, ...],
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
) -> list[str]:
    """Replace eligible Linear/Conv2d layers under `target_prefixes` with adapters.

    Returns the qualified names that were wrapped, so a run can record exactly
    where its adapters sit.

    Note on attention layers: torch's nn.MultiheadAttention hands
    `in_proj_weight` and `out_proj.weight` straight to
    F.multi_head_attention_forward as raw tensors instead of calling the
    submodule, so wrapping those would be silently ignored (or crash). Their
    children are therefore skipped, and for ViT the adapters land on the Linear
    layers of every transformer block's MLP instead.
    """
    wrapped: list[str] = []

    for parent_name, parent in list(model.named_modules()):
        if isinstance(parent, nn.MultiheadAttention):
            continue

        for child_name, child in list(parent.named_children()):
            qualified_name = f"{parent_name}.{child_name}" if parent_name else child_name
            if not qualified_name.startswith(target_prefixes):
                continue

            if isinstance(child, nn.Linear):
                adapter: nn.Module = LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout)
            elif isinstance(child, nn.Conv2d):
                adapter = LoRAConv2d(child, rank=rank, alpha=alpha, dropout=dropout)
            else:
                continue

            setattr(parent, child_name, adapter)
            wrapped.append(qualified_name)

    return wrapped


def lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Collect only the adapter parameters (what 3C actually optimises)."""
    return [
        parameter
        for name, parameter in model.named_parameters()
        if "lora_" in name and parameter.requires_grad
    ]


def count_lora_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in lora_parameters(model))
