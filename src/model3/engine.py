"""Training / evaluation loops shared by all four variants.

Kept deliberately strategy-agnostic: whatever model_builder.py hands over is
trained the same way, so runtime differences between 3A-3D come from the model
and not from a different training recipe. Mixed precision uses bfloat16, which
needs no gradient scaler.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader

from config import ExperimentConfig
from model_builder import TransferModel


@dataclass
class EvalResult:
    """Outcome of one pass over an evaluation loader."""

    loss: float
    accuracy: float
    y_true: np.ndarray
    y_pred: np.ndarray
    seconds: float

    @property
    def images_per_second(self) -> float:
        return len(self.y_true) / self.seconds if self.seconds > 0 else 0.0


@dataclass
class TrainingOutcome:
    """Everything the training loop learned about a run."""

    history: list[dict] = field(default_factory=list)
    best_epoch: int = 0
    best_val_accuracy: float = 0.0
    best_val_f1_macro: float = 0.0
    epochs_ran: int = 0
    early_stopped: bool = False
    total_train_seconds: float = 0.0
    mean_epoch_seconds: float = 0.0
    peak_gpu_memory_mb: float = 0.0


def train_one_epoch(
    model: TransferModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = True,
    grad_clip: float = 0.0,
    limit_batches: int | None = None,
) -> tuple[float, float, float]:
    """One pass over the training loader. Returns (loss, accuracy, seconds)."""
    model.train()
    started = time.perf_counter()

    running_loss = 0.0
    correct = 0
    seen = 0

    for batch_index, (images, targets) in enumerate(loader):
        if limit_batches is not None and batch_index >= limit_batches:
            break

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)

        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], grad_clip
            )
        optimizer.step()

        running_loss += loss.item() * targets.size(0)
        correct += (logits.argmax(dim=1) == targets).sum().item()
        seen += targets.size(0)

    if device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    return running_loss / max(seen, 1), correct / max(seen, 1), seconds


@torch.no_grad()
def evaluate(
    model: TransferModel,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = True,
    limit_batches: int | None = None,
) -> EvalResult:
    """One pass over an evaluation loader, collecting predictions for metrics."""
    model.eval()
    started = time.perf_counter()

    running_loss = 0.0
    all_targets: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []

    for batch_index, (images, targets) in enumerate(loader):
        if limit_batches is not None and batch_index >= limit_batches:
            break

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)

        running_loss += loss.item() * targets.size(0)
        all_targets.append(targets.cpu().numpy())
        all_predictions.append(logits.argmax(dim=1).cpu().numpy())

    if device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - started

    y_true = np.concatenate(all_targets) if all_targets else np.array([])
    y_pred = np.concatenate(all_predictions) if all_predictions else np.array([])
    accuracy = float((y_true == y_pred).mean()) if y_true.size else 0.0
    return EvalResult(
        loss=running_loss / max(y_true.size, 1),
        accuracy=accuracy,
        y_true=y_true,
        y_pred=y_pred,
        seconds=seconds,
    )


@torch.no_grad()
def benchmark_inference_speed(
    model: TransferModel,
    device: torch.device,
    batch_size: int = 32,
    image_size: int = 224,
    warmup_batches: int = 10,
    timed_batches: int = 30,
    use_amp: bool = True,
) -> dict:
    """Measure pure forward speed on synthetic batches.

    Separate from the test-loader timing on purpose: that one includes JPEG
    decoding and DataLoader worker startup, which says more about the disk and
    the OS than about the model. This one is the model's own cost.
    """
    model.eval()
    images = torch.randn(batch_size, 3, image_size, image_size, device=device)

    for _ in range(warmup_batches):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            model(images)
    if device.type == "cuda":
        torch.cuda.synchronize()

    started = time.perf_counter()
    for _ in range(timed_batches):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            model(images)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    return {
        "batch_size": batch_size,
        "timed_batches": timed_batches,
        "ms_per_batch": elapsed / timed_batches * 1000,
        "images_per_second": batch_size * timed_batches / elapsed,
    }


def fit(
    config: ExperimentConfig,
    model: TransferModel,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    use_amp: bool = True,
    limit_batches: int | None = None,
    log: Callable[[str], None] = print,
) -> TrainingOutcome:
    """Train with cosine LR decay, early stopping and best-checkpoint saving.

    Model selection uses validation accuracy only - the test split is never
    consulted during training.
    """
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    outcome = TrainingOutcome()
    epochs_without_improvement = 0
    config.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    training_started = time.perf_counter()

    for epoch in range(1, config.epochs + 1):
        train_loss, train_accuracy, epoch_seconds = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            use_amp=use_amp,
            grad_clip=config.grad_clip,
            limit_batches=limit_batches,
        )
        validation = evaluate(
            model, val_loader, criterion, device, use_amp=use_amp, limit_batches=limit_batches
        )
        val_f1_macro = (
            float(f1_score(validation.y_true, validation.y_pred, average="macro", zero_division=0))
            if validation.y_true.size
            else 0.0
        )

        current_lrs = [group["lr"] for group in optimizer.param_groups]
        scheduler.step()

        outcome.history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": validation.loss,
                "val_accuracy": validation.accuracy,
                "val_f1_macro": val_f1_macro,
                "epoch_seconds": epoch_seconds,
                "val_seconds": validation.seconds,
                "learning_rates": current_lrs,
            }
        )
        outcome.epochs_ran = epoch

        improved = validation.accuracy > outcome.best_val_accuracy
        if improved:
            outcome.best_val_accuracy = validation.accuracy
            outcome.best_val_f1_macro = val_f1_macro
            outcome.best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "experiment": config.name,
                    "variant": config.variant,
                    "backbone": config.backbone,
                    "strategy": config.strategy,
                    "epoch": epoch,
                    "val_accuracy": validation.accuracy,
                    "model_state_dict": model.state_dict(),
                },
                config.checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        log(
            f"  epoch {epoch:>2}/{config.epochs}  "
            f"train loss {train_loss:.4f} acc {train_accuracy:.4f}  |  "
            f"val loss {validation.loss:.4f} acc {validation.accuracy:.4f} f1 {val_f1_macro:.4f}  |  "
            f"{epoch_seconds:.1f}s{'  <- best' if improved else ''}"
        )

        if epochs_without_improvement >= config.patience:
            outcome.early_stopped = True
            log(f"  early stopping: no val improvement for {config.patience} epochs")
            break

    outcome.total_train_seconds = time.perf_counter() - training_started
    epoch_times = [entry["epoch_seconds"] for entry in outcome.history]
    outcome.mean_epoch_seconds = sum(epoch_times) / len(epoch_times) if epoch_times else 0.0
    if device.type == "cuda":
        outcome.peak_gpu_memory_mb = torch.cuda.max_memory_allocated() / (1024**2)
    return outcome


def load_best_weights(model: TransferModel, config: ExperimentConfig, device: torch.device) -> int:
    """Restore the best checkpoint before the final test evaluation."""
    checkpoint = torch.load(config.checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return int(checkpoint["epoch"])
