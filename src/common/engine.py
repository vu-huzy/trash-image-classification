import time
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common.data import prefetch_batches, prepare_images


@dataclass
class ClassificationEpoch:
    """Aggregated results from one standard image-classification epoch."""

    loss: float
    accuracy: float
    seconds: float
    targets: np.ndarray
    predictions: np.ndarray


def run_classification_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    amp_dtype: torch.dtype | None = None,
    grad_clip: float = 0.0,
    limit_batches: int | None = None,
    collect_predictions: bool = False,
    channels_last: bool = False,
) -> ClassificationEpoch:
    """Run a standard train/evaluation epoch over image and label batches."""
    training = optimizer is not None
    model.train(training)
    loss_sum = torch.zeros((), device=device)
    correct = torch.zeros((), dtype=torch.long, device=device)
    seen = 0
    all_targets: list[torch.Tensor] = []
    all_predictions: list[torch.Tensor] = []
    started = time.perf_counter()

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_index, (images, targets) in enumerate(loader):
            if limit_batches is not None and batch_index >= limit_batches:
                break

            images = images.to(
                device,
                non_blocking=device.type == "cuda",
                memory_format=(
                    torch.channels_last
                    if channels_last and device.type == "cuda"
                    else torch.contiguous_format
                ),
            )
            targets = targets.to(device, non_blocking=device.type == "cuda")

            if training:
                optimizer.zero_grad(set_to_none=True)

            autocast_context = (
                torch.amp.autocast(device.type, dtype=amp_dtype)
                if amp_dtype is not None and device.type == "cuda"
                else nullcontext()
            )
            with autocast_context:
                logits = model(images)
                loss = criterion(logits, targets)

            if training:
                loss.backward()
                if grad_clip > 0:
                    trainable_parameters = [
                        parameter for parameter in model.parameters() if parameter.requires_grad
                    ]
                    torch.nn.utils.clip_grad_norm_(trainable_parameters, grad_clip)
                optimizer.step()

            batch_size = targets.size(0)
            loss_sum += loss.detach() * batch_size
            predictions = logits.argmax(dim=1)
            correct += (predictions == targets).sum()
            seen += batch_size

            if collect_predictions:
                all_targets.append(targets.detach())
                all_predictions.append(predictions.detach())

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    if seen == 0:
        raise RuntimeError("The data loader produced no batches.")

    return ClassificationEpoch(
        loss=loss_sum.item() / seen,
        accuracy=correct.item() / seen,
        seconds=time.perf_counter() - started,
        targets=torch.cat(all_targets).cpu().numpy() if all_targets else np.array([]),
        predictions=torch.cat(all_predictions).cpu().numpy() if all_predictions else np.array([]),
    )


def run_epoch(
    model,
    loader,
    device: torch.device,
    image_size: int,
    optimizer=None,
    scaler=None,
    max_batches: int | None = None,
    channels_last: bool = False,
    collect_predictions: bool = False,
):
    """Train for one epoch when optimizer is supplied; otherwise evaluate."""
    training = optimizer is not None
    model.train(training)
    loss_function = nn.CrossEntropyLoss()
    total_loss = torch.zeros((), device=device)
    total_correct = torch.zeros((), device=device)
    total_examples = 0
    predictions, targets = [], []

    batch_iterator = prefetch_batches(loader)
    try:
        with torch.set_grad_enabled(training):
            for batch_index, (images, labels) in enumerate(batch_iterator):
                images = prepare_images(images, device, image_size, training, channels_last)
                labels = labels.to(device, non_blocking=True)

                if training:
                    optimizer.zero_grad(set_to_none=True)

                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    logits = model(images)
                    loss = loss_function(logits, labels)

                if training:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                batch_size = labels.size(0)
                predicted = logits.argmax(dim=1)
                # Keep these values on GPU until the epoch ends to avoid a CPU sync each batch.
                total_loss += loss.detach() * batch_size
                total_correct += (predicted == labels).sum()
                total_examples += batch_size
                if collect_predictions:
                    predictions.extend(predicted.detach().cpu().tolist())
                    targets.extend(labels.detach().cpu().tolist())

                if max_batches is not None and batch_index + 1 >= max_batches:
                    break
    finally:
        # Stops the prefetch thread if early stopping or --max-batches breaks the loop.
        close = getattr(batch_iterator, "close", None)
        if close is not None:
            close()

    if total_examples == 0:
        raise RuntimeError("The data loader produced no batches.")

    return {
        "loss": total_loss.item() / total_examples,
        "accuracy": total_correct.item() / total_examples,
        "predictions": predictions,
        "targets": targets,
    }
