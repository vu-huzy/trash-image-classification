import torch
import torch.nn as nn

from common.data import prefetch_batches, prepare_images


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
