"""Train and evaluate one of the standalone scratch architectures."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from models import build_models


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DEFAULT_DATA = PROJECT / "data" / "VN_trash_classification_preprocessing"

SUMMARY_PATH = HERE / "results" / "summary.csv"

SUMMARY_FIELDS = [
    "model",
    "run_number",
    "epochs_run",
    "best_epoch",
    "best_val_accuracy",
    "test_accuracy",
    "test_loss",
    "elapsed_seconds",
    "checkpoint_bytes",
    "checkpoint_mib",
    "run_timestamp",
]


def update_summary(result: dict[str, object], checkpoint: Path) -> None:
    """Append this run to summary.csv, importing existing per-model results once."""

    if not SUMMARY_PATH.exists():
        existing_rows = []

        for result_file in sorted((HERE / "results").glob("*.json")):
            try:
                previous = json.loads(
                    result_file.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError):
                continue

            if not isinstance(previous, dict) or "model" not in previous:
                continue

            previous_checkpoint = (
                HERE / "checkpoints" / f"{previous['model']}.pt"
            )

            checkpoint_bytes = (
                previous_checkpoint.stat().st_size
                if previous_checkpoint.exists()
                else ""
            )

            existing_rows.append({
                "model": previous["model"],
                "run_number": 1,
                "epochs_run": len(previous.get("history", [])),
                "best_epoch": previous.get("best_epoch", ""),
                "best_val_accuracy": previous.get(
                    "best_val_accuracy", ""
                ),
                "test_accuracy": previous.get(
                    "test_accuracy", ""
                ),
                "test_loss": previous.get("test_loss", ""),
                "elapsed_seconds": previous.get(
                    "elapsed_seconds", ""
                ),
                "checkpoint_bytes": checkpoint_bytes,
                "checkpoint_mib": (
                    round(
                        checkpoint_bytes / (1024 ** 2),
                        3,
                    )
                    if checkpoint_bytes != ""
                    else ""
                ),
                "run_timestamp": "existing result (timestamp unavailable)",
            })

        with SUMMARY_PATH.open(
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=SUMMARY_FIELDS,
            )
            writer.writeheader()
            writer.writerows(existing_rows)

    with SUMMARY_PATH.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        rows = list(csv.DictReader(file))

    prior_runs = sum(
        row.get("model") == result["model"]
        for row in rows
    )

    checkpoint_bytes = checkpoint.stat().st_size

    row = {
        "model": result["model"],
        "run_number": prior_runs + 1,
        "epochs_run": len(result["history"]),
        "best_epoch": result["best_epoch"],
        "best_val_accuracy": result["best_val_accuracy"],
        "test_accuracy": result["test_accuracy"],
        "test_loss": result["test_loss"],
        "elapsed_seconds": result["elapsed_seconds"],
        "checkpoint_bytes": checkpoint_bytes,
        "checkpoint_mib": round(
            checkpoint_bytes / (1024 ** 2),
            3,
        ),
        "run_timestamp": time.strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
    }

    with SUMMARY_PATH.open(
        "a",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=SUMMARY_FIELDS,
        )
        writer.writerow(row)


def get_device() -> torch.device:
    """Use CPU for training."""

    return torch.device("cpu")


def make_loader(
    path: Path,
    batch_size: int,
    train: bool,
    workers: int,
) -> DataLoader:
    """Create an ImageFolder DataLoader."""

    ops = [
        transforms.Resize((224, 224))
    ]

    if train:
        ops.extend([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
            ),
        ])

    ops.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
        ),
    ])

    dataset = ImageFolder(
        path,
        transform=transforms.Compose(ops),
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=workers,
        pin_memory=False,
        persistent_workers=workers > 0,
    )


def epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    """Run one training or evaluation epoch."""

    training = optimizer is not None
    model.train(training)

    loss_sum = 0.0
    correct = 0
    seen = 0

    context = (
        torch.enable_grad()
        if training
        else torch.inference_mode()
    )

    with context:
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            if training:
                optimizer.zero_grad(
                    set_to_none=True
                )

            logits = model(images)
            loss = loss_fn(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

            count = labels.size(0)

            loss_sum += loss.item() * count

            correct += (
                (logits.argmax(1) == labels)
                .sum()
                .item()
            )

            seen += count

    return (
        loss_sum / max(seen, 1),
        correct / max(seen, 1),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--model",
        choices=tuple(build_models().keys()),
        required=True,
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA,
        help=(
            "Folder containing train/, val/, "
            "and test/ ImageFolder splits"
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    # ============================================================
    # Reproducibility
    # ============================================================

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ============================================================
    # Device
    # ============================================================

    device = get_device()

    print("=" * 60)
    print(f"PyTorch version : {torch.__version__}")
    print("Device          : cpu")
    print("GPU             : DISABLED")
    print("Training        : CPU")
    print("=" * 60)

    # ============================================================
    # Check dataset
    # ============================================================

    for split in ("train", "val", "test"):
        split_path = args.data_dir / split

        if not split_path.is_dir():
            raise FileNotFoundError(
                f"Missing {split_path}; "
                "run src/preprocessing.py first"
            )

    # ============================================================
    # Data loaders
    # ============================================================

    train_loader = make_loader(
        args.data_dir / "train",
        args.batch_size,
        True,
        args.workers,
    )

    val_loader = make_loader(
        args.data_dir / "val",
        args.batch_size,
        False,
        args.workers,
    )

    test_loader = make_loader(
        args.data_dir / "test",
        args.batch_size,
        False,
        args.workers,
    )

    class_names = train_loader.dataset.classes

    for loader in (val_loader, test_loader):
        if loader.dataset.classes != class_names:
            raise ValueError(
                "train, val, and test must contain "
                "the same class folders"
            )

    print(f"Classes         : {len(class_names)}")
    print(f"Train samples   : {len(train_loader.dataset)}")
    print(f"Val samples     : {len(val_loader.dataset)}")
    print(f"Test samples    : {len(test_loader.dataset)}")
    print(f"Batch size      : {args.batch_size}")
    print(f"Workers         : {args.workers}")
    print(f"Model           : {args.model}")

    # ============================================================
    # Model
    # ============================================================

    model = build_models(
        len(class_names)
    )[args.model].to(device)

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        patience=2,
        factor=0.5,
    )

    best_accuracy = -1.0

    history: list[dict[str, float | int]] = []

    started = time.perf_counter()

    # ============================================================
    # Training
    # ============================================================

    for current_epoch in range(
        1,
        args.epochs + 1,
    ):
        print(
            f"\nEpoch {current_epoch}/{args.epochs}"
        )

        train_loss, train_accuracy = epoch(
            model,
            train_loader,
            loss_fn,
            device,
            optimizer,
        )

        val_loss, val_accuracy = epoch(
            model,
            val_loader,
            loss_fn,
            device,
        )

        scheduler.step(val_accuracy)

        row = {
            "epoch": current_epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        }

        history.append(row)

        print(
            json.dumps(
                row,
                indent=2,
            ),
            flush=True,
        )

        # ========================================================
        # Save best checkpoint
        # ========================================================

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy

            checkpoint_path = (
                HERE
                / "checkpoints"
                / f"{args.model}.pt"
            )

            torch.save(
                {
                    "model": model.state_dict(),
                    "model_name": args.model,
                    "class_names": class_names,
                    "epoch": current_epoch,
                    "val_accuracy": val_accuracy,
                },
                checkpoint_path,
            )

            print(
                f"Best checkpoint saved: "
                f"{checkpoint_path}"
            )

    # ============================================================
    # Load best checkpoint
    # ============================================================

    checkpoint = (
        HERE
        / "checkpoints"
        / f"{args.model}.pt"
    )

    state = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(
        state["model"]
    )

    # ============================================================
    # Test
    # ============================================================

    test_loss, test_accuracy = epoch(
        model,
        test_loader,
        loss_fn,
        device,
    )

    # ============================================================
    # Save result
    # ============================================================

    result = {
        "model": args.model,
        "class_names": class_names,
        "best_epoch": state["epoch"],
        "best_val_accuracy": best_accuracy,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "elapsed_seconds": (
            time.perf_counter() - started
        ),
        "history": history,
    }

    result_path = (
        HERE
        / "results"
        / f"{args.model}.json"
    )

    result_path.write_text(
        json.dumps(
            result,
            indent=2,
        ),
        encoding="utf-8",
    )

    update_summary(
        result,
        checkpoint,
    )

    # ============================================================
    # Final output
    # ============================================================

    print("\n" + "=" * 60)
    print("TRAINING FINISHED")
    print("=" * 60)

    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k != "history"
            },
            indent=2,
        )
    )

    print(
        f"\nSummary updated: {SUMMARY_PATH}"
    )


if __name__ == "__main__":
    (HERE / "checkpoints").mkdir(exist_ok=True)
    (HERE / "results").mkdir(exist_ok=True)
    main()
