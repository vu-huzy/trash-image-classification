"""Train and evaluate one of the standalone scratch architectures."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

SRC = Path(__file__).resolve().parents[1]

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from common.checkpoint import load_checkpoint, save_checkpoint
from common.engine import run_classification_epoch
from common.logging_utils import RunLogger
from common.metrics import (
    compute_metrics,
    save_classification_report,
    save_confusion_matrix,
    save_history_plot,
)
from common.utils import count_parameters, save_json, set_seed
from models import MODEL_NAMES, build_model


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent

DEFAULT_DATA = PROJECT / "data" / "VN_trash_classification_preprocessing"
SUMMARY_PATH = HERE / "results" / "summary.csv"

SUMMARY_FIELDS = [
    "model",
    "run_number",
    "epochs_run",
    "best_epoch",
    "best_val_accuracy",
    "best_val_f1_macro",
    "test_accuracy",
    "test_loss",
    "elapsed_seconds",
    "checkpoint_bytes",
    "checkpoint_mib",
    "run_timestamp",
]


def update_summary(
    result: dict[str, object],
    checkpoint: Path,
) -> None:
    """Append current training result to summary.csv."""

    SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not SUMMARY_PATH.exists():
        existing_rows = []

        for result_file in sorted(
            (HERE / "results").glob("*.json")
        ):
            try:
                previous = json.loads(
                    result_file.read_text(
                        encoding="utf-8"
                    )
                )
            except (OSError, json.JSONDecodeError):
                continue

            if (
                not isinstance(previous, dict)
                or "model" not in previous
            ):
                continue

            previous_checkpoint = (
                HERE
                / "checkpoints"
                / f"{previous['model']}.pt"
            )

            checkpoint_bytes = (
                previous_checkpoint.stat().st_size
                if previous_checkpoint.exists()
                else ""
            )

            existing_rows.append(
                {
                    "model": previous["model"],
                    "run_number": 1,
                    "epochs_run": len(
                        previous.get("history", [])
                    ),
                    "best_epoch": previous.get(
                        "best_epoch",
                        "",
                    ),
                    "best_val_accuracy": previous.get(
                        "best_val_accuracy",
                        "",
                    ),
                    "best_val_f1_macro": previous.get(
                        "best_val_f1_macro",
                        "",
                    ),
                    "test_accuracy": previous.get(
                        "test_accuracy",
                        "",
                    ),
                    "test_loss": previous.get(
                        "test_loss",
                        "",
                    ),
                    "elapsed_seconds": previous.get(
                        "elapsed_seconds",
                        "",
                    ),
                    "checkpoint_bytes": checkpoint_bytes,
                    "checkpoint_mib": (
                        round(
                            checkpoint_bytes / (1024**2),
                            3,
                        )
                        if checkpoint_bytes != ""
                        else ""
                    ),
                    "run_timestamp": (
                        "existing result "
                        "(timestamp unavailable)"
                    ),
                }
            )

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
        "best_val_accuracy": result[
            "best_val_accuracy"
        ],
        "best_val_f1_macro": result[
            "best_val_f1_macro"
        ],
        "test_accuracy": result["test_accuracy"],
        "test_loss": result["test_loss"],
        "elapsed_seconds": result[
            "elapsed_seconds"
        ],
        "checkpoint_bytes": checkpoint_bytes,
        "checkpoint_mib": round(
            checkpoint_bytes / (1024**2),
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


def get_device(
    requested: str,
) -> torch.device:
    """Select training device."""

    if (
        requested == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA was requested but is unavailable. "
            "Install CUDA-enabled PyTorch and "
            "check the NVIDIA driver."
        )

    if requested == "auto":
        requested = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    return torch.device(requested)


def make_loader(
    path: Path,
    batch_size: int,
    train: bool,
    workers: int,
    pin_memory: bool,
    image_size: int,
) -> DataLoader:
    """Create ImageFolder DataLoader."""

    if train:
        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(0.75, 1.0),
                    ratio=(0.85, 1.15),
                ),
                transforms.RandomHorizontalFlip(
                    p=0.5
                ),
                transforms.RandomRotation(15),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.05,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(
                        0.485,
                        0.456,
                        0.406,
                    ),
                    std=(
                        0.229,
                        0.224,
                        0.225,
                    ),
                ),
                transforms.RandomErasing(
                    p=0.2,
                    scale=(0.02, 0.12),
                    ratio=(0.5, 2.0),
                ),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size)
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(
                        0.485,
                        0.456,
                        0.406,
                    ),
                    std=(
                        0.229,
                        0.224,
                        0.225,
                    ),
                ),
            ]
        )

    dataset = ImageFolder(
        path,
        transform=transform,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=workers > 0,
    )


def epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer
    | None = None,
    *,
    amp: bool = False,
    channels_last: bool = False,
    limit_batches: int | None = None,
    collect_predictions: bool = False,
):
    """Run one train or validation epoch."""

    return run_classification_epoch(
        model,
        loader,
        loss_fn,
        device,
        optimizer=optimizer,
        amp_dtype=(
            torch.bfloat16
            if amp
            else None
        ),
        limit_batches=limit_batches,
        collect_predictions=collect_predictions,
        channels_last=channels_last,
    )


def write_run_artifacts(
    run_dir: Path,
    result: dict[str, object],
    history: list[
        dict[str, float | int]
    ],
    class_names: list[str],
    targets,
    predictions,
) -> None:
    """Save metrics and plots."""

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_json(
        run_dir / "metrics.json",
        result,
    )

    pd.DataFrame(history).to_csv(
        run_dir / "history.csv",
        index=False,
    )

    save_history_plot(
        history,
        run_dir / "curves.png",
        title=str(result["model"]),
    )

    save_classification_report(
        targets,
        predictions,
        class_names,
        run_dir
        / "classification_report_test.txt",
    )

    save_confusion_matrix(
        targets,
        predictions,
        class_names,
        run_dir
        / "confusion_matrix_test.png",
        title=f"{result['model']} - test",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--model",
        choices=MODEL_NAMES,
        required=True,
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA,
        help=(
            "Folder containing "
            "train/, val/, test/"
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Square input image size; smaller images train faster.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--device",
        choices=(
            "auto",
            "cuda",
            "cpu",
        ),
        default="auto",
    )

    parser.add_argument(
        "--amp",
        action="store_true",
        help=(
            "Enable CUDA "
            "bfloat16 mixed precision"
        ),
    )

    parser.add_argument(
        "--channels-last",
        action="store_true",
        help="Use CUDA channels-last tensors for faster convolution.",
    )

    parser.add_argument(
        "--limit-batches",
        type=int,
        default=None,
    )

    args = parser.parse_args()
    minimum_image_size = (
        24
        if args.model == "resnet50_bottleneck"
        else 32
    )

    if (
        args.epochs < 1
        or args.batch_size < 1
        or args.image_size < minimum_image_size
        or args.workers < 0
    ):
        parser.error(
            "--epochs and --batch-size must be positive; "
            f"--image-size must be at least {minimum_image_size} "
            f"for {args.model}; "
            "--workers cannot be negative"
        )

    if (
        args.limit_batches is not None
        and args.limit_batches < 1
    ):
        parser.error(
            "--limit-batches must be positive"
        )

    return args


def run_training(
    args: argparse.Namespace,
) -> dict[str, object]:

    set_seed(args.seed)

    device = get_device(
        args.device
    )

    smoke_test = (
        args.limit_batches is not None
    )

    checkpoint_path = (
        HERE
        / "checkpoints"
        / f"{args.model}.pt"
    )

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    run_stamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    run_dir = (
        HERE
        / "results"
        / args.model
        / run_stamp
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    log = RunLogger(
        run_dir / "train_log.txt"
    )

    try:
        for split in (
            "train",
            "val",
            "test",
        ):
            split_path = (
                args.data_dir / split
            )

            if not split_path.is_dir():
                raise FileNotFoundError(
                    f"Missing {split_path}; "
                    "run preprocessing first."
                )

        train_loader = make_loader(
            args.data_dir / "train",
            args.batch_size,
            True,
            args.workers,
            device.type == "cuda",
            args.image_size,
        )

        val_loader = make_loader(
            args.data_dir / "val",
            args.batch_size,
            False,
            args.workers,
            device.type == "cuda",
            args.image_size,
        )

        test_loader = make_loader(
            args.data_dir / "test",
            args.batch_size,
            False,
            args.workers,
            device.type == "cuda",
            args.image_size,
        )

        class_names = (
            train_loader.dataset.classes
        )

        for loader in (
            val_loader,
            test_loader,
        ):
            if (
                loader.dataset.classes
                != class_names
            ):
                raise ValueError(
                    "train, val and test "
                    "must contain the same "
                    "class folders."
                )

        model = build_model(
            args.model,
            len(class_names),
        ).to(
            device,
            memory_format=(
                torch.channels_last
                if args.channels_last and device.type == "cuda"
                else torch.contiguous_format
            ),
        )

        train_targets = torch.tensor(
            train_loader.dataset.targets,
            dtype=torch.long,
        )

        class_counts = torch.bincount(
            train_targets,
            minlength=len(class_names),
        ).float()

        class_weights = (
            class_counts.sum()
            / (
                len(class_names)
                * class_counts.clamp_min(1)
            )
        )

        class_weights = (
            class_weights.to(device)
        )

        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=0.05,
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=1e-4,
        )

        scheduler = (
            torch.optim.lr_scheduler.
            ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=0.5,
                patience=3,
                min_lr=1e-6,
            )
        )

        use_amp = (
            args.amp
            and device.type == "cuda"
        )
        use_channels_last = (
            args.channels_last
            and device.type == "cuda"
        )

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(
                device
            )

        log(
            f"=== Scratch CNN | "
            f"{args.model} ==="
        )

        log(
            f"PyTorch={torch.__version__}"
        )

        log(
            f"device={device} "
            f"gpu="
            f"{torch.cuda.get_device_name(device) if device.type == 'cuda' else 'not in use'}"
        )

        log(
            f"amp="
            f"{'bfloat16' if use_amp else 'off'}"
        )

        log(
            f"channels_last={use_channels_last}"
        )

        log(
            f"splits="
            f"train:{len(train_loader.dataset)} "
            f"val:{len(val_loader.dataset)} "
            f"test:{len(test_loader.dataset)}"
        )

        log(
            f"classes={len(class_names)} "
            f"batch_size={args.batch_size} "
            f"image_size={args.image_size} "
            f"workers={args.workers}"
        )

        log(
            f"parameters="
            f"{count_parameters(model):,} "
            f"optimizer=AdamW "
            f"lr={args.lr}"
        )

        log(
            f"class_counts="
            f"{class_counts.tolist()}"
        )

        log(
            f"class_weights="
            f"{class_weights.cpu().tolist()}"
        )

        if smoke_test:
            log(
                "SMOKE TEST enabled."
            )

        best_accuracy = -1.0
        best_f1 = -1.0
        best_epoch = 0

        best_model_state = None

        history: list[
            dict[str, float | int]
        ] = []

        training_started = (
            time.perf_counter()
        )

        for current_epoch in range(
            1,
            args.epochs + 1,
        ):
            train_stats = epoch(
                model,
                train_loader,
                criterion,
                device,
                optimizer,
                amp=use_amp,
                channels_last=use_channels_last,
                limit_batches=(
                    args.limit_batches
                ),
            )

            val_stats = epoch(
                model,
                val_loader,
                criterion,
                device,
                amp=use_amp,
                channels_last=use_channels_last,
                limit_batches=(
                    args.limit_batches
                ),
                collect_predictions=True,
            )

            val_metrics = compute_metrics(
                val_stats.targets,
                val_stats.predictions,
                class_names,
            )

            learning_rate = float(
                optimizer.param_groups[
                    0
                ]["lr"]
            )

            current_f1 = float(
                val_metrics["f1_macro"]
            )

            scheduler.step(
                current_f1
            )

            improved = (
                current_f1 > best_f1
            )

            row = {
                "epoch": current_epoch,
                "train_loss": (
                    train_stats.loss
                ),
                "train_accuracy": (
                    train_stats.accuracy
                ),
                "val_loss": (
                    val_stats.loss
                ),
                "val_accuracy": (
                    val_stats.accuracy
                ),
                "val_f1_macro": (
                    current_f1
                ),
                "epoch_seconds": (
                    train_stats.seconds
                    + val_stats.seconds
                ),
                "learning_rate": (
                    learning_rate
                ),
            }

            history.append(row)

            log(
                f"epoch "
                f"{current_epoch:>2}/"
                f"{args.epochs} "
                f"train loss "
                f"{train_stats.loss:.4f} "
                f"acc "
                f"{train_stats.accuracy:.4f} | "
                f"val loss "
                f"{val_stats.loss:.4f} "
                f"acc "
                f"{val_stats.accuracy:.4f} "
                f"f1 "
                f"{current_f1:.4f} | "
                f"lr "
                f"{learning_rate:.2e} "
                f"| "
                f"{row['epoch_seconds']:.1f}s"
                f"{' <- best' if improved else ''}"
            )

            if improved:
                best_f1 = current_f1
                best_accuracy = (
                    val_stats.accuracy
                )
                best_epoch = (
                    current_epoch
                )

                if smoke_test:
                    best_model_state = (
                        copy.deepcopy(
                            model.state_dict()
                        )
                    )
                else:
                    save_checkpoint(
                        checkpoint_path,
                        {
                            "model": (
                                model.state_dict()
                            ),
                            "model_name": (
                                args.model
                            ),
                            "class_names": (
                                class_names
                            ),
                            "epoch": (
                                current_epoch
                            ),
                            "val_accuracy": (
                                best_accuracy
                            ),
                            "val_f1_macro": (
                                best_f1
                            ),
                        },
                    )

        training_seconds = (
            time.perf_counter()
            - training_started
        )

        if smoke_test:
            if best_model_state is None:
                raise RuntimeError(
                    "Training did not produce "
                    "a best model."
                )

            model.load_state_dict(
                best_model_state
            )

        else:
            state = load_checkpoint(
                checkpoint_path,
                map_location=device,
                weights_only=True,
            )

            model.load_state_dict(
                state["model"]
            )

        test_stats = epoch(
            model,
            test_loader,
            criterion,
            device,
            amp=use_amp,
            channels_last=use_channels_last,
            limit_batches=(
                args.limit_batches
            ),
            collect_predictions=True,
        )

        test_metrics = compute_metrics(
            test_stats.targets,
            test_stats.predictions,
            class_names,
        )

        peak_gpu_memory_mb = (
            torch.cuda.max_memory_allocated(
                device
            )
            / (1024**2)
            if device.type == "cuda"
            else 0.0
        )

        result: dict[
            str,
            object,
        ] = {
            "model": args.model,
            "class_names": class_names,
            "data": {
                "directory": str(
                    args.data_dir.resolve()
                ),
                "splits": {
                    "train": len(
                        train_loader.dataset
                    ),
                    "val": len(
                        val_loader.dataset
                    ),
                    "test": len(
                        test_loader.dataset
                    ),
                },
            },
            "training": {
                "epochs_ran": len(
                    history
                ),
                "epochs_configured": (
                    args.epochs
                ),
                "best_epoch": (
                    best_epoch
                ),
                "best_val_accuracy": (
                    best_accuracy
                ),
                "best_val_f1_macro": (
                    best_f1
                ),
                "history": history,
            },
            "best_epoch": best_epoch,
            "best_val_accuracy": (
                best_accuracy
            ),
            "best_val_f1_macro": (
                best_f1
            ),
            "history": history,
            "timing": {
                "train_seconds": (
                    training_seconds
                ),
                "test_seconds": (
                    test_stats.seconds
                ),
                "test_images_per_second": (
                    len(test_stats.targets)
                    / test_stats.seconds
                    if test_stats.seconds > 0
                    else 0.0
                ),
                "run_seconds": (
                    time.perf_counter()
                    - training_started
                ),
                "peak_gpu_memory_mb": (
                    peak_gpu_memory_mb
                ),
            },
            "environment": {
                "device": (
                    torch.cuda.get_device_name(
                        device
                    )
                    if device.type == "cuda"
                    else "cpu"
                ),
                "torch": (
                    torch.__version__
                ),
                "amp_dtype": (
                    "bfloat16"
                    if use_amp
                    else "off"
                ),
            },
            "test_loss": (
                test_stats.loss
            ),
            "test_accuracy": (
                test_stats.accuracy
            ),
            "test_metrics": (
                test_metrics
            ),
            "elapsed_seconds": (
                time.perf_counter()
                - training_started
            ),
        }

        write_run_artifacts(
            run_dir,
            result,
            history,
            class_names,
            test_stats.targets,
            test_stats.predictions,
        )

        log(
            f"TEST accuracy "
            f"{test_stats.accuracy:.4f} "
            f"balanced_accuracy "
            f"{test_metrics['balanced_accuracy']:.4f} "
            f"f1_macro "
            f"{test_metrics['f1_macro']:.4f} "
            f"f1_weighted "
            f"{test_metrics['f1_weighted']:.4f}"
        )

        log(
            f"best_epoch="
            f"{best_epoch} "
            f"best_val_accuracy="
            f"{best_accuracy:.4f} "
            f"best_val_f1_macro="
            f"{best_f1:.4f} "
            f"train="
            f"{training_seconds:.1f}s "
            f"test="
            f"{test_stats.seconds:.1f}s "
            f"peak_gpu="
            f"{peak_gpu_memory_mb:.0f} MB"
        )

        log(
            f"reports -> {run_dir}"
        )

        if smoke_test:
            log(
                "Smoke test: checkpoint "
                "and summary were not changed."
            )

        else:
            save_json(
                HERE
                / "results"
                / f"{args.model}.json",
                result,
            )

            update_summary(
                result,
                checkpoint_path,
            )

            log(
                f"checkpoint -> "
                f"{checkpoint_path}"
            )

            log(
                f"summary -> "
                f"{SUMMARY_PATH}"
            )

        return result

    finally:
        log.close()


def main() -> None:
    run_training(
        parse_args()
    )


if __name__ == "__main__":
    main()