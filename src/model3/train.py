"""Run one model3 experiment end to end and write its results.

Usage (from the project root, with the venv python):

    .venv/Scripts/python.exe src/model3/train.py --experiment 3a
    .venv/Scripts/python.exe src/model3/train.py --experiment 3b --backbone efficientnet_b0
    .venv/Scripts/python.exe src/model3/train.py --experiment 3c --backbone resnet50
    .venv/Scripts/python.exe src/model3/train.py --experiment 3d --backbone resnet50 --epochs 2

Artefacts per run land in src/model3/results/<run name>/ and the best weights
in src/model3/checkpoints/<run name>.pt.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import CHECKPOINT_DIR, ExperimentConfig, NUM_WORKERS, RANDOM_SEED, RESULTS_DIR, build_experiment
from data import DataBundle, build_dataloaders
from engine import benchmark_inference_speed, evaluate, fit, load_best_weights
from metrics import (
    compute_metrics,
    save_classification_report,
    save_confusion_matrix,
    save_history_plot,
)
from model_builder import build_model, build_optimizer


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Make a run repeatable (up to cuDNN's non-deterministic kernels)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class RunLogger:
    """Print to stdout and tee the same lines into the run directory."""

    def __init__(self, log_path: Path) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = log_path.open("w", encoding="utf-8")

    def __call__(self, message: str = "") -> None:
        print(message, flush=True)
        self._file.write(message + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def run_experiment(
    config: ExperimentConfig,
    bundle: DataBundle | None = None,
    num_workers: int = NUM_WORKERS,
    limit_batches: int | None = None,
    use_amp: bool = True,
) -> dict:
    """Train, evaluate and persist one experiment. Returns its metrics dict."""
    run_started = time.perf_counter()
    config.run_dir.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    log = RunLogger(config.run_dir / "train_log.txt")

    try:
        set_seed()
        torch.backends.cudnn.benchmark = True
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_amp = use_amp and device.type == "cuda"

        if bundle is None or bundle.train_loader.batch_size != config.batch_size:
            bundle = build_dataloaders(batch_size=config.batch_size, num_workers=num_workers)

        log(f"=== {config.variant} | {config.name} ===")
        log(f"{config.description}")
        log(f"device={device} amp={'bfloat16' if use_amp else 'off'} batch_size={config.batch_size}")
        log(f"splits={bundle.split_sizes} classes={bundle.num_classes}")

        model, info = build_model(config, num_classes=bundle.num_classes)
        model.to(device)
        optimizer = build_optimizer(model, config)

        log(f"backbone={info.backbone_label} ({info.feature_dim}-d features)  head={info.head}")
        log(
            f"strategy={info.strategy}  trainable={info.trainable_params:,} / "
            f"{info.total_params:,} params ({info.trainable_fraction:.2%})"
        )
        if info.lora_modules:
            log(
                f"LoRA: rank={config.lora_rank} alpha={config.lora_alpha} -> "
                f"{len(info.lora_modules)} adapted layers, {info.lora_params:,} adapter params"
            )
            log(f"      first adapted layers: {', '.join(info.lora_modules[:4])} ...")
        log(f"optimizer=AdamW head_lr={config.head_lr} backbone_lr={config.backbone_lr}")
        log("")

        outcome = fit(
            config,
            model,
            optimizer,
            bundle.train_loader,
            bundle.val_loader,
            device,
            use_amp=use_amp,
            limit_batches=limit_batches,
            log=log,
        )

        best_epoch = load_best_weights(model, config, device)
        log("")
        log(f"restored best checkpoint from epoch {best_epoch}")

        criterion = torch.nn.CrossEntropyLoss()
        val_result = evaluate(model, bundle.val_loader, criterion, device, use_amp=use_amp)

        # First pass warms the test loader's worker processes (on Windows that
        # costs seconds) so the timed pass below measures the pipeline, not
        # process startup. Predictions are identical - the transform is deterministic.
        evaluate(model, bundle.test_loader, criterion, device, use_amp=use_amp)
        test_result = evaluate(model, bundle.test_loader, criterion, device, use_amp=use_amp)
        speed = benchmark_inference_speed(
            model, device, batch_size=config.batch_size, use_amp=use_amp
        )

        val_metrics = compute_metrics(val_result.y_true, val_result.y_pred, bundle.class_names)
        test_metrics = compute_metrics(test_result.y_true, test_result.y_pred, bundle.class_names)

        log(
            f"VAL   acc {val_metrics['accuracy']:.4f}  f1_macro {val_metrics['f1_macro']:.4f}  "
            f"precision_macro {val_metrics['precision_macro']:.4f}  recall_macro {val_metrics['recall_macro']:.4f}"
        )
        log(
            f"TEST  acc {test_metrics['accuracy']:.4f}  f1_macro {test_metrics['f1_macro']:.4f}  "
            f"precision_macro {test_metrics['precision_macro']:.4f}  recall_macro {test_metrics['recall_macro']:.4f}"
        )
        log(
            f"train time {outcome.total_train_seconds:.1f}s over {outcome.epochs_ran} epochs "
            f"({outcome.mean_epoch_seconds:.1f}s/epoch), peak GPU {outcome.peak_gpu_memory_mb:.0f} MB"
        )
        log(
            f"inference: {test_result.images_per_second:.0f} img/s end-to-end (test loader) | "
            f"{speed['images_per_second']:.0f} img/s model-only "
            f"({speed['ms_per_batch']:.1f} ms / batch of {speed['batch_size']})"
        )

        payload = {
            "experiment": config.name,
            "variant": config.variant,
            "description": config.description,
            "model": asdict(info) | {"trainable_fraction": info.trainable_fraction},
            "config": asdict(config),
            "data": {"splits": bundle.split_sizes, "class_names": bundle.class_names},
            "training": {
                "epochs_ran": outcome.epochs_ran,
                "epochs_configured": config.epochs,
                "best_epoch": outcome.best_epoch,
                "early_stopped": outcome.early_stopped,
                "best_val_accuracy": outcome.best_val_accuracy,
                "best_val_f1_macro": outcome.best_val_f1_macro,
            },
            "timing": {
                "train_total_seconds": outcome.total_train_seconds,
                "mean_epoch_seconds": outcome.mean_epoch_seconds,
                # End-to-end: JPEG decode + transform + model, on a warmed-up loader.
                "test_inference_seconds": test_result.seconds,
                "test_images_per_second": test_result.images_per_second,
                # Model-only: synthetic batches already on the GPU.
                "model_ms_per_batch": speed["ms_per_batch"],
                "model_images_per_second": speed["images_per_second"],
                "benchmark_batch_size": speed["batch_size"],
                "run_total_seconds": time.perf_counter() - run_started,
                "peak_gpu_memory_mb": outcome.peak_gpu_memory_mb,
            },
            "val": val_metrics,
            "test": test_metrics,
            "environment": {
                "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
                "torch": torch.__version__,
                "amp_dtype": "bfloat16" if use_amp else "fp32",
            },
        }

        _write_artefacts(config, payload, outcome.history, test_result, bundle, info)
        log(f"artefacts -> {config.run_dir}")
        return payload

    finally:
        log.close()


def _write_artefacts(config, payload, history, test_result, bundle, info) -> None:
    """metrics.json + history.csv + report + confusion matrix + curves."""
    with (config.run_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)

    if history:
        history_frame = pd.DataFrame(
            [
                {
                    key: value
                    for key, value in entry.items()
                    if key != "learning_rates"
                }
                | {
                    "lr_head": entry["learning_rates"][0],
                    "lr_backbone": (
                        entry["learning_rates"][1] if len(entry["learning_rates"]) > 1 else 0.0
                    ),
                }
                for entry in history
            ]
        )
        history_frame.to_csv(config.run_dir / "history.csv", index=False)
        save_history_plot(
            history,
            config.run_dir / "curves.png",
            title=f"{config.variant} {info.backbone_label} ({info.strategy})",
        )

    save_classification_report(
        test_result.y_true,
        test_result.y_pred,
        bundle.class_names,
        config.run_dir / "classification_report_test.txt",
    )
    save_confusion_matrix(
        test_result.y_true,
        test_result.y_pred,
        bundle.class_names,
        config.run_dir / "confusion_matrix_test.png",
        title=f"{config.variant} {info.backbone_label} ({info.strategy}) - test",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", required=True, help="3a | 3b | 3c | 3d")
    parser.add_argument("--backbone", default=None, help="required for 3b; overrides the default for 3c/3d")
    parser.add_argument("--epochs", type=int, default=None, help="override the configured epoch count")
    parser.add_argument("--batch-size", type=int, default=None, help="override the configured batch size")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument(
        "--limit-batches",
        type=int,
        default=None,
        help="stop each epoch after N batches (smoke testing only)",
    )
    parser.add_argument("--no-amp", action="store_true", help="disable bfloat16 mixed precision")
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    config = build_experiment(arguments.experiment, arguments.backbone)
    if arguments.epochs is not None:
        config.epochs = arguments.epochs
    if arguments.batch_size is not None:
        config.batch_size = arguments.batch_size

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_experiment(
        config,
        num_workers=arguments.num_workers,
        limit_batches=arguments.limit_batches,
        use_amp=not arguments.no_amp,
    )


if __name__ == "__main__":
    main()
