"""Workflow train/evaluate dùng chung cho mọi thí nghiệm Model 1.

File này không chứa kiến trúc. Mỗi module trong ``model1/models`` truyền vào hàm
``build_model`` của chính nó, còn data loading, metric, early stopping, checkpoint
và report được tái sử dụng nhất quán ở đây.
"""

import copy
import time
from pathlib import Path
from typing import Callable

import torch

from common.data import make_loaders
from common.engine import run_epoch
from common.metrics import classification_report
from common.utils import append_experiment, count_parameters, save_json, set_seed


PROJECT_DIR = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_DIR / "data" / "image_cache_224"
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints" / "model1"
REPORT_DIR = PROJECT_DIR / "reports" / "model1"


def make_optimizer(model: torch.nn.Module, config: dict, use_fused_optimizer: bool):
    """Dùng fused Adam/AdamW trên CUDA nếu phiên bản PyTorch hỗ trợ."""
    optimizer_class = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}.get(
        config["optimizer"].lower()
    )
    if optimizer_class is None:
        raise ValueError(f"Unsupported optimizer: {config['optimizer']}")

    arguments = {"lr": config["learning_rate"], "weight_decay": config["weight_decay"]}
    if use_fused_optimizer:
        try:
            return optimizer_class(model.parameters(), fused=True, **arguments)
        except (RuntimeError, TypeError):
            print("Fused optimizer is unavailable; using the standard optimizer.")
    return optimizer_class(model.parameters(), **arguments)


def run_model_workflow(
    config: dict,
    args,
    device: torch.device,
    class_names: list[str],
    build_model: Callable[[dict, int], torch.nn.Module],
    *,
    is_cnn: bool,
    dropout_description: str,
) -> None:
    """Train một model, chọn checkpoint tốt nhất theo validation loss rồi test đúng một lần.

    ``build_model`` là phần duy nhất thay đổi giữa các M1. Các model vì vậy được
    so sánh công bằng: cùng split dữ liệu, augmentation, metric và early stopping.
    """
    config = dict(config)  # Không làm thay đổi JSON config được nạp bởi model kế tiếp.
    if args.epochs is not None:
        config["max_epochs"] = args.epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.workers is not None:
        config["workers"] = args.workers

    set_seed(config["seed"])
    train_loader, val_loader, test_loader = make_loaders(
        CACHE_DIR, config["batch_size"], config["workers"], config["seed"]
    )
    model = build_model(config, len(class_names)).to(device)
    if is_cnn and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    optimizer = make_optimizer(model, config, use_fused_optimizer=device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    smoke_test = args.max_batches is not None
    checkpoint_path = CHECKPOINT_DIR / f"{config['id']}.pt"
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (PROJECT_DIR / "reports" / "figures").mkdir(parents=True, exist_ok=True)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    print(
        f"\n{'=' * 64}\n{config['id']} | Dropout={dropout_description} | "
        f"parameters={count_parameters(model):,}"
    )
    best_val_loss = float("inf")
    best_epoch = 0
    best_model_state = None
    wait = 0
    history = []
    started_at = time.perf_counter()

    for epoch in range(1, config["max_epochs"] + 1):
        train_stats = run_epoch(
            model,
            train_loader,
            device,
            config["input_size"],
            optimizer=optimizer,
            scaler=scaler,
            max_batches=args.max_batches,
            channels_last=is_cnn,
        )
        val_stats = run_epoch(
            model,
            val_loader,
            device,
            config["input_size"],
            max_batches=args.max_batches,
            channels_last=is_cnn,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_stats["loss"],
                "train_accuracy": train_stats["accuracy"],
                "val_loss": val_stats["loss"],
                "val_accuracy": val_stats["accuracy"],
            }
        )
        print(
            f"Epoch {epoch:02d}/{config['max_epochs']} | "
            f"train loss {train_stats['loss']:.4f}, acc {train_stats['accuracy']:.2%} | "
            f"val loss {val_stats['loss']:.4f}, acc {val_stats['accuracy']:.2%}"
        )

        if val_stats["loss"] < best_val_loss - config["min_delta"]:
            best_val_loss = val_stats["loss"]
            best_epoch = epoch
            wait = 0
            if smoke_test:
                # Smoke test không được ghi đè checkpoint/report của lần train thật.
                best_model_state = copy.deepcopy(model.state_dict())
            else:
                torch.save(
                    {"model_state": model.state_dict(), "config": config, "epoch": epoch},
                    checkpoint_path,
                )
        else:
            wait += 1
            if wait >= config["patience"]:
                print(f"Early stopping at epoch {epoch}.")
                break

    if smoke_test:
        model.load_state_dict(best_model_state)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])

    test_stats = run_epoch(
        model,
        test_loader,
        device,
        config["input_size"],
        max_batches=args.max_batches,
        channels_last=is_cnn,
        collect_predictions=True,
    )
    test_metrics = classification_report(test_stats["targets"], test_stats["predictions"], class_names)
    if device.type == "cuda":
        torch.cuda.synchronize(device)  # CUDA bất đồng bộ; đồng bộ trước khi tính thời gian.
    elapsed_minutes = (time.perf_counter() - started_at) / 60
    peak_vram_gb = (
        torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0
    )

    summary = {
        "model": config["id"],
        "dropout": dropout_description,
        "parameters": count_parameters(model),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "test_loss": test_stats["loss"],
        "test_accuracy": test_stats["accuracy"],
        "minutes": elapsed_minutes,
        "peak_vram_gb": peak_vram_gb,
    }
    if not smoke_test:
        save_json(
            REPORT_DIR / f"{config['id'].lower().replace('-', '_')}.json",
            {"config": config, "history": history, "summary": summary, "test_metrics": test_metrics},
        )
        append_experiment(
            PROJECT_DIR / "reports" / "experiments.csv",
            {
                **summary,
                "macro_precision": test_metrics["macro_precision"],
                "macro_recall": test_metrics["macro_recall"],
                "macro_f1": test_metrics["macro_f1"],
                "weighted_f1": test_metrics["weighted_f1"],
            },
        )
    else:
        print("Smoke test: checkpoint and report were not saved.")
    print(
        f"{config['id']} test | accuracy {test_stats['accuracy']:.2%} | "
        f"macro F1 {test_metrics['macro_f1']:.4f} | peak VRAM {peak_vram_gb:.2f} GB"
    )

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
