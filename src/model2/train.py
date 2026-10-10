"""Train and evaluate scratch CNN architectures on the 224x224 trash dataset."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from collections import Counter
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

HERE = Path(__file__).resolve().parent
SRC = HERE.parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from model2.models import build_models

PROJECT = HERE.parent.parent
DEFAULT_DATA = PROJECT / "data" / "VN_trash_classification_preprocessing"
SUMMARY_PATH = HERE / "results" / "summary.csv"
SUMMARY_FIELDS = [
    "model", "run_number", "epochs_run", "best_epoch", "best_val_accuracy",
    "test_accuracy", "test_loss", "elapsed_seconds", "checkpoint_bytes",
    "checkpoint_mib", "run_timestamp",
]


def get_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable")
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_loader(path: Path, batch_size: int, train: bool, workers: int) -> DataLoader:
    ops = [transforms.Resize((224, 224))]
    if train:
        ops.extend([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        ])
    ops.extend([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset = ImageFolder(path, transform=transforms.Compose(ops))
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=train, num_workers=workers,
        pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0,
    )


def epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    use_amp: bool = False,
    grad_clip: float = 1.0,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    loss_sum, correct, seen = 0.0, 0, 0
    grad_context = torch.enable_grad() if training else torch.inference_mode()
    amp_context = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                   if use_amp and device.type == "cuda" else nullcontext())
    with grad_context:
        with amp_context:
            for batch_idx, (images, labels) in enumerate(loader, 1):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                if training:
                    optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = loss_fn(logits, labels)
                if not torch.isfinite(logits).all().item() or not torch.isfinite(loss).item():
                    raise RuntimeError(
                        f"Non-finite logits/loss at batch {batch_idx}; "
                        "try lower --lr and keep --amp off"
                    )
                if training:
                    loss.backward()
                    if grad_clip > 0:
                        norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip,
                                                        error_if_nonfinite=True)
                        if batch_idx == 1:
                            print(f"  Gradient norm (first batch): {norm.item():.4f}")
                    optimizer.step()
                count = labels.size(0)
                loss_sum += float(loss.item()) * count
                correct += int((logits.argmax(dim=1) == labels).sum().item())
                seen += count
    return loss_sum / max(seen, 1), correct / max(seen, 1)


def per_class_accuracy(model: nn.Module, loader: DataLoader,
                       device: torch.device) -> dict[str, float]:
    model.eval()
    correct, total = Counter(), Counter()
    with torch.inference_mode():
        for images, labels in loader:
            predictions = model(images.to(device)).argmax(1).cpu()
            for label, prediction in zip(labels, predictions):
                name = loader.dataset.classes[int(label.item())]
                total[name] += 1
                correct[name] += int(label.item() == prediction.item())
    return {name: correct[name] / total[name] for name in loader.dataset.classes if total[name]}


def update_summary(result: dict, checkpoint: Path) -> None:
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing_rows = []
    if SUMMARY_PATH.exists():
        with SUMMARY_PATH.open("r", newline="", encoding="utf-8-sig") as f:
            existing_rows = list(csv.DictReader(f))
    else:
        # Preserve compatibility with old one-file-per-model results.
        for old in sorted((HERE / "results").glob("*.json")):
            try:
                data = json.loads(old.read_text(encoding="utf-8"))
                if "model" not in data:
                    continue
                old_ckpt = HERE / "checkpoints" / f"{data['model']}.pt"
                size = old_ckpt.stat().st_size if old_ckpt.exists() else None
                existing_rows.append({
                    "model": data["model"], "run_number": 1,
                    "epochs_run": len(data.get("history", [])),
                    "best_epoch": data.get("best_epoch", ""),
                    "best_val_accuracy": data.get("best_val_accuracy", ""),
                    "test_accuracy": data.get("test_accuracy", ""),
                    "test_loss": data.get("test_loss", ""),
                    "elapsed_seconds": data.get("elapsed_seconds", ""),
                    "checkpoint_bytes": size if size is not None else "",
                    "checkpoint_mib": round(size / 1024**2, 3) if size is not None else "",
                    "run_timestamp": "existing result",
                })
            except (OSError, ValueError, TypeError, KeyError):
                continue
    run_number = 1 + sum(row.get("model") == result["model"] for row in existing_rows)
    size = checkpoint.stat().st_size
    row = {
        "model": result["model"], "run_number": run_number,
        "epochs_run": len(result["history"]), "best_epoch": result["best_epoch"],
        "best_val_accuracy": result["best_val_accuracy"],
        "test_accuracy": result["test_accuracy"], "test_loss": result["test_loss"],
        "elapsed_seconds": result["elapsed_seconds"], "checkpoint_bytes": size,
        "checkpoint_mib": round(size / 1024**2, 3),
        "run_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with SUMMARY_PATH.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows([{key: r.get(key, "") for key in SUMMARY_FIELDS}
                          for r in (*existing_rows, row)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(build_models().keys()), required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--class-balanced", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--early-stopping", type=int, default=12)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", help="Enable CUDA bfloat16 autocast; off by default")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error("--epochs, --batch-size and --lr must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = get_device(args.device)
    use_amp = args.amp and device.type == "cuda"
    if use_amp and not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA bfloat16 not supported; remove --amp")
    print(f"PyTorch: {torch.__version__} | Device: {device} | AMP: {use_amp} | LR: {args.lr}")
    for split in ("train", "val", "test"):
        if not (args.data_dir / split).is_dir():
            raise FileNotFoundError(f"Missing {args.data_dir / split}; run preprocessing first")
    train_loader = make_loader(args.data_dir / "train", args.batch_size, True, args.workers)
    val_loader = make_loader(args.data_dir / "val", args.batch_size, False, args.workers)
    test_loader = make_loader(args.data_dir / "test", args.batch_size, False, args.workers)
    classes = train_loader.dataset.classes
    if any(loader.dataset.classes != classes for loader in (val_loader, test_loader)):
        raise ValueError("Class names/order differ between train, val and test")
    print(f"Classes ({len(classes)}): {classes}")
    print(f"Samples: train={len(train_loader.dataset)} val={len(val_loader.dataset)} test={len(test_loader.dataset)}")
    if not all(len(loader.dataset) > 0 for loader in (train_loader, val_loader, test_loader)):
        raise ValueError("A dataset split is empty")
    model = build_models(len(classes), pretrained=args.pretrained)[args.model].to(device)
    counts = Counter(train_loader.dataset.targets)
    print("Training class counts:", {classes[i]: counts[i] for i in range(len(classes))})
    if any(counts[i] == 0 for i in range(len(classes))):
        raise ValueError("Training set has an empty class")
    weights = None
    if args.class_balanced:
        weights = torch.tensor([
            len(train_loader.dataset) / (len(classes) * counts[i])
            for i in range(len(classes))
        ], device=device, dtype=torch.float32)
    is_cnn = args.model.startswith("cnn_")
    label_smoothing = 0.0 if is_cnn else 0.05
    weight_decay = 1e-4 if is_cnn else 5e-4
    loss_fn = nn.CrossEntropyLoss(
        weight=weights,
        label_smoothing=label_smoothing,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    history = []
    best_accuracy, best_loss, best_epoch = -1.0, float("inf"), 0
    stalled = 0
    started = time.perf_counter()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint = HERE / "checkpoints" / f"{args.model}_{run_id}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for current_epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {current_epoch}/{args.epochs}")
        train_loss, train_accuracy = epoch(
            model, train_loader, loss_fn, device, optimizer, use_amp, args.grad_clip)
        val_loss, val_accuracy = epoch(model, val_loader, loss_fn, device, use_amp=use_amp)
        row = {
            "epoch": current_epoch, "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss, "train_accuracy": train_accuracy,
            "val_loss": val_loss, "val_accuracy": val_accuracy,
        }
        history.append(row)
        print(json.dumps(row, indent=2))
        improved = val_accuracy > best_accuracy or (
            val_accuracy == best_accuracy and val_loss < best_loss)
        if improved:
            best_accuracy, best_loss, best_epoch = val_accuracy, val_loss, current_epoch
            stalled = 0
            torch.save({
                "model": model.state_dict(), "model_name": args.model,
                "class_names": classes, "epoch": current_epoch,
                "val_accuracy": val_accuracy, "val_loss": val_loss,
            }, checkpoint)
            print(f"Best checkpoint saved: {checkpoint}")
        else:
            stalled += 1
        scheduler.step()
        if args.early_stopping > 0 and stalled >= args.early_stopping:
            print(f"Early stopping at epoch {current_epoch}")
            break
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
    model.to(device)
    test_loss, test_accuracy = epoch(model, test_loader, loss_fn, device, use_amp=use_amp)
    test_per_class = per_class_accuracy(model, test_loader, device)
    elapsed = time.perf_counter() - started
    result = {
        "model": args.model, "class_names": classes, "best_epoch": best_epoch,
        "best_val_accuracy": best_accuracy, "best_val_loss": best_loss,
        "test_loss": test_loss, "test_accuracy": test_accuracy,
        "test_per_class_accuracy": test_per_class,
        "device": str(device), "class_balanced": args.class_balanced,
        "pretrained": args.pretrained, "amp": use_amp,
        "lr": args.lr, "grad_clip": args.grad_clip,
        "label_smoothing": label_smoothing, "weight_decay": weight_decay,
        "elapsed_seconds": elapsed, "history": history,
        "checkpoint": str(checkpoint),
    }
    # Keep old result path for scripts that read results/<model>.json.
    result_path = HERE / "results" / f"{args.model}.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    update_summary(result, checkpoint)
    print("\nTRAINING FINISHED")
    print(json.dumps({key: value for key, value in result.items() if key != "history"}, indent=2))
    print(f"Result: {result_path}\nSummary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()

