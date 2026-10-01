"""CLI điều phối bốn Model 1; kiến trúc nằm trong ``model1/models``.

Chạy toàn bộ Model 1:
    python src/model1/train.py

Smoke test (một batch/model, không ghi checkpoint hay report):
    python src/model1/train.py --epochs 1 --max-batches 1 --batch-size 32
"""

import argparse
import json
import sys
from pathlib import Path

import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model1.registry import get_model_module


MODEL_IDS = ("M1-01", "M1-02", "M1-03", "M1-04")
CACHE_DIR = PROJECT_DIR / "data" / "image_cache_224"
CONFIG_DIR = PROJECT_DIR / "configs"


def load_config(model_id: str) -> dict:
    """Nạp hyperparameter của model từ config JSON tương ứng."""
    file_name = model_id.lower().replace("-", "_") + ".json"
    with (CONFIG_DIR / file_name).open(encoding="utf-8") as file:
        return json.load(file)


def configure_cuda(device: torch.device) -> None:
    """Bật các tối ưu CUDA áp dụng chung cho tất cả CNN/MLP M1."""
    if device.type != "cuda":
        return
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("CUDA acceleration: AMP FP16 | TF32 | channels-last CNN | fused optimizer | pinned-memory prefetch")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Model 1 Simple NN / Simple CNN variants")
    parser.add_argument("--models", nargs="+", choices=MODEL_IDS, default=MODEL_IDS)
    parser.add_argument("--epochs", type=int, default=None, help="Override max_epochs in JSON configs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch_size in JSON configs")
    parser.add_argument("--workers", type=int, default=None, help="Override workers in JSON configs")
    parser.add_argument("--max-batches", type=int, default=None, help="Use 1 for a quick smoke test")
    args = parser.parse_args()

    if not CACHE_DIR.exists():
        raise FileNotFoundError(f"Missing image cache: {CACHE_DIR}")
    with (CACHE_DIR / "metadata.json").open(encoding="utf-8") as file:
        class_names = json.load(file)["classes"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    configure_cuda(device)

    for model_id in args.models:
        config = load_config(model_id)
        model_module = get_model_module(config["architecture"])
        print(f"\nSelected {model_module.MODEL_ID}: {model_module.ARCHITECTURE}")
        model_module.run(config, args, device, class_names)


if __name__ == "__main__":
    main()
