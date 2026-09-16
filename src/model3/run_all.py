"""Run the whole model3 study in order: 3A -> 3B (x3) -> 3C -> 3D.

3C and 3D reuse whichever 3B backbone reached the highest *validation* accuracy,
so the LoRA and full-fine-tune strategies are compared against the frozen
baseline on identical features.

    .venv/Scripts/python.exe src/model3/run_all.py                  # full study
    .venv/Scripts/python.exe src/model3/run_all.py --epochs 1 --limit-batches 3   # smoke test
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    RESULTS_DIR,
    ExperimentConfig,
    NUM_WORKERS,
    experiment_3a,
    experiment_3c,
    experiment_3d,
    experiments_3b,
)
from data import build_dataloaders
from report import write_report
from train import run_experiment


def _apply_overrides(config: ExperimentConfig, arguments: argparse.Namespace) -> ExperimentConfig:
    if arguments.epochs is not None:
        config.epochs = arguments.epochs
    if arguments.batch_size is not None:
        config.batch_size = arguments.batch_size
    return config


def main() -> None:
    arguments = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    study_started = time.perf_counter()

    # DataLoaders are cached per batch size: 3A/3B use one size, 3C/3D another.
    bundles: dict[int, object] = {}

    def bundle_for(config: ExperimentConfig):
        if config.batch_size not in bundles:
            bundles[config.batch_size] = build_dataloaders(
                batch_size=config.batch_size, num_workers=arguments.num_workers
            )
        return bundles[config.batch_size]

    def execute(config: ExperimentConfig) -> dict | None:
        config = _apply_overrides(config, arguments)
        if arguments.skip_existing and (config.run_dir / "metrics.json").exists():
            print(f"--- skipping {config.name} (metrics.json already exists)")
            return None
        print(f"\n{'=' * 78}\n{config.variant}: {config.name}\n{'=' * 78}")
        try:
            return run_experiment(
                config,
                bundle=bundle_for(config),
                num_workers=arguments.num_workers,
                limit_batches=arguments.limit_batches,
                use_amp=not arguments.no_amp,
            )
        except Exception:
            config.run_dir.mkdir(parents=True, exist_ok=True)
            (config.run_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            print(f"!!! {config.name} failed; traceback saved to {config.run_dir / 'error.txt'}")
            traceback.print_exc()
            return None

    selected = set(arguments.only) if arguments.only else {"3a", "3b", "3c", "3d"}
    results: list[dict] = []

    # ---- 3A: frozen light backbone -------------------------------------------------
    if "3a" in selected:
        payload = execute(experiment_3a())
        if payload:
            results.append(payload)

    # ---- 3B: frozen stronger backbones ---------------------------------------------
    payloads_3b: list[dict] = []
    if "3b" in selected:
        for config in experiments_3b():
            payload = execute(config)
            if payload:
                payloads_3b.append(payload)
                results.append(payload)

    # ---- pick the 3B winner on validation accuracy ---------------------------------
    winner = arguments.winner
    if winner is None and payloads_3b:
        best = max(payloads_3b, key=lambda run: run["training"]["best_val_accuracy"])
        winner = best["model"]["backbone"]
        print(
            f"\n>>> 3B winner: {best['model']['backbone_label']} "
            f"(val acc {best['training']['best_val_accuracy']:.4f}) -> used for 3C and 3D"
        )

    if winner is None and selected & {"3c", "3d"}:
        print(
            "\n!!! No 3B result available to choose a backbone from. "
            "Re-run with --winner <backbone> to force one."
        )
    else:
        # ---- 3C: LoRA on the winning backbone --------------------------------------
        if "3c" in selected:
            payload = execute(experiment_3c(winner))
            if payload:
                results.append(payload)

        # ---- 3D: full fine-tune of the winning backbone ----------------------------
        if "3d" in selected:
            payload = execute(experiment_3d(winner))
            if payload:
                results.append(payload)

    print(f"\n{'=' * 78}")
    print(f"Study finished in {(time.perf_counter() - study_started) / 60:.1f} min")
    write_report()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        metavar="VARIANT",
        help="run only these variants, e.g. --only 3c 3d",
    )
    parser.add_argument(
        "--winner",
        default=None,
        help="force the backbone used by 3C/3D instead of taking the best 3B run",
    )
    parser.add_argument("--epochs", type=int, default=None, help="override every experiment's epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="override every batch size")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument(
        "--limit-batches", type=int, default=None, help="stop each epoch after N batches (smoke test)"
    )
    parser.add_argument(
        "--skip-existing", action="store_true", help="skip runs that already wrote metrics.json"
    )
    parser.add_argument("--no-amp", action="store_true", help="disable bfloat16 mixed precision")
    return parser.parse_args()


if __name__ == "__main__":
    main()
