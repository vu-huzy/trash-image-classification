"""Aggregate every finished run into results/summary.csv and results/REPORT.md."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from config import RESULTS_DIR


SUMMARY_COLUMNS = [
    ("variant", "Variant"),
    ("experiment", "Run"),
    ("backbone_label", "Backbone"),
    ("strategy", "Strategy"),
    ("trainable_params", "Trainable params"),
    ("trainable_fraction", "Trainable %"),
    ("test_accuracy", "Test acc"),
    ("test_balanced_accuracy", "Test bal-acc"),
    ("test_precision_macro", "Test P (macro)"),
    ("test_recall_macro", "Test R (macro)"),
    ("test_f1_macro", "Test F1 (macro)"),
    ("test_f1_weighted", "Test F1 (weighted)"),
    ("val_accuracy", "Val acc"),
    ("best_epoch", "Best epoch"),
    ("epochs_ran", "Epochs"),
    ("train_total_seconds", "Train time (s)"),
    ("mean_epoch_seconds", "Epoch (s)"),
    ("model_ms_per_batch", "ms/batch"),
    ("model_images_per_second", "Model img/s"),
    ("test_images_per_second", "Pipeline img/s"),
    ("peak_gpu_memory_mb", "Peak GPU (MB)"),
]


def collect_runs(results_dir: Path = RESULTS_DIR) -> list[dict]:
    """Load every results/<run>/metrics.json, sorted by variant then run name."""
    runs = []
    for metrics_path in sorted(results_dir.glob("*/metrics.json")):
        with metrics_path.open(encoding="utf-8") as file:
            runs.append(json.load(file))
    return sorted(runs, key=lambda run: (run["variant"], run["experiment"]))


def flatten_run(run: dict) -> dict:
    """Pull the reported numbers of one run into a single flat row."""
    return {
        "variant": run["variant"],
        "experiment": run["experiment"],
        "description": run["description"],
        "backbone": run["model"]["backbone"],
        "backbone_label": run["model"]["backbone_label"],
        "strategy": run["model"]["strategy"],
        "total_params": run["model"]["total_params"],
        "trainable_params": run["model"]["trainable_params"],
        "trainable_fraction": run["model"]["trainable_fraction"],
        "lora_params": run["model"].get("lora_params", 0),
        "lora_layers": len(run["model"].get("lora_modules", [])),
        "batch_size": run["config"]["batch_size"],
        "head_lr": run["config"]["head_lr"],
        "backbone_lr": run["config"]["backbone_lr"],
        "epochs_ran": run["training"]["epochs_ran"],
        "best_epoch": run["training"]["best_epoch"],
        "early_stopped": run["training"]["early_stopped"],
        "val_accuracy": run["val"]["accuracy"],
        "val_f1_macro": run["val"]["f1_macro"],
        "test_accuracy": run["test"]["accuracy"],
        "test_balanced_accuracy": run["test"]["balanced_accuracy"],
        "test_precision_macro": run["test"]["precision_macro"],
        "test_recall_macro": run["test"]["recall_macro"],
        "test_f1_macro": run["test"]["f1_macro"],
        "test_precision_weighted": run["test"]["precision_weighted"],
        "test_recall_weighted": run["test"]["recall_weighted"],
        "test_f1_weighted": run["test"]["f1_weighted"],
        "train_total_seconds": run["timing"]["train_total_seconds"],
        "mean_epoch_seconds": run["timing"]["mean_epoch_seconds"],
        "test_inference_seconds": run["timing"]["test_inference_seconds"],
        "test_images_per_second": run["timing"]["test_images_per_second"],
        "model_ms_per_batch": run["timing"].get("model_ms_per_batch", float("nan")),
        "model_images_per_second": run["timing"].get("model_images_per_second", float("nan")),
        "run_total_seconds": run["timing"]["run_total_seconds"],
        "peak_gpu_memory_mb": run["timing"]["peak_gpu_memory_mb"],
    }


def build_summary_frame(runs: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([flatten_run(run) for run in runs])


def _format_cell(key: str, value) -> str:
    if isinstance(value, float):
        if key == "trainable_fraction":
            return f"{value:.2%}"
        if key in {"test_images_per_second", "model_images_per_second", "peak_gpu_memory_mb"}:
            return f"{value:.0f}"
        if key == "model_ms_per_batch" or key.endswith(("_seconds", "_second")):
            return f"{value:.1f}"
        return f"{value:.4f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _markdown_table(frame: pd.DataFrame, columns: list[tuple[str, str]]) -> str:
    keys = [key for key, _ in columns if key in frame.columns]
    headers = [label for key, label in columns if key in frame.columns]

    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join([" --- "] * len(headers)) + "|"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(_format_cell(key, row[key]) for key in keys) + " |")
    return "\n".join(lines)


def _per_class_table(run: dict) -> str:
    per_class = run["test"]["per_class"]
    lines = [
        "| Class | Precision | Recall | F1 | Support |",
        "| --- | --- | --- | --- | --- |",
    ]
    for class_name, scores in per_class.items():
        lines.append(
            f"| {class_name} | {scores['precision']:.4f} | {scores['recall']:.4f} | "
            f"{scores['f1']:.4f} | {scores['support']} |"
        )
    return "\n".join(lines)


def write_report(results_dir: Path = RESULTS_DIR) -> Path | None:
    """Write summary.csv + REPORT.md across all runs found on disk."""
    runs = collect_runs(results_dir)
    if not runs:
        print(f"No metrics.json found under {results_dir}")
        return None

    frame = build_summary_frame(runs)
    frame.to_csv(results_dir / "summary.csv", index=False)

    best_run = max(runs, key=lambda run: run["test"]["accuracy"])
    total_seconds = float(frame["run_total_seconds"].sum())

    sections = [
        "# Model 3 - Transfer learning results",
        "",
        "Pretrained ImageNet backbones on the preprocessed VN trash dataset "
        "(9 classes, 9,542 train / 1,685 val / 864 test images). "
        "Model selection is on validation accuracy; every number below is on the held-out test split "
        "unless stated otherwise.",
        "",
        f"- Runs: **{len(runs)}**",
        f"- Total wall time: **{total_seconds / 60:.1f} min**",
        f"- Best test accuracy: **{best_run['experiment']}** at **{best_run['test']['accuracy']:.4f}** "
        f"(F1 macro {best_run['test']['f1_macro']:.4f})",
        f"- Device: {runs[0]['environment']['device']}, torch {runs[0]['environment']['torch']}, "
        f"AMP {runs[0]['environment']['amp_dtype']}",
        "",
        "## Summary",
        "",
        _markdown_table(frame, SUMMARY_COLUMNS),
        "",
        "## Variants",
        "",
    ]

    for run in runs:
        model = run["model"]
        training = run["training"]
        timing = run["timing"]
        sections += [
            f"### {run['variant']} - {run['experiment']}",
            "",
            f"{run['description']}",
            "",
            f"- Backbone: {model['backbone_label']} ({model['feature_dim']}-d features)",
            f"- Head: {model['head']}",
            f"- Trainable: {model['trainable_params']:,} / {model['total_params']:,} "
            f"({model['trainable_fraction']:.2%})",
        ]
        if model.get("lora_params"):
            sections.append(
                f"- LoRA: rank {run['config']['lora_rank']}, alpha {run['config']['lora_alpha']}, "
                f"{len(model['lora_modules'])} adapted layers, {model['lora_params']:,} adapter params"
            )
        sections += [
            f"- Epochs: {training['epochs_ran']} ran (best at {training['best_epoch']}"
            f"{', early stopped' if training['early_stopped'] else ''})",
            f"- Test: acc {run['test']['accuracy']:.4f} | balanced acc {run['test']['balanced_accuracy']:.4f} | "
            f"P {run['test']['precision_macro']:.4f} | R {run['test']['recall_macro']:.4f} | "
            f"F1 {run['test']['f1_macro']:.4f} (weighted F1 {run['test']['f1_weighted']:.4f})",
            f"- Val: acc {run['val']['accuracy']:.4f} | F1 macro {run['val']['f1_macro']:.4f}",
            f"- Time: {timing['train_total_seconds']:.1f}s training "
            f"({timing['mean_epoch_seconds']:.1f}s/epoch), "
            f"inference {timing.get('model_ms_per_batch', float('nan')):.1f} ms/batch "
            f"= {timing.get('model_images_per_second', float('nan')):.0f} img/s model-only "
            f"({timing['test_images_per_second']:.0f} img/s end-to-end over the test loader), "
            f"peak GPU {timing['peak_gpu_memory_mb']:.0f} MB",
            "",
            "Per-class test scores:",
            "",
            _per_class_table(run),
            "",
            f"Artefacts: `results/{run['experiment']}/` "
            f"(curves.png, confusion_matrix_test.png, classification_report_test.txt, history.csv)",
            "",
        ]

    report_path = results_dir / "REPORT.md"
    report_path.write_text("\n".join(sections), encoding="utf-8")
    print(f"Wrote {report_path} and {results_dir / 'summary.csv'} ({len(runs)} runs)")
    return report_path


if __name__ == "__main__":
    write_report()
