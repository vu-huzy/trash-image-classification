"""Metric computation and reporting artefacts for one evaluated split.

Every run records accuracy, balanced accuracy, and precision / recall / F1 in
both macro (treats all 9 classes equally) and weighted (weights by class
support) form, plus the per-class breakdown, a confusion matrix image and a
text classification report.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no GUI backend - these plots are written to disk

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> dict:
    """Accuracy plus macro/weighted/per-class precision, recall and F1."""
    labels = list(range(len(class_names)))

    macro = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    weighted = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )
    per_class = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(macro[0]),
        "recall_macro": float(macro[1]),
        "f1_macro": float(macro[2]),
        "precision_weighted": float(weighted[0]),
        "recall_weighted": float(weighted[1]),
        "f1_weighted": float(weighted[2]),
        "per_class": {
            class_name: {
                "precision": float(per_class[0][index]),
                "recall": float(per_class[1][index]),
                "f1": float(per_class[2][index]),
                "support": int(per_class[3][index]),
            }
            for index, class_name in enumerate(class_names)
        },
    }


def save_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    output_path: Path,
) -> None:
    """Write sklearn's text report (the familiar per-class table)."""
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    output_path.write_text(report, encoding="utf-8")


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    output_path: Path,
    title: str = "Confusion matrix",
) -> None:
    """Render a row-normalised confusion matrix (recall per class on the diagonal)."""
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalised = np.divide(matrix, row_sums, out=np.zeros_like(matrix, dtype=float), where=row_sums > 0)

    figure, axes = plt.subplots(figsize=(9, 7.5))
    image = axes.imshow(normalised, cmap="Blues", vmin=0.0, vmax=1.0)

    axes.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    axes.set_yticks(range(len(class_names)), class_names)
    axes.set_xlabel("Predicted")
    axes.set_ylabel("True")
    axes.set_title(title)

    for row in range(len(class_names)):
        for column in range(len(class_names)):
            share = normalised[row, column]
            if share <= 0:
                continue
            axes.text(
                column,
                row,
                f"{share:.2f}\n({matrix[row, column]})",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if share > 0.5 else "black",
            )

    figure.colorbar(image, ax=axes, fraction=0.046, label="Share of true class")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_history_plot(history: list[dict], output_path: Path, title: str) -> None:
    """Plot train/val loss and accuracy across epochs."""
    if not history:
        return

    epochs = [entry["epoch"] for entry in history]
    figure, (loss_axes, accuracy_axes) = plt.subplots(1, 2, figsize=(12, 4.5))

    loss_axes.plot(epochs, [entry["train_loss"] for entry in history], label="train")
    loss_axes.plot(epochs, [entry["val_loss"] for entry in history], label="val")
    loss_axes.set_xlabel("Epoch")
    loss_axes.set_ylabel("Loss")
    loss_axes.set_title("Loss")
    loss_axes.legend()
    loss_axes.grid(alpha=0.3)

    accuracy_axes.plot(epochs, [entry["train_accuracy"] for entry in history], label="train")
    accuracy_axes.plot(epochs, [entry["val_accuracy"] for entry in history], label="val")
    accuracy_axes.set_xlabel("Epoch")
    accuracy_axes.set_ylabel("Accuracy")
    accuracy_axes.set_title("Accuracy")
    accuracy_axes.legend()
    accuracy_axes.grid(alpha=0.3)

    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
