from pathlib import Path

import matplotlib

import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report as sklearn_classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> dict:
    """Return accuracy and macro, weighted, and per-class classification metrics."""
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
        "balanced_accuracy": float(
            np.mean(per_class[1][per_class[3] > 0])
            if np.any(per_class[3] > 0)
            else 0.0
        ),
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


def classification_report(targets, predictions, class_names):
    """Return the test metrics required by docs/model.md as JSON-safe values."""
    labels = list(range(len(class_names)))
    precision, recall, f1, support = precision_recall_fscore_support(
        targets,
        predictions,
        labels=labels,
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        targets, predictions, average="macro", zero_division=0
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        targets, predictions, average="weighted", zero_division=0
    )
    return {
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "confusion_matrix": confusion_matrix(targets, predictions, labels=labels).tolist(),
        "per_class": {
            class_name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
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
    """Write a text classification report with a stable class order."""
    report = sklearn_classification_report(
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
    """Save a row-normalised confusion matrix and its raw counts."""
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalised = np.divide(
        matrix,
        row_sums,
        out=np.zeros_like(matrix, dtype=float),
        where=row_sums > 0,
    )

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
            if share > 0:
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
    """Save loss and accuracy curves for a training run."""
    if not history:
        return

    epochs = [entry["epoch"] for entry in history]
    figure, (loss_axes, accuracy_axes) = plt.subplots(1, 2, figsize=(12, 4.5))
    loss_axes.plot(epochs, [entry["train_loss"] for entry in history], label="train")
    loss_axes.plot(epochs, [entry["val_loss"] for entry in history], label="val")
    loss_axes.set(xlabel="Epoch", ylabel="Loss", title="Loss")
    loss_axes.legend()
    loss_axes.grid(alpha=0.3)

    accuracy_axes.plot(
        epochs, [entry["train_accuracy"] for entry in history], label="train"
    )
    accuracy_axes.plot(
        epochs, [entry["val_accuracy"] for entry in history], label="val"
    )
    accuracy_axes.set(xlabel="Epoch", ylabel="Accuracy", title="Accuracy")
    accuracy_axes.legend()
    accuracy_axes.grid(alpha=0.3)

    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
