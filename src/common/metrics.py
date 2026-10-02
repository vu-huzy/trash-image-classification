from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


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
