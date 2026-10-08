"""Các metric dùng chung cho bốn model M1."""

from sklearn.metrics import precision_recall_fscore_support


def calculate_metrics(true_labels, predicted_labels):
    """Tính accuracy, macro precision/recall/F1 và weighted F1 từ nhãn test."""
    correct = sum(true == predicted for true, predicted in zip(true_labels, predicted_labels))
    accuracy = correct / len(true_labels)

    # Macro coi mọi lớp có cùng trọng số; weighted F1 phản ánh cả số mẫu mỗi lớp.
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        average="macro",
        zero_division=0,
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        average="weighted",
        zero_division=0,
    )

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }
