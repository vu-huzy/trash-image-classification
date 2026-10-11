"""Các chỉ số đánh giá dùng cho các mô hình trong Model 2."""

from sklearn.metrics import precision_recall_fscore_support


def calculate_metrics(true_labels, predicted_labels):
    """Tính accuracy, precision, recall và F1 trên tập dự đoán."""
    if len(true_labels) == 0:
        raise ValueError("Danh sách nhãn không được rỗng")

    if len(true_labels) != len(predicted_labels):
        raise ValueError("Số lượng nhãn thật và nhãn dự đoán phải bằng nhau")

    accuracy = sum(
        true == predicted
        for true, predicted in zip(true_labels, predicted_labels)
    ) / len(true_labels)

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
