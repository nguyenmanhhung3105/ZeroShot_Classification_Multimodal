"""
Evaluation — đúng bộ chỉ số đã chốt:
- Accuracy: chỉ để tham khảo, KHÔNG dùng làm kết luận chính (dataset lệch lớp)
- Macro-F1: chỉ số CHÍNH cho mọi bài single-label lệch lớp
- AUROC: riêng cho binary classification
- Micro/Macro-F1 multi-label: riêng cho MM-IMDb
"""

from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report,
)
import numpy as np


def evaluate_single_label(y_true: list, y_pred: list, label_order: list,
                           low_sample_classes: set = None) -> dict:
    """
    Dùng cho Fakeddit và CrisisMMD (single-label classification).

    low_sample_classes: tập các lớp có cỡ mẫu quá nhỏ (đã phát hiện ở CrisisMMD
    Humanitarian) — nếu có, sẽ in cảnh báo riêng, không loại khỏi kết quả.
    """
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=label_order, zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average="micro", labels=label_order, zero_division=0)

    result = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
    }

    print(f"Accuracy : {acc:.4f}  (chỉ tham khảo, có thể gây hiểu lầm nếu lệch lớp)")
    print(f"Macro-F1 : {macro_f1:.4f}  (chỉ số chính, phản ánh đúng cả lớp hiếm)")
    print(f"Micro-F1 : {micro_f1:.4f}")

    print("\nBáo cáo chi tiết theo từng lớp:")
    print(classification_report(y_true, y_pred, labels=label_order, zero_division=0))

    if low_sample_classes:
        print(f"\n⚠️  CẢNH BÁO: các lớp sau có cỡ mẫu quá nhỏ, kết quả không đủ tin cậy "
              f"thống kê để kết luận chắc chắn: {low_sample_classes}")

    return result


def evaluate_binary(y_true: list, y_pred: list, y_scores: np.ndarray,
                     positive_label) -> dict:
    """
    Dùng riêng cho bài toán binary (CrisisMMD Informativeness, Fakeddit 2-way).
    y_scores: similarity score thô của lớp positive_label (chưa qua argmax),
    dùng để tính AUROC không phụ thuộc ngưỡng.
    """
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, pos_label=positive_label, zero_division=0)

    y_true_binary = [1 if y == positive_label else 0 for y in y_true]
    try:
        auroc = roc_auc_score(y_true_binary, y_scores)
    except ValueError as e:
        print(f"⚠️ Không tính được AUROC: {e}")
        auroc = None

    print(f"Accuracy : {acc:.4f}")
    print(f"F1       : {f1:.4f}")
    print(f"AUROC    : {auroc:.4f}" if auroc is not None else "AUROC    : N/A")

    return {"accuracy": acc, "f1": f1, "auroc": auroc}


def evaluate_multi_label(y_true: list, y_pred: list, label_order: list) -> dict:
    """
    Dùng riêng cho MM-IMDb (multi-label).
    y_true, y_pred: list các list label, ví dụ y_true[0] = ["Comedy", "Family"]
    """
    def to_binary_matrix(label_lists):
        matrix = np.zeros((len(label_lists), len(label_order)))
        for i, labels in enumerate(label_lists):
            for label in labels:
                if label in label_order:
                    matrix[i, label_order.index(label)] = 1
        return matrix

    y_true_matrix = to_binary_matrix(y_true)
    y_pred_matrix = to_binary_matrix(y_pred)

    macro_f1 = f1_score(y_true_matrix, y_pred_matrix, average="macro", zero_division=0)
    micro_f1 = f1_score(y_true_matrix, y_pred_matrix, average="micro", zero_division=0)
    # Hamming accuracy: tỷ lệ nhãn dự đoán đúng trên tổng số nhãn (đúng cả có/không thuộc lớp)
    hamming_acc = (y_true_matrix == y_pred_matrix).mean()

    print(f"Macro-F1      : {macro_f1:.4f}  (chỉ số chính cho multi-label)")
    print(f"Micro-F1      : {micro_f1:.4f}")
    print(f"Hamming Acc.  : {hamming_acc:.4f}")

    return {"macro_f1": macro_f1, "micro_f1": micro_f1, "hamming_accuracy": hamming_acc}
