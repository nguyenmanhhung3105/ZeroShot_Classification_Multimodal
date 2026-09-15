"""
logging_utils.py
========================================================================
Lưu log chi tiết từng mẫu sau mỗi lần chạy inference, phục vụ phân tích
lỗi (error analysis) để tinh chỉnh prompt — KHÔNG chỉ lưu điểm tổng hợp.

2 loại file được lưu mỗi lần chạy (1 model x 1 dataset x 1 phiên bản prompt):
  1. raw_predictions  : TOÀN BỘ mẫu, kèm điểm số thô từng lớp -> dùng để
                        tra cứu lại bất kỳ mẫu nào, vẽ biểu đồ threshold...
  2. error_analysis   : CHỈ những mẫu bị đoán SAI, sắp xếp theo độ tự tin
                        giảm dần -> ưu tiên xem các ca "tự tin nhưng sai"
                        trước, vì đó là dấu hiệu prompt đang hiểu nhầm
                        khái niệm gì đó một cách hệ thống.
========================================================================
"""

import os
import json
from datetime import datetime
import numpy as np
import pandas as pd


def _make_run_id(model_name: str, dataset_name: str, task_name: str, prompt_version: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{model_name}__{dataset_name}__{task_name}__{prompt_version}__{ts}"


def save_single_label_log(df_valid: pd.DataFrame, y_true: list, y_pred: list, sims: np.ndarray,
                           label_order: list, id_col: str, text_col: str,
                           model_name: str, dataset_name: str, task_name: str,
                           output_dir: str, prompt_version: str = "v1") -> str:
    """Dùng cho Fakeddit / CrisisMMD (single-label)."""
    run_id = _make_run_id(model_name, dataset_name, task_name, prompt_version)
    run_dir = os.path.join(output_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    rows = []
    for i in range(len(y_true)):
        row = {
            "id": df_valid.iloc[i][id_col],
            "text": df_valid.iloc[i][text_col],
            "image_path": df_valid.iloc[i].get("image_path", ""),
            "y_true": y_true[i],
            "y_pred": y_pred[i],
            "correct": y_true[i] == y_pred[i],
            "confidence": float(np.max(sims[i])),          # điểm của lớp được chọn
            "margin": float(np.sort(sims[i])[-1] - np.sort(sims[i])[-2]),  # khoảng cách top1 vs top2
        }
        # lưu thêm điểm thô từng lớp để tiện tra cứu / vẽ biểu đồ sau này
        for j, label in enumerate(label_order):
            row[f"score_{label}"] = float(sims[i, j])
        rows.append(row)

    df_log = pd.DataFrame(rows)
    raw_path = os.path.join(run_dir, "raw_predictions.tsv")
    df_log.to_csv(raw_path, sep="\t", index=False)

    # error_analysis: chỉ mẫu sai, sắp theo confidence giảm dần
    # (sai mà điểm tự tin cao -> dấu hiệu prompt/model hiểu nhầm khái niệm, đáng xem trước)
    df_error = df_log[~df_log["correct"]].sort_values("confidence", ascending=False)
    error_path = os.path.join(run_dir, "error_analysis.tsv")
    df_error.to_csv(error_path, sep="\t", index=False)

    _save_run_manifest(run_dir, model_name, dataset_name, task_name, prompt_version,
                        n_total=len(df_log), n_errors=len(df_error))

    print(f"Đã lưu log tại: {run_dir}  ({len(df_error)}/{len(df_log)} mẫu sai)")
    return run_dir


def save_multi_label_log(df_valid: pd.DataFrame, y_true: list, y_pred: list, probs: np.ndarray,
                          label_order: list, id_col: str, text_col: str,
                          model_name: str, dataset_name: str, task_name: str,
                          output_dir: str, prompt_version: str = "v1") -> str:
    """Dùng cho MM-IMDb (multi-label) — 'sai' được tách thành thiếu nhãn / thừa nhãn."""
    run_id = _make_run_id(model_name, dataset_name, task_name, prompt_version)
    run_dir = os.path.join(output_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    rows = []
    for i in range(len(y_true)):
        true_set = set(y_true[i])
        pred_set = set(y_pred[i])
        missing = true_set - pred_set     # model bỏ sót genre đúng ra phải có
        extra = pred_set - true_set       # model gắn thừa genre không có thật

        row = {
            "id": df_valid.iloc[i][id_col],
            "text": df_valid.iloc[i][text_col],
            "image_path": df_valid.iloc[i].get("image_path", ""),
            "y_true": "|".join(sorted(true_set)),
            "y_pred": "|".join(sorted(pred_set)),
            "missing_genres": "|".join(sorted(missing)),   # thiếu gì
            "extra_genres": "|".join(sorted(extra)),        # thừa gì
            "n_missing": len(missing),
            "n_extra": len(extra),
            "exact_match": true_set == pred_set,
        }
        for j, label in enumerate(label_order):
            row[f"prob_{label}"] = float(probs[i, j])
        rows.append(row)

    df_log = pd.DataFrame(rows)
    raw_path = os.path.join(run_dir, "raw_predictions.tsv")
    df_log.to_csv(raw_path, sep="\t", index=False)

    # error_analysis: chỉ mẫu KHÔNG khớp hoàn toàn, sắp theo tổng lỗi (thiếu+thừa) giảm dần
    df_error = df_log[~df_log["exact_match"]].copy()
    df_error["total_errors"] = df_error["n_missing"] + df_error["n_extra"]
    df_error = df_error.sort_values("total_errors", ascending=False)
    error_path = os.path.join(run_dir, "error_analysis.tsv")
    df_error.to_csv(error_path, sep="\t", index=False)

    _save_run_manifest(run_dir, model_name, dataset_name, task_name, prompt_version,
                        n_total=len(df_log), n_errors=len(df_error))

    print(f"Đã lưu log tại: {run_dir}  ({len(df_error)}/{len(df_log)} mẫu không khớp hoàn toàn)")
    return run_dir


def _save_run_manifest(run_dir: str, model_name: str, dataset_name: str, task_name: str,
                        prompt_version: str, n_total: int, n_errors: int) -> None:
    """
    Ghi lại 'bối cảnh' của lần chạy này — để sau này nhìn vào thư mục log
    biết ngay đây là chạy với model/prompt/threshold nào, không cần đoán
    qua tên file. Đây là phần quan trọng để SO SÁNH GIỮA CÁC LẦN chỉnh prompt.
    """
    manifest = {
        "model": model_name,
        "dataset": dataset_name,
        "task": task_name,
        "prompt_version": prompt_version,
        "n_total_samples": n_total,
        "n_errors": n_errors,
        "error_rate": n_errors / n_total if n_total else None,
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(run_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)