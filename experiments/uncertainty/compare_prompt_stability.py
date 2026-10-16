"""Đo bất đồng baseline/DCLIP trên cùng model và mẫu từ hai log Kaggle."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.uncertainty.diagnose_single_model import read_one
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from results_layout import dataset_folder, model_folder, sample_folder, task_folder


POLICY = "prompt_stability_v1"


def compare(base_manifest, base_rows, adapted_manifest, adapted_rows):
    if ((base_manifest.get("dataset"), base_manifest.get("task"), base_manifest.get("model")) !=
            (adapted_manifest.get("dataset"), adapted_manifest.get("task"), adapted_manifest.get("model"))
            or len(base_rows) != len(adapted_rows)):
        raise ValueError("Hai lượt khác model, task hoặc số mẫu")
    keys = ("id", "text", "image_path", "y_true")
    if any(any(a[key] != b[key] for key in keys) for a, b in zip(base_rows, adapted_rows)):
        raise ValueError("Thứ tự mẫu/nhãn không khớp")
    labels = [key.removeprefix("score_") for key in base_rows[0]
              if key.startswith("score_") and not key.startswith(("score_image_", "score_text_"))]
    if len(labels) < 2 or any(f"score_{label}" not in adapted_rows[0] for label in labels):
        raise ValueError("Thứ tự nhãn không khớp")
    truth = np.array([row["y_true"] for row in base_rows])
    def scores(rows, prefix):
        matrix = np.array([[float(row[f"{prefix}{label}"]) for label in labels] for row in rows])
        if not np.isfinite(matrix).all() or not np.allclose(matrix.sum(axis=1), 1, atol=1e-5):
            raise ValueError("Điểm nhánh không phải phân phối hợp lệ")
        return np.asarray(labels)[matrix.argmax(axis=1)]
    base = scores(base_rows, "score_")
    adapted = scores(adapted_rows, "score_")
    if (any(row["y_pred"] != pred for row, pred in zip(base_rows, base))
            or any(row["y_pred"] != pred for row, pred in zip(adapted_rows, adapted))):
        raise ValueError("Nhãn dự đoán không khớp điểm")
    image_base = scores(base_rows, "score_image_")
    image_adapted = scores(adapted_rows, "score_image_")
    error = base != truth
    def detection(flag):
        n = int(flag.sum())
        wrong = int(np.sum(flag & error))
        return {"n_disagreement": n, "baseline_errors_in_disagreement": wrong,
                "baseline_error_rate_disagreement": wrong / n if n else None,
                "baseline_error_rate_agreement": (int(error.sum()) - wrong) / (len(error) - n)
                if n < len(error) else None,
                "error_auroc": float(roc_auc_score(error, flag)) if 0 < error.sum() < len(error) else None}
    return {"policy": POLICY, "dataset": base_manifest["dataset"], "task": base_manifest["task"],
            "model": base_manifest["model"], "n_samples": len(base_rows),
            "baseline_sample_set_id": base_manifest.get("sample_set_id"),
            "adapted_sample_set_id": adapted_manifest.get("sample_set_id"),
            "sample_set_id_equal": base_manifest.get("sample_set_id") == adapted_manifest.get("sample_set_id"),
            "split_verified": bool(base_manifest.get("split_verified", False)
                                   and adapted_manifest.get("split_verified", False)),
            "baseline_accuracy": float(accuracy_score(truth, base)),
            "adapted_accuracy": float(accuracy_score(truth, adapted)),
            "wrong_to_right": int(np.sum(error & (adapted == truth))),
            "right_to_wrong": int(np.sum(~error & (adapted != truth))),
            "final_prediction_disagreement": detection(base != adapted),
            "image_branch_disagreement": detection(image_base != image_adapted)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--adapted-dir", required=True)
    parser.add_argument("--model", help="Chọn model nếu archive baseline có nhiều model")
    args = parser.parse_args()
    bm, br = read_one(args.baseline_dir, args.model)
    am, ar = read_one(args.adapted_dir, args.model)
    report = compare(bm, br, am, ar)
    report["baseline_run"] = args.baseline_dir
    report["adapted_run"] = args.adapted_dir
    output = (Path("results/analysis") / POLICY
              / dataset_folder(report["dataset"], report["task"])
              / task_folder(report["dataset"], report["task"])
              / sample_folder(report["n_samples"])
              / model_folder(models=[report["model"]])
              / (Path(args.baseline_dir).name + "_" + Path(args.adapted_dir).name + "_" + POLICY))
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                         encoding="utf-8")
    print(f"[RESULT] {output / 'summary.json'}")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
