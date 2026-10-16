"""Thử split conformal trên log Fakeddit 2way đã lưu; không chạy model.

Nhãn thật chỉ dùng cho phần calibration của từng fold và báo cáo hậu nghiệm.
Lượt này là phân tích thăm dò, không chứng nhận coverage trên dữ liệu mới.
"""

import argparse
import csv
import io
import json
import math
from pathlib import Path
import shutil
import tempfile
import zipfile

import numpy as np

from analyze_review import ROOT, _prediction_member, analysis_path
from sweep_fusion_weight import _group_folds


POLICY = "split_conformal_group5_v2"


def conformal_threshold(scores, alpha):
    """Lượng tử hữu hạn-sample của split conformal; k>n cho ngưỡng vô hạn."""
    if not 0 < alpha < 1:
        raise ValueError("alpha phải nằm trong (0, 1)")
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Điểm calibration phải là vector hữu hạn không rỗng")
    k = math.ceil((len(values) + 1) * (1 - alpha))
    return (float(np.sort(values)[k - 1]) if k <= len(values) else math.inf), k


def _read_log(run_dir, model=None):
    archive_path = run_dir / "run_artifacts.zip"
    receipt = json.loads((run_dir / "run_receipt.json").read_text(encoding="utf-8"))
    if receipt.get("run_id") != run_dir.name:
        raise ValueError("run_receipt.json không khớp thư mục")
    with zipfile.ZipFile(archive_path) as archive:
        member = _prediction_member(archive, model)
        manifest = json.loads(archive.read(str(Path(member).parent / "manifest.json")))
        if manifest.get("dataset") != "fakeddit" or manifest.get("task") != "2way":
            raise ValueError("Script hiện dành cho Fakeddit 2way")
        with archive.open(member) as stream:
            reader = csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8"), delimiter="\t")
            labels = [field.removeprefix("score_") for field in reader.fieldnames or []
                      if field.startswith("score_") and not field.startswith(
                          ("score_image_", "score_text_"))]
            if len(labels) != 2 or any(field not in (reader.fieldnames or [])
                                       for field in ("id", "text", "y_true", "y_pred")):
                raise ValueError("Log thiếu nhãn hoặc trường căn bản")
            ids, texts, truth, predicted, probabilities = [], [], [], [], []
            for row in reader:
                if row["y_true"] not in labels or row["y_pred"] not in labels:
                    raise ValueError("Nhãn dự đoán hoặc nhãn thật không thuộc score")
                ids.append(row["id"])
                texts.append(row["text"])
                truth.append(labels.index(row["y_true"]))
                predicted.append(labels.index(row["y_pred"]))
                probabilities.append([float(row[f"score_{label}"]) for label in labels])
    p = np.asarray(probabilities, dtype=float)
    y, pred = np.asarray(truth), np.asarray(predicted)
    if len(p) != manifest.get("n_total_samples") or len(p) < 5:
        raise ValueError("Số dòng không khớp manifest hoặc không đủ 5 fold")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)) or \
            not np.allclose(p.sum(axis=1), 1, atol=1e-5):
        raise ValueError("Điểm lớp phải hữu hạn, nằm trong [0,1] và cộng thành 1")
    if not np.array_equal(p.argmax(axis=1), pred):
        raise ValueError("Nhãn dự đoán không khớp argmax điểm lớp")
    return manifest, labels, ids, texts, y, pred, p


def analyze(run_dir, alpha=0.10, output_dir=None, model=None, mode="pooled"):
    if mode not in ("pooled", "classwise"):
        raise ValueError("mode phải là pooled hoặc classwise")
    run_dir = Path(run_dir).resolve()
    manifest, labels, ids, texts, truth, predicted, p = _read_log(run_dir, model)
    folds, n_groups = _group_folds(ids, texts)
    sets = np.zeros_like(p, dtype=bool)
    fold_rows = []
    for fold in range(5):
        calibration, held_out = folds != fold, folds == fold
        if not calibration.any() or not held_out.any():
            raise ValueError("Fold rỗng")
        if mode == "pooled":
            calibration_scores = 1 - p[calibration, truth[calibration]]
            q, k = conformal_threshold(calibration_scores, alpha)
            thresholds = np.repeat(q, len(labels))
            details = {"all": {"n": int(calibration.sum()), "k": k, "q": q}}
        else:
            details = {}
            thresholds = np.empty(len(labels), dtype=float)
            for label_index, label in enumerate(labels):
                selected = calibration & (truth == label_index)
                if not selected.any():
                    raise ValueError(f"Fold calibration không có nhãn {label}")
                q, k = conformal_threshold(1 - p[selected, label_index], alpha)
                thresholds[label_index] = q
                details[label] = {"n": int(selected.sum()), "k": k, "q": q}
        sets[held_out] = 1 - p[held_out] <= thresholds
        held_size = sets[held_out].sum(axis=1)
        held_truth = truth[held_out]
        held_covered = sets[held_out, held_truth]
        fold_rows.append({"fold": fold, "n_calibration": int(calibration.sum()),
                          "n_held_out": int(held_out.sum()),
                          "thresholds": json.dumps(details, sort_keys=True),
                          "empirical_coverage": float(held_covered.mean()),
                          "n_singleton": int(np.sum(held_size == 1)),
                          "n_two_labels": int(np.sum(held_size == 2)),
                          "n_empty": int(np.sum(held_size == 0))})
    size = sets.sum(axis=1)
    covered = sets[np.arange(len(truth)), truth]
    correct = predicted == truth
    singleton = size == 1
    review = ~singleton
    singleton_predicted = sets.argmax(axis=1)
    singleton_correct = singleton & (singleton_predicted == truth)
    singleton_changed = singleton & (singleton_predicted != predicted)
    summary = {"source_run_id": run_dir.name, "model": manifest.get("model"),
               "dataset": manifest.get("dataset"), "task": manifest.get("task"),
               "policy": POLICY, "mode": mode, "alpha": alpha, "n_samples": len(truth),
               "n_groups": n_groups,
               "split_method": "group_by_shared_id_or_exact_caption; deterministic_sha256; 5_folds",
               "nonconformity_score": "1 - score_of_true_label_on_calibration",
               "n_singleton": int(singleton.sum()), "n_two_labels": int(np.sum(size == 2)),
               "n_empty": int(np.sum(size == 0)),
               "singleton_rate": float(singleton.mean()),
               "empirical_label_coverage": float(covered.mean()),
               "average_set_size": float(size.mean()),
               "baseline_accuracy_all": float(correct.mean()),
               "singleton_accuracy": float(singleton_correct[singleton].mean()) if singleton.any() else None,
               "baseline_accuracy_on_singleton": float(correct[singleton].mean()) if singleton.any() else None,
               "n_singleton_changed_from_baseline": int(singleton_changed.sum()),
               "n_errors_singleton": int(np.sum(~correct & singleton)),
               "n_errors_review": int(np.sum(~correct & review)),
               "n_correct_review": int(np.sum(correct & review)),
               "per_class": {label: {"n": int(np.sum(truth == i)),
                                     "empirical_coverage": float(covered[truth == i].mean()),
                                     "n_singleton": int(np.sum(singleton & (truth == i))),
                                     "singleton_accuracy": (float(singleton_correct[singleton & (truth == i)].mean())
                                                            if np.any(singleton & (truth == i)) else None)}
                             for i, label in enumerate(labels)},
               "source_split_verified": manifest.get("split_verified", False),
               "independent_test": False,
               "guaranteed_coverage_claimed": False}
    target = (Path(output_dir).resolve() if output_dir else
              ROOT / analysis_path(POLICY, run_dir,
                      f"{run_dir.name}_{POLICY}_{mode}_alpha{int(alpha * 1000):03d}", model=model))
    if target.exists():
        raise FileExistsError(f"Kết quả đã tồn tại: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".conformal_", dir=target.parent))
    try:
        (temporary / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        with (temporary / "folds.tsv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(fold_rows[0]), delimiter="\t",
                                    lineterminator="\n")
            writer.writeheader()
            writer.writerows(fold_rows)
        with (temporary / "prediction_sets.tsv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=("row_number", "id", "fold", "y_true",
                                                       "baseline_pred", "prediction_set", "set_size", "singleton_pred",
                                                       "needs_review", "contains_true_label", "margin"),
                                    delimiter="\t", lineterminator="\n")
            writer.writeheader()
            for i in range(len(truth)):
                writer.writerow({"row_number": i + 1, "id": ids[i], "fold": int(folds[i]),
                                 "y_true": labels[truth[i]], "baseline_pred": labels[predicted[i]],
                                 "prediction_set": "|".join(labels[j] for j in range(2) if sets[i, j]),
                                 "set_size": int(size[i]),
                                 "singleton_pred": labels[singleton_predicted[i]] if singleton[i] else "",
                                 "needs_review": str(bool(review[i])).lower(),
                                 "contains_true_label": str(bool(covered[i])).lower(),
                                 "margin": float(abs(p[i, 0] - p[i, 1]))})
        temporary.rename(target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return target, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--mode", choices=("pooled", "classwise"), default="pooled")
    parser.add_argument("--model")
    parser.add_argument("--output-dir")
    args = parser.parse_args(argv)
    target, summary = analyze(args.run_dir, args.alpha, args.output_dir, args.model, args.mode)
    print(f"[RESULT] {target}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
