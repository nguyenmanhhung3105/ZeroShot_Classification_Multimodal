"""Thử tỷ trọng ảnh/text từ log đã lưu, chọn trọng số bằng grouped 5-fold CV.

Đây là phân tích hậu nghiệm: các fold chỉ giảm overfit khi chọn trọng số,
không biến bộ mẫu từng dùng chọn backbone/prompt thành test độc lập.
"""

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import tempfile
import zipfile

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from analyze_review import ROOT, _prediction_member


POLICY = "fusion_weight_group_cv_v2"
WEIGHTS = tuple(round(i / 20, 2) for i in range(21))


def _metric(truth, pred, labels):
    return {"accuracy": float(accuracy_score(truth, pred)),
            "macro_f1": float(f1_score(truth, pred, labels=list(range(len(labels))),
                                       average="macro", zero_division=0))}


def _group_folds(ids, texts, n_folds=5):
    """Giữ chung fold nếu trùng ID hoặc caption; cân bằng xấp xỉ số dòng."""
    parent = list(range(len(ids)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    seen = {}
    for i, (sample_id, caption) in enumerate(zip(ids, texts)):
        for key in (("id", sample_id), ("text", caption)):
            if key in seen:
                parent[find(i)] = find(seen[key])
            else:
                seen[key] = i
    groups = {}
    for i in range(len(ids)):
        groups.setdefault(find(i), []).append(i)
    ordered = []
    for members in groups.values():
        digest = min(hashlib.sha256((ids[i] + "\0" + texts[i]).encode("utf-8")).hexdigest()
                     for i in members)
        ordered.append((digest, members))
    sizes = [0] * n_folds
    folds = np.empty(len(ids), dtype=int)
    for _, members in sorted(ordered):
        fold = min(range(n_folds), key=lambda index: (sizes[index], index))
        folds[members] = fold
        sizes[fold] += len(members)
    return folds, len(groups)


def _read_run(run_dir):
    with zipfile.ZipFile(run_dir / "run_artifacts.zip") as archive:
        member = _prediction_member(archive)
        manifest = json.loads(archive.read(str(Path(member).parent / "manifest.json")))
        if manifest.get("dataset") != "crisismmd" or manifest.get("task") != "informativeness":
            raise ValueError("Script hiện dành cho CrisisMMD informativeness")
        if manifest.get("relation_enabled"):
            raise ValueError("Không thể tái tạo relation class-aware từ điểm hai nhánh")
        if not isinstance(manifest.get("image_weight"), (int, float)):
            raise ValueError("Manifest thiếu image_weight")
        with archive.open(member) as stream:
            reader = csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8"), delimiter="\t")
            columns = reader.fieldnames or []
            labels = [column.removeprefix("score_") for column in columns
                      if column.startswith("score_") and not column.startswith(
                          ("score_image_", "score_text_"))]
            if len(labels) != 2 or any(f"score_{branch}_{label}" not in columns
                                       for branch in ("image", "text") for label in labels):
                raise ValueError("Log thiếu điểm từng nhánh của hai lớp")
            ids, texts, truth, logged, empty, image, text, fused = [], [], [], [], [], [], [], []
            for row in reader:
                ids.append(row["id"])
                texts.append(row["text"])
                if row["y_true"] not in labels or row["y_pred"] not in labels:
                    raise ValueError("Nhãn ngoài danh sách score")
                truth.append(labels.index(row["y_true"]))
                logged.append(labels.index(row["y_pred"]))
                if row["_text_was_empty"] not in ("True", "False"):
                    raise ValueError("Cờ text rỗng không hợp lệ")
                empty.append(row["_text_was_empty"] == "True")
                image.append([float(row[f"score_image_{label}"]) for label in labels])
                text.append([float(row[f"score_text_{label}"]) for label in labels]
                            if not empty[-1] else [0.0] * len(labels))
                fused.append([float(row[f"score_{label}"]) for label in labels])
    if len(truth) != manifest.get("n_total_samples") or len(truth) < 5:
        raise ValueError("Số mẫu không khớp manifest hoặc không đủ 5 fold")
    image, text, fused = map(np.asarray, (image, text, fused))
    if (any(not np.isfinite(matrix).all() for matrix in (image, text, fused))
            or not np.allclose(image.sum(axis=1), 1, atol=1e-5)
            or not np.allclose(text[~np.asarray(empty)].sum(axis=1), 1, atol=1e-5)
            or not np.allclose(fused.sum(axis=1), 1, atol=1e-5)):
        raise ValueError("Điểm không hữu hạn hoặc không cộng thành 1")
    weight = float(manifest["image_weight"])
    reconstructed = np.where(np.asarray(empty)[:, None], image, weight * image + (1 - weight) * text)
    if not np.allclose(reconstructed, fused, atol=1e-5) or \
            not np.array_equal(fused.argmax(axis=1), np.asarray(logged)):
        raise ValueError("Không tái tạo đúng dự đoán gốc; không thể quét tỷ trọng")
    return manifest, labels, ids, texts, np.asarray(truth), np.asarray(logged), \
        np.asarray(empty), image, text


def analyze(run_dir, output_dir=None):
    run_dir = Path(run_dir).resolve()
    manifest, labels, ids, texts, truth, original, empty, image, text = _read_run(run_dir)
    baseline_weight = float(manifest["image_weight"])
    folds, n_groups = _group_folds(ids, texts)
    predictions = {weight: np.where(empty[:, None], image,
                                    weight * image + (1 - weight) * text).argmax(axis=1)
                   for weight in WEIGHTS}
    if baseline_weight not in predictions:
        predictions[baseline_weight] = original
    selected = np.empty(len(truth), dtype=int)
    selections = []
    for fold in range(5):
        train, held = folds != fold, folds == fold
        if not np.any(train) or not np.any(held):
            raise ValueError("Fold rỗng")
        scores = {weight: _metric(truth[train], pred[train], labels)
                  for weight, pred in predictions.items()}
        weight = max(sorted(predictions), key=lambda value: (
            scores[value]["macro_f1"], scores[value]["accuracy"],
            -abs(value - baseline_weight), -value))
        selected[held] = predictions[weight][held]
        baseline = _metric(truth[held], original[held], labels)
        trial = _metric(truth[held], selected[held], labels)
        selections.append({"fold": fold, "n_train": int(train.sum()), "n_held_out": int(held.sum()),
                           "selected_image_weight": weight,
                           "train_macro_f1": scores[weight]["macro_f1"],
                           "held_baseline_accuracy": baseline["accuracy"],
                           "held_selected_accuracy": trial["accuracy"],
                           "held_baseline_macro_f1": baseline["macro_f1"],
                           "held_selected_macro_f1": trial["macro_f1"]})
    fixed = (original != truth) & (selected == truth)
    broken = (original == truth) & (selected != truth)
    n_fixed, n_broken = int(fixed.sum()), int(broken.sum())
    n_discordant = n_fixed + n_broken
    mcnemar_p = (min(1.0, 2 * sum(math.comb(n_discordant, i)
                                  for i in range(min(n_fixed, n_broken) + 1)) / 2 ** n_discordant)
                 if n_discordant else 1.0)
    baseline_scores = np.where(empty[:, None], image,
                               baseline_weight * image + (1 - baseline_weight) * text)
    low_margin = np.sort(baseline_scores, axis=1)[:, -1] - \
        np.sort(baseline_scores, axis=1)[:, -2] <= 0.10
    branch_disagreement = (~empty) & (image.argmax(axis=1) != text.argmax(axis=1))
    changed = selected != original
    full_sweep = [{"image_weight": weight, **_metric(truth, pred, labels),
                   "changed_vs_baseline": int(np.sum(pred != original)),
                   "wrong_to_right": int(np.sum((original != truth) & (pred == truth))),
                   "right_to_wrong": int(np.sum((original == truth) & (pred != truth)))}
                  for weight, pred in sorted(predictions.items())]
    summary = {"source_run_id": run_dir.name, "policy": POLICY,
               "n_samples": len(truth), "n_groups": n_groups,
               "split_method": "group_by_shared_id_or_exact_caption; deterministic_sha256; 5_folds",
               "selection_metric": "macro_f1_then_accuracy_then_closest_to_baseline",
               "baseline_image_weight": baseline_weight,
               "candidate_weights": list(WEIGHTS),
               "selected_weights": [item["selected_image_weight"] for item in selections],
               "n_branch_disagreement": int(branch_disagreement.sum()),
               "n_baseline_margin_le_0_10": int(low_margin.sum()),
               "baseline": _metric(truth, original, labels),
               "cross_fitted": {**_metric(truth, selected, labels),
                                "changed_vs_baseline": int(changed.sum()),
                                "changed_with_branch_disagreement": int(np.sum(changed & branch_disagreement)),
                                "changed_with_baseline_margin_le_0_10": int(np.sum(changed & low_margin)),
                                "wrong_to_right": n_fixed,
                                "right_to_wrong": n_broken,
                                "mcnemar_exact_two_sided_p_exploratory": mcnemar_p},
               "per_class": {label: {"n": int(np.sum(truth == i)),
                                     "wrong_to_right": int(np.sum(fixed & (truth == i))),
                                     "right_to_wrong": int(np.sum(broken & (truth == i)))}
                             for i, label in enumerate(labels)},
               "source_split_verified": manifest.get("split_verified", False),
               "independent_test": False}
    target = (Path(output_dir).resolve() if output_dir else
              ROOT / "results/analysis/uncertainty" / f"{run_dir.name}_{POLICY}")
    if target.exists():
        raise FileExistsError(f"Kết quả đã tồn tại: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".fusion_", dir=target.parent))
    try:
        (temporary / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                                   encoding="utf-8")
        for name, rows in (("folds.tsv", selections), ("full_cohort_sweep.tsv", full_sweep)):
            with (temporary / name).open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t",
                                        lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
        temporary.rename(target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return target, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir")
    args = parser.parse_args(argv)
    target, summary = analyze(args.run_dir, args.output_dir)
    print(f"[RESULT] {target}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
