"""Đối chiếu fusion 0,5/0,5 với trọng số theo entropy, không fit nhãn thật.

Đọc điểm hai nhánh đã lưu của một model CrisisMMD; không chạy inference.
"""

import argparse
import csv
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from results_layout import analysis_path


POLICY = "entropy_weighted_fusion_v1"


def _read_run(run_dir, model):
    archive_path = run_dir / "run_artifacts.zip"
    with zipfile.ZipFile(archive_path) as archive:
        members = [name for name in archive.namelist() if name.endswith("/raw_predictions.tsv")]
        if model:
            members = [name for name in members if Path(name).parent.name.startswith(model + "__")]
        if len(members) != 1:
            raise ValueError("Cần đúng một log model; dùng --model nếu archive có nhiều model")
        member = members[0]
        manifest = json.loads(archive.read(str(Path(member).parent / "manifest.json")))
        with archive.open(member) as stream:
            rows = list(csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8"), delimiter="\t"))
    return archive_path, manifest, rows


def compare(run_dir, model=None):
    archive_path, manifest, rows = _read_run(Path(run_dir), model)
    if (manifest.get("dataset") != "crisismmd"
            or manifest.get("task") not in ("informativeness", "humanitarian")
            or manifest.get("image_weight") != 0.5):
        raise ValueError("Phép thử chỉ dùng log CrisisMMD single-label với fusion 0,5/0,5")
    if not rows or len(rows) != manifest.get("n_total_samples"):
        raise ValueError("Số dự đoán không khớp manifest")
    labels = [key.removeprefix("score_image_") for key in rows[0]
              if key.startswith("score_image_")]
    required = {"id", "y_true", "y_pred", "_text_was_empty"} | {
        f"score_{branch}_{label}" for branch in ("image", "text") for label in labels
    }
    if len(labels) < 2 or not required.issubset(rows[0]):
        raise ValueError("Log thiếu điểm của hai nhánh hoặc nhãn")
    if any(row["y_true"] not in labels or row["_text_was_empty"] not in ("True", "False")
           for row in rows):
        raise ValueError("Nhãn hoặc cờ text rỗng không hợp lệ")

    image = np.array([[float(row[f"score_image_{label}"]) for label in labels] for row in rows])
    text = np.array([[float(row[f"score_text_{label}"]) for label in labels] for row in rows])
    empty = np.array([row["_text_was_empty"] == "True" for row in rows])
    if (not np.isfinite(image).all() or not np.isfinite(text).all()
            or np.any(image < 0) or np.any(text[~empty] < 0)
            or not np.allclose(image.sum(axis=1), 1, atol=1e-5)
            or not np.allclose(text[~empty].sum(axis=1), 1, atol=1e-5)):
        raise ValueError("Điểm nhánh phải là phân phối hữu hạn cộng thành 1")

    baseline = (image + text) / 2
    baseline[empty] = image[empty]
    if not manifest.get("relation_enabled"):
        logged = np.array([[float(row[f"score_{label}"]) for label in labels] for row in rows])
        if (not np.allclose(baseline, logged, atol=1e-5)
                or any(labels[baseline[i].argmax()] != row["y_pred"] for i, row in enumerate(rows))):
            raise ValueError("Không tái tạo được baseline đã lưu")

    def entropy(probabilities):
        return -np.sum(probabilities * np.log(np.maximum(probabilities, 1e-300)), axis=1) / np.log(len(labels))

    image_entropy = np.clip(entropy(image), 0, 1)
    text_entropy = np.clip(entropy(text), 0, 1)
    image_confidence = np.clip(1 - image_entropy, 0, 1)
    text_confidence = np.clip(1 - text_entropy, 0, 1)
    denominator = image_confidence + text_confidence
    image_weight = np.divide(image_confidence, denominator, out=np.full(len(rows), 0.5),
                             where=denominator > 1e-12)
    image_weight[empty] = 1.0
    adaptive = image_weight[:, None] * image + (1 - image_weight[:, None]) * text
    truth = np.array([row["y_true"] for row in rows])
    old = np.array(labels)[baseline.argmax(axis=1)]
    new = np.array(labels)[adaptive.argmax(axis=1)]
    image_choice = np.array(labels)[image.argmax(axis=1)]
    text_choice = np.array(labels)[text.argmax(axis=1)]
    changed = old != new
    if np.any(changed & (image_choice == text_choice)):
        raise AssertionError("Fusion lồi không thể đổi nhãn nếu hai nhánh cùng chọn một lớp")

    def metrics(prediction):
        precision, recall, f1, support = precision_recall_fscore_support(
            truth, prediction, labels=labels, zero_division=0)
        return {
            "accuracy": float(accuracy_score(truth, prediction)),
            "macro_f1": float(f1_score(truth, prediction, labels=labels,
                                       average="macro", zero_division=0)),
            "per_class": {label: {"support": int(support[i]), "precision": float(precision[i]),
                                  "recall": float(recall[i]), "f1": float(f1[i])}
                          for i, label in enumerate(labels)},
        }

    report = {
        "policy": POLICY, "source_archive": str(archive_path),
        "model": manifest["model"], "dataset": "crisismmd", "task": manifest["task"],
        "n_samples": len(rows), "sample_set_id": manifest.get("sample_set_id"),
        "source_relation_enabled": bool(manifest.get("relation_enabled")),
        "source_split_verified": bool(manifest.get("split_verified", False)),
        "uses_y_true_for_weight": False,
        "formula": "c_b=1-H(p_b)/log(C); w_image=c_image/(c_image+c_text); P=w_image*p_image+(1-w_image)*p_text; both_zero=>w_image=0.5; empty_text=>w_image=1",
        "baseline": metrics(old), "entropy_weighted": metrics(new),
        "n_branch_disagreement": int(np.sum(~empty & (image_choice != text_choice))),
        "n_changed": int(changed.sum()),
        "wrong_to_right": int(np.sum((old != truth) & (new == truth))),
        "right_to_wrong": int(np.sum((old == truth) & (new != truth))),
        "wrong_to_wrong_changed": int(np.sum(changed & (old != truth) & (new != truth))),
        "mean_image_weight": float(image_weight.mean()),
        "median_image_weight": float(np.median(image_weight)),
        "n_image_weight_gt_half": int(np.sum(image_weight > 0.5)),
    }
    details = [
        {"id": row["id"], "y_true": row["y_true"], "baseline_pred": old[i],
         "entropy_pred": new[i], "image_weight": float(image_weight[i]),
         "image_entropy": float(image_entropy[i]), "text_entropy": float(text_entropy[i])}
        for i, row in enumerate(rows)
    ]
    return report, details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--model", help="Chọn model khi archive có nhiều model")
    args = parser.parse_args()
    report, details = compare(args.run_dir, args.model)
    output = analysis_path(POLICY, args.run_dir, f"{Path(args.run_dir).name}_{POLICY}",
                           model=report["model"])
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                          encoding="utf-8")
    with (output / "predictions.tsv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(details[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(details)
    print(f"[RESULT] {output / 'summary.json'}")
    print(f"Macro-F1: {report['baseline']['macro_f1']:.4f} -> "
          f"{report['entropy_weighted']['macro_f1']:.4f}; "
          f"sai→đúng {report['wrong_to_right']}, đúng→sai {report['right_to_wrong']}")


if __name__ == "__main__":
    main()
