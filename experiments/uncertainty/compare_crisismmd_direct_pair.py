"""Thử offline direct_pair_fusion trên log agreement CrisisMMD có cùng điểm nhánh."""

import argparse
import csv
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from pair_relation import apply_class_relation, validate_relation_options
from results_layout import analysis_path
from compare_crisismmd_agreement import compare as check_source


def compare_direct(run_dir, model=None, weight=0.25):
    source = check_source(run_dir, model)  # Kiểm log nguồn và công thức của lượt đã chạy.
    archive_path = Path(run_dir) / "run_artifacts.zip"
    with zipfile.ZipFile(archive_path) as archive:
        members = [name for name in archive.namelist() if name.endswith("/raw_predictions.tsv")]
        members = [name for name in members if Path(name).parent.name.startswith(source["model"] + "__")]
        if len(members) != 1:
            raise ValueError("Không tìm thấy duy nhất log model đã xác minh")
        with archive.open(members[0]) as stream:
            rows = list(csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8"), delimiter="\t"))
        manifest = json.loads(archive.read(str(Path(members[0]).parent / "manifest.json")))
    labels = [key.removeprefix("score_image_") for key in rows[0] if key.startswith("score_image_")]
    if "image_text_similarity" not in rows[0]:
        raise ValueError("Log thiếu cosine ảnh–text")
    image = np.array([[float(row[f"score_image_{label}"]) for label in labels] for row in rows])
    empty = np.array([row["_text_was_empty"].lower() == "true" for row in rows])
    text = np.array([[0.0 if is_empty else float(row[f"score_text_{label}"]) for label in labels]
                     for row, is_empty in zip(rows, empty)])
    baseline = float(manifest["image_weight"]) * image + (1 - float(manifest["image_weight"])) * text
    baseline[empty] = image[empty]
    similarity = [None if row["image_text_similarity"] in ("", "None", "nan")
                  else float(row["image_text_similarity"]) for row in rows]
    scores = {"image_text_similarity": similarity, "relation_applied": [False] * len(rows)}
    options = validate_relation_options({"enabled": True, "mode": "direct_pair_fusion", "weight": weight})
    direct = apply_class_relation(baseline, labels, scores, options, float(manifest["image_weight"]),
                                  p_image=image, p_text=text)
    truth = np.array([row["y_true"] for row in rows])
    old = np.array(labels)[baseline.argmax(axis=1)]
    new = np.array(labels)[direct.argmax(axis=1)]

    def metrics(prediction, probability):
        values = {"accuracy": float(accuracy_score(truth, prediction)),
                  "macro_f1": float(f1_score(truth, prediction, labels=labels,
                                             average="macro", zero_division=0))}
        if manifest["task"] == "informativeness" and set(truth) == {"informative", "not_informative"}:
            values["auroc_informative_positive"] = float(roc_auc_score(
                truth == "informative", probability[:, labels.index("informative")]))
        return values

    old_correct, new_correct = old == truth, new == truth
    changed = old != new
    pair_weight = np.array([value for value in scores["image_text_pair_match_weight"] if value is not None])
    return {"source": str(archive_path), "model": source["model"], "task": source["task"],
            "n_samples": len(rows), "n_relation_applied": int(sum(scores["relation_applied"])),
            "direct_branch_weight": weight,
            "mean_pair_match_weight": float(pair_weight.mean()) if len(pair_weight) else None,
            "two_signal": metrics(old, baseline), "direct_pair": metrics(new, direct),
            "n_changed": int(changed.sum()),
            "wrong_to_right": int(np.sum(~old_correct & new_correct)),
            "right_to_wrong": int(np.sum(old_correct & ~new_correct)),
            "wrong_to_wrong_changed": int(np.sum(changed & ~old_correct & ~new_correct)),
            "split": manifest.get("split"), "split_verified": bool(manifest.get("split_verified", False))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--model")
    parser.add_argument("--output", help="Đường dẫn JSON; mặc định theo phương pháp/dataset/task/số mẫu/model")
    args = parser.parse_args()
    report = compare_direct(args.run_dir, args.model)
    output = Path(args.output) if args.output else analysis_path(
        "direct_pair_fusion", args.run_dir, f"{Path(args.run_dir).name}_{report['model']}",
        model=report["model"]) / "summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[RESULT] {output}")
    print(f"Macro-F1: {report['two_signal']['macro_f1']:.4f} -> {report['direct_pair']['macro_f1']:.4f}; "
          f"đổi nhãn {report['n_changed']}; sai→đúng {report['wrong_to_right']}, đúng→sai {report['right_to_wrong']}")


if __name__ == "__main__":
    main()
