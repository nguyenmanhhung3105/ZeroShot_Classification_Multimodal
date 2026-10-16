"""So fusion 2 tín hiệu với 3 tín hiệu từ cùng một log CrisisMMD đã chạy."""

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
from results_layout import analysis_path


def compare(run_dir, model=None):
    archive_path = Path(run_dir) / "run_artifacts.zip"
    with zipfile.ZipFile(archive_path) as archive:
        members = [name for name in archive.namelist() if name.endswith("/raw_predictions.tsv")]
        if model:
            members = [name for name in members if Path(name).parent.name.startswith(model + "__")]
        if len(members) != 1:
            raise ValueError(f"Cần đúng một log model; tìm thấy {len(members)}. Truyền --model nếu cần.")
        member = members[0]
        folder = Path(member).parent
        manifest = json.loads(archive.read(str(folder / "manifest.json")))
        if (manifest.get("dataset") != "crisismmd" or manifest.get("relation_mode") != "agreement_fusion"
                or manifest.get("relation_enabled") is not True):
            raise ValueError("Log phải là lượt CrisisMMD bật agreement_fusion")
        with archive.open(member) as stream:
            rows = list(csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8"), delimiter="\t"))
    if not rows:
        raise ValueError("Log rỗng")
    labels = [key.removeprefix("score_image_") for key in rows[0] if key.startswith("score_image_")]
    needed = {"y_true", "y_pred", "_text_was_empty", "image_text_fusion_gate",
              "relation_applied"} | {f"{prefix}{label}" for prefix in ("score_", "score_image_", "score_text_")
                                                 for label in labels}
    if len(labels) < 2 or not needed.issubset(rows[0]):
        raise ValueError("Log thiếu điểm nhánh, điểm cuối hoặc cột quan hệ")
    weight = float(manifest["image_weight"])
    if not 0 < weight < 1:
        raise ValueError("Lượt này không fusion hai modality")
    truth = np.array([row["y_true"] for row in rows])
    empty = np.array([row["_text_was_empty"].lower() == "true" for row in rows])
    image = np.array([[float(row[f"score_image_{label}"]) for label in labels] for row in rows])
    text = np.array([[0.0 if is_empty else float(row[f"score_text_{label}"])
                      for label in labels] for row, is_empty in zip(rows, empty)])
    final = np.array([[float(row[f"score_{label}"]) for label in labels] for row in rows])
    gate = np.array([float(row["image_text_fusion_gate"]) for row in rows])
    baseline = weight * image + (1 - weight) * text
    baseline[empty] = image[empty]
    if (not np.isfinite(image).all() or not np.isfinite(text).all() or not np.isfinite(final).all()
            or not np.isfinite(gate).all() or not np.allclose(final.sum(axis=1), 1, atol=1e-6)
            or np.any(gate < 0) or np.any(gate > float(manifest["relation_weight"]) + 1e-9)):
        raise ValueError("Điểm trong log không hợp lệ")
    product = image * text
    denominator = product.sum(axis=1)
    active = gate > 0
    if np.any(active & (denominator <= 1e-15)):
        raise ValueError("Gate dương nhưng tích điểm nhánh bằng 0")
    expected = baseline.copy()
    expected[active] = ((1 - gate[active, None]) * baseline[active]
                        + gate[active, None] * product[active] / denominator[active, None])
    if not np.allclose(final, expected, atol=1e-6):
        raise ValueError("Điểm cuối không khớp công thức agreement_fusion")
    applied = np.array([row["relation_applied"].lower() == "true" for row in rows])
    if not np.array_equal(applied, active):
        raise ValueError("Cột relation_applied không khớp gate")
    old = np.array(labels)[baseline.argmax(axis=1)]
    new = np.array(labels)[final.argmax(axis=1)]
    if not np.array_equal(new, [row["y_pred"] for row in rows]):
        raise ValueError("Điểm cuối không khớp y_pred đã lưu")

    def metrics(prediction, scores):
        result = {"accuracy": float(accuracy_score(truth, prediction)),
                  "macro_f1": float(f1_score(truth, prediction, labels=labels,
                                             average="macro", zero_division=0))}
        if manifest["task"] == "informativeness" and set(truth) == {"informative", "not_informative"}:
            result["auroc_informative_positive"] = float(roc_auc_score(
                truth == "informative", scores[:, labels.index("informative")]))
        return result

    old_correct, new_correct = old == truth, new == truth
    changed = old != new
    return {
        "source": str(archive_path), "model": manifest["model"], "task": manifest["task"],
        "n_samples": len(rows), "n_relation_applied": int(active.sum()),
        "mean_gate": float(gate.mean()), "max_gate": float(gate.max()),
        "two_signal": metrics(old, baseline), "three_signal": metrics(new, final),
        "n_changed": int(changed.sum()),
        "wrong_to_right": int(np.sum(~old_correct & new_correct)),
        "right_to_wrong": int(np.sum(old_correct & ~new_correct)),
        "wrong_to_wrong_changed": int(np.sum(changed & ~old_correct & ~new_correct)),
        "split": manifest.get("split"), "split_verified": bool(manifest.get("split_verified", False)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--model", help="Cần khi archive có nhiều model")
    parser.add_argument("--output", help="Đường dẫn JSON; mặc định theo phương pháp/dataset/task/số mẫu/model")
    args = parser.parse_args()
    report = compare(args.run_dir, args.model)
    output = Path(args.output) if args.output else analysis_path(
        "agreement_fusion", args.run_dir, f"{Path(args.run_dir).name}_{report['model']}",
        model=report["model"]) / "summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[RESULT] {output}")
    print(f"Macro-F1: {report['two_signal']['macro_f1']:.4f} -> {report['three_signal']['macro_f1']:.4f}; "
          f"sai→đúng {report['wrong_to_right']}, đúng→sai {report['right_to_wrong']}")


if __name__ == "__main__":
    main()
