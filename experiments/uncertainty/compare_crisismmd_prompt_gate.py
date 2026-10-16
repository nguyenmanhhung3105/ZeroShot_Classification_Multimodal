"""Đối chiếu cổng margin chọn prompt DCLIP trên log CrisisMMD đã lưu.

Đây là phân tích offline, không chạy inference hoặc gửi Kaggle job.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from compare_prompt_stability import compare as check_paired_runs
from diagnose_single_model import read_one

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from results_layout import dataset_folder, model_folder, sample_folder, task_folder


MODEL = "metaclip2_vith14"
POLICY = "prompt_margin_gate_v1"


def evaluate(base_manifest, base_rows, adapted_manifest, adapted_rows, threshold=0.10):
    if not 0 <= threshold <= 1:
        raise ValueError("margin threshold phải trong [0, 1]")
    if (not base_rows or len(base_rows) != base_manifest.get("n_total_samples")
            or len(adapted_rows) != adapted_manifest.get("n_total_samples")):
        raise ValueError("Số dòng không khớp manifest")
    check_paired_runs(base_manifest, base_rows, adapted_manifest, adapted_rows)
    if base_manifest["dataset"] != "crisismmd" or base_manifest["model"] != MODEL:
        raise ValueError("Phép thử chỉ dùng MetaCLIP2 trên CrisisMMD")

    labels = [key.removeprefix("score_") for key in base_rows[0]
              if key.startswith("score_") and not key.startswith(("score_image_", "score_text_"))]
    scores = np.array([[float(row[f"score_{label}"]) for label in labels] for row in base_rows])
    margins = np.sort(scores, axis=1)[:, -1] - np.sort(scores, axis=1)[:, -2]
    gate = margins <= threshold
    truth = np.asarray([row["y_true"] for row in base_rows])
    baseline = np.asarray([row["y_pred"] for row in base_rows])
    adapted = np.asarray([row["y_pred"] for row in adapted_rows])
    selected = np.where(gate, adapted, baseline)

    def metrics(prediction):
        return {
            "accuracy": float(accuracy_score(truth, prediction)),
            "macro_f1": float(f1_score(truth, prediction, labels=labels,
                                       average="macro", zero_division=0)),
            "changed_vs_baseline": int(np.sum(prediction != baseline)),
            "wrong_to_right": int(np.sum((baseline != truth) & (prediction == truth))),
            "right_to_wrong": int(np.sum((baseline == truth) & (prediction != truth))),
            "wrong_to_wrong_changed": int(np.sum((baseline != truth) &
                                                   (prediction != truth) & (prediction != baseline))),
        }

    report = {
        "policy": POLICY, "dataset": "crisismmd", "task": base_manifest["task"],
        "model": MODEL, "n_samples": len(truth), "labels": labels,
        "margin_threshold": threshold, "gate_count": int(gate.sum()),
        "baseline_sample_set_id": base_manifest.get("sample_set_id"),
        "adapted_sample_set_id": adapted_manifest.get("sample_set_id"),
        "paired_rows_verified": True,
        "split_verified": bool(base_manifest.get("split_verified", False)
                               and adapted_manifest.get("split_verified", False)),
        "methods": {"baseline": metrics(baseline), "adapted_dclip": metrics(adapted),
                    "prompt_margin_gate": metrics(selected)},
    }
    details = [{"id": row["id"], "y_true": truth[i], "baseline_pred": baseline[i],
                "adapted_pred": adapted[i], "gated_pred": selected[i],
                "baseline_margin": float(margins[i]), "gate_active": bool(gate[i])}
               for i, row in enumerate(base_rows)]
    return report, details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--adapted-dir", required=True)
    parser.add_argument("--margin-threshold", type=float, default=0.10)
    args = parser.parse_args()

    base_manifest, base_rows = read_one(args.baseline_dir, MODEL)
    adapted_manifest, adapted_rows = read_one(args.adapted_dir, MODEL)
    report, details = evaluate(base_manifest, base_rows, adapted_manifest, adapted_rows,
                               args.margin_threshold)
    report["baseline_run"] = args.baseline_dir
    report["adapted_run"] = args.adapted_dir
    output = (Path("results/analysis") / POLICY
              / dataset_folder(report["dataset"], report["task"])
              / task_folder(report["dataset"], report["task"])
              / sample_folder(report["n_samples"]) / model_folder(models=[MODEL])
              / (Path(args.baseline_dir).name + "_" + Path(args.adapted_dir).name + "_" + POLICY))
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                         encoding="utf-8")
    with (output / "predictions.tsv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(details[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(details)
    print(f"[RESULT] {output / 'summary.json'}")
    print(json.dumps(report["methods"], ensure_ascii=False))


if __name__ == "__main__":
    main()
