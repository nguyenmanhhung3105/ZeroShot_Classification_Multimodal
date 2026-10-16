"""Đo tín hiệu phân vân từ một log Kaggle đã lưu, không chạy model/dataset."""

import argparse
import csv
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from results_layout import dataset_folder, model_folder, sample_folder, task_folder


POLICY = "uncertainty_diagnostic_v1"


def read_one(run_dir, model=None):
    run_dir = Path(run_dir)
    archive_path = run_dir / "run_artifacts.zip"
    if not (run_dir / "run_receipt.json").is_file() or not archive_path.is_file():
        raise ValueError("Cần run_receipt.json và run_artifacts.zip của lượt Kaggle")
    with zipfile.ZipFile(archive_path) as archive:
        members = [name for name in archive.namelist() if name.endswith("/raw_predictions.tsv")]
        if model:
            members = [name for name in members if Path(name).parent.name.startswith(model + "__")]
        if len(members) != 1:
            raise ValueError("Cần đúng một log; dùng --model nếu archive có nhiều model")
        name = members[0]
        manifest = json.loads(archive.read(str(Path(name).parent / "manifest.json")))
        with archive.open(name) as stream:
            rows = list(csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8"), delimiter="\t"))
    return manifest, rows


def diagnose(manifest, rows):
    if not rows or len(rows) != manifest.get("n_total_samples"):
        raise ValueError("Số dòng không khớp manifest")
    labels = [key.removeprefix("score_") for key in rows[0]
              if key.startswith("score_") and not key.startswith(("score_image_", "score_text_"))]
    if len(labels) < 2:
        raise ValueError("Cần ít nhất hai nhãn với điểm lớp")
    score = np.array([[float(row[f"score_{label}"]) for label in labels] for row in rows])
    image = np.array([[float(row[f"score_image_{label}"]) for label in labels] for row in rows])
    text = np.array([[float(row[f"score_text_{label}"]) for label in labels] for row in rows])
    if (not np.isfinite(score).all() or not np.isfinite(image).all() or not np.isfinite(text).all()
            or not np.allclose(score.sum(axis=1), 1, atol=1e-5)):
        raise ValueError("Điểm lớp thiếu, không hữu hạn hoặc không chuẩn hóa")
    true = np.array([row["y_true"] for row in rows])
    predicted = np.asarray(labels)[score.argmax(axis=1)]
    if any(row["y_pred"] != pred for row, pred in zip(rows, predicted)):
        raise ValueError("Nhãn dự đoán không khớp điểm")
    error = (true != predicted).astype(int)
    def entropy(prob):
        return -np.sum(prob * np.log(np.maximum(prob, 1e-300)), axis=1) / np.log(len(labels))
    sorted_scores = np.sort(score, axis=1)
    mean_branch = (image + text) / 2
    risks = {"one_minus_margin": 1 - (sorted_scores[:, -1] - sorted_scores[:, -2]),
             "one_minus_top1_score": 1 - sorted_scores[:, -1],
             "normalized_entropy": entropy(score),
             "image_text_vote_disagreement": (image.argmax(axis=1) != text.argmax(axis=1)).astype(int),
             "image_text_js_divergence": np.clip(
                 entropy(mean_branch) - (entropy(image) + entropy(text)) / 2, 0, 1)}
    def detection(risk):
        n_review = max(1, int(round(len(rows) * .2)))
        selected = np.argsort(-risk, kind="stable")[:n_review]
        return {"error_auroc": float(roc_auc_score(error, risk)) if 0 < error.sum() < len(error) else None,
                "error_average_precision": float(average_precision_score(error, risk)) if error.any() else None,
                "errors_in_top_20_percent": int(error[selected].sum()),
                "n_review": n_review,
                "remaining_accuracy": float(1 - (error.sum() - error[selected].sum()) / (len(rows) - n_review))}
    order = np.argsort(score, axis=1)
    top2 = np.asarray(labels)[order[:, -2:]]
    top2_contains = np.any(top2 == true[:, None], axis=1)
    return {"policy": POLICY, "dataset": manifest["dataset"], "task": manifest["task"],
            "model": manifest["model"], "n_samples": len(rows), "sample_set_id": manifest.get("sample_set_id"),
            "split_verified": bool(manifest.get("split_verified", False)),
            "labels": labels, "n_errors": int(error.sum()),
            "top2_contains_true": int(top2_contains.sum()),
            "errors_with_true_as_second_choice": int(np.sum((error == 1) & top2_contains)),
            "uncertainty_detection": {name: detection(risk) for name, risk in risks.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--model", help="Chọn model nếu archive chứa nhiều model")
    args = parser.parse_args()
    manifest, rows = read_one(args.run_dir, args.model)
    result = diagnose(manifest, rows)
    output = (Path("results/analysis") / POLICY
              / dataset_folder(result["dataset"], result["task"])
              / task_folder(result["dataset"], result["task"])
              / sample_folder(result["n_samples"])
              / model_folder(models=[result["model"]])
              / (Path(args.run_dir).name + "_" + POLICY))
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                                         encoding="utf-8")
    print(f"[RESULT] {output / 'summary.json'}")
    print(json.dumps(result["uncertainty_detection"], ensure_ascii=False))


if __name__ == "__main__":
    main()
