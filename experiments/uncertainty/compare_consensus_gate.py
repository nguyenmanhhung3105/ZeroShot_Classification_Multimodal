"""Thử cổng đồng thuận ba model trên log đã lưu; không chạy inference.

Quy tắc được cố định trước khi đọc nhãn thật: chỉ đổi nhãn MetaCLIP2 khi
margin <= 0.10 và cả DFN5B lẫn SigLIP2 chọn cùng một nhãn khác.
Đầu ra là phân tích hậu nghiệm, không phải metric của một lượt Kaggle mới.
"""

import argparse
import csv
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from results_layout import dataset_folder, sample_folder, task_folder


MODELS = ("metaclip2_vith14", "dfn5b_vith14", "siglip2_so400m")
POLICY = "consensus_gate_v1"


def read_logs(run_dir, dataset=None, task=None):
    run_dir = Path(run_dir)
    archive_path = run_dir / "run_artifacts.zip"
    runs = {}
    if archive_path.is_file():
        with zipfile.ZipFile(archive_path) as archive:
            members = [name for name in archive.namelist() if name.endswith("/raw_predictions.tsv")]
            for name in members:
                folder = Path(name).parent
                model = folder.name.split("__", 1)[0]
                if model not in MODELS:
                    continue
                manifest = json.loads(archive.read(str(folder / "manifest.json")))
                if dataset and (manifest.get("dataset"), manifest.get("task")) != (dataset, task):
                    continue
                with archive.open(name) as stream:
                    rows = list(csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8"), delimiter="\t"))
                if model in runs:
                    raise ValueError(f"Trùng log model {model}; chọn --dataset/--task")
                runs[model] = (manifest, rows)
        source = "kaggle_archive" if (run_dir / "run_receipt.json").is_file() else "archive_no_receipt"
    else:
        for path in sorted((run_dir / "model_logs").glob("*/raw_predictions.tsv")):
            model = path.parent.name.split("__", 1)[0]
            if model not in MODELS:
                continue
            manifest = json.loads((path.parent / "manifest.json").read_text(encoding="utf-8"))
            if dataset and (manifest.get("dataset"), manifest.get("task")) != (dataset, task):
                continue
            with path.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream, delimiter="\t"))
            if model in runs:
                raise ValueError(f"Trùng log model {model}; chọn --dataset/--task")
            runs[model] = (manifest, rows)
        source = "local_logs"
    if set(runs) != set(MODELS):
        raise ValueError(f"Cần đủ ba model {MODELS}; có {sorted(runs)}")
    return runs, source


def compare(runs, margin_threshold=0.10):
    if not 0 <= margin_threshold <= 1:
        raise ValueError("margin_threshold phải trong [0, 1]")
    manifest, reference = runs[MODELS[0]]
    if not reference or len(reference) != manifest.get("n_total_samples"):
        raise ValueError("Log MetaCLIP2 thiếu mẫu hoặc khác manifest")
    labels = [column.removeprefix("score_") for column in reference[0]
              if column.startswith("score_") and not column.startswith(("score_image_", "score_text_"))]
    if len(labels) < 2:
        raise ValueError("Log cần ít nhất hai cột score lớp")
    keys = ("id", "text", "image_path", "y_true")
    matrices, predictions = {}, {}
    for model in MODELS:
        current_manifest, rows = runs[model]
        if ((current_manifest.get("dataset"), current_manifest.get("task")) !=
                (manifest.get("dataset"), manifest.get("task")) or len(rows) != len(reference)
                or current_manifest.get("sample_set_id") != manifest.get("sample_set_id")):
            raise ValueError(f"Cohort/manifest không khớp ở {model}")
        if any(any(row[key] != base[key] for key in keys) for row, base in zip(rows, reference)):
            raise ValueError(f"Thứ tự mẫu không khớp ở {model}")
        matrix = np.array([[float(row[f"score_{label}"]) for label in labels] for row in rows])
        if (not np.isfinite(matrix).all() or (matrix < 0).any()
                or not np.allclose(matrix.sum(axis=1), 1, atol=1e-5)):
            raise ValueError(f"Điểm lớp không phải phân phối hợp lệ ở {model}")
        pred = np.asarray(labels)[matrix.argmax(axis=1)]
        if any(str(value) != row["y_pred"] for value, row in zip(pred, rows)):
            raise ValueError(f"Nhãn dự đoán không khớp điểm ở {model}")
        matrices[model], predictions[model] = matrix, pred

    truth = np.asarray([row["y_true"] for row in reference])
    primary = predictions[MODELS[0]]
    aux1, aux2 = predictions[MODELS[1]], predictions[MODELS[2]]
    base_scores = matrices[MODELS[0]]
    margin = np.sort(base_scores, axis=1)[:, -1] - np.sort(base_scores, axis=1)[:, -2]
    low_margin = margin <= margin_threshold
    aux_consensus = (aux1 == aux2) & (aux1 != primary)
    mean_scores = np.mean(np.stack([matrices[model] for model in MODELS]), axis=0)
    mean_pred = np.asarray(labels)[mean_scores.argmax(axis=1)]
    predictions_out = {
        "primary": primary,
        "equal_mean": mean_pred,
        "margin_gate_mean": np.where(low_margin, mean_pred, primary),
        "aux_consensus_all": np.where(aux_consensus, aux1, primary),
        "aux_consensus_low_margin": np.where(low_margin & aux_consensus, aux1, primary),
    }
    def entropy(prob):
        return -np.sum(prob * np.log(np.maximum(prob, 1e-300)), axis=1) / np.log(len(labels))
    error = (primary != truth).astype(int)
    model_disagreement = (aux1 != primary).astype(int) + (aux2 != primary).astype(int)
    model_js = np.clip(entropy(mean_scores) - np.mean(
        np.stack([entropy(matrices[model]) for model in MODELS]), axis=0), 0, 1)
    uncertainty_scores = {"one_minus_primary_margin": 1 - margin,
                          "model_vote_disagreement": model_disagreement,
                          "model_js_divergence": model_js}
    if all(f"score_image_{label}" in reference[0] and f"score_text_{label}" in reference[0]
           for label in labels):
        image = np.array([[float(row[f"score_image_{label}"]) for label in labels] for row in reference])
        text = np.array([[float(row[f"score_text_{label}"]) for label in labels] for row in reference])
        if np.isfinite(image).all() and np.isfinite(text).all():
            uncertainty_scores["image_text_vote_disagreement"] = (
                image.argmax(axis=1) != text.argmax(axis=1)).astype(int)
            uncertainty_scores["image_text_js_divergence"] = np.clip(
                entropy((image + text) / 2) - (entropy(image) + entropy(text)) / 2, 0, 1)
    def detection(scores):
        top_count = max(1, int(round(len(scores) * .2)))
        selected = np.argsort(-scores, kind="stable")[:top_count]
        return {"error_auroc": float(roc_auc_score(error, scores)) if 0 < error.sum() < len(error) else None,
                "error_average_precision": float(average_precision_score(error, scores)) if error.any() else None,
                "errors_in_top_20_percent": int(error[selected].sum()),
                "top_20_percent_count": top_count}
    def metrics(pred):
        fixed = (primary != truth) & (pred == truth)
        broken = (primary == truth) & (pred != truth)
        return {"accuracy": float(accuracy_score(truth, pred)),
                "macro_f1": float(f1_score(truth, pred, labels=labels, average="macro", zero_division=0)),
                "changed": int(np.sum(pred != primary)),
                "wrong_to_right": int(np.sum(fixed)), "right_to_wrong": int(np.sum(broken)),
                "wrong_to_wrong_changed": int(np.sum((pred != primary) & (primary != truth) & (pred != truth)))}
    def flag_metrics(flag):
        count = int(flag.sum())
        errors = int(np.sum(flag & (primary != truth)))
        accepted_count = len(truth) - count
        accepted_errors = int(np.sum(primary != truth)) - errors
        return {"n": count, "primary_errors": errors,
                "primary_error_rate": errors / count if count else None,
                "share_of_primary_errors": errors / int(np.sum(primary != truth)) if np.any(primary != truth) else None,
                "accepted_n": accepted_count,
                "accepted_accuracy": 1 - accepted_errors / accepted_count if accepted_count else None}
    report = {"policy": POLICY, "dataset": manifest["dataset"], "task": manifest["task"],
              "n_samples": len(reference), "sample_set_id": manifest.get("sample_set_id"),
              "split_verified": bool(manifest.get("split_verified", False)),
              "threshold_fixed_without_labels": margin_threshold,
              "labels": labels, "models": list(MODELS),
              "methods": {name: metrics(pred) for name, pred in predictions_out.items()},
              "recoverability": {
                  "primary_errors": int(np.sum(primary != truth)),
                  "primary_errors_any_aux_correct": int(np.sum((primary != truth) &
                      ((aux1 == truth) | (aux2 == truth)))),
                  "primary_errors_both_aux_correct": int(np.sum((primary != truth) &
                      (aux1 == truth) & (aux2 == truth))),
                  "primary_errors_all_models_wrong": int(np.sum((primary != truth) &
                      (aux1 != truth) & (aux2 != truth))),
              },
              "predicted_class_counts": {model: {label: int(np.sum(predictions[model] == label))
                                                for label in labels} for model in MODELS},
              "uncertainty_detection": {name: detection(scores)
                                        for name, scores in uncertainty_scores.items()},
              "uncertainty_groups": {
                  "low_margin": flag_metrics(low_margin),
                  "any_model_disagreement": flag_metrics((aux1 != primary) | (aux2 != primary)),
                  "aux_consensus_opposite": flag_metrics(aux_consensus),
                  "low_margin_and_aux_consensus": flag_metrics(low_margin & aux_consensus),
              }}
    details = [{"id": row["id"], "y_true": truth[i], "primary_pred": primary[i],
                "mean_pred": mean_pred[i], "consensus_pred": predictions_out["aux_consensus_low_margin"][i],
                "primary_margin": float(margin[i]), "low_margin": bool(low_margin[i]),
                "aux_consensus_opposite": bool(aux_consensus[i])}
               for i, row in enumerate(reference)]
    return report, details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dataset", help="Bắt buộc nếu archive có nhiều task")
    parser.add_argument("--task", help="Bắt buộc nếu archive có nhiều task")
    parser.add_argument("--margin-threshold", type=float, default=0.10)
    args = parser.parse_args()
    if bool(args.dataset) != bool(args.task):
        parser.error("--dataset và --task phải đi cùng nhau")
    runs, source = read_logs(args.run_dir, args.dataset, args.task)
    report, details = compare(runs, args.margin_threshold)
    report["source_type"] = source
    report["source_run"] = args.run_dir
    output = (Path("results/analysis") / POLICY
              / dataset_folder(report["dataset"], report["task"])
              / task_folder(report["dataset"], report["task"])
              / sample_folder(report["n_samples"]) / "multi_model"
              / (Path(args.run_dir).name + "_" + POLICY))
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
