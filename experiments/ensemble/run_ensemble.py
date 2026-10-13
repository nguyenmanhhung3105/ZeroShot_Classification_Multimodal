"""Thử ensemble tùy chọn trên Fakeddit, CrisisMMD và MM-IMDb.

Launcher tạo job YAML đã cố định trước khi chạy. Không fit ngưỡng bằng nhãn thật.
"""

import argparse
from collections import Counter
from copy import deepcopy
import csv
import gc
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
import yaml

from config import load_config
from models.model_registry import MODEL_CONFIGS, load_model
from prompts import crisismmd_prompts, fakeddit_prompts, mmimdb_prompts
from run.run_crisismmd import run_crisismmd
from run.run_fakeddit import run_fakeddit
from run.run_mmimdb import run_mmimdb

DEFAULT_MODELS = ("metaclip2_vith14", "dfn5b_vith14", "siglip2_so400m")
RUNNERS = {"fakeddit": run_fakeddit, "crisismmd": run_crisismmd, "mmimdb": run_mmimdb}
TASKS = {"fakeddit": ("2way", "6way"), "crisismmd": ("informativeness", "humanitarian"),
         "mmimdb": ("genres",)}
UNCERTAIN_MARGIN = 0.10


def labels_for(dataset, task, task_config):
    if dataset == "fakeddit":
        return list(fakeddit_prompts.get_prompt_set(task, task_config.get("prompt_variant", "current")))
    if dataset == "crisismmd":
        return list(crisismmd_prompts.get_prompt_set(task))
    return list(mmimdb_prompts.get_prompt_set())


def read_run(log_dir, labels, max_samples, score_prefix="score"):
    """Đọc đúng tối đa số mẫu đã cho, kể cả khi log bị thay thế ngoài ý muốn."""
    path = Path(log_dir) / "raw_predictions.tsv"
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        required = {"id", "text", "image_path", "y_true", "y_pred"} | {
            f"{score_prefix}_{c}" for c in labels}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Log thiếu cột: {sorted(required - set(reader.fieldnames or []))}")
        rows = []
        for i, row in enumerate(reader):
            if i >= max_samples:
                raise ValueError(f"Log có hơn {max_samples} mẫu: {path}")
            rows.append(row)
    if not rows:
        raise ValueError(f"Log rỗng: {path}")
    return rows


def combine_runs(runs, labels, baseline_model=DEFAULT_MODELS[0], margin_threshold=UNCERTAIN_MARGIN,
                 methods=("equal_mean", "margin_gate")):
    """Ghép đúng thứ tự mẫu/lớp rồi tính ensemble và điều kiện margin cố định."""
    if baseline_model not in runs or len(runs) < 2:
        raise ValueError("Cần baseline và ít nhất một model đối chứng")
    names = list(runs)
    reference = runs[baseline_model]
    if not reference:
        raise ValueError("Không có mẫu để so sánh")
    keys = ("id", "text", "image_path", "y_true")
    for name, rows in runs.items():
        if len(rows) != len(reference):
            raise ValueError(f"Số mẫu của {name} khác baseline")
        for i, (row, base) in enumerate(zip(rows, reference)):
            if any(row[key] != base[key] for key in keys):
                raise ValueError(f"Cohort không khớp ở {name}, dòng {i + 1}")

    matrices = {}
    predictions = {}
    for name, rows in runs.items():
        matrix = np.array([[float(row[f"score_{label}"]) for label in labels] for row in rows])
        if not np.isfinite(matrix).all() or (matrix < -1e-6).any() or (matrix > 1 + 1e-6).any():
            raise ValueError(f"Điểm của {name} ngoài khoảng [0, 1] hoặc không hữu hạn")
        if not np.allclose(matrix.sum(axis=1), 1, atol=1e-5):
            raise ValueError(f"Điểm của {name} không cộng thành 1")
        pred = np.asarray([str(label) for label in labels], dtype=object)[matrix.argmax(axis=1)]
        if any(str(pred[i]) != row["y_pred"] for i, row in enumerate(rows)):
            raise ValueError(f"Nhãn trong log {name} khác argmax điểm")
        matrices[name], predictions[name] = matrix, pred

    avg = np.mean(np.stack(list(matrices.values())), axis=0)
    ensemble = np.asarray([str(label) for label in labels], dtype=object)[avg.argmax(axis=1)]
    base_scores = matrices[baseline_model]
    margins = np.sort(base_scores, axis=1)[:, -1] - np.sort(base_scores, axis=1)[:, -2]
    uncertain = margins <= margin_threshold
    gated = np.where(uncertain, ensemble, predictions[baseline_model])
    truth = np.asarray([row["y_true"] for row in reference], dtype=object)

    string_labels = [str(label) for label in labels]

    def metrics(name, pred, scores):
        item = {"method": name, "n_samples": len(truth), "accuracy": float(accuracy_score(truth, pred)),
                "macro_f1": float(f1_score(truth, pred, labels=string_labels, average="macro", zero_division=0)),
                "balanced_accuracy_observed": float(balanced_accuracy_score(truth, pred)),
                "changed_vs_baseline": int(np.sum(pred != predictions[baseline_model])),
                "wrong_to_right": int(np.sum((pred == truth) & (predictions[baseline_model] != truth))),
                "right_to_wrong": int(np.sum((pred != truth) & (predictions[baseline_model] == truth)))}
        if len(labels) == 2:
            positive = string_labels[1]
            item["auroc"] = (float(roc_auc_score(truth == positive, scores[:, 1]))
                             if len(set(truth)) == 2 else None)
        return item

    summary = [metrics(name, predictions[name], matrices[name]) for name in names]
    if "equal_mean" in methods:
        summary.append(metrics("equal_mean", ensemble, avg))
    if "margin_gate" in methods:
        summary.append(metrics("margin_gate", gated, np.where(uncertain[:, None], avg, base_scores)))
    comparison = []
    for i, row in enumerate(reference):
        comparison.append({"id": row["id"], "y_true": truth[i],
                           "baseline_pred": predictions[baseline_model][i],
                           "baseline_margin": float(margins[i]), "uncertain": bool(uncertain[i]),
                           "ensemble_pred": ensemble[i], "gated_pred": gated[i],
                           **{f"pred_{name}": predictions[name][i] for name in names},
                           **{f"ensemble_score_{label}": float(avg[i, j]) for j, label in enumerate(labels)}})
    return summary, comparison, {"class_support": dict(Counter(truth)),
                                 "uncertain_count": int(uncertain.sum()),
                                 "margin_threshold": margin_threshold}


def combine_multilabel_runs(runs, labels, task_config, baseline_model, methods=("equal_mean",)):
    """Ghép điểm theo từng nhãn; không chuẩn hóa softmax cho bài toán multi-label."""
    if baseline_model not in runs or len(runs) < 2:
        raise ValueError("Cần baseline và ít nhất một model đối chứng")
    if any(method != "equal_mean" for method in methods):
        raise ValueError("MM-IMDb chỉ hỗ trợ equal_mean cho điểm nhiều nhãn")
    reference = runs[baseline_model]
    keys = ("id", "text", "image_path", "y_true")
    for name, rows in runs.items():
        if len(rows) != len(reference) or any(
                any(row[key] != base[key] for key in keys) for row, base in zip(rows, reference)):
            raise ValueError(f"Cohort không khớp ở {name}")
    relative = task_config.get("decision_mode", "relative_z") == "relative_z"
    prefix = "zscore" if relative else "prob"
    matrices = {name: np.array([[float(row[f"{prefix}_{label}"]) for label in labels]
                                for row in rows]) for name, rows in runs.items()}
    if any(not np.isfinite(matrix).all() for matrix in matrices.values()):
        raise ValueError("Điểm multi-label không hữu hạn")
    if not relative and any(((matrix < 0) | (matrix > 1)).any() for matrix in matrices.values()):
        raise ValueError("Xác suất multi-label ngoài [0, 1]")
    mean = np.mean(np.stack(list(matrices.values())), axis=0)
    threshold = (task_config.get("relative_z_threshold", 1.0) if relative
                 else task_config.get("threshold", 0.22))
    fallback = task_config.get("fallback_top1", True)

    def decide(matrix):
        result = []
        for row in matrix:
            selected = [str(labels[i]) for i, value in enumerate(row)
                        if (value > threshold if relative else value >= threshold)]
            if not selected and fallback:
                selected = [str(labels[int(row.argmax())])]
            result.append(set(selected))
        return result

    truth = [set(filter(None, row["y_true"].split("|"))) for row in reference]
    baseline = [set(filter(None, row["y_pred"].split("|"))) for row in reference]
    unknown = set().union(*truth, *baseline) - {str(label) for label in labels}
    if unknown:
        raise ValueError(f"Nhãn multi-label ngoài prompt: {sorted(unknown)}")
    predicted = {name: decide(matrix) for name, matrix in matrices.items()}
    if any(predicted[name] != [set(filter(None, row["y_pred"].split("|"))) for row in runs[name]]
           for name in runs):
        raise ValueError("Nhãn multi-label trong log khác quy tắc quyết định")
    predicted["equal_mean"] = decide(mean)
    from sklearn.preprocessing import MultiLabelBinarizer
    mlb = MultiLabelBinarizer(classes=[str(label) for label in labels])
    truth_matrix = mlb.fit_transform(truth)
    summary = []
    for name in [*runs, "equal_mean"]:
        pred = predicted[name]
        pred_matrix = mlb.transform(pred)
        summary.append({"method": name, "n_samples": len(truth),
                        "macro_f1": float(f1_score(truth_matrix, pred_matrix, average="macro", zero_division=0)),
                        "micro_f1": float(f1_score(truth_matrix, pred_matrix, average="micro", zero_division=0)),
                        "hamming_accuracy": float((truth_matrix == pred_matrix).mean()),
                        "exact_match": float(np.mean([item == actual for item, actual in zip(pred, truth)])),
                        "changed_vs_baseline": sum(item != base for item, base in zip(pred, baseline)),
                        "wrong_to_right": sum(item == actual and base != actual for item, base, actual
                                              in zip(pred, baseline, truth)),
                        "right_to_wrong": sum(item != actual and base == actual for item, base, actual
                                              in zip(pred, baseline, truth))})
    comparison = [{"id": row["id"], "y_true": row["y_true"],
                   "baseline_pred": "|".join(sorted(baseline[i])),
                   "ensemble_pred": "|".join(sorted(predicted["equal_mean"][i])),
                   **{f"ensemble_score_{label}": float(mean[i, j]) for j, label in enumerate(labels)}}
                  for i, row in enumerate(reference)]
    return summary, comparison, {"decision_mode": task_config.get("decision_mode", "relative_z"),
                                 "threshold": threshold, "fallback_top1": fallback,
                                 "class_support": dict(Counter(label for item in truth for label in item))}


def write_tsv(path, rows):
    if not rows:
        raise ValueError("Không có kết quả để ghi")
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment_config.yaml")
    parser.add_argument("--job-config", required=True)
    args = parser.parse_args(argv)
    job = yaml.safe_load(Path(args.job_config).read_text(encoding="utf-8"))
    dataset, task = job["dataset"], job["task"]
    models, limit = job["models"], job["max_samples"]
    methods = job["ensemble"]["methods"]
    margin_threshold = job["ensemble"].get("margin_threshold", UNCERTAIN_MARGIN)
    if dataset not in TASKS or task not in TASKS[dataset]:
        parser.error("Dataset/task không hợp lệ")
    if type(limit) is not int or limit < 1:
        parser.error("max_samples phải là số nguyên dương")
    if len(models) < 2 or len(models) != len(set(models)) or any(name not in MODEL_CONFIGS for name in models):
        parser.error("Ensemble cần ít nhất hai model hợp lệ, không trùng tên")
    if (not methods or any(method not in ("equal_mean", "margin_gate") for method in methods)
            or len(methods) != len(set(methods)) or (dataset == "mmimdb" and "margin_gate" in methods)):
        parser.error("Phương pháp ensemble không hợp lệ cho dataset")
    if not isinstance(margin_threshold, (int, float)) or not 0 <= margin_threshold <= 1:
        parser.error("margin_threshold phải trong [0, 1]")
    if job.get("offline", False):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    cfg = deepcopy(load_config(args.config))
    cfg["models"] = models
    cfg.setdefault("evaluation", {}).update(enabled=True, sample_lock_mode="off")
    task_config = deepcopy(cfg["datasets"][dataset])
    task_config.update(task=task, max_samples=limit, _audit_scope=f"ensemble_{job['id']}")
    if len(task_config.get("prompt_variants", [task_config.get("prompt_variant", "current")])) != 1:
        parser.error("Ensemble cần đúng một prompt_variant")
    if len(task_config.get("prompt_aggregations", [task_config.get("prompt_aggregation", "mean_embed")])) != 1:
        parser.error("Ensemble cần đúng một prompt_aggregation")
    task_config["prompt_variant"] = task_config.pop("prompt_variants", [task_config.get("prompt_variant", "current")])[0]
    task_config["prompt_aggregation"] = task_config.pop("prompt_aggregations", [task_config.get("prompt_aggregation", "mean_embed")])[0]
    cfg["datasets"][dataset] = task_config
    parent = Path(job["output_dir"])
    parent.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix=f"{job['id']}_", dir=parent))
    cfg["output"]["raw_predictions_dir"] = str(root / "model_logs")
    plan = {"protocol": "bounded_no_fit", "dataset": dataset, "task": task, "models": models,
            "max_samples": limit, "baseline": models[0], "methods": methods,
            "margin_threshold": margin_threshold, "device_requested": job.get("device"),
            "offline": job.get("offline", False), "config": cfg}
    (root / "frozen_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[PLAN] {root / 'frozen_plan.json'}", flush=True)

    logs, manifests = {}, {}
    try:
        labels = labels_for(dataset, task, task_config)
        score_prefix = ("zscore" if task_config.get("decision_mode", "relative_z") == "relative_z"
                        else "prob") if dataset == "mmimdb" else "score"
        for name in models:
            vlm = None
            try:
                vlm = load_model(name, device=job.get("device"))
                result = RUNNERS[dataset](vlm, task_config, cfg)
                log_dir = result["prediction_log_dir"]
                manifest = json.loads((Path(log_dir) / "manifest.json").read_text(encoding="utf-8"))
                manifests[name] = {"log_dir": log_dir, "sample_set_id": manifest["sample_set_id"],
                                   "prompt_id": manifest["prompt_id"]}
                logs[name] = read_run(log_dir, labels, limit, score_prefix)
            finally:
                if vlm is not None:
                    vlm.unload()
                    del vlm
                    gc.collect()

        cohorts = {entry["sample_set_id"] for entry in manifests.values()}
        prompts = {entry["prompt_id"] for entry in manifests.values()}
        if len(cohorts) != 1 or len(prompts) != 1:
            raise ValueError("Mẫu hoặc prompt không đồng nhất giữa các model")
        if dataset == "mmimdb":
            summary, comparison, detail = combine_multilabel_runs(logs, labels, task_config, models[0], methods)
        else:
            summary, comparison, detail = combine_runs(logs, labels, models[0], margin_threshold, methods)
        all_summary = [{"dataset": dataset, "task": task, **item} for item in summary]
        write_tsv(root / "comparison.tsv", comparison)
        details = {**detail, "sample_set_id": next(iter(cohorts)),
                   "prompt_id": next(iter(prompts)), "runs": manifests}
        write_tsv(root / "summary.tsv", all_summary)
        (root / "details.json").write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[RESULT] {root / 'summary.tsv'}", flush=True)
        for row in all_summary:
            print(f"{row['dataset']}/{row['task']} {row['method']:24} n={row['n_samples']:3} "
                  f"macro-F1={row['macro_f1']:.4f} acc={row.get('accuracy', row.get('exact_match')):.4f} "
                  f"sửa={row['wrong_to_right']} hỏng={row['right_to_wrong']}", flush=True)
    except Exception as exc:
        (root / "failure.txt").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
