"""Đối chứng quan hệ ảnh–text trên tối đa năm dòng đầu Fakeddit, encode một lần.

Chạy từ gốc: venv/bin/python -B src/research/run_relation_ablation.py
Không tải model mới khi chọn --offline. Không tìm thêm mẫu sau năm dòng đầu.
"""
import argparse
from copy import deepcopy
import csv
import hashlib
from itertools import islice
import json
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from config import load_config
from pair_relation import relation_settings, relation_evidence, apply_class_relation


def bounded_copy(source, target, limit=5):
    """Chỉ đọc schema và năm dòng đầu, giới hạn ngay tại iterator."""
    with Path(source).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        fields = reader.fieldnames
        rows = list(islice(reader, limit))
    if not rows:
        raise ValueError("Fakeddit không có mẫu trong năm dòng đầu")
    with target.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def compare_rows(rows, options, image_weight):
    """Tính lại điểm từ log cùng lượt, không dùng nhãn thật để chọn tham số."""
    labels = list(range(6))
    base = np.array([[float(row[f"score_{c}"]) for c in labels] for row in rows])
    pi = np.array([[float(row[f"score_image_{c}"]) for c in labels] for row in rows])
    pt = np.array([[float(row[f"score_text_{c}"]) for c in labels] for row in rows])
    r = [float(row["image_text_similarity"]) if row.get("image_text_similarity") else None for row in rows]
    # Dựng vector hai chiều chỉ để tái dùng công thức evidence từ cosine đã ghi.
    image = np.tile([1., 0.], (len(rows), 1))
    text = np.array([[x, np.sqrt(max(0., 1 - x*x))] if x is not None else [0., 0.] for x in r])
    evidence = relation_evidence(image, text, pi, pt, np.array([x is None for x in r]))
    new = apply_class_relation(base, labels, evidence, options, image_weight)
    legacy = base.copy()
    for i, cosine in enumerate(r):
        if cosine is not None:
            boost = 0.1 * (1 - np.clip(cosine, -1, 1)) / 2
            legacy[i, 2] += boost
            legacy[i] /= 1 + boost
    truth = [int(row["y_true"]) for row in rows]
    matrices = {"baseline": base, "legacy_false_connection_0.1": legacy, "class_aware": new}
    summary, comparison = [], []
    for method, scores in matrices.items():
        prediction = scores.argmax(axis=1)
        summary.append({"method": method, "n_samples": len(rows), "accuracy": accuracy_score(truth, prediction),
                        "macro_f1": f1_score(truth, prediction, labels=labels, average="macro", zero_division=0),
                        "balanced_accuracy": balanced_accuracy_score(truth, prediction),
                        "relation_weight": options["weight"] if method == "class_aware" else 0.1 if method.startswith("legacy") else 0,
                        "changed_predictions": int(np.sum(prediction != base.argmax(axis=1)))})
        for i, row in enumerate(rows):
            comparison.append({"id": row["id"], "y_true": truth[i], "method": method,
                "y_pred": int(prediction[i]), "image_text_similarity": r[i],
                "js_divergence": evidence["image_text_js_divergence"][i],
                "reliability": evidence["image_text_reliability"][i],
                "match_evidence": evidence["image_text_match_evidence"][i],
                "mismatch_evidence": evidence["image_text_mismatch_evidence"][i],
                **{f"score_{c}": float(scores[i,c]) for c in labels}})
    return summary, comparison


def write_tsv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment_config.yaml")
    parser.add_argument("--model", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--max-samples", type=int, default=5)
    parser.add_argument("--output-dir", default="results/relation_ablation")
    args = parser.parse_args(argv)
    if not 1 <= args.max_samples <= 5:
        parser.error("--max-samples phải trong [1, 5] cho ablation này")
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    base = load_config(args.config)
    if base["datasets"]["fakeddit"]["task"] != "6way":
        raise ValueError("Đối chứng này cần Fakeddit task 6way")
    options, _ = relation_settings(base["datasets"]["fakeddit"], "fakeddit", base)
    options.update(enabled=True, mode="class_aware")
    if not options["class_policies"]:
        options["class_policies"] = {0: "match", 2: "mismatch"}
    root_parent = Path(args.output_dir)
    root_parent.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix=f"smoke{args.max_samples}_", dir=root_parent))
    bounded = root / "input_bounded.tsv"
    selected = bounded_copy(base["datasets"]["fakeddit"]["data_path"], bounded, args.max_samples)
    cfg = deepcopy(base)
    task = cfg["datasets"]["fakeddit"]
    task.update(data_path=str(bounded), max_samples=selected, _audit_scope="relation_smoke5",
                image_text_relation={"enabled": True, "mode": "diagnostic"})
    cfg.setdefault("evaluation", {}).update(enabled=True, sample_lock_mode="off")
    cfg["output"]["raw_predictions_dir"] = str(root / "baseline_log")
    cfg["inference"]["batch_size"] = min(selected, 5)
    model = args.model or base["models"][0]
    from run.common import get_image_weight
    weight = get_image_weight(cfg, "fakeddit")
    if not 0 < weight < 1:
        raise ValueError("Đối chứng quan hệ cần hai nhánh phân loại (0 < image_weight < 1)")
    plan = {"model": model, "device_requested": args.device, "base_config": base,
            "relation_options": options, "legacy_weight": 0.1, "source_rows_limit": args.max_samples,
            "input_sha256": hashlib.sha256(bounded.read_bytes()).hexdigest(),
            "protocol": "frozen_smoke_only_no_fit", "bounded_input": str(bounded)}
    source_root = Path(__file__).resolve().parents[1]
    plan["source_sha256"] = {str(p.relative_to(source_root)): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sorted(source_root.rglob("*.py")) if "__pycache__" not in p.parts}
    (root / "frozen_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2))
    print(f"[PLAN] {root / 'frozen_plan.json'}", flush=True)
    vlm = None
    try:
        from models.model_registry import load_model
        from run.run_fakeddit import run_fakeddit
        vlm = load_model(model, device=args.device)
        result = run_fakeddit(vlm, task, cfg)
        with (Path(result["prediction_log_dir"]) / "raw_predictions.tsv").open(encoding="utf-8", newline="") as stream:
            rows = list(islice(csv.DictReader(stream, delimiter="\t"), selected))
        summary, comparison = compare_rows(rows, options, weight)
        for row in summary:
            row.update(model=model, device=vlm.device, sample_set_id=result.get("sample_set_id"))
        write_tsv(root / "summary.tsv", summary)
        write_tsv(root / "comparison.tsv", comparison)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"[RESULTS] {root}")
    except Exception as exc:
        (root / "failure.json").write_text(json.dumps({"model": model, "error": str(exc)}, ensure_ascii=False, indent=2))
        raise
    finally:
        if vlm is not None:
            vlm.unload()
    return root


if __name__ == "__main__":
    main()
