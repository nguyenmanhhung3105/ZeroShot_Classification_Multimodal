"""So ghép cặp hoặc tổng hợp seed từ các run đã hoàn tất; không gọi model."""
import argparse
import csv
import io
import json
from pathlib import Path
import sys
import uuid
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.waffle_lab.run import ROOT, digest, evaluate, write_json
import numpy as np


def load_run(directory, allow_synthetic=False):
    directory = Path(directory)
    def artifact(name):
        direct = directory / name
        if direct.is_file():
            return direct.read_text(encoding="utf-8")
        archive = directory / "run_artifacts.zip"
        if archive.is_file():
            with zipfile.ZipFile(archive) as stream:
                return stream.read(name).decode("utf-8")
        raise ValueError(f"Thiếu {name} trong run hoặc run_artifacts.zip")
    manifest = json.loads(artifact("run_manifest.json"))
    if manifest.get("status") != "complete":
        raise ValueError("Chỉ phân tích run đã hoàn tất")
    if not allow_synthetic and manifest.get("inference_backend") != "openai_clip_real":
        raise ValueError("Đầu ra giả/custom không được dùng làm kết quả model thật")
    cohort = json.loads(artifact("cohort.json"))
    if digest(cohort["rows"]) != manifest.get("cohort_fingerprint"):
        raise ValueError("Cohort manifest sai fingerprint")
    plan = json.loads(artifact("frozen_plan.json"))
    if digest(plan["protocol"]) != manifest.get("protocol_fingerprint"):
        raise ValueError("Protocol sai fingerprint")
    with io.StringIO(artifact("predictions.tsv"), newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if not rows or len(rows) != manifest["samples"]:
        raise ValueError("Thiếu dự đoán")
    if [(r["id"], r["y_true"]) for r in rows] != [(r["id"], r["y_true"]) for r in cohort["rows"]]:
        raise ValueError("Dự đoán khác ID/thứ tự/nhãn của cohort")
    labels = plan["label_order"]
    scores = np.asarray([json.loads(row["cosine_scores"]) for row in rows], dtype=float)
    if scores.shape != (len(rows), len(labels)) or not np.isfinite(scores).all():
        raise ValueError("Điểm cosine không hợp lệ")
    predicted = [labels[i] for i in scores.argmax(axis=1)]
    if predicted != [row["y_pred"] for row in rows]:
        raise ValueError("y_pred không khớp argmax cosine")
    metrics = evaluate([r["y_true"] for r in rows], predicted, scores, labels)
    return {"directory": str(directory), "manifest": manifest, "plan": plan,
            "rows": rows, "metrics": metrics}


def compare(baseline, candidate, bootstraps=2000, seed=42):
    """CI ghép cặp thăm dò; không chọn tham số bằng kết quả so sánh này."""
    from scipy.stats import binomtest
    left, right = baseline["manifest"], candidate["manifest"]
    for key in ("task", "model", "samples", "cohort_fingerprint", "protocol_fingerprint", "inference_backend",
                "device", "compute_dtype"):
        if left.get(key) != right.get(key):
            raise ValueError(f"Hai run khác {key}; không so như cùng phép thử")
    if type(bootstraps) is not int or bootstraps < 1:
        raise ValueError("bootstraps phải là số nguyên dương")
    labels = baseline["plan"]["label_order"]
    order = {label: i for i, label in enumerate(labels)}
    truth = np.array([order[r["y_true"]] for r in baseline["rows"]])
    old = np.array([order[r["y_pred"]] for r in baseline["rows"]])
    new = np.array([order[r["y_pred"]] for r in candidate["rows"]])
    old_ok, new_ok = truth == old, truth == new
    b, c = int((old_ok & ~new_ok).sum()), int((~old_ok & new_ok).sum())
    flips = {"wrong_to_correct": c, "correct_to_wrong": b,
             "wrong_to_wrong_changed": int((~old_ok & ~new_ok & (old != new)).sum()),
             "unchanged": int((old == new).sum())}

    def macro_f1(actual, predicted):
        matrix = np.bincount(actual * len(labels) + predicted,
                             minlength=len(labels)**2).reshape(len(labels), len(labels))
        denominator = matrix.sum(0) + matrix.sum(1)
        return np.divide(2 * matrix.diagonal(), denominator,
                         out=np.zeros(len(labels)), where=denominator != 0).mean()

    rng = np.random.default_rng(seed)
    deltas = np.empty(bootstraps)
    for index in range(bootstraps):
        sampled = rng.integers(len(truth), size=len(truth))
        deltas[index] = macro_f1(truth[sampled], new[sampled]) - macro_f1(truth[sampled], old[sampled])
    by_event = {}
    events = np.array([r["event"] for r in baseline["rows"]])
    for event in sorted(set(events)):
        mask = events == event
        by_event[event] = {"samples": int(mask.sum()), "wrong_to_correct": int((~old_ok & new_ok & mask).sum()),
                           "correct_to_wrong": int((old_ok & ~new_ok & mask).sum())}
    return {"scope": "exploratory_paired_comparison", "baseline": baseline["directory"],
            "candidate": candidate["directory"], "samples": len(truth), "flips": flips,
            "baseline_metrics": baseline["metrics"], "candidate_metrics": candidate["metrics"],
            "delta_accuracy": candidate["metrics"]["accuracy"] - baseline["metrics"]["accuracy"],
            "delta_macro_f1": candidate["metrics"]["macro_f1"] - baseline["metrics"]["macro_f1"],
            "delta_macro_f1_paired_bootstrap_95pct": np.quantile(deltas, [.025, .975]).tolist(),
            "bootstrap_repetitions": bootstraps, "bootstrap_seed": seed,
            "mcnemar_exact_p_nominal": float(binomtest(b, b+c, p=.5).pvalue) if b+c else 1.,
            "multiple_comparison_correction": "not_applied_exploratory", "by_event": by_event}


def summarize(runs):
    """Tổng hợp cùng cohort/protocol; không xếp hạng hoặc chọn seed tốt nhất."""
    groups = {}
    contexts = set()
    for run in runs:
        metadata = run["manifest"]
        contexts.add(tuple(metadata.get(key) for key in ("cohort_fingerprint", "protocol_fingerprint", "inference_backend",
                                                        "device", "compute_dtype")))
        groups.setdefault(metadata["variant"], []).append(run)
    if len(contexts) != 1:
        raise ValueError("Tổng hợp seed cần cùng cohort và protocol")
    rows = []
    for variant, group in groups.items():
        seeds = [run["manifest"]["seed"] for run in group]
        if len(seeds) != len(set(seeds)):
            raise ValueError("Lặp seed trong cùng biến thể; chọn đúng một run/seed")
        row = {"variant": variant, "seeds": sorted(seeds), "n_runs": len(group)}
        for metric in ("accuracy", "macro_f1", "auroc"):
            values = [run["metrics"][metric] for run in group if run["metrics"][metric] is not None]
            row[metric + "_mean"] = float(np.mean(values)) if values else None
            row[metric + "_std"] = float(np.std(values, ddof=0)) if len(values) > 1 else None
        rows.append(row)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    paired = sub.add_parser("compare")
    paired.add_argument("--baseline", required=True)
    paired.add_argument("--candidate", required=True)
    paired.add_argument("--bootstraps", type=int, default=2000)
    paired.add_argument("--seed", type=int, default=42)
    seeds = sub.add_parser("seed-summary")
    seeds.add_argument("runs", nargs="+")
    args = parser.parse_args(argv)
    if args.action == "compare":
        runs = [load_run(args.baseline), load_run(args.candidate)]
        report = compare(*runs, args.bootstraps, args.seed)
    else:
        runs = [load_run(path) for path in args.runs]
        report = {"seed_summary": summarize(runs), "std_definition": "population_ddof_0"}
    first = runs[0]["manifest"]
    directory = (ROOT / "results/analysis/waffle_lab/crisismmd" / first["task"]
                 / f"samples_{first['samples']}" / first["model"] / uuid.uuid4().hex[:12])
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "analysis.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[ANALYSIS] {directory / 'analysis.json'}")


if __name__ == "__main__":
    main()
