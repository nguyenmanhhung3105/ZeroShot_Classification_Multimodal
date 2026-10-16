"""Chạy MetaCLIP2 hai bộ prompt và dùng DCLIP adapted khi margin thấp."""

import argparse
from copy import deepcopy
import csv
import gc
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import yaml

from config import load_config
from models.model_registry import load_model
from prompts import crisismmd_prompts
from research.adapted_methods import load_descriptors
from run.run_crisismmd import run_crisismmd
from compare_crisismmd_prompt_gate import MODEL, evaluate


def read_log(path, limit):
    directory = Path(path)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    with (directory / "raw_predictions.tsv").open(encoding="utf-8", newline="") as stream:
        rows = []
        for index, row in enumerate(csv.DictReader(stream, delimiter="\t")):
            if index >= limit:
                raise ValueError("Log có nhiều mẫu hơn giới hạn đã đóng băng")
            rows.append(row)
    if len(rows) != manifest.get("n_total_samples"):
        raise ValueError("Số mẫu log khác manifest")
    return manifest, rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment_config.yaml")
    parser.add_argument("--job-config", required=True)
    args = parser.parse_args(argv)
    job = yaml.safe_load(Path(args.job_config).read_text(encoding="utf-8"))
    if (job.get("dataset") != "crisismmd" or job.get("task") not in
            ("informativeness", "humanitarian") or job.get("models") != [MODEL]
            or job.get("prompt_gate", {}).get("method") != "dclip_adapted"):
        parser.error("Job phải là CrisisMMD, MetaCLIP2 và prompt_gate dclip_adapted")
    threshold = job["prompt_gate"].get("margin_threshold", 0.10)
    if type(threshold) not in (int, float) or not 0 <= threshold <= 1:
        parser.error("margin_threshold phải trong [0, 1]")

    cfg = deepcopy(load_config(args.config))
    cfg["models"] = [MODEL]
    cfg.setdefault("evaluation", {})["sample_lock_mode"] = "off"
    task = deepcopy(cfg["datasets"]["crisismmd"])
    task.update(task=job["task"], max_samples=job["max_samples"],
                _audit_scope=f"prompt_margin_gate_{job['id']}")
    if len(task.get("prompt_aggregations", [task.get("prompt_aggregation", "mean_embed")])) != 1:
        parser.error("Cần đúng một prompt_aggregation")
    task["prompt_aggregation"] = task.pop("prompt_aggregations", [task.get("prompt_aggregation", "mean_embed")])[0]
    task.pop("prompt_variants", None)
    if task.get("image_text_relation", {}).get("enabled") is not False:
        parser.error("prompt_gate cần tắt image_text_relation")
    if task.get("inference", {}).get("fusion_mode", "fixed") != "fixed":
        parser.error("prompt_gate cần fusion cố định")
    cfg["datasets"]["crisismmd"] = task

    descriptor = ROOT / "configs/research/descriptors" / (
        "crisismmd_info.json" if job["task"] == "informativeness" else "crisismmd_humanitarian.json")
    prompts, provenance = load_descriptors(descriptor, crisismmd_prompts.get_label_names(job["task"]))
    parent = Path(job["output_dir"])
    parent.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix=f"{job['task']}_", dir=parent))
    plan = {"dataset": "crisismmd", "task": job["task"], "model": MODEL,
            "max_samples": job["max_samples"], "margin_threshold": threshold,
            "methods": ["baseline", "dclip_adapted", "prompt_margin_gate"],
            "descriptor_file": str(descriptor.relative_to(ROOT)), **provenance}
    (root / "frozen_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n",
                                          encoding="utf-8")
    print(f"[PLAN] {root / 'frozen_plan.json'}", flush=True)

    vlm = None
    try:
        vlm = load_model(MODEL, device=job.get("device"))
        logs = {}
        for method in ("baseline", "dclip_adapted"):
            active = deepcopy(task)
            if method == "dclip_adapted":
                active["_research_image_prompts"] = prompts
                active["_research_metadata"] = {"image_method": "dclip", **provenance}
            run_config = deepcopy(cfg)
            run_config["datasets"]["crisismmd"] = active
            run_config["output"]["raw_predictions_dir"] = str(root / "raw_predictions" / method)
            print(f"[RUN] {method}", flush=True)
            result = run_crisismmd(vlm, active, run_config)
            logs[method] = read_log(result["prediction_log_dir"], job["max_samples"])
            gc.collect()
        report, details = evaluate(*logs["baseline"], *logs["dclip_adapted"], threshold)
        report.update(source_type="inference_run", descriptor_file=plan["descriptor_file"],
                      **provenance)
        (root / "details.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                            encoding="utf-8")
        with (root / "comparison.tsv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(details[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(details)
        with (root / "summary.tsv").open("w", encoding="utf-8", newline="") as stream:
            fields = ["dataset", "task", "model", "method", "n_samples", "margin_threshold",
                      "gate_count", "accuracy", "macro_f1", "changed_vs_baseline",
                      "wrong_to_right", "right_to_wrong", "wrong_to_wrong_changed"]
            writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            for name, metrics in report["methods"].items():
                writer.writerow({"dataset": "crisismmd", "task": job["task"], "model": MODEL,
                                 "method": name, "n_samples": report["n_samples"],
                                 "margin_threshold": threshold, "gate_count": report["gate_count"],
                                 **metrics})
        print(f"[RESULT] {root / 'summary.tsv'}", flush=True)
    except Exception as exc:
        (root / "failure.txt").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        raise
    finally:
        if vlm is not None:
            vlm.unload()


if __name__ == "__main__":
    main()
