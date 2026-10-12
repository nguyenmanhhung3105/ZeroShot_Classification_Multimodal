"""
Đối chứng phương pháp trên backbone hiện có bằng runner từng dataset.

Chạy từ thư mục gốc. --prepare-only lưu prompt/cấu hình trước khi đánh giá;
chế độ này không đọc dataset, tải trọng số hoặc suy luận.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import yaml

from research.adapted_methods import ROOT, official_words, waffle_prompts, load_descriptors
from prompts import fakeddit_prompts, crisismmd_prompts, mmimdb_prompts


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def git_revision(path):
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def prompt_spec(dataset, task):
    if dataset == "fakeddit":
        return fakeddit_prompts.get_label_names(task)
    if dataset == "crisismmd":
        return crisismmd_prompts.get_label_names(task)
    if dataset == "mmimdb" and task == "genres":
        return mmimdb_prompts.get_label_names()
    raise ValueError("Unsupported dataset/task")


def baseline_prompt_set(dataset, task):
    if dataset == "fakeddit":
        return fakeddit_prompts.get_prompt_set(task["task"], task["prompt_variant"])
    if dataset == "crisismmd":
        return crisismmd_prompts.get_prompt_set(task["task"])
    return mmimdb_prompts.get_prompt_set()


def prepare_trials(settings, base):
    dataset = settings["dataset"]
    task = deepcopy(base["datasets"][dataset])
    task["task"] = settings.get("task", task.get("task", "genres"))
    task["enabled"] = True
    limit = settings.get("max_samples", 5)
    if type(limit) is not int or limit < 1:
        raise ValueError("max_samples must be a positive integer")
    task["max_samples"] = limit
    # Chỉ nhận một biến thể và một cách tổng hợp làm mốc đối chứng.
    for plural, singular, default in (("prompt_variants", "prompt_variant", "current"),
                                      ("prompt_aggregations", "prompt_aggregation", "mean_embed")):
        choices = task.pop(plural, [task.get(singular, default)])
        if not isinstance(choices, list) or len(choices) != 1:
            raise ValueError(f"Paper runner requires exactly one {plural}")
        task[singular] = choices[0]
    names = prompt_spec(dataset, task["task"])
    methods = settings.get("methods", ["baseline", "class_name", "waffle"])
    if not isinstance(methods, list) or not methods or any(
            not isinstance(m, str) or m not in ("baseline", "current_mean_sim", "class_name", "waffle", "dclip") for m in methods):
        raise ValueError("Unknown/empty methods")
    if len(set(methods)) != len(methods):
        raise ValueError("Duplicate methods")
    seeds = settings.get("seeds", [0, 1, 2, 3, 4])
    if (not isinstance(seeds, list) or not seeds or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("seeds must be unique integers in [0, 2**32)")
    words, vocabulary_hash = official_words() if "waffle" in methods else (None, None)
    trials = []
    for method in methods:
        for seed in seeds if method == "waffle" else [None]:
            active = deepcopy(task)
            metadata = {"image_method": method, "research_seed": seed,
                        "method_protocol": "frozen_no_fit_exploratory",
                        "implementation": "image_branch_adapter_v1",
                        "image_prompt_aggregation": active["prompt_aggregation"] if method == "baseline" else "mean_sim"}
            prompts = None
            if method == "current_mean_sim":
                prompts = baseline_prompt_set(dataset, task)
            elif method == "class_name":
                prompts = {key: [f"A photo of a {name.replace('_', ' ')}."] for key, name in names.items()}
            elif method == "waffle":
                pairs = settings.get("waffle_pairs", 15)
                prompts = waffle_prompts(names, words, pairs, seed)
                metadata.update(waffle_pairs=pairs, vocabulary_sha256=vocabulary_hash,
                                waffle_alphabet="ascii_lowercase")
            elif method == "dclip":
                if not settings.get("descriptor_file"):
                    raise ValueError("dclip requires a fixed descriptor_file; no automatic generation")
                prompts, origin = load_descriptors(settings["descriptor_file"], names)
                metadata.update(origin)
            active["_research_metadata"] = metadata
            if prompts is not None:
                active["_research_image_prompts"] = prompts
            trials.append(active)
    return trials


def aggregate_results(rows):
    """Tính độ lệch qua seed; không chọn seed tốt nhất hoặc coi là khoảng tin cậy."""
    frame = pd.DataFrame(rows)
    output = []
    for (model, method), group in frame.groupby(["model", "image_method"], sort=False):
        row = {"model": model, "image_method": method, "n_runs": len(group)}
        for metric in ("macro_f1", "micro_f1", "balanced_accuracy", "samples_f1",
                       "image_macro_f1", "text_macro_f1"):
            if metric in group:
                row[metric + "_mean"] = group[metric].mean()
                row[metric + "_std"] = group[metric].std(ddof=0) if len(group) > 1 else None
        output.append(row)
    return pd.DataFrame(output)


def paper_summary(rows):
    from run.common import compact_summary
    frame = compact_summary(rows)
    # compact_summary bỏ các cột image_* không nhận diện;
    # hai cột này xác định phương pháp thí nghiệm nên cần giữ lại.
    for key in ("image_method", "image_prompt_aggregation"):
        frame[key] = [row.get(key) for row in rows]
    return frame


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/research/adapted.yaml")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)
    settings = yaml.safe_load(Path(args.config).read_text())
    base = yaml.safe_load(Path(settings["base_config"]).read_text())
    trials = prepare_trials(settings, base)
    sources = json.loads((ROOT / "configs/research/sources.json").read_text())
    for name, spec in sources.items():
        source = ROOT / "third_party" / name
        if git_revision(source) != spec["commit"]:
            raise ValueError(f"{name}: checkout must match configs/research/sources.json")
        if subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True).strip():
            raise ValueError(f"{name}: upstream checkout modified; keep reference code unchanged")
    output = Path(settings.get("output_dir", "results/paper_methods"))
    output.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix=f"{settings['dataset']}_", dir=output))
    source_files = sorted((ROOT / "src").rglob("*.py"))
    write_json(root / "frozen_plan.json", {"settings": settings, "base_config": base,
               "trials": trials, "sources": sources, "git_hash": git_revision(ROOT),
               "baseline_text_prompts": baseline_prompt_set(settings["dataset"], trials[0]),
               "source_hashes": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in source_files}})
    print(f"[PLAN] {root / 'frozen_plan.json'}; {len(trials)} trials/model")
    if args.prepare_only:
        print("Prepared only: no dataset or model accessed.")
        return

    from models.model_registry import load_model
    from run.run_fakeddit import run_fakeddit
    from run.run_crisismmd import run_crisismmd
    from run.run_mmimdb import run_mmimdb
    runner = {"fakeddit": run_fakeddit, "crisismmd": run_crisismmd, "mmimdb": run_mmimdb}[settings["dataset"]]
    rows = []
    cohort = None
    for model in base["models"]:
        vlm = load_model(model)
        try:
            for active in trials:
                cfg = deepcopy(base)
                cfg["datasets"][settings["dataset"]] = active
                cfg.setdefault("evaluation", {}).update(enabled=True, sample_lock_mode="off")
                cfg.setdefault("output", {})["raw_predictions_dir"] = str(root / "raw_predictions")
                metadata = active["_research_metadata"]
                print(f"[RESEARCH] {model}: {metadata['image_method']} seed={metadata['research_seed']}")
                result = runner(vlm, active, cfg)
                sample_id = result.get("sample_set_id")
                if sample_id is None or (cohort is not None and sample_id != cohort):
                    raise RuntimeError("Cohort mismatch; comparison stopped, previous logs preserved")
                cohort = sample_id
                rows.append({"model": model, "dataset": settings["dataset"], "task": active["task"], **result})
                paper_summary(rows).to_csv(root / "summary.tsv", sep="\t", index=False)
        except Exception as exc:
            write_json(root / "failure.json", {"model": model, "error": str(exc), "completed_runs": len(rows)})
            raise  # không coi nhóm seed chưa chạy hết là đánh giá đã hoàn thành
        finally:
            vlm.unload()
    aggregate_results(rows).to_csv(root / "seed_summary.tsv", sep="\t", index=False)
    print(f"[RESULTS] {root}; summary.tsv + seed_summary.tsv + raw_predictions/")


if __name__ == "__main__":
    main()
