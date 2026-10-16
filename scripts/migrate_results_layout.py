"""Chuyển kết quả cũ sang results/<method>/<dataset>/<task>/samples_N/<model>/<run>.

Mặc định chỉ in kế hoạch. --apply kiểm tra xung đột trước khi di chuyển và ghi
results/_system/migration_map.json để tra lại đường dẫn gốc.
--insert-task-level chuyển cây bốn cấp đã sắp xếp trước đó sang cây năm cấp.
--insert-model-level chuyển cây năm cấp sang cây sáu cấp.
"""

import argparse
import csv
import json
from pathlib import Path
import re
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from results_layout import (dataset_folder, job_path, method_for_job, model_folder, run_path,
                            sample_folder, task_folder)

KNOWN_MODELS = ("metaclip2_vith14", "dfn5b_vith14", "siglip2_so400m",
                "openai_clip_vit_b_32", "openai_clip_vit_l_14", "openai_clip_rn50")


def first_rows(path):
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def distinct(rows, field):
    values = {row.get(field, "") for row in rows if row.get(field, "")}
    return next(iter(values)) if len(values) == 1 else None


def stage_jobs(results):
    stages = {}
    stage_files = list((results / "launches/staging").glob("*/job.yaml"))
    stage_files += list((results / "_system/launches/staging").glob("*/job.yaml"))
    for stage in stage_files:
        try:
            job = yaml.safe_load(stage.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            continue
        if isinstance(job, dict):
            if job.get("kind") == "upstream" and not job.get("backbones"):
                research = stage.parent / "research.yaml"
                if research.is_file():
                    job["backbones"] = yaml.safe_load(research.read_text(encoding="utf-8"))["backbones"]
            stages[stage.parent.name[:12]] = job
    return stages


def _model_values(values):
    values = {str(value) for value in values if value}
    if not values:
        return None
    return next(iter(values)) if len(values) == 1 else "multi_model"


def infer_run_model(source, rows=None, job=None, source_models=None, method=None):
    """Dùng metadata của lượt, không đoán model từ điểm hoặc nhãn thật."""
    source = Path(source)
    rows = rows if rows is not None else first_rows(source / "summary.tsv")
    if method == "ensemble" or (len(source.parts) >= 5 and source.parts[-5] == "ensemble"):
        return "multi_model"
    values = [row.get("model") for row in rows if row.get("model")]
    if values:
        return _model_values(values)
    values = [row.get("method") for row in rows if row.get("method") in KNOWN_MODELS]
    if values:
        return _model_values(values)
    report = source / "summary.json" if source.is_dir() else source
    if report.is_file() and report.suffix == ".json":
        data = json.loads(report.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            if data.get("model"):
                return str(data["model"])
            if data.get("primary_model") and data.get("candidate_model"):
                return _model_values((data["primary_model"], data["candidate_model"]))
            if data.get("models"):
                return _model_values(data["models"])
            for key in ("source_run_id", "primary_run_id"):
                if source_models and data.get(key) in source_models:
                    return source_models[data[key]]
    manifest = source / "manifest.json"
    if manifest.is_file():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        for key in ("model", "model_name"):
            if data.get(key):
                return str(data[key])
    receipt = source / "run_receipt.json"
    if not job and receipt.is_file():
        data = json.loads(receipt.read_text(encoding="utf-8"))
        job = yaml.safe_load(data.get("job_yaml") or "") or None
    plan = source / "frozen_plan.json"
    if plan.is_file():
        data = json.loads(plan.read_text(encoding="utf-8"))
        settings = data.get("settings", {})
        if settings.get("backbones"):
            return model_folder(backbones=settings["backbones"])
        if settings.get("models"):
            return model_folder(models=settings["models"])
        if data.get("base_config", {}).get("models"):
            return model_folder(models=data["base_config"]["models"])
    configs = sorted(source.glob("config_*.yaml")) if source.is_dir() else []
    if configs:
        backbones = set()
        for config in configs:
            backbones.update(yaml.safe_load(config.read_text(encoding="utf-8")).get("backbones", []))
        if backbones:
            return model_folder(backbones=sorted(backbones))
    if job:
        if job.get("kind") == "upstream" and job.get("backbones"):
            return model_folder(backbones=job["backbones"])
        if job.get("models"):
            return model_folder(models=job["models"])
    matched = [name for name in KNOWN_MODELS if name in source.name]
    if matched:
        return _model_values(matched)
    if source_models:
        matched = [model for run_id, model in source_models.items() if run_id in source.name]
        if matched:
            return _model_values(matched)
    raise ValueError(f"Không xác định được model từ metadata: {source}")


def infer_job_method(job_name, rows, family):
    if family == "runs":
        relation = distinct(rows, "relation_mode")
        if relation in ("agreement_fusion", "direct_pair_fusion", "class_aware"):
            return relation
        return "baseline"
    if job_name.startswith("adapted_"):
        method = distinct(rows, "image_method")
        return f"adapted_{method}" if method else "adapted_suite"
    if job_name.startswith("upstream_"):
        method = distinct(rows, "method")
        return f"upstream_{method}" if method else "upstream_suite"
    return "ensemble" if any(row.get("method", "").startswith(("equal_mean", "margin_gate")) for row in rows) else "mixed_suite"


def infer_dataset_task(rows, hint=""):
    dataset, task = distinct(rows, "dataset"), distinct(rows, "task")
    if dataset and task:
        return dataset, task
    for dataset_name, marker, task_name in (
            ("crisismmd", "humanitarian", "humanitarian"),
            ("crisismmd", "info", "informativeness"),
            ("fakeddit", "6way", "6way"), ("fakeddit", "2way", "2way"),
            ("mmimdb", "genres", "genres")):
        if dataset_name in hint and marker in hint:
            return dataset_name, task_name
    return "mixed", "dataset"


def infer_samples(rows, fallback=None):
    values = {row.get("n_samples", "") for row in rows if row.get("n_samples", "")}
    if len(values) == 1:
        return next(iter(values))
    return fallback


def add(moves, source, target, results):
    if source.exists():
        if not source.is_relative_to(results) or not target.is_relative_to(results):
            raise ValueError("Nguồn/đích phải nằm trong results/")
        moves.append((source, target))


def plan(root=ROOT):
    results = Path(root) / "results"
    moves = []
    stages = stage_jobs(results)
    run_meta = {}
    for family in ("runs", "experiments"):
        for source in sorted((results / family).glob("*/*")):
            if not source.is_dir():
                continue
            rows = first_rows(source / "summary.tsv")
            job = stages.get(source.name)
            if job is None and (source / "run_receipt.json").is_file():
                receipt = json.loads((source / "run_receipt.json").read_text(encoding="utf-8"))
                if receipt.get("job_yaml"):
                    candidate = yaml.safe_load(receipt["job_yaml"])
                    if isinstance(candidate, dict) and candidate.get("id") == source.parent.name:
                        job = candidate
            if job:
                count = infer_samples(rows, job["max_samples"])
                model = infer_run_model(source, rows, job)
                if job.get("kind") in ("adapted", "upstream") and len(job.get("methods") or []) != 1:
                    target = results.parent / run_path(job["kind"] + "_suite", job["dataset"],
                                                       job["task"], count, model, source.name)
                else:
                    candidate = (run_path(method_for_job(job), job["dataset"], job["task"],
                                          job["max_samples"], model, source.name)
                                 if job.get("kind") == "upstream" and not job.get("backbones")
                                 else job_path(job, source.name))
                    target = results.parent / (candidate if count == job["max_samples"]
                                               else run_path(candidate.parts[1], job["dataset"], job["task"],
                                                             count, model, source.name))
                dataset, task, samples = job["dataset"], job["task"], count
            else:
                dataset, task = infer_dataset_task(rows, source.parent.name)
                samples = infer_samples(rows)
                method = infer_job_method(source.parent.name, rows, family)
                model = infer_run_model(source, rows, method=method)
                target = results.parent / run_path(method, dataset, task, samples, model, source.name)
            add(moves, source, target, results)
            run_meta[source.name] = (dataset, task, samples, model)

    # Log và summary kiểu cũ: không ghép cặp khi thiếu bằng chứng rằng chúng là cùng lượt.
    for source in sorted((results / "raw_predictions").glob("*")):
        if not source.is_dir():
            continue
        manifest_path = source / "manifest.json"
        if not manifest_path.is_file():
            target = results / "_system/uncategorized" / source.name
        else:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            target = results.parent / run_path("baseline", manifest.get("dataset", "mixed"),
                manifest.get("task", "dataset"), manifest.get("n_total_samples") or manifest.get("max_samples"),
                infer_run_model(source, method="baseline"), source.name)
        add(moves, source, target, results)
    for source in sorted(results.glob("summary_table_*.tsv")):
        rows = first_rows(source)
        dataset, task = infer_dataset_task(rows, source.name)
        samples = infer_samples(rows)
        target = results.parent / run_path("baseline", dataset, task, samples,
                                          infer_run_model(source, rows, method="baseline"),
                                          "legacy_" + source.stem) / "summary.tsv"
        add(moves, source, target, results)

    for family, method in (("paper_methods", "adapted_suite"),
                           ("upstream_projects", "upstream_suite"),
                           ("uncertainty_ensemble", "ensemble"),
                           ("kaggle_runs", "ensemble"),
                           ("relation_ablation", "relation_ablation")):
        for source in sorted((results / family).glob("*")):
            if not source.is_dir():
                continue
            rows = first_rows(source / "summary.tsv")
            if family == "relation_ablation":
                dataset, task = "fakeddit", "6way"
                samples = infer_samples(rows, 5 if source.name.startswith("smoke5") else None)
            else:
                dataset, task = infer_dataset_task(rows, source.name)
                samples = infer_samples(rows)
                if samples is None and (source / "frozen_plan.json").is_file():
                    plan_data = json.loads((source / "frozen_plan.json").read_text(encoding="utf-8"))
                    settings = plan_data.get("settings", plan_data)
                    dataset = settings.get("dataset", dataset)
                    task = settings.get("task", task)
                    samples = settings.get("max_samples")
            if family == "upstream_projects" and source.name.startswith("smoke_suite"):
                dataset, task = "mixed", "dataset"
                if samples is None:
                    counts = {infer_samples(first_rows(summary))
                              for summary in source.glob("*/summary.tsv")}
                    if len(counts) == 1:
                        samples = counts.pop()
            model = infer_run_model(source, rows, method=method)
            target = results.parent / run_path(method, dataset, task, samples, model, source.name)
            add(moves, source, target, results)

    for source in sorted((results / "analysis").glob("*")):
        # Các nhóm đã chuyển sang results/analysis/<policy>/<dataset>/...
        # không còn là đầu vào legacy của bước migration ban đầu.
        if source.is_dir() and any((source / dataset).is_dir()
                                   for dataset in ("crisismmd", "fakeddit", "mmimdb", "mixed")):
            continue
        if source.is_file():
            items = [source]
        elif source.is_dir():
            items = list(source.iterdir())
        else:
            continue
        for item in items:
            if item.is_dir():
                report = item / "summary.json"
                if not report.is_file():
                    report = item / "analysis_manifest.json"
            else:
                report = item
            data = json.loads(report.read_text(encoding="utf-8")) if report.is_file() and report.suffix == ".json" else {}
            if not isinstance(data, dict):
                data = {}
            name = item.name
            policy = data.get("policy") or data.get("review_policy") or source.name
            if source.name == "fallback":
                policy = "fallback"
            if source.name == "crisismmd_agreement":
                policy = "agreement_fusion"
            elif source.name == "crisismmd_direct_pair":
                policy = "direct_pair_fusion"
            elif source.name == "fakeddit_text_exploratory.json":
                policy = "text_diagnosis"
            elif source.name == "uncertainty" and policy == "uncertainty":
                policy = name.split("_", 1)[-1]
            policy = str(policy).removesuffix(".json")
            source_id = data.get("source_run_id") or data.get("primary_run_id") or name[:12]
            meta = run_meta.get(source_id, (None, None, None, None))
            dataset = data.get("dataset") or meta[0] or ("fakeddit" if "fakeddit" in name else "mixed")
            task = data.get("task") or meta[1] or ("2way" if dataset == "fakeddit" else "dataset")
            samples = data.get("n_samples") or data.get("n_six_way_rows") or meta[2]
            model = infer_run_model(item, source_models={key: value[3] for key, value in run_meta.items()},
                                    method="analysis_" + policy)
            run_name = name if item.is_dir() else item.stem
            target = (results / "analysis" / policy / dataset_folder(dataset, task)
                      / task_folder(dataset, task) / sample_folder(samples) / model_folder(models=[model])
                      / run_name)
            if not item.is_dir():
                target /= "summary.json"
            add(moves, item, target, results)

    for source, target in ((results / "launches", results / "_system/launches"),
                           (results / "kaggle_launcher", results / "_system/kaggle_launcher"),
                           (results / "label_distribution", results / "_system/label_distribution"),
                           (results / "INDEX.md", results / "_system/old_index/INDEX.md"),
                           (results / "INDEX.tsv", results / "_system/old_index/INDEX.tsv")):
        add(moves, source, target, results)
    return moves


def apply(moves, root=ROOT):
    results = Path(root) / "results"
    targets = [target for _, target in moves]
    if len(targets) != len(set(targets)):
        raise ValueError("Hai nguồn trỏ cùng một đích; chưa di chuyển gì")
    conflicts = [str(target) for target in targets if target.exists()]
    if conflicts:
        raise ValueError(f"Đích đã tồn tại; chưa di chuyển gì: {conflicts[:5]}")
    record = [{"from": str(source.relative_to(results)), "to": str(target.relative_to(results))}
              for source, target in moves]
    manifest = results / "_system/migration_map.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    if manifest.exists():
        raise FileExistsError(f"Đã có migration map: {manifest}")
    manifest.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for source, target in moves:
        target.parent.mkdir(parents=True, exist_ok=True)
        source.rename(target)
    for old in ("runs", "experiments", "raw_predictions", "paper_methods", "upstream_projects",
                "uncertainty_ensemble", "kaggle_runs", "relation_ablation", "analysis"):
        directory = results / old
        if directory.is_dir():
            for child in sorted(directory.glob("*")):
                if child.is_dir() and not any(child.iterdir()):
                    child.rmdir()
            if not any(directory.iterdir()):
                directory.rmdir()
    return manifest


def plan_task_level(root=ROOT):
    """Chèn task vào giữa dataset và samples cho các lượt đã phân loại."""
    results = Path(root) / "results"
    moves = []
    tasks = ("informativeness", "humanitarian", "multi_label", "genres", "2way", "6way", "dataset")
    for source in sorted(results.glob("*/*/samples_*/*")):
        if not source.is_dir():
            continue
        task = next((name for name in tasks if source.name.startswith(name + "_")), None)
        if task is None:
            raise ValueError(f"Không xác định được task của lượt: {source}")
        dataset = source.parent.parent.name
        target = source.parent.parent / task_folder(dataset, task) / source.parent.name / source.name
        moves.append((source, target))
    return moves


def apply_task_level(moves, root=ROOT):
    results = Path(root) / "results"
    manifest = results / "_system/migration_map.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"Thiếu bảng di chuyển trước đó: {manifest}")
    if len({target for _, target in moves}) != len(moves):
        raise ValueError("Hai lượt trỏ cùng một đích; chưa di chuyển gì")
    if any(not source.is_dir() or target.exists() for source, target in moves):
        raise ValueError("Nguồn thiếu hoặc đích đã tồn tại; chưa di chuyển gì")
    rows = json.loads(manifest.read_text(encoding="utf-8"))
    prefixes = [(source.relative_to(results), target.relative_to(results))
                for source, target in moves]
    for row in rows:
        current = Path(row["to"])
        matches = [(source, target) for source, target in prefixes
                   if current == source or current.is_relative_to(source)]
        if len(matches) > 1:
            raise ValueError(f"Bảng di chuyển có nhiều đích cho {current}")
        if matches:
            source, target = matches[0]
            row["to"] = str(target / current.relative_to(source))
    for source, target in moves:
        target.parent.mkdir(parents=True, exist_ok=True)
        source.rename(target)
    for parent in sorted({source.parent for source, _ in moves}, reverse=True):
        if parent.is_dir() and not any(parent.iterdir()):
            parent.rmdir()
    manifest.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def plan_model_level(root=ROOT):
    """Chèn model sau samples_N, giữ nguyên archive của lượt nhiều model."""
    results = Path(root) / "results"
    stages = stage_jobs(results)
    runs = sorted(source for source in results.glob("*/*/*/samples_*/*")
                  if source.is_dir() and any(item.is_file() for item in source.iterdir()))
    source_models = {}
    resolved = {}
    for source in runs:
        method = source.relative_to(results).parts[0]
        if method.startswith("analysis_"):
            continue
        match = re.search(r"[0-9a-f]{12}", source.name)
        job = stages.get(match.group() if match else "")
        model = infer_run_model(source, job=job, method=method)
        resolved[source] = model
        if match:
            source_models[match.group()] = model
    for source in runs:
        if source in resolved:
            continue
        method = source.relative_to(results).parts[0]
        resolved[source] = infer_run_model(source, source_models=source_models, method=method)
    return [(source, source.parent / resolved[source] / source.name) for source in runs]


def apply_model_level(moves, root=ROOT):
    results = Path(root) / "results"
    manifest = results / "_system/migration_map.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"Thiếu bảng di chuyển trước đó: {manifest}")
    if len({target for _, target in moves}) != len(moves):
        raise ValueError("Hai lượt trỏ cùng một đích; chưa di chuyển gì")
    if any(not source.is_dir() or target.exists() for source, target in moves):
        raise ValueError("Nguồn thiếu hoặc đích đã tồn tại; chưa di chuyển gì")
    rows = json.loads(manifest.read_text(encoding="utf-8"))
    prefixes = [(source.relative_to(results), target.relative_to(results))
                for source, target in moves]
    for row in rows:
        current = Path(row["to"])
        matches = [(source, target) for source, target in prefixes
                   if current == source or current.is_relative_to(source)]
        if len(matches) > 1:
            raise ValueError(f"Bảng di chuyển có nhiều đích cho {current}")
        if matches:
            source, target = matches[0]
            row["to"] = str(target / current.relative_to(source))
    for source, target in moves:
        target.parent.mkdir(parents=True, exist_ok=True)
        source.rename(target)
    manifest.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def plan_analysis_group(root=ROOT):
    """Gom các thư mục analysis_* lịch sử dưới một thư mục analysis chung."""
    results = Path(root) / "results"
    return [(source, results / "analysis" / source.name.removeprefix("analysis_"))
            for source in sorted(results.glob("analysis_*")) if source.is_dir()]


def apply_analysis_group(moves, root=ROOT):
    results = Path(root) / "results"
    manifest = results / "_system/migration_map.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"Thiếu bảng di chuyển trước đó: {manifest}")
    if len({target for _, target in moves}) != len(moves):
        raise ValueError("Hai nhóm phân tích trỏ cùng một đích; chưa di chuyển gì")
    if any(not source.is_dir() or target.exists() or source.parent != results
           or target.parent != results / "analysis" for source, target in moves):
        raise ValueError("Nguồn thiếu hoặc đích xung đột; chưa di chuyển gì")
    rows = json.loads(manifest.read_text(encoding="utf-8"))
    for source, target in moves:
        old, new = source.relative_to(results), target.relative_to(results)
        for row in rows:
            current = Path(row["to"])
            if current == old or current.is_relative_to(old):
                row["to"] = (new / current.relative_to(old)).as_posix()
        rows.append({"from": old.as_posix(), "to": new.as_posix()})
    for source, target in moves:
        target.parent.mkdir(parents=True, exist_ok=True)
        source.rename(target)
    manifest.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def plan_image_only_runs(root=ROOT):
    """Chỉ chuyển lượt upstream mà mọi dòng summary đều là original_image."""
    results = Path(root) / "results"
    moves = []
    for summary in sorted(results.glob("upstream_*/**/summary.tsv")):
        source = summary.parent
        parts = source.relative_to(results).parts
        if len(parts) != 6 or not parts[3].startswith("samples_"):
            continue
        rows = first_rows(summary)
        if rows and all(row.get("output_variant") == "original_image" for row in rows):
            target = results / ("image_only_" + parts[0]) / Path(*parts[1:])
            moves.append((source, target))
    return moves


def apply_image_only_runs(moves, root=ROOT):
    results = Path(root) / "results"
    manifest = results / "_system/migration_map.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"Thiếu bảng di chuyển trước đó: {manifest}")
    if len({target for _, target in moves}) != len(moves):
        raise ValueError("Hai lượt ảnh-only trỏ cùng một đích; chưa di chuyển gì")
    if any(not source.is_dir() or target.exists() or not source.is_relative_to(results)
           or not target.is_relative_to(results) for source, target in moves):
        raise ValueError("Nguồn thiếu hoặc đích xung đột; chưa di chuyển gì")
    rows = json.loads(manifest.read_text(encoding="utf-8"))
    for source, target in moves:
        old, new = source.relative_to(results), target.relative_to(results)
        for row in rows:
            current = Path(row["to"])
            if current == old or current.is_relative_to(old):
                row["to"] = (new / current.relative_to(old)).as_posix()
        rows.append({"from": old.as_posix(), "to": new.as_posix()})
    for source, target in moves:
        target.parent.mkdir(parents=True, exist_ok=True)
        source.rename(target)
        parent = source.parent
        while parent != results and parent.is_dir() and not any(parent.iterdir()):
            parent.rmdir()
            parent = parent.parent
    manifest.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Thực sự di chuyển sau khi kiểm tra xung đột")
    parser.add_argument("--insert-task-level", action="store_true",
                        help="Thêm cấp task cho cây kết quả đã sắp xếp trước đó")
    parser.add_argument("--insert-model-level", action="store_true",
                        help="Thêm cấp model cho cây kết quả đã có task")
    parser.add_argument("--group-analysis", action="store_true",
                        help="Gom analysis_* vào results/analysis/<policy>")
    parser.add_argument("--group-image-only", action="store_true",
                        help="Chuyển lượt upstream chỉ có original_image vào image_only_*")
    args = parser.parse_args()
    if sum((args.insert_task_level, args.insert_model_level, args.group_analysis,
            args.group_image_only)) > 1:
        parser.error("Chỉ chọn một bước chuyển cấu trúc")
    moves = (plan_image_only_runs() if args.group_image_only else
             plan_analysis_group() if args.group_analysis else plan_model_level() if args.insert_model_level
             else plan_task_level() if args.insert_task_level else plan())
    print(f"[PLAN] {len(moves)} mục sẽ chuyển")
    for source, target in moves:
        print(f"{source.relative_to(ROOT)} -> {target.relative_to(ROOT)}")
    if args.apply:
        manifest = (apply_image_only_runs(moves) if args.group_image_only else
                    apply_analysis_group(moves) if args.group_analysis else
                    apply_model_level(moves) if args.insert_model_level else
                    apply_task_level(moves) if args.insert_task_level else apply(moves))
        print(f"[DONE] {manifest.relative_to(ROOT)}")
    else:
        print("[DRY RUN] Chưa di chuyển; thêm --apply để thực hiện")


if __name__ == "__main__":
    main()
