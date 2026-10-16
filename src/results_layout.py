"""Đường dẫn thống nhất: phương pháp / dataset / task / samples_N / model / lượt."""

import re
from pathlib import Path
from copy import deepcopy
import uuid


def _slug(value):
    value = str(value).lower().replace("-", "_")
    if not re.fullmatch(r"[a-z0-9_]+", value):
        raise ValueError(f"Tên thư mục kết quả không hợp lệ: {value!r}")
    return value


def sample_folder(samples):
    if samples is None:
        return "samples_unknown"
    samples = int(samples)
    if samples < 1:
        raise ValueError("Số mẫu phải dương")
    return f"samples_{samples}"


def dataset_folder(dataset, task):
    return _slug(dataset)


def task_folder(dataset, task):
    """Tên task hiện hành; multi_label là tên lịch sử của MM-IMDb genres."""
    if _slug(dataset) == "mmimdb" and _slug(task) == "multi_label":
        return "genres"
    return _slug(task)


def model_folder(models=None, backbones=None):
    """Một model giữ tên riêng; một lượt dùng nhiều model giữ nguyên dưới multi_model."""
    if backbones is not None:
        names = ["openai_clip_" + _slug(name.replace("/", "_")) for name in backbones]
    else:
        names = [_slug(name) for name in (models or [])]
    if not names:
        raise ValueError("Cần khai báo model/backbone để đặt thư mục kết quả")
    return names[0] if len(set(names)) == 1 else "multi_model"


def method_for_job(job):
    """Phương pháp là biến thể thuật toán; model và seed vẫn ở trong lượt chạy."""
    if job.get("result_method"):
        return _slug(job["result_method"])
    if job.get("result_protocol") == "image_only":
        base = {**job}
        base.pop("result_protocol")
        return "image_only_" + method_for_job(base)
    kind = job["kind"]
    if job.get("ensemble", {}).get("enabled"):
        return "ensemble"
    if kind == "standard":
        relation = job.get("image_text_relation") or {}
        if relation.get("enabled") and relation.get("mode") not in (None, "auto", "diagnostic"):
            return _slug(relation["mode"])
        return "baseline"
    if kind in ("adapted", "upstream"):
        methods = job.get("methods") or []
        if len(methods) != 1:
            return f"{kind}_suite"
        base = f"{kind}_{_slug(methods[0])}"
        relation = job.get("image_text_relation") or {}
        if relation.get("enabled"):
            mode = relation.get("mode", "diagnostic")
            if mode == "auto":
                mode = "class_aware" if job["dataset"] == "fakeddit" and job["task"] == "6way" else "diagnostic"
            if mode != "diagnostic":
                base += "_" + _slug(mode)
        return base
    if kind in ("demo", "relation_ablation"):
        return kind
    raise ValueError(f"Loại job không hợp lệ: {kind}")


def run_path(method, dataset, task, samples, model, run_id):
    return (Path("results") / _slug(method) / dataset_folder(dataset, task)
            / task_folder(dataset, task) / sample_folder(samples) / _slug(model)
            / f"{_slug(task)}_{_slug(run_id)}")


def job_path(job, run_id):
    model = (model_folder(backbones=job.get("backbones")) if job["kind"] == "upstream"
             else model_folder(models=job.get("models")))
    return run_path(method_for_job(job), job["dataset"], job["task"], job["max_samples"], model, run_id)


def standalone_method(config, dataset):
    settings = config["datasets"][dataset]
    relation = {**(config.get("inference", {}).get("image_text_relation") or {}),
                **(settings.get("image_text_relation") or {})}
    if not relation.get("enabled"):
        return "baseline"
    mode = relation.get("mode", "diagnostic")
    if mode == "auto":
        mode = "class_aware" if dataset == "fakeddit" and settings.get("task") == "6way" else "diagnostic"
    return "baseline" if mode == "diagnostic" else _slug(mode)


def standalone_config(config, dataset, run_id=None):
    """Đổi output của config gốc sang cây mới; config launcher đã đóng băng giữ nguyên."""
    output = config["output"]
    if Path(output["results_dir"]) not in (Path("results"), Path("auto")):
        return config
    active = deepcopy(config)
    task = active["datasets"][dataset]
    path = run_path(standalone_method(active, dataset), dataset, task["task"], task["max_samples"],
                    model_folder(models=active["models"]), run_id or uuid.uuid4().hex[:12])
    active["output"].update(results_dir=str(path), raw_predictions_dir=str(path / "raw_predictions"),
                            summary_table=str(path / "summary_table.tsv"))
    return active


def analysis_path(policy, source_run, result_name, model=None):
    """Gom phân tích dưới results/analysis, kế thừa dataset/task/số mẫu/model."""
    run = Path(source_run)
    inherited_model = run.parent.name
    sample = run.parent.parent.name if run.parent.parent.name.startswith("samples_") else "samples_unknown"
    if sample == "samples_unknown":
        dataset, task = "mixed", "dataset"
    else:
        task = run.parent.parent.parent.name
        dataset = run.parent.parent.parent.parent.name
    return (Path("results") / "analysis" / _slug(policy) / dataset / task
            / sample / _slug(model or inherited_model) / _slug(result_name))
