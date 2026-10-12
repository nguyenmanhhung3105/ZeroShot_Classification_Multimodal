"""Đọc cấu hình và kiểm tra lựa chọn trước khi tải model hoặc đọc dữ liệu."""
from pathlib import Path

import yaml
from pair_relation import relation_settings

DEFAULT_CONFIG = "configs/experiment_config.yaml"
TASKS = {"fakeddit": {"2way", "6way"}, "crisismmd": {"informativeness", "humanitarian"},
         "mmimdb": {"genres"}}


def load_config(path=DEFAULT_CONFIG):
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Cấu hình phải là một ánh xạ YAML")
    if not isinstance(config.get("models"), list) or not config["models"]:
        raise ValueError("models phải là danh sách model không rỗng")
    for dataset, settings in config.get("datasets", {}).items():
        if dataset not in TASKS or settings.get("task") not in TASKS[dataset]:
            raise ValueError(f"Dataset/task không hợp lệ: {dataset}/{settings.get('task')}")
        relation_settings(settings, dataset, config)
        expected = "multi_label" if dataset == "mmimdb" else "single_label"
        if settings.get("task_type", expected) != expected:
            raise ValueError(f"{dataset}: task_type phải là {expected}")
        limit = settings.get("max_samples")
        if limit is not None and (type(limit) is not int or limit < 1):
            raise ValueError(f"{dataset}: max_samples phải là số nguyên dương hoặc null")
        options = settings.get("inference", {})
        if "image_weight" in options and (isinstance(options["image_weight"], bool)
                or not isinstance(options["image_weight"], (int, float))
                or not 0 <= options["image_weight"] <= 1):
            raise ValueError(f"{dataset}: image_weight phải nằm trong [0, 1]")
    return config
