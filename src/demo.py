"""Chạy một nhóm mẫu nhỏ và vẽ kết quả bằng cùng runner với dự án chính."""
import argparse
from copy import deepcopy
import math
import os

import pandas as pd

from run import common as shared
from run.common import (load_config, load_dataframe, resolve_override, resolve_image_paths,
                        load_images_safe, validate_columns, parse_multilabel,
                        normalize_single_labels, get_image_weight, ImagePathSequence)
from run.run_fakeddit import run_fakeddit
from run.run_crisismmd import run_crisismmd
from run.run_mmimdb import run_mmimdb

CONFIG_PATH = "configs/experiment_config.yaml"
N_SAMPLES = 20
BATCH_SIZE = 4
LOG_DIR = "results/demo_logs"
OUTPUT_DIR = "results/demo"
PROMPT_VERSION = "v1"
RUNNERS = {"fakeddit": run_fakeddit, "crisismmd": run_crisismmd, "mmimdb": run_mmimdb}


def load_model(*args, **kwargs):
    from models.model_registry import load_model as load
    return load(*args, **kwargs)


def visualize(df, images, text_col, true_col, dataset_name, task, model_name):
    import matplotlib.pyplot as plt
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    n_cols = min(5, len(df))
    n_rows = math.ceil(len(df) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 5 * n_rows), squeeze=False)
    axes = axes.flatten()
    for ax, (_, row), image in zip(axes, df.iterrows(), images):
        text = str(row[text_col])
        ax.imshow(image)
        if hasattr(images, "paths"):
            image.close()
        ax.axis("off")
        ax.set_title(f"{text[:60]}\nNhãn thật: {row[true_col]}\nDự đoán: {row['predicted']}", fontsize=9)
    for ax in axes[len(df):]:
        ax.axis("off")
    plt.suptitle(f"{dataset_name} — {task} — {model_name}")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"demo_{dataset_name}_{task}_{model_name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OUTPUT] {path}")


def run_dataset(vlm, model_name, dataset_name, task_config, image_weight, config=None):
    active = deepcopy(task_config)
    active["max_samples"] = min(N_SAMPLES, active.get("max_samples") or N_SAMPLES)
    active["_audit_scope"] = "demo"
    active["prompt_version"] = PROMPT_VERSION
    active["image_dir"] = active.get("image_dir") or os.path.join(os.path.dirname(active["data_path"]), "images")
    active.setdefault("inference", {})["image_weight"] = image_weight
    effective = deepcopy(config or {})
    effective.setdefault("datasets", {})[dataset_name] = active
    effective.setdefault("inference", {})["batch_size"] = BATCH_SIZE
    effective.setdefault("output", {})["raw_predictions_dir"] = LOG_DIR
    try:
        result = RUNNERS[dataset_name](vlm, active, effective)
    except RuntimeError as exc:
        if "không load được ảnh hợp lệ" in str(exc):
            return {"error": "no_valid_images"}
        raise
    # Dùng log vừa tạo để vẽ; không đọc lại dữ liệu nguồn hay suy luận lần hai.
    rows = pd.read_csv(os.path.join(result["prediction_log_dir"], "raw_predictions.tsv"), sep="\t")
    rows["predicted"] = rows["y_pred"]
    images = ImagePathSequence(rows["image_path"].tolist())
    visualize(rows, images, "text", "y_true", dataset_name, active.get("task", "genres"), model_name)
    return result


def main(argv=None):
    global N_SAMPLES, BATCH_SIZE, LOG_DIR, OUTPUT_DIR, PROMPT_VERSION
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=CONFIG_PATH)
    args = parser.parse_args(argv if argv is not None else [])
    config = load_config(args.config)
    settings = config.get("demo", {})
    N_SAMPLES = settings.get("n_samples", N_SAMPLES)
    BATCH_SIZE = settings.get("batch_size", BATCH_SIZE)
    LOG_DIR = settings.get("log_dir", LOG_DIR)
    OUTPUT_DIR = settings.get("visualization_dir", OUTPUT_DIR)
    PROMPT_VERSION = settings.get("prompt_version", PROMPT_VERSION)
    rows = []
    for model_name in config["models"]:
        vlm = load_model(model_name, device=config.get("device"))
        try:
            for dataset, task in config["datasets"].items():
                if task.get("enabled", False):
                    rows.extend(shared.run_prompt_trials(
                        lambda active, name=dataset: run_dataset(vlm, model_name, name, active,
                                  get_image_weight(config, name), config), model_name, dataset, task))
        finally:
            vlm.unload()
    if rows:
        shared.save_summary_tables(rows, os.path.join(OUTPUT_DIR, "demo_summary.tsv"))
    return rows


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
