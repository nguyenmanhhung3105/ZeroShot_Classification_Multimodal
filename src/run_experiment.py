"""Chạy model × dataset bằng cùng runner với các lệnh chạy riêng.

Chạy từ thư mục gốc: python3 src/run_experiment.py --config configs/experiment_config.yaml
"""
import argparse
import os

import pandas as pd

from run import common as shared
from run.common import (load_config, load_dataframe, resolve_override, resolve_image_paths,
                        load_images_safe, validate_columns, parse_multilabel,
                        normalize_single_labels, get_image_weight)
from run.run_fakeddit import run_fakeddit
from run.run_crisismmd import run_crisismmd
from run.run_mmimdb import run_mmimdb

RUNNERS = {"fakeddit": run_fakeddit, "crisismmd": run_crisismmd, "mmimdb": run_mmimdb}
DATASET_RUNNERS = RUNNERS


def load_model(*args, **kwargs):
    from models.model_registry import load_model as load
    return load(*args, **kwargs)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment_config.yaml")
    args = parser.parse_args(argv if argv is not None else [])
    config = load_config(args.config)
    os.makedirs(config["output"]["results_dir"], exist_ok=True)
    rows = []
    for model_name in config["models"]:
        vlm = load_model(model_name, device=config.get("device"))
        try:
            for dataset, settings in config["datasets"].items():
                if settings.get("enabled", False):
                    rows.extend(shared.run_prompt_trials(
                        lambda active, name=dataset: RUNNERS[name](vlm, active, config),
                        model_name, dataset, settings))
        finally:
            vlm.unload()
            del vlm
    paths = shared.save_summary_tables(rows, config["output"]["summary_table"])
    print("Đã lưu summary:", ", ".join(paths))
    print(shared.compact_summary(pd.DataFrame(rows)).to_string(index=False))
    return rows


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
