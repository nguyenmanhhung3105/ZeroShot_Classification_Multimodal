"""Tương thích lệnh CrisisMMD cũ; dùng scripts/launch_runs.py cho job mới."""

import argparse
from pathlib import Path
import yaml

from launch_runs import ROOT, run_one, validate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launcher-config", default=str(ROOT / "configs/crisismmd_launcher.yaml"))
    parser.add_argument("--use-kaggle", choices=("true", "false"))
    parser.add_argument("--task", choices=("both", "informativeness", "humanitarian"))
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    legacy = yaml.safe_load(Path(args.launcher_config).read_text(encoding="utf-8"))
    config = yaml.safe_load((ROOT / "configs/launcher.yaml").read_text(encoding="utf-8"))
    config["use_kaggle"] = legacy["use_kaggle"] if args.use_kaggle is None else args.use_kaggle == "true"
    config["kaggle"].update(legacy.get("kaggle", {}))
    task = args.task or legacy.get("task", "both")
    limit = args.max_samples or legacy.get("max_samples", 30)
    names = (["crisismmd_info", "crisismmd_humanitarian"] if task == "both" else
             ["crisismmd_info" if task == "informativeness" else "crisismmd_humanitarian"])
    for name in names:
        config["jobs"][name]["max_samples"] = limit
        config["jobs"][name]["ensemble"]["enabled"] = True
    validate(config, names)
    for name in names:
        run_one(config, name, args.dry_run)


if __name__ == "__main__":
    main()
