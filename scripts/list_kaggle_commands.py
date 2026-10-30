"""In từng lệnh Kaggle riêng biệt từ launcher.yaml, không gửi job hay đọc dữ liệu."""

import argparse
from pathlib import Path
import shlex

import yaml

from launch_runs import ADAPTED_METHODS, UPSTREAM_BACKBONES


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "configs/launcher.yaml"
BASE = "venv/bin/python -B scripts/launch_runs.py"
KINDS = ("standard", "ensemble", "uncertainty", "adapted", "upstream", "demo", "relation_ablation")
ADAPTED_ORDER = ("baseline", "current_mean_sim", "class_name", "waffle", "dclip")
UPSTREAM_ORDER = ("clip", "waffle", "dclip_labels", "dclip")
BACKBONE_ORDER = ("RN50", "ViT-B/32", "ViT-L/14")


def commands(config, *, kind_filter=None, job_filter=None, launcher_path=None, configured_only=False):
    """Sinh đúng một lệnh cho mỗi cấu hình; không chạy launcher."""
    adapted = yaml.safe_load((ROOT / "configs/research/adapted.yaml").read_text(encoding="utf-8"))
    upstream = yaml.safe_load((ROOT / "configs/research/upstream.yaml").read_text(encoding="utf-8"))
    base = BASE + (f" --launcher-config {shlex.quote(str(launcher_path))}" if launcher_path else "")
    for name, job in config["jobs"].items():
        if job_filter and name != job_filter:
            continue
        job_kind = job["kind"]
        if job_kind == "standard":
            if job.get("result_method") in ("margin_majority", "prompt_margin_gate"):
                if not kind_filter or kind_filter == "uncertainty":
                    yield "uncertainty", name, f"{base} --job {name} --use-kaggle true"
                continue
            if not kind_filter or kind_filter == "standard":
                for model in job["models"]:
                    yield "standard", name, f"{base} --job {name} --model {model} --ensemble false --use-kaggle true"
            if (not kind_filter or kind_filter == "ensemble") and job.get("ensemble", {}).get("methods"):
                yield "ensemble", name, f"{base} --job {name} --ensemble true --use-kaggle true"
        elif job_kind == "adapted" and (not kind_filter or kind_filter == "adapted"):
            for model in job["models"]:
                methods = job["methods"] if configured_only or job.get("result_protocol") == "image_only" else ADAPTED_ORDER
                for method in methods:
                    if method not in ADAPTED_METHODS:
                        continue
                    seeds = job.get("seeds", adapted["seeds"]) if method == "waffle" else (None,)
                    for seed in seeds:
                        suffix = f" --seed {seed}" if seed is not None else ""
                        yield "adapted", name, (f"{base} --job {name} --model {model} "
                                                f"--method {method}{suffix} --use-kaggle true")
        elif job_kind == "upstream" and (not kind_filter or kind_filter == "upstream"):
            image_only = job.get("result_protocol") == "image_only"
            for backbone in (job.get("backbones", BACKBONE_ORDER) if image_only else BACKBONE_ORDER):
                if backbone not in UPSTREAM_BACKBONES:
                    continue
                for method in (job["methods"] if configured_only or image_only else UPSTREAM_ORDER):
                    for output in (job["outputs"] if image_only else ("original_image", "multimodal")):
                        seeds = job.get("seeds", upstream["seeds"]) if method == "waffle" else (None,)
                        for seed in seeds:
                            suffix = f" --seed {seed}" if seed is not None else ""
                            yield "upstream", name, (f"{base} --job {name} --backbone {backbone} "
                                                     f"--method {method} --output {output}{suffix} --use-kaggle true")
        elif job_kind in ("demo", "relation_ablation") and (not kind_filter or kind_filter == job_kind):
            for model in config["jobs"]["fakeddit_2way"]["models"]:
                yield job_kind, name, f"{base} --job {name} --model {model} --use-kaggle true"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=KINDS, help="Chỉ liệt kê một nhóm lệnh")
    parser.add_argument("--job", help="Chỉ liệt kê một job trong configs/launcher.yaml")
    parser.add_argument("--launcher-config", help="Đọc hồ sơ riêng và ghi đúng đường dẫn này trong từng lệnh")
    parser.add_argument("--configured-only", action="store_true", help="Chỉ in các phương pháp đã khai báo trong job")
    args = parser.parse_args(argv)
    config = yaml.safe_load(Path(args.launcher_config or LAUNCHER).read_text(encoding="utf-8"))
    if args.job and args.job not in config["jobs"]:
        parser.error(f"Không có job {args.job}")
    last = None
    count = 0
    for kind, name, command in commands(config, kind_filter=args.kind, job_filter=args.job,
                                       launcher_path=args.launcher_config, configured_only=args.configured_only):
        if (kind, name) != last:
            print(f"\n# {kind}: {name} (mặc định {config['jobs'][name]['max_samples']} mẫu)")
            last = kind, name
        print(command)
        count += 1
    if not count:
        parser.error("Bộ lọc không có lệnh nào")
    print(f"\n# Tổng cộng: {count} lệnh độc lập. Chỉ chạy một lệnh mỗi lần.")


if __name__ == "__main__":
    main()
