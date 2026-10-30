"""Xác thực bốn gói mount trên Kaggle bằng CPU; không nạp model/GPU."""
import argparse
import json
from pathlib import Path
import re
import tempfile
import uuid

import yaml

import launch_runs as launcher

AUDIT = '''
def audit_only(settings):
    from datetime import datetime, timezone
    sources = []
    mounts = {}
    for slug, expected in settings["trees"].items():
        print("[CPU AUDIT] checking", slug, flush=True)
        mount = find_dataset("/kaggle/input", slug, expected)
        mounts[slug] = mount
        sources.append({"slug": slug, "tree_sha256": expected,
                        "files": sum(path.is_file() for path in mount.rglob("*"))})
        print("[CPU AUDIT] SHA verified", slug, flush=True)
    cohorts = []
    for spec in settings["cohorts"]:
        cohorts.append(verify_cohort(mounts[spec["slug"]], {"cohort": spec["cohort"]}))
    result = {"audit_id": settings["audit_id"], "checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "gpu_used": False, "models_loaded": False, "sources": sources, "cohorts": cohorts}
    output = Path("/kaggle/working/kaggle_data_check.json")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)

if __name__ == "__main__":
    audit_only(AUDIT_SETTINGS)
'''


def prepare(config, stage):
    stage = Path(stage)
    stage.mkdir(parents=True, exist_ok=True)
    kernel = stage / "kernel"
    kernel.mkdir()
    experiment = launcher.load_config(launcher.project_path(config["experiment_config"]))
    settings = {"audit_id": uuid.uuid4().hex, "trees": {}, "cohorts": []}
    kaggle = config["kaggle"]
    if kaggle.get("data_mode") != "full_dataset":
        raise ValueError("Audit cần data_mode: full_dataset")
    with tempfile.TemporaryDirectory(dir=stage) as directory:
        directory = Path(directory)
        code = directory / "code.zip"
        launcher.archive_code(code)
        settings["trees"][kaggle["code_dataset_slug"]] = launcher.archive_tree_sha256(code)
        for dataset, tasks in (("fakeddit", ("2way", "6way")),
                               ("crisismmd", ("informativeness", "humanitarian")), ("mmimdb", ("genres",))):
            task = experiment["datasets"][dataset]
            manifest_path = task.get("cohort_manifest")
            if not manifest_path:
                raise ValueError(f"{dataset}: thiếu cohort_manifest")
            manifest = json.loads(launcher.project_path(manifest_path).read_text())
            bundle = directory / f"{dataset}.zip"
            launcher.archive_full_data(bundle, experiment, dataset, 1)
            slug = f"{kaggle['data_dataset_slug']}-{dataset}"
            settings["trees"][slug] = launcher.archive_tree_sha256(bundle)
            for name in tasks:
                label = (task[f"label_col_{name}"] if dataset == "fakeddit" else
                         task["label_col_info" if name == "informativeness" else "label_col_human"]
                         if dataset == "crisismmd" else task["label_col"])
                spec = {"manifest_path": manifest_path, "cohort_id": manifest["cohort_id"],
                        "samples_path": task["data_path"], "samples": manifest["selected"]["samples"],
                        "id_col": task["id_col"], "label_col": label, "dataset": dataset,
                        "task": name, "limit": manifest["selected"]["samples"]}
                settings["cohorts"].append({"slug": slug, "cohort": spec})
    template = (launcher.ROOT / "scripts/kaggle_worker.py").read_text()
    (kernel / "worker.py").write_text(template + "\nAUDIT_SETTINGS = " + repr(settings) + "\n" + AUDIT)
    kernel_id = f"{kaggle['username']}/vlm-data-check-3000"
    metadata = {"id": kernel_id, "title": "VLM Data Check 3000", "code_file": "worker.py",
                "language": "python", "kernel_type": "script", "is_private": "true",
                "enable_gpu": "false", "enable_internet": "false",
                "dataset_sources": [f"{kaggle['username']}/{slug}" for slug in settings["trees"]],
                "competition_sources": [], "kernel_sources": [], "model_sources": []}
    (kernel / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))
    (stage / "expected.json").write_text(json.dumps(settings, indent=2))
    return settings, kernel_id


def validate_result(result, expected):
    trees = {entry["slug"]: entry["tree_sha256"] for entry in result["sources"]}
    if (result.get("audit_id") != expected["audit_id"] or trees != expected["trees"]
            or result.get("gpu_used") is not False or result.get("models_loaded") is not False):
        raise ValueError("Kết quả audit không khớp lượt/gói dữ liệu vừa yêu cầu")
    wanted = {(item["cohort"]["dataset"], item["cohort"]["task"]): item["cohort"] for item in expected["cohorts"]}
    # Cohort IDs for two tasks can be shared, so compare task and ID together.
    found = {(item["task"], item["cohort_id"]): item for item in result["cohorts"]}
    if len(found) != len(wanted):
        raise ValueError("Audit thiếu task")
    for (_, task), spec in wanted.items():
        actual = found.get((task, spec["cohort_id"]), {})
        if actual.get("cohort_samples") != spec["samples"] or actual.get("requested_samples") != spec["limit"]:
            raise ValueError("Audit sai số mẫu")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launcher-config", default="configs/launcher_3000.yaml")
    parser.add_argument("--submit", action="store_true", help="Gửi notebook CPU private và tải biên nhận; không dùng GPU")
    parser.add_argument("--resume-stage", help="Tiếp tục chờ/tải đúng lượt CPU đã gửi; không gửi notebook mới")
    args = parser.parse_args(argv)
    if args.submit and args.resume_stage:
        parser.error("Chỉ dùng --submit hoặc --resume-stage")
    config = yaml.safe_load(Path(args.launcher_config).read_text())
    root = launcher.launch_root() / "cohort_checks"
    root.mkdir(parents=True, exist_ok=True)
    if args.resume_stage:
        stage = Path(args.resume_stage).resolve()
        expected = json.loads((stage / "expected.json").read_text())
        metadata = json.loads((stage / "kernel/kernel-metadata.json").read_text())
        kernel_id = metadata["id"]
        version_number = int((stage / "kernel_version.txt").read_text())
    else:
        stage = Path(tempfile.mkdtemp(prefix="check_", dir=root))
        expected, kernel_id = prepare(config, stage)
    print(f"[CPU AUDIT] stage={stage}; kernel={kernel_id}", flush=True)
    if not args.submit and not args.resume_stage:
        print("Chỉ chuẩn bị; chưa gửi notebook.")
        return
    kaggle_bin = launcher.find_kaggle_cli()
    if not kaggle_bin:
        raise RuntimeError("Chưa cài Kaggle CLI")
    if not args.resume_stage:
        pushed = launcher.run_cli([kaggle_bin, "kernels", "push", "-p", str(stage / "kernel"), "--timeout", "3600"])
        version = re.search(r"Kernel version (\d+) successfully pushed", pushed.stdout + pushed.stderr, re.I)
        if not version:
            raise RuntimeError(f"Kaggle chưa xác nhận gửi CPU audit: {pushed.stdout or pushed.stderr}")
        version_number = int(version.group(1))
        (stage / "kernel_version.txt").write_text(str(version_number))
    print(f"[CPU AUDIT] version={version_number}", flush=True)
    launcher.await_kernel(kaggle_bin, f"{kernel_id}/{version_number}", 20, 1200)
    output = stage / "output"
    output.mkdir(exist_ok=True)
    launcher.run_cli([kaggle_bin, "kernels", "output", f"{kernel_id}/{version_number}", "-p", str(output), "--force"])
    result = json.loads((output / "kaggle_data_check.json").read_text())
    validate_result(result, expected)
    result.update(kernel=kernel_id, kernel_version=version_number, expected=expected)
    destination = launcher.ROOT / "docs/experiments/kaggle_3000_remote_check.json"
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"[VERIFIED] 4 gói và 5 task trên Kaggle CPU; {destination}", flush=True)


if __name__ == "__main__":
    main()
