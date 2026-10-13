"""Điều phối các job chuẩn/nghiên cứu trên máy hoặc Kaggle theo configs/launcher.yaml."""

import argparse
import ast
from copy import deepcopy
import csv
import hashlib
import io
from itertools import islice
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
import zipfile

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from config import TASKS, load_config
from models.model_registry import MODEL_CONFIGS

DEFAULT_CONFIG = ROOT / "configs/launcher.yaml"
SLUG = re.compile(r"^[a-z0-9][a-z0-9-]{2,48}[a-z0-9]$")
JOB_ID = re.compile(r"^[a-z][a-z0-9_]{1,48}$")
KINDS = {"standard", "demo", "adapted", "upstream", "relation_ablation"}
ADAPTED_METHODS = {"baseline", "current_mean_sim", "class_name", "waffle", "dclip"}
UPSTREAM_METHODS = {"clip", "waffle", "dclip_labels", "dclip"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def archive_tree_sha256(path):
    digest = hashlib.sha256()
    with zipfile.ZipFile(path) as archive:
        for name in sorted(info.filename for info in archive.infolist() if not info.is_dir()):
            digest.update(name.encode("utf-8") + b"\0")
            with archive.open(name) as stream:
                item = hashlib.sha256()
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    item.update(chunk)
            digest.update(item.digest())
    return digest.hexdigest()


def stable_info(name):
    info = zipfile.ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    return info


def add_file(archive, source, target):
    with Path(source).open("rb") as input_stream, archive.open(stable_info(target), "w") as output_stream:
        shutil.copyfileobj(input_stream, output_stream)


def project_path(value):
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def under_processed(value):
    path = project_path(value)
    if not path.is_relative_to((ROOT / "data/processed").resolve()):
        raise ValueError(f"Chỉ đóng gói đường dẫn trong data/processed/: {value}")
    return path


def valid_choice_list(value, allowed, field):
    if not isinstance(value, list) or not value or len(value) != len(set(value)) or any(
            not isinstance(item, str) or item not in allowed for item in value):
        raise ValueError(f"{field} phải là danh sách không lặp thuộc {sorted(allowed)}")


def validate(config, selected=None):
    if not isinstance(config, dict) or type(config.get("use_kaggle")) is not bool:
        raise ValueError("use_kaggle phải là true hoặc false")
    experiment = project_path(config.get("experiment_config", ""))
    if not experiment.is_file() or not experiment.is_relative_to((ROOT / "configs").resolve()):
        raise ValueError("experiment_config phải là file trong configs/")
    jobs = config.get("jobs")
    if not isinstance(jobs, dict) or not jobs:
        raise ValueError("jobs phải là ánh xạ không rỗng")
    names = selected if selected else [name for name, job in jobs.items() if job.get("enabled") is True]
    if not names:
        raise ValueError("Chưa bật job nào; dùng enabled: true hoặc --job")
    for name in names:
        if name not in jobs or not JOB_ID.fullmatch(name):
            raise ValueError(f"Job không hợp lệ: {name}")
        job = jobs[name]
        kind, dataset, task = job.get("kind"), job.get("dataset"), job.get("task")
        if kind not in KINDS or dataset not in TASKS or task not in TASKS[dataset]:
            raise ValueError(f"{name}: kind/dataset/task không hợp lệ")
        limit = job.get("max_samples")
        if type(limit) is not int or limit < 1 or limit > 100000:
            raise ValueError(f"{name}: max_samples phải trong [1, 100000]")
        scan = job.get("max_source_rows")
        if scan is not None and (type(scan) is not int or scan < limit):
            raise ValueError(f"{name}: max_source_rows phải >= max_samples")
        if kind == "relation_ablation" and (dataset, task) != ("fakeddit", "6way"):
            raise ValueError(f"{name}: relation_ablation chỉ hỗ trợ Fakeddit 6way")
        if kind == "relation_ablation" and limit > 5:
            raise ValueError(f"{name}: relation_ablation tối đa 5 mẫu")
        models = job.get("models")
        valid_choice_list(models, MODEL_CONFIGS, f"{name}.models")
        ensemble = job.get("ensemble", {"enabled": False})
        if not isinstance(ensemble, dict) or type(ensemble.get("enabled")) is not bool:
            raise ValueError(f"{name}.ensemble.enabled phải là true hoặc false")
        if ensemble["enabled"]:
            if kind != "standard" or len(models) < 2:
                raise ValueError(f"{name}: ensemble cần kind=standard và ít nhất 2 model")
            valid_choice_list(ensemble.get("methods"), {"equal_mean", "margin_gate"}, f"{name}.ensemble.methods")
            if dataset == "mmimdb" and "margin_gate" in ensemble["methods"]:
                raise ValueError(f"{name}: MM-IMDb không hỗ trợ margin_gate")
            margin = ensemble.get("margin_threshold", 0.1)
            if type(margin) not in (float, int) or not 0 <= margin <= 1:
                raise ValueError(f"{name}: margin_threshold phải trong [0, 1]")
        if kind in ("adapted", "upstream"):
            valid_choice_list(job.get("methods"), ADAPTED_METHODS if kind == "adapted" else UPSTREAM_METHODS,
                              f"{name}.methods")
        if kind in ("standard", "demo"):
            for field in ("prompt_variants", "prompt_aggregations"):
                if field in job and (not isinstance(job[field], list) or not job[field]):
                    raise ValueError(f"{name}.{field} phải là danh sách không rỗng")
                if ensemble["enabled"] and len(job.get(field, ["current"])) != 1:
                    raise ValueError(f"{name}: ensemble cần đúng một {field}")
    if config["use_kaggle"]:
        kaggle = config.get("kaggle", {})
        username = kaggle.get("username") or os.environ.get("KAGGLE_USERNAME", "")
        if not re.fullmatch(r"[a-zA-Z0-9_-]{2,40}", username):
            raise ValueError("Điền kaggle.username hoặc KAGGLE_USERNAME")
        for key in ("code_dataset_slug", "data_dataset_slug", "kernel_slug"):
            if not SLUG.fullmatch(kaggle.get(key, "")):
                raise ValueError(f"Slug Kaggle không hợp lệ: {key}")
        if kaggle.get("accelerator") != "NvidiaTeslaT4":
            raise ValueError("Launcher hiện hỗ trợ NvidiaTeslaT4")
        if type(kaggle.get("timeout_seconds")) is not int or not 300 <= kaggle["timeout_seconds"] <= 43200:
            raise ValueError("timeout_seconds phải trong [300, 43200]")
        if type(kaggle.get("poll_seconds")) is not int or not 5 <= kaggle["poll_seconds"] <= 300:
            raise ValueError("poll_seconds phải trong [5, 300]")
        kaggle["username"] = username
    return names


def job_plan(config, name, stage):
    job = deepcopy(config["jobs"][name])
    experiment = load_config(project_path(config["experiment_config"]))
    dataset, kind = job["dataset"], job["kind"]
    for settings in experiment["datasets"].values():
        settings["enabled"] = False
    task = experiment["datasets"][dataset]
    task.update(enabled=True, task=job["task"], max_samples=job["max_samples"])
    for key in ("prompt_variants", "prompt_aggregations"):
        if key in job:
            task[key] = job[key]
    experiment["models"] = job["models"]
    run_id = stage.name.split("_", 1)[0]
    family = "experiments" if job.get("ensemble", {}).get("enabled") or kind != "standard" else "runs"
    output = Path("results") / family / name / run_id
    if family == "runs":
        experiment["output"].update(results_dir=str(output), raw_predictions_dir=str(output / "raw_predictions"),
                                    summary_table=str(output / "summary_table.tsv"))
    else:
        experiment["output"]["raw_predictions_dir"] = str(output / "raw_predictions")
    if kind == "demo":
        experiment.setdefault("demo", {}).update(n_samples=job["max_samples"],
                                                  log_dir=str(output / "demo_logs"),
                                                  visualization_dir=str(output))
    active = stage / "active.yaml"
    active.write_text(yaml.safe_dump(experiment, sort_keys=False, allow_unicode=True), encoding="utf-8")
    job.update(id=name, output_dir=str(output))
    job_file = stage / "job.yaml"
    job_file.write_text(yaml.safe_dump(job, sort_keys=False, allow_unicode=True), encoding="utf-8")
    remote_config = "configs/active.yaml"
    research_file = None
    if job.get("ensemble", {}).get("enabled"):
        command = ["experiments/ensemble/run_ensemble.py", "--config", remote_config,
                   "--job-config", "configs/job.yaml"]
    elif kind == "standard":
        command = ["src/run_experiment.py", "--config", remote_config]
    elif kind == "demo":
        command = ["src/demo.py", "--config", remote_config]
    elif kind == "relation_ablation":
        command = ["src/research/run_relation_ablation.py", "--config", remote_config,
                   "--model", job["models"][0], "--max-samples", str(job["max_samples"]),
                   "--output-dir", str(output)]
    else:
        source = project_path(job.get("research_config", f"configs/research/{kind}.yaml"))
        if not source.is_file() or not source.is_relative_to((ROOT / "configs/research").resolve()):
            raise ValueError(f"{name}: research_config phải nằm trong configs/research/")
        research = yaml.safe_load(source.read_text(encoding="utf-8"))
        research["base_config"] = remote_config
        research["output_dir"] = str(output)
        research["methods"] = job["methods"]
        if kind == "adapted":
            research.update(dataset=dataset, task=job["task"], max_samples=job["max_samples"])
            descriptor = research.get("descriptor_file")
            if "dclip" in job["methods"]:
                suffix = "info" if (dataset, job["task"]) == ("crisismmd", "informativeness") else job["task"]
                research["descriptor_file"] = f"configs/research/descriptors/{dataset}_{suffix}.json"
            elif descriptor:
                research.pop("descriptor_file", None)
            entry = "src/research/run_adapted_methods.py"
        else:
            for ds, options in research["datasets"].items():
                options["enabled"] = ds == dataset
                if ds == dataset:
                    options.update(task=job["task"], max_samples=job["max_samples"])
            entry = "src/research/run_upstream.py"
        research_file = stage / "research.yaml"
        research_file.write_text(yaml.safe_dump(research, sort_keys=False, allow_unicode=True), encoding="utf-8")
        command = [entry, "--config", "configs/research/active.yaml"]
    return experiment, job, output, command, active, job_file, research_file


def selected_rows(source, task, limit, scan_limit):
    """Đọc tuần tự, dừng ngay khi đủ mẫu hợp lệ hoặc chạm trần scan."""
    minimum = task.get("min_text_length", 0)
    text_col = task.get("text_col", "clean_title")
    rows, scanned, valid = [], 0, 0
    if source.suffix in (".tsv", ".csv"):
        with source.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream, delimiter="\t" if source.suffix == ".tsv" else ",")
            fields = reader.fieldnames or []
            for row in reader:
                scanned += 1
                rows.append(row)
                if not minimum or len(row.get(text_col, "").replace(" ", "")) > minimum:
                    valid += 1
                    if valid >= limit:
                        break
                if scanned >= scan_limit:
                    raise ValueError("Chạm max_source_rows trước khi chọn đủ mẫu; tăng giới hạn trong YAML")
        return fields, rows
    if source.suffix == ".parquet":
        import pyarrow.parquet as pq
        parquet = pq.ParquetFile(source)
        fields = parquet.schema_arrow.names
        valid = 0
        for batch in parquet.iter_batches(batch_size=min(512, scan_limit)):
            for row in batch.to_pylist():
                scanned += 1
                rows.append(row)
                if not minimum or len(str(row.get(text_col, "")).replace(" ", "")) > minimum:
                    valid += 1
                    if valid >= limit:
                        break
                if scanned >= scan_limit:
                    raise ValueError("Chạm max_source_rows trước khi chọn đủ mẫu")
            if valid >= limit:
                break
        return fields, rows
    raise ValueError(f"Định dạng dữ liệu chưa hỗ trợ: {source.suffix}")


def image_for_row(row, task):
    image_root = under_processed(task["image_dir"])
    raw = str(row.get(task.get("image_col", "image_path"), "")).strip()
    sample_id = str(row.get(task.get("id_col", "id"), "")).strip()
    candidates = []
    if raw:
        candidates += [project_path(raw), image_root / raw, image_root / Path(raw).name]
    if sample_id:
        candidates += [image_root / sample_id]
        candidates += [image_root / f"{sample_id}{ext}" for ext in
                       (task.get("image_ext", ".jpg"), ".jpeg", ".jpg", ".JPEG", ".JPG")]
    for candidate in candidates:
        path = candidate.resolve()
        if path.is_relative_to(image_root) and path.is_file():
            return path
    return None


def archive_data(destination, experiment, dataset, limit, scan_limit=None):
    task = experiment["datasets"][dataset]
    source = under_processed(task["data_path"])
    image_root = under_processed(task["image_dir"])
    if not source.is_file() or not image_root.is_dir():
        raise FileNotFoundError(f"Thiếu dữ liệu đã xử lý của {dataset}")
    fields, rows = selected_rows(source, task, limit, scan_limit or max(limit * 50, 100))
    if not rows:
        raise ValueError(f"{dataset}: file dữ liệu rỗng")
    root = ROOT.resolve()
    images = {path.relative_to(root).as_posix(): path for row in rows
              if (path := image_for_row(row, task)) is not None}
    if not images:
        raise ValueError(f"{dataset}: không tìm được ảnh trong các dòng đã chọn")
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        if source.suffix in (".tsv", ".csv"):
            text = io.StringIO(newline="")
            writer = csv.DictWriter(text, fieldnames=fields,
                                    delimiter="\t" if source.suffix == ".tsv" else ",",
                                    lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            archive.writestr(stable_info(source.relative_to(root).as_posix()), text.getvalue())
        else:
            import pyarrow as pa
            import pyarrow.parquet as pq
            with tempfile.TemporaryDirectory() as temp:
                bounded = Path(temp) / source.name
                table = pa.Table.from_pylist(rows, schema=pq.ParquetFile(source).schema_arrow)
                pq.write_table(table, bounded)
                add_file(archive, bounded, source.relative_to(root).as_posix())
        for relative, path in sorted(images.items()):
            add_file(archive, path, relative)
    valid_rows = sum(not task.get("min_text_length", 0) or
                     len(str(row.get(task.get("text_col", "clean_title"), "")).replace(" ", "")) >
                     task.get("min_text_length", 0) for row in rows)
    return sha256(destination), valid_rows, len(images), len(rows)


def archive_code(destination):
    files = {path: path.relative_to(ROOT).as_posix() for base in ("src", "experiments")
             for path in (ROOT / base).rglob("*.py") if "__pycache__" not in path.parts}
    for directory in ("configs/research/descriptors", "configs/prompts"):
        for path in (ROOT / directory).glob("*.json"):
            files[path] = path.relative_to(ROOT).as_posix()
    for relative in ("configs/research/sources.json", "requirements.txt", "requirements/research.txt"):
        files[ROOT / relative] = relative
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source, target in sorted(files.items(), key=lambda item: item[1]):
            add_file(archive, source, target)
    return sha256(destination)


def metadata(path, dataset_id):
    title = dataset_id.split("/", 1)[1].replace("-", " ").title()
    path.write_text(json.dumps({"title": title, "id": dataset_id,
                                "licenses": [{"name": "unknown"}]}, indent=2), encoding="utf-8")


def run_cli(args, *, check=True):
    result = subprocess.run(args, cwd=ROOT, text=True, capture_output=True)
    if check and result.returncode:
        raise RuntimeError(f"Lệnh Kaggle thất bại: {' '.join(args[:3])}\n{result.stderr or result.stdout}")
    return result


def find_kaggle_cli():
    candidate = Path(sys.executable).with_name("kaggle")
    return str(candidate) if candidate.is_file() else shutil.which("kaggle")


def sync_dataset(staged_dir, name, digest, state, kaggle_bin):
    if state.get(name) == digest:
        print(f"[KAGGLE] Giữ dataset {name}: SHA-256 không đổi")
        return
    source = next(staged_dir.glob("*.zip"))
    expected = archive_tree_sha256(source)

    listing = run_cli([kaggle_bin, "datasets", "list", "-m", "-s",
                       name.split("/", 1)[1], "--page-size", "200", "--format", "json"])
    exists = re.search(r'"ref"\s*:\s*"' + re.escape(name) + r'"', listing.stdout) is not None

    def remote_tree():
        with tempfile.TemporaryDirectory(prefix="kaggle_verify_") as directory:
            result = run_cli([kaggle_bin, "datasets", "download", name,
                              "-p", directory, "-q", "-o"], check=False)
            if result.returncode:
                message = result.stderr or result.stdout
                if re.search(r"\b(404|not found|does not exist)\b", message, re.I):
                    return None
                raise RuntimeError(f"Không kiểm tra được dataset {name}: {message}")
            downloaded = Path(directory) / f"{name.split('/', 1)[1]}.zip"
            if not downloaded.is_file():
                raise RuntimeError(f"Kaggle không trả ZIP dataset {name}")
            return archive_tree_sha256(downloaded)

    current = remote_tree() if exists else None
    if current != expected:
        if current is None:
            args = [kaggle_bin, "datasets", "create", "-p", str(staged_dir), "-t"]
        else:
            args = [kaggle_bin, "datasets", "version", "-p", str(staged_dir),
                    "-m", f"Update bounded VLM bundle {digest[:12]}", "-t"]
        run_cli(args)
    deadline = time.monotonic() + 300
    while remote_tree() != expected:
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Dataset {name} chưa có nội dung đúng sau 5 phút")
        time.sleep(10)
    state[name] = digest
    print(f"[KAGGLE] Dataset {name} đã xác minh nội dung", flush=True)


def make_kernel(staged_dir, settings, kaggle):
    template = (ROOT / "scripts/kaggle_worker.py").read_text(encoding="utf-8")
    (staged_dir / "worker.py").write_text(
        template + "\nSETTINGS = " + repr(settings) +
        "\nif __name__ == '__main__':\n    run(SETTINGS)\n", encoding="utf-8")
    kernel_id = f"{kaggle['username']}/{kaggle['kernel_slug']}"
    (staged_dir / "kernel-metadata.json").write_text(json.dumps({
        "id": kernel_id, "title": kaggle["kernel_slug"].replace("-", " ").title(),
        "code_file": "worker.py", "language": "python", "kernel_type": "script",
        "is_private": "true", "enable_gpu": "true", "enable_internet": "true",
        "machine_shape": kaggle["accelerator"],
        "dataset_sources": [f"{kaggle['username']}/{kaggle['code_dataset_slug']}",
                            f"{kaggle['username']}/{kaggle['data_dataset_slug']}"],
        "competition_sources": [], "kernel_sources": [], "model_sources": []
    }, indent=2), encoding="utf-8")
    return kernel_id


def await_kernel(kaggle_bin, kernel_id, interval, timeout):
    deadline = time.monotonic() + timeout + 300
    access_failures = 0
    while time.monotonic() < deadline:
        result = run_cli([kaggle_bin, "kernels", "status", kernel_id], check=False)
        if result.returncode:
            message = result.stderr or result.stdout
            if re.search(r"Authentication required|kernels\.get.*denied", message, re.I):
                access_failures += 1
                if access_failures < 3:
                    print(f"[KAGGLE] Mất quyền xem kernel tạm thời ({access_failures}/3); thử lại", flush=True)
                    time.sleep(interval)
                    continue
                raise RuntimeError("Kaggle CLI không xem được kernel. "
                                   "Chạy `kaggle auth login --force` rồi tiếp tục lượt đã gửi. "
                                   "Nếu vẫn bị từ chối, kiểm tra quyền truy cập notebook trên Kaggle; "
                                   f"phản hồi CLI: {message.strip()}")
            raise RuntimeError(f"Không đọc được trạng thái Kaggle: {message.strip()}")
        access_failures = 0
        status = (result.stdout + result.stderr).lower()
        print("[KAGGLE]", status.strip(), flush=True)
        if re.search(r"\b(complete|completed)\b", status):
            return
        if re.search(r"\b(error|failed|cancelled|canceled)\b", status):
            logs = run_cli([kaggle_bin, "kernels", "logs", kernel_id], check=False)
            raw = logs.stdout or logs.stderr
            try:
                raw = "".join(item.get("data", "") for item in json.loads(raw))
            except (ValueError, TypeError):
                pass
            raise RuntimeError(f"Kaggle Kernel lỗi:\n{raw[-5000:]}")
        time.sleep(interval)
    raise TimeoutError(f"Hết thời gian chờ Kaggle Kernel {kernel_id}")


def staged_run(stage):
    """Đọc metadata của lượt đã gửi, không thực thi worker.py hay tạo job mới."""
    stage = Path(stage).resolve()
    staging_root = (ROOT / "results/launches/staging").resolve()
    if not stage.is_dir() or not stage.is_relative_to(staging_root):
        raise ValueError("--resume-stage phải là thư mục lượt chạy trong results/launches/staging/")
    worker = stage / "kernel/worker.py"
    metadata_file = stage / "kernel/kernel-metadata.json"
    if not worker.is_file() or not metadata_file.is_file():
        raise ValueError("Thư mục staging thiếu worker.py hoặc kernel-metadata.json")
    assignments = [node for node in ast.parse(worker.read_text(encoding="utf-8")).body
                   if isinstance(node, ast.Assign) and any(
                       isinstance(target, ast.Name) and target.id == "SETTINGS" for target in node.targets)]
    if len(assignments) != 1:
        raise ValueError("Không đọc được SETTINGS của lượt chạy")
    settings = ast.literal_eval(assignments[0].value)
    metadata_value = json.loads(metadata_file.read_text(encoding="utf-8"))
    if not isinstance(settings, dict) or not isinstance(settings.get("output_dir"), str):
        raise ValueError("SETTINGS của lượt chạy không hợp lệ")
    run_id, job_id = settings.get("run_id"), settings.get("job_id")
    output_relative = Path(settings["output_dir"])
    output = (ROOT / output_relative).resolve()
    if (not isinstance(run_id, str) or not re.fullmatch(r"[0-9a-f]{12}", run_id)
            or not isinstance(job_id, str) or not JOB_ID.fullmatch(job_id)
            or not stage.name.startswith(f"{run_id}_{job_id}_")
            or output_relative.is_absolute() or not output.is_relative_to((ROOT / "results").resolve())
            or output.name != run_id or output.parent.name != job_id
            or not all(isinstance(settings.get(key), str) and re.fullmatch(r"[0-9a-f]{64}", settings[key])
                       for key in ("code_sha256", "data_sha256"))):
        raise ValueError("Metadata staging không hợp lệ")
    kernel_id = metadata_value.get("id")
    if not isinstance(kernel_id, str) or not re.fullmatch(r"[a-zA-Z0-9_-]+/[a-z0-9-]+", kernel_id):
        raise ValueError("ID kernel trong staging không hợp lệ")
    return settings, kernel_id, output


def verify_result(directory, settings):
    receipt_path = directory / "run_receipt.json"
    if not receipt_path.is_file() or not (directory / "summary.tsv").is_file():
        raise RuntimeError("Kaggle không trả đủ summary.tsv và run_receipt.json")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if any(receipt.get(key) != settings[key] for key in ("run_id", "job_id", "code_sha256", "data_sha256")):
        raise RuntimeError("Kết quả tải về không khớp lượt chạy vừa gửi")


def collect_result(kaggle_bin, kernel_id, settings, output):
    if output.exists():
        if not output.is_dir():
            raise RuntimeError(f"Đường dẫn kết quả đã tồn tại và không phải thư mục: {output}")
        verify_result(output, settings)
        print(f"[RESULT] {output / 'summary.tsv'}", flush=True)
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}_", dir=output.parent))
    try:
        run_cli([kaggle_bin, "kernels", "output", kernel_id, "-p", str(temporary)])
        verify_result(temporary, settings)
        temporary.rename(output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    print(f"[RESULT] {output / 'summary.tsv'}", flush=True)
    return output


def resume_run(stage, config):
    settings, kernel_id, output = staged_run(stage)
    kaggle_bin = find_kaggle_cli()
    if kaggle_bin is None:
        raise RuntimeError("Chưa cài Kaggle CLI")
    await_kernel(kaggle_bin, kernel_id, config["kaggle"]["poll_seconds"],
                 config["kaggle"]["timeout_seconds"])
    return collect_result(kaggle_bin, kernel_id, settings, output)


def run_one(config, name, dry_run=False):
    staging = ROOT / "results/launches/staging"
    staging.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex[:12]
    stage = Path(tempfile.mkdtemp(prefix=f"{run_id}_{name}_", dir=staging))
    experiment, job, output, command, active, job_file, research_file = job_plan(config, name, stage)
    local = [sys.executable, "-B", command[0], *command[1:]]
    local = [str(active) if arg == "configs/active.yaml" else
             str(job_file) if arg == "configs/job.yaml" else
             str(research_file) if arg == "configs/research/active.yaml" else arg for arg in local]
    if not config["use_kaggle"]:
        print("[LOCAL]", " ".join(local), flush=True)
        if not dry_run:
            subprocess.run(local, cwd=ROOT, check=True)
        return output

    code_dir, data_dir, kernel_dir = (stage / part for part in ("code", "data", "kernel"))
    for directory in (code_dir, data_dir, kernel_dir):
        directory.mkdir()
    code_sha = archive_code(code_dir / "code_bundle.zip")
    data_sha, selected, images, scanned = archive_data(
        data_dir / "data_bundle.zip", experiment, job["dataset"],
        job["max_samples"], job.get("max_source_rows"))
    kaggle = config["kaggle"]
    username = kaggle["username"]
    code_id, data_id = (f"{username}/{kaggle[key]}" for key in ("code_dataset_slug", "data_dataset_slug"))
    metadata(code_dir / "dataset-metadata.json", code_id)
    metadata(data_dir / "dataset-metadata.json", data_id)
    settings = {"run_id": run_id, "job_id": name, "kind": job["kind"],
                "output_dir": str(output), "command": command,
                "active_yaml": active.read_text(encoding="utf-8"),
                "job_yaml": job_file.read_text(encoding="utf-8"),
                "research_yaml": research_file.read_text(encoding="utf-8") if research_file else None,
                "code_dataset_slug": kaggle["code_dataset_slug"],
                "data_dataset_slug": kaggle["data_dataset_slug"],
                "code_sha256": code_sha, "data_sha256": data_sha,
                "code_tree_sha256": archive_tree_sha256(code_dir / "code_bundle.zip"),
                "data_tree_sha256": archive_tree_sha256(data_dir / "data_bundle.zip")}
    kernel_id = make_kernel(kernel_dir, settings, kaggle)
    print(f"[PLAN] {name}: {selected} mẫu từ {scanned} dòng, {images} ảnh; "
          f"stage={stage}; kernel={kernel_id}", flush=True)
    if dry_run:
        print("[DRY RUN] Chưa gọi Kaggle CLI và chưa chạy model")
        return output
    kaggle_bin = find_kaggle_cli()
    if kaggle_bin is None:
        raise RuntimeError("Chưa cài Kaggle CLI: pip install kaggle; sau đó kaggle auth login")
    quota = run_cli([kaggle_bin, "quota"])
    print(f"[KAGGLE QUOTA]\n{quota.stdout.strip()}", flush=True)
    state_path = ROOT / "results/launches/dataset_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
    for directory, dataset_id, digest in ((code_dir, code_id, code_sha), (data_dir, data_id, data_sha)):
        sync_dataset(directory, dataset_id, digest, state, kaggle_bin)
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    pushed = run_cli([kaggle_bin, "kernels", "push", "-p", str(kernel_dir),
                      "--accelerator", kaggle["accelerator"], "--timeout", str(kaggle["timeout_seconds"])])
    marker = re.search(r"Kernel version (\d+) successfully pushed", pushed.stdout + pushed.stderr, re.I)
    if not marker:
        raise RuntimeError("Kaggle CLI chưa xác nhận đã gửi kernel; không theo dõi lượt chạy cũ. "
                           f"Phản hồi: {(pushed.stderr or pushed.stdout).strip()}\n"
                           f"Kiểm tra Kaggle trước khi chạy lại. Staging: {stage}")
    (stage / "kernel_version.txt").write_text(marker.group(1), encoding="utf-8")
    print(f"[KAGGLE] Kernel version {marker.group(1)} đã gửi", flush=True)
    try:
        await_kernel(kaggle_bin, kernel_id, kaggle["poll_seconds"], kaggle["timeout_seconds"])
        return collect_result(kaggle_bin, kernel_id, settings, output)
    except (RuntimeError, TimeoutError) as exc:
        raise RuntimeError(f"{exc}\nLượt Kaggle đã được gửi; đừng chạy lại job. "
                           f"Sau khi khôi phục quyền truy cập, dùng: "
                           f"{sys.executable} -B scripts/launch_runs.py --resume-stage {stage}") from exc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launcher-config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--job", action="append", help="Chọn job theo ID; lặp flag để chọn nhiều job")
    parser.add_argument("--use-kaggle", choices=("true", "false"))
    parser.add_argument("--ensemble", choices=("true", "false"), help="Ghi đè ensemble.enabled của job đã chọn")
    parser.add_argument("--max-samples", type=int, help="Ghi đè giới hạn mẫu của job đã chọn")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume-stage", help="Tiếp tục chờ/tải kết quả lượt Kaggle đã gửi; không chạy GPU mới")
    args = parser.parse_args(argv)
    config = yaml.safe_load(Path(args.launcher_config).read_text(encoding="utf-8"))
    if args.resume_stage:
        if args.job or args.use_kaggle is not None or args.ensemble is not None or args.max_samples is not None or args.dry_run:
            parser.error("--resume-stage không dùng cùng các tùy chọn tạo job")
        resume_run(args.resume_stage, config)
        return
    # YAML anchor dùng để giảm lặp trong file; mỗi job vẫn phải có cấu hình độc lập.
    config["jobs"] = {name: deepcopy(job) for name, job in config["jobs"].items()}
    if args.use_kaggle is not None:
        config["use_kaggle"] = args.use_kaggle == "true"
    names = args.job or [name for name, job in config["jobs"].items() if job.get("enabled") is True]
    for name in names:
        if name not in config["jobs"]:
            parser.error(f"Không có job {name}")
        job = config["jobs"][name]
        if args.max_samples is not None:
            job["max_samples"] = args.max_samples
        if args.ensemble is not None:
            job.setdefault("ensemble", {})["enabled"] = args.ensemble == "true"
    validate(config, names)
    for name in names:
        run_one(config, name, args.dry_run)


if __name__ == "__main__":
    main()
