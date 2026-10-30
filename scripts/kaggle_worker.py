"""Worker Kaggle cho các job trong launcher; chỉ ghi kết quả vào /kaggle/working."""

import hashlib
import csv
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tempfile
import zipfile


def unpack_archive(archive, destination):
    """Tiện ích kiểm tra ZIP cho test/local; Kaggle tự giải nén dataset upload."""
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as stream:
        for item in stream.infolist():
            name = PurePosixPath(item.filename)
            if name.is_absolute() or ".." in name.parts or not item.filename:
                raise ValueError(f"Đường dẫn không hợp lệ trong archive: {item.filename}")
            target = destination.joinpath(*name.parts)
            if not target.resolve().is_relative_to(destination):
                raise ValueError(f"Đường dẫn thoát archive: {item.filename}")
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with stream.open(item) as source, target.open("wb") as sink:
                    shutil.copyfileobj(source, sink)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(directory):
    digest = hashlib.sha256()
    files = (path for path in Path(directory).rglob("*") if path.is_file())
    for path in sorted(files, key=lambda item: item.relative_to(directory).as_posix()):
        digest.update(path.relative_to(directory).as_posix().encode("utf-8") + b"\0")
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def find_dataset(root, slug, expected):
    root = Path(root)
    # Kaggle mounts use either /input/<slug> or /input/datasets/<owner>/<slug>.
    # Avoid recursively scanning thousands of images just to locate that root.
    candidates = [root / slug, *root.glob(f"*/{slug}"), *root.glob(f"datasets/*/{slug}")]
    candidates = list(dict.fromkeys(path for path in candidates if path.is_dir()))
    if not candidates:
        candidates = [path for path in root.rglob(slug) if path.is_dir()]
    for source in candidates:
        if tree_sha256(source) == expected:
            return source
    visible = sorted(path.relative_to(root).as_posix() for path in root.iterdir())
    details = [{"path": str(path), "actual": tree_sha256(path),
                "count": sum(item.is_file() for item in path.rglob("*")),
                "first_files": sorted(item.relative_to(path).as_posix() for item in path.rglob("*")
                                      if item.is_file())[:12]} for path in candidates]
    raise ValueError(f"Dataset {slug} thiếu hoặc sai SHA-256; "
                     f"expected={expected}; candidates={details}; input={visible}")


def clone_sources(root):
    sources = json.loads((root / "configs/research/sources.json").read_text(encoding="utf-8"))
    for name, spec in sources.items():
        url, commit = spec["url"], spec["commit"]
        if not url.startswith("https://github.com/") or len(commit) != 40:
            raise ValueError(f"Nguồn nghiên cứu không hợp lệ: {name}")
        path = root / "third_party" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-q", "--filter=blob:none", url, str(path)], check=True)
        subprocess.run(["git", "-C", str(path), "checkout", "-q", commit], check=True)


def verify_cohort(root, settings):
    """Check the mounted cohort and task-specific y_true before loading models."""
    spec = settings.get("cohort")
    if not spec:
        return None
    root = Path(root).resolve()
    def project_file(value):
        path = (root / value).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Thiếu file cohort trong gói: {value}")
        return path
    manifest = json.loads(project_file(spec["manifest_path"]).read_text(encoding="utf-8"))
    samples = project_file(spec["samples_path"])
    if (manifest.get("status") != "ready" or manifest.get("cohort_id") != spec["cohort_id"]
            or manifest.get("dataset") != spec["dataset"] or sha256(samples) != manifest["samples_sha256"]):
        raise ValueError("Cohort mount không khớp ID/checksum TSV.")
    if spec.get("final_protocol_sha256") and manifest.get("final_protocol_sha256") != spec["final_protocol_sha256"]:
        raise ValueError("Cohort test khác protocol DN-unit đã khóa.")
    if spec.get("final_protocol_sha256") and spec["limit"] != spec["samples"]:
        raise ValueError("Final test phải chạy đủ cohort đã khóa, không chọn lại số mẫu.")
    entries = {entry["id"]: entry for entry in manifest["samples"]}
    seen = set()
    with samples.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        for row in reader:
            key = row[spec["id_col"]]
            if key not in entries or key in seen:
                raise ValueError("ID mẫu thiếu/lặp so với cohort manifest.")
            truth = entries[key]["y_true"]
            actual = row[spec["label_col"]]
            if spec["dataset"] == "crisismmd" and isinstance(truth, dict):
                expected = truth[spec["task"]]
            elif spec["dataset"] == "fakeddit" and manifest["task"] == "6way" and spec["task"] == "2way":
                expected = "1" if str(truth) == "0" else "0"
            elif spec["dataset"] == "mmimdb":
                parsed = json.loads(actual) if actual.startswith("[") else actual.split("|")
                expected, actual = sorted(truth), sorted(parsed)
            else:
                expected = str(truth)
            if actual != expected or row.get("source_split") != manifest["source_split"]:
                raise ValueError(f"y_true/split không khớp manifest cho {key}.")
            if spec.get("secondary_task"):
                if not isinstance(truth, dict) or row.get(spec["secondary_label_col"]) != truth.get(spec["secondary_task"]):
                    raise ValueError(f"Nhãn task thứ hai không khớp manifest cho {key}.")
            project_file(row["image_path"])
            seen.add(key)
    if len(seen) != spec["samples"] or set(entries) != seen or not 0 < spec["limit"] <= len(seen):
        raise ValueError("Cohort không đủ số mẫu hoặc manifest thiếu ID.")
    result = {"cohort_id": spec["cohort_id"], "cohort_samples": len(seen),
              "requested_samples": spec["limit"], "source_split": manifest["source_split"],
              "label_source": manifest["label_source"], "task": spec["task"]}
    print("[COHORT VERIFIED]", json.dumps(result, ensure_ascii=False), flush=True)
    return result


def run(settings):
    root = Path(tempfile.mkdtemp(prefix="vlm_run_"))
    print(f"[WORKER] Bắt đầu job {settings['job_id']} ({settings['run_id']})", flush=True)
    for slug, expected in ((settings["code_dataset_slug"], settings["code_tree_sha256"]),
                           (settings["data_dataset_slug"], settings["data_tree_sha256"])):
        print(f"[WORKER] Kiểm tra dataset {slug}", flush=True)
        source = find_dataset("/kaggle/input", slug, expected)
        shutil.copytree(source, root, dirs_exist_ok=True)
        print(f"[WORKER] Đã nạp dataset {slug}", flush=True)
    cohort_receipt = verify_cohort(root, settings)
    (root / "configs/research").mkdir(parents=True, exist_ok=True)
    (root / "configs/active.yaml").write_text(settings["active_yaml"], encoding="utf-8")
    (root / "configs/job.yaml").write_text(settings["job_yaml"], encoding="utf-8")
    if settings["research_yaml"] is not None:
        (root / "configs/research/active.yaml").write_text(settings["research_yaml"], encoding="utf-8")
    print("[WORKER] Cài thư viện chính", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
                   cwd=root, check=True)
    if settings["kind"] in ("adapted", "upstream", "waffle_lab", "clip_two_stage") or settings["command"][0] == "experiments/fakeddit_calip/run.py":
        print("[WORKER] Cài thư viện và nguồn nghiên cứu", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements/research.txt"],
                       cwd=root, check=True)
        clone_sources(root)
    if settings["kind"] == "vlm_reasoning":
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements/vlm_reasoning.txt"],
                       cwd=root, check=True)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Kaggle Kernel không có GPU CUDA")
    command = [sys.executable, "-u", "-B", *settings["command"]]
    environment = {**os.environ, "SOURCE_REVISION": settings["code_sha256"],
                   "PYTHONUNBUFFERED": "1"}
    print(f"[WORKER] Chạy {settings['command'][0]}", flush=True)
    subprocess.run(command, cwd=root, env=environment, check=True)
    print("[WORKER] Đóng gói kết quả", flush=True)
    output_dir = (root / settings["output_dir"]).resolve()
    if not output_dir.is_relative_to(root / "results") or not output_dir.is_dir():
        raise RuntimeError("Không tìm thấy thư mục kết quả trong results/")
    summaries = list(output_dir.rglob("summary.tsv"))
    if not summaries:
        summaries = list(output_dir.rglob("summary_table*.tsv"))
    if not summaries:
        summaries = list(output_dir.rglob("demo_summary*.tsv"))
    if not summaries:
        raise RuntimeError("Kết quả thiếu summary.tsv")
    output = Path("/kaggle/working")
    shutil.copy2(sorted(summaries)[0], output / "summary.tsv")
    shutil.make_archive(str(output / "run_artifacts"), "zip", root_dir=output_dir)
    receipt = {"run_id": settings["run_id"], "job_id": settings["job_id"],
               "kind": settings["kind"], "code_sha256": settings["code_sha256"],
               "data_sha256": settings["data_sha256"], "gpu": torch.cuda.get_device_name(0)}
    if cohort_receipt:
        receipt["cohort"] = cohort_receipt
    (output / "run_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2),
                                              encoding="utf-8")
    print("[KAGGLE_RESULT]", (output / "summary.tsv").read_text(encoding="utf-8"), flush=True)
