"""Worker Kaggle cho các job trong launcher; chỉ ghi kết quả vào /kaggle/working."""

import hashlib
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


def run(settings):
    root = Path(tempfile.mkdtemp(prefix="vlm_run_"))
    print(f"[WORKER] Bắt đầu job {settings['job_id']} ({settings['run_id']})", flush=True)
    for slug, expected in ((settings["code_dataset_slug"], settings["code_tree_sha256"]),
                           (settings["data_dataset_slug"], settings["data_tree_sha256"])):
        print(f"[WORKER] Kiểm tra dataset {slug}", flush=True)
        source = find_dataset("/kaggle/input", slug, expected)
        shutil.copytree(source, root, dirs_exist_ok=True)
        print(f"[WORKER] Đã nạp dataset {slug}", flush=True)
    (root / "configs/research").mkdir(parents=True, exist_ok=True)
    (root / "configs/active.yaml").write_text(settings["active_yaml"], encoding="utf-8")
    (root / "configs/job.yaml").write_text(settings["job_yaml"], encoding="utf-8")
    if settings["research_yaml"] is not None:
        (root / "configs/research/active.yaml").write_text(settings["research_yaml"], encoding="utf-8")
    print("[WORKER] Cài thư viện chính", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
                   cwd=root, check=True)
    if settings["kind"] in ("adapted", "upstream"):
        print("[WORKER] Cài thư viện và nguồn nghiên cứu", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements/research.txt"],
                       cwd=root, check=True)
        clone_sources(root)
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
    (output / "run_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2),
                                              encoding="utf-8")
    print("[KAGGLE_RESULT]", (output / "summary.tsv").read_text(encoding="utf-8"), flush=True)
