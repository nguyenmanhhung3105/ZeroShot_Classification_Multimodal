"""Xuất bản code Kaggle gọn, không kèm dataset, cache, token hoặc weights."""
import hashlib
import json
import zipfile
from .io import ROOT, path, sha256, write_json

def notebook():
    cells = []

    def cell(kind, text):
        value = {'cell_type': kind, 'metadata': {}, 'id': f'{kind}-{len(cells)}', 'source': text.splitlines(keepends=True)}
        if kind == 'code':
            value.update(execution_count=None, outputs=[])
        cells.append(value)
    cell('markdown', '# MetaCLIP2 — bản dự án cuối\n1. Bật **Internet** và **GPU** trong Settings.\n2. Add Input: dataset chứa ZIP code này, `vlm-public-benchmarks-eval-v1` (benchmark), hoặc Fakeddit v2 đúng split.\n3. Chọn TASK bên dưới, rồi chạy lần lượt các cell. Mặc định thử DTD, sau đó chạy đủ 1.880 mẫu.\n4. Với Fakeddit: chạy dev → notebook tự tạo lock → lưu kết quả. Test cần đúng lock đó, không thay code/prompt sau khi khóa.\n')
    cell('code', 'TASK = "benchmark"  # benchmark hoặc fakeddit\nDATASET = "dtd"      # một trong 8 bộ benchmark\nSPLIT = "dev"        # Fakeddit: dev trước, test sau khi khóa\nDATA_ROOT = "auto"   # nếu không tìm thấy mount: điền thư mục chứa manifest và images\nTEST_LOCK_INPUT = "" # test phiên mới: path tới fakeddit_test_lock.json đã lưu từ dev\n')
    cell('code', 'from pathlib import Path\nimport hashlib, json, shutil, subprocess, sys, zipfile\n\n# Hỗ trợ Kaggle tự giải nén ZIP hoặc giữ file ZIP trong Input.\nbase = Path("/kaggle/input")\npatterns = ["*/final_model_source.zip", "*/*/final_model_source.zip",\n            "datasets/*/*/final_model_source.zip", "datasets/*/*/*/final_model_source.zip"]\narchives = sorted({p for pattern in patterns for p in base.glob(pattern)})\npatterns = [p.replace("final_model_source.zip", "source_bundle_manifest.json") for p in patterns]\nmanifests = sorted({p for pattern in patterns for p in base.glob(pattern)})\nif len(archives) + len(manifests) != 1:\n    raise RuntimeError("Add Input đúng một bundle final_model_source; bỏ bundle cũ nếu có")\nif archives:\n    with zipfile.ZipFile(archives[0]) as archive:\n        manifest = json.loads(archive.read("source_bundle_manifest.json"))\n        code_id = hashlib.sha256(json.dumps(manifest["files"], sort_keys=True).encode()).hexdigest()[:12]\n        PROJECT = Path("/kaggle/working") / ("vlm-final-" + code_id)\n        PROJECT.mkdir(exist_ok=True)\n        for name, checksum in manifest["files"].items():\n            target = (PROJECT / name).resolve()\n            if not target.is_relative_to(PROJECT.resolve()):\n                raise ValueError("Path ngoài bundle")\n            content = archive.read(name)\n            if hashlib.sha256(content).hexdigest() != checksum:\n                raise ValueError("Sai checksum: " + name)\n            if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != checksum:\n                raise ValueError("Code tại working đã được sửa; không ghi đè: " + name)\n            target.parent.mkdir(parents=True, exist_ok=True)\n            target.write_bytes(content)\n        (PROJECT / "source_bundle_manifest.json").write_text(json.dumps(manifest, indent=2))\nelse:\n    manifest = json.loads(manifests[0].read_text())\n    code_id = hashlib.sha256(json.dumps(manifest["files"], sort_keys=True).encode()).hexdigest()[:12]\n    PROJECT = Path("/kaggle/working") / ("vlm-final-" + code_id)\n    PROJECT.mkdir(exist_ok=True)\n    for name, checksum in manifest["files"].items():\n        source = manifests[0].parent / name\n        target = (PROJECT / name).resolve()\n        if not target.is_relative_to(PROJECT.resolve()):\n            raise ValueError("Path ngoài bundle")\n        if hashlib.sha256(source.read_bytes()).hexdigest() != checksum:\n            raise ValueError("Sai checksum: " + name)\n        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != checksum:\n            raise ValueError("Code tại working đã được sửa; không ghi đè: " + name)\n        target.parent.mkdir(parents=True, exist_ok=True)\n        shutil.copy2(source, target)\n    shutil.copy2(manifests[0], PROJECT / "source_bundle_manifest.json")\n\nsubprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(PROJECT / "requirements.txt")], check=True)\nsubprocess.run([sys.executable, str(PROJECT / "run.py"), "sources"], cwd=PROJECT, check=True)\nsys.path.insert(0, str(PROJECT / "src"))\nfrom vlm_project.io import config, locate_data\nsettings = config()\nif TEST_LOCK_INPUT:\n    destination = PROJECT / settings["fakeddit"]["lock_file"]\n    content = Path(TEST_LOCK_INPUT).read_bytes()\n    if destination.exists() and destination.read_bytes() != content:\n        raise ValueError("Đã có lock khác trong phiên; không ghi đè")\n    destination.write_bytes(content)\nprint("Code:", PROJECT)\n')
    cell('code', '''if TASK == "benchmark":
    from vlm_project.benchmark import run
    name = DATASET
elif TASK == "fakeddit":
    from vlm_project.fakeddit import run
    name = SPLIT
else:
    raise ValueError("TASK chỉ nhận benchmark hoặc fakeddit")
root = locate_data(TASK, name, settings, DATA_ROOT)
print("Dữ liệu:", root)
run_options = {}
benchmark_backend = None
if TASK == "benchmark":
    from vlm_project.backbone import Backbone
    benchmark_backend = Backbone(settings["device"])
    run_options["backend"] = benchmark_backend
# Smoke benchmark chỉ CLIP, 5 ảnh, đủ lớp; full giữ mọi phương pháp/seed.
if TASK == "benchmark" or SPLIT != "test":
    try:
        smoke_output = run(settings, name, root, smoke=True, **run_options)
        print("Smoke OK:", smoke_output)
    except BaseException:
        if benchmark_backend is not None:
            benchmark_backend.unload()
            benchmark_backend = None
        raise
else:
    from vlm_project.protocol import validate_lock
    validate_lock(settings)
    print("Lock hợp lệ; chạy test đầy đủ ở cell sau")
''')
    cell('code', '''try:
    output = run(settings, name, root, smoke=False, **run_options)
finally:
    if benchmark_backend is not None:
        benchmark_backend.unload()
        benchmark_backend = None
from vlm_project.protocol import report, freeze
report(output)
if TASK == "fakeddit" and SPLIT == "dev":
    lock_file = PROJECT / settings["fakeddit"]["lock_file"]
    if not lock_file.exists():
        freeze(settings, output)
    else:
        print("Lock đã tồn tại; giữ nguyên, không tự ghi đè:", lock_file)
''')
    cell('code', '# ZIP kết quả và lock; Save Version để giữ Output sau khi kết thúc phiên.\nresult_zip = Path("/kaggle/working") / ("metrics_" + output.name + ".zip")\nwith zipfile.ZipFile(result_zip, "w", zipfile.ZIP_DEFLATED) as archive:\n    for file in sorted(output.iterdir()):\n        if file.is_file():\n            archive.write(file, "run/" + file.name)\n    lock_file = PROJECT / settings["fakeddit"]["lock_file"]\n    if TASK == "fakeddit" and lock_file.is_file():\n        archive.write(lock_file, "fakeddit_test_lock.json")\nprint("Tải và lưu kết quả:", result_zip)\n')
    return {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python'}, 'kaggle': {'isInternetEnabled': True, 'isGpuEnabled': True}}, 'cells': cells}

def package(destination):
    output = path(destination)
    output.mkdir(parents=True, exist_ok=True)
    files = [ROOT / 'run.py', ROOT / 'requirements.txt', ROOT / 'README.md']
    for directory in ['src/vlm_project', 'src/preprocessing', 'scripts', 'configs', 'docs', 'tests']:
        files.extend((p for p in sorted((ROOT / directory).rglob('*')) if p.is_file() and p.suffix in ['.py', '.yaml', '.json', '.md'] and ('__pycache__' not in p.parts)))
    # Bản chụp CSV tác giả để đối chiếu offline, cùng checksum trong registry.
    files.extend(sorted((ROOT / 'configs/paper_results/sources').glob('*.csv')))
    files = list(dict.fromkeys(files))
    manifest = {'version': 1, 'model': 'metaclip2_vith14', 'files': {str(p.relative_to(ROOT)): sha256(p) for p in files}, 'scope': 'Final code only; data, weights, cache, secrets and legacy archive excluded.'}
    archive = output / 'final_model_source.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for p in files:
            bundle.write(p, str(p.relative_to(ROOT)))
        bundle.writestr('source_bundle_manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2))
    write_json(output / 'Final_MetaCLIP2_Kaggle.ipynb', notebook())
    write_json(output / 'source_bundle_manifest.json', manifest)
    receipt = {'zip': str(archive), 'sha256': sha256(archive), 'files': len(files), 'notebook': str(output / 'Final_MetaCLIP2_Kaggle.ipynb')}
    write_json(output / 'package_receipt.json', receipt)
    return receipt
