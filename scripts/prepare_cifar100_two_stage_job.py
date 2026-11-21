"""Build a whitelist-only standalone Kaggle notebook; never upload or push."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def prepare(mode="smoke", output=None, dev_kernel=None, variant="hierarchy"):
    if variant not in ["hierarchy", "rerank", "em_stata"]:
        raise ValueError("Unknown experiment variant")
    rerank = variant == "rerank"
    adapted = variant == "em_stata"
    if mode not in ["smoke", "dev", "test"] or (mode == "test" and not dev_kernel):
        raise ValueError("Test requires --dev-kernel from the full dev job")
    config_file = "configs/cifar100_em_stata.json" if adapted else "configs/cifar100_rerank.json" if rerank else "configs/cifar100_two_stage.json"
    names = [config_file, "requirements.txt", "src/vlm_project/__init__.py",
             *[f"src/vlm_project/{name}.py" for name in
               ["cifar100_two_stage", "backbone", "io", "evaluate", "shared_data"]]]
    if rerank:
        names.extend(["src/vlm_project/cifar100_rerank.py", "configs/descriptors/cifar100_visual_v1.json"])
    if adapted:
        names.extend(["src/vlm_project/cifar100_em_stata.py", "src/vlm_project/adaptation.py"])
    files = {name: (ROOT / name).read_text() for name in names}
    checksums = {name: hashlib.sha256(content.encode()).hexdigest() for name, content in files.items()}
    identity = hashlib.sha256(json.dumps({"files": checksums, "mode": mode, "dev_kernel": dev_kernel}, sort_keys=True).encode()).hexdigest()[:12]
    destination = Path(output or ROOT / "results/_system/cifar100_two_stage" / f"{variant}_{mode}_{identity}")
    destination.mkdir(parents=True, exist_ok=True)
    payload = repr(json.dumps(files))
    bootstrap = f'''import json, hashlib, subprocess, sys
from pathlib import Path
files = json.loads({payload})
PROJECT = Path("/kaggle/working/cifar100_experiment_source")
for name, content in files.items():
    file = PROJECT / name
    if file.exists() and file.read_text() != content:
        raise ValueError("Existing source differs; start a clean session")
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(content)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(PROJECT / "requirements.txt")], check=True)
sys.path.insert(0, str(PROJECT / "src"))
from vlm_project.cifar100_two_stage import locate_source, run
print("Code ready; mode={mode}; backbone=MetaCLIP2 ViT-H/14")
'''
    execute = f'''root = locate_source()
dev_run = None
MODE = {mode!r}
if MODE == "test":
    candidates = []
    for file in Path("/kaggle/input").rglob("run.json"):
        value = json.loads(file.read_text())
        if value.get("kind") == {('cifar100_em_stata' if adapted else 'cifar100_rerank' if rerank else 'cifar100_two_stage')!r} and value.get("split") == "dev" and value.get("mode") == "full" and value.get("status") == "complete":
            candidates.append(file.parent)
    if len(candidates) != 1:
        raise ValueError("Attach exactly one completed full dev Output")
    dev_run = candidates[0]
output = run(root, "/kaggle/working/cifar100_two_stage_results",
             split="test" if MODE == "test" else "dev", smoke=MODE == "smoke",
             dev_run=dev_run, spec_file=PROJECT / {config_file!r}, device="cuda")
print("Completed:", output)
import zipfile
result_zip = Path("/kaggle/working") / ("metrics_" + output.name + ".zip")
with zipfile.ZipFile(result_zip, "w", zipfile.ZIP_DEFLATED) as archive:
    for file in sorted(output.iterdir()):
        if file.is_file():
            archive.write(file, "run/" + file.name)
print("Metrics ZIP:", result_zip)
'''
    cells = []
    for kind, content in [("markdown", "# CIFAR-100: " + variant + "\nMode: " + mode + ". No fine-tuning. Smoke is not a quality estimate."),
                          ("code", bootstrap), ("code", execute)]:
        cell = {"cell_type": kind, "id": f"cifar-experiment-{len(cells)}", "metadata": {}, "source": content.splitlines(True)}
        if kind == "code":
            compile(content, "generated-notebook", "exec")
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
                "nbformat": 4, "nbformat_minor": 5}
    slug = "em-stata" if adapted else "rerank" if rerank else "two-stage"
    metadata = {"id": f"nguyenhung3105/vlm-cifar100-{slug}-{mode}-v1",
                "title": f"VLM CIFAR100 {slug.title()} {mode.title()} V1", "code_file": "run.ipynb",
                "language": "python", "kernel_type": "notebook", "is_private": "true",
                "enable_gpu": "true", "enable_internet": "true", "machine_shape": "NvidiaTeslaT4",
                "dataset_sources": ["nguyenhung3105/vlm-cifar100-sources-v1"],
                "competition_sources": [], "kernel_sources": [dev_kernel] if mode == "test" else []}
    for filename, value in [("run.ipynb", notebook), ("kernel-metadata.json", metadata),
                            ("source_manifest.json", {"sha256": checksums, "mode": mode, "variant": variant, "source_id": identity})]:
        (destination / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "dev", "test"], default="smoke")
    parser.add_argument("--output")
    parser.add_argument("--dev-kernel")
    parser.add_argument("--variant", choices=["hierarchy", "rerank", "em_stata"], default="hierarchy")
    args = parser.parse_args()
    print(prepare(args.mode, args.output, args.dev_kernel, args.variant))
