"""Prepare Kaggle notebook or serve local UI; never loads model weights locally."""
import argparse
import base64
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def prepare(destination, owner, slug, backbones, datasets=(), assets_path=None, assets_dir=None):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', owner) or not re.fullmatch(r'[a-z0-9-]+', slug):
        raise ValueError('Owner/slug Kaggle không hợp lệ')
    if assets_dir and assets_path:
        raise ValueError('Chọn assets-dir hoặc assets-path, không đồng thời cả hai')
    if assets_path and (not assets_path.startswith('/kaggle/input/') or '..' in Path(assets_path).parts):
        raise ValueError('Assets phải là thư mục được mount dưới /kaggle/input/')
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    archive = io.BytesIO()
    config_names = ['fakeddit_prompts.json', 'fakeddit_task1.yaml', 'research_sources.json',
                    'fakeddit_hierarchy_dev.json', 'fakeddit_fewshot_cascade_min5.json',
                    'descriptors/fakeddit_2way.json', 'descriptors/fakeddit_6way.json']
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for file in (ROOT / 'src/vlm_project').glob('*.py'):
            bundle.write(file, 'src/vlm_project/' + file.name)
        for name in config_names:
            bundle.write(ROOT / 'configs' / name, 'configs/' + name)
        bundle.write(ROOT / 'requirements.txt', 'requirements.txt')
        if assets_dir:
            asset_root = Path(assets_dir).resolve()
            manifest = json.loads((asset_root / 'manifest.json').read_text())
            bundle.write(asset_root / 'manifest.json', 'demo_assets/manifest.json')
            for config in manifest.values():
                for kind in ('binary', 'cascade', 'reference'):
                    if kind not in config:
                        continue
                    entry = config[kind]
                    file = (asset_root / entry['file']).resolve()
                    if not file.is_relative_to(asset_root) or file.suffix != '.npz':
                        raise ValueError('Đường dẫn asset không hợp lệ')
                    if hashlib.sha256(file.read_bytes()).hexdigest() != entry['sha256']:
                        raise ValueError('Asset sai checksum')
                    bundle.write(file, 'demo_assets/' + entry['file'])
            assets_path = '/kaggle/working/vlm_demo/demo_assets'
    encoded = base64.b64encode(archive.getvalue()).decode()
    # Only an explicit Kaggle-mounted asset path, never local datasets or credentials.
    bootstrap = (
        'import base64, io, zipfile, pathlib, os, sys, subprocess\n'
        'root = pathlib.Path("/kaggle/working/vlm_demo")\nroot.mkdir(exist_ok=True)\n'
        f'zipfile.ZipFile(io.BytesIO(base64.b64decode({encoded!r}))).extractall(root)\n'
        'subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", str(root / "requirements.txt")])\n'
        'os.chdir(root)\nsys.path.insert(0, str(root / "src"))\n'
        'from vlm_project.demo_worker import run\n'
        f'run(backbones={backbones!r}, assets={assets_path!r})\n')
    notebook = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {'kernelspec': {
        'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}, 'cells': [
            {'cell_type': 'markdown', 'id': 'guide', 'metadata': {}, 'source': [
                '# VLM demo — GPU worker\n',
                'Bật Internet/GPU. Thêm và bật quyền truy cập Kaggle Secrets: '
                '`DEMO_BROKER_URL`, `DEMO_WORKER_TOKEN`. Không ghi token vào code.\n',
                'Worker giữ model, lấy yêu cầu từ localhost qua HTTPS gateway. '
                'Xem docs/LOCAL_DEMO.md trong repo.\n']},
            {'cell_type': 'code', 'id': 'worker', 'metadata': {}, 'execution_count': None,
             'outputs': [], 'source': bootstrap.splitlines(keepends=True)}]}
    (destination / 'demo.ipynb').write_text(json.dumps(notebook, ensure_ascii=False))
    metadata = {'id': f'{owner}/{slug}', 'title': 'VLM multimodal local demo worker',
                'code_file': 'demo.ipynb', 'language': 'python', 'kernel_type': 'notebook',
                'is_private': True, 'enable_gpu': True, 'enable_internet': True,
                'dataset_sources': list(datasets), 'competition_sources': [], 'kernel_sources': []}
    (destination / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2))
    digest = hashlib.sha256(archive.getvalue()).hexdigest()
    (destination / 'bundle-info.json').write_text(json.dumps({'sha256': digest, 'backbones': backbones,
                                                            'scope': 'source/config and optional explicitly exported heads; no images, backbone weights or credentials'}))
    return destination


def export_assets(destination, binary=None, cascade=None, reference=None):
    """Read only explicitly supplied completed-run artifacts, never dataset directories."""
    import numpy as np
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    entries = {}
    for kind, source in [('binary', binary), ('cascade', cascade), ('reference', reference)]:
        if not source:
            continue
        source = Path(source).resolve()
        run_bytes = (source / 'run.json').read_bytes()
        run = json.loads(run_bytes)
        complete = json.loads((source / 'completed.json').read_text())
        if (complete.get('status') != 'complete' or complete.get('run_sha256') != hashlib.sha256(run_bytes).hexdigest()
                or run.get('model') != 'metaclip2_vith14' or run.get('split') != 'dev'
                or run.get('mode') != 'full'):
            raise ValueError('Chỉ xuất completed full dev MetaCLIP2 đúng checksum; không nhận test/smoke')
        filename = 'features.npz' if kind == 'reference' else 'linear_models.npz'
        with np.load(source / filename, allow_pickle=False) as archive:
            if kind == 'reference':
                # Use TRAIN features only, no query/dev labels and no access to image files.
                arrays = {k: archive['train_' + k][:74] for k in ('image', 'title')}
                if len(arrays['image']) < 3:
                    raise ValueError('Thiếu đặc trưng train tham chiếu')
            else:
                arrays = {k: archive[k] for k in archive.files}
            output = destination / f'{kind}.npz'
            np.savez_compressed(output, **arrays)
        entries[kind] = {'file': output.name, 'sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
                         'source_run_sha256': complete['run_sha256'], 'model_runtime': run.get('model_runtime')}
        if kind == 'cascade':
            spec = run['specification']
            entries['thresholds'] = {f"{c['policy']}_s{c['seed']}": c['threshold']
                                     for c in spec['gate_candidates'] if c['decision'] != 'default'}
    if not entries:
        raise ValueError('Cần ít nhất một artifact nguồn')
    manifest = {'metaclip2': entries}
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    local = commands.add_parser('serve', help='Web local và worker gateway, không nạp model')
    local.add_argument('--port', type=int, default=8765)
    commands.add_parser('doctor', help='Kiểm tra dependency, không đọc credentials')
    package = commands.add_parser('prepare', help='Tạo notebook GPU offline, chưa gửi Kaggle')
    package.add_argument('--output', default='bundles/demo')
    package.add_argument('--owner', required=True)
    package.add_argument('--slug', default='vlm-local-demo-worker')
    package.add_argument('--backbones', nargs='+', choices=['metaclip2', 'openai_clip'], default=['metaclip2'])
    package.add_argument('--dataset', action='append', default=[])
    package.add_argument('--assets-path')
    package.add_argument('--assets-dir', help='Nhúng head đã export vào notebook, không cần tạo Dataset riêng')
    package.add_argument('--push', action='store_true', help='Gửi notebook lên tài khoản Kaggle và chạy GPU')
    export = commands.add_parser('export-assets', help='Xuất head và nhóm tham chiếu từ artifact dev đã hoàn tất')
    export.add_argument('--output', required=True)
    export.add_argument('--binary-run')
    export.add_argument('--cascade-run')
    export.add_argument('--reference-run')
    args = parser.parse_args()
    if args.command == 'serve':
        from vlm_project.demo_service import serve
        serve(args.port)
    elif args.command == 'doctor':
        print(json.dumps({
            'python': sys.version.split()[0],
            'pillow': bool(importlib.util.find_spec('PIL')),
            'kaggle_cli': bool(importlib.util.find_spec('kaggle')),
            'local_model_loading': False,
            'credentials': 'Không đọc; chạy python -m kaggle kernels list --mine để kiểm tra đăng nhập',
            'tunnel': 'Cần HTTPS tunnel tới worker gateway cổng 8766'}, ensure_ascii=False, indent=2))
    elif args.command == 'prepare':
        folder = prepare(args.output, args.owner, args.slug, args.backbones, args.dataset, args.assets_path, args.assets_dir)
        print(f'Đã chuẩn bị notebook: {folder}')
        if args.push:
            subprocess.run([sys.executable, '-m', 'kaggle', 'kernels', 'push', '-p', str(folder)], check=True)
    else:
        export_assets(args.output, args.binary_run, args.cascade_run, args.reference_run)
        print(f'Đã xuất assets: {args.output}')


if __name__ == '__main__':
    main()
