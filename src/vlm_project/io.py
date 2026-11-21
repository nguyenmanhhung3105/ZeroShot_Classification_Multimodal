"""Cấu hình, checksum và artifact dùng chung; không truy cập dữ liệu khi import."""
import hashlib
import json
from pathlib import Path
import uuid
import importlib.metadata
import yaml
ROOT = Path(__file__).resolve().parents[2]
MODEL = 'metaclip2_vith14'
BENCHMARKS = {
    'imagenet': (1000, 50000, 'val'),
    'imagenetv2': (1000, 10000, 'matched-frequency'),
    'pets': (37, 3669, 'test'),
    'dtd': (47, 1880, 'test-partition-1'),
    'eurosat': (10, 27000, 'all-images-as-waffle-upstream'),
    'cub': (200, 5794, 'test'),
    'food101': (101, 25250, 'test'),
    'places365': (365, 36500, 'val-small-256'),
}

def path(value):
    """Chuẩn hóa đường dẫn và chặn vùng dữ liệu cấm, kể cả symlink."""
    p = Path(value)
    p = (p if p.is_absolute() else ROOT / p).resolve()
    if {'dataraw', 'data_process'} & set(p.parts):
        raise ValueError('Không truy cập dataraw/ hoặc data_process/')
    return p

def sha256(file):
    h = hashlib.sha256()
    with path(file).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def digest(value):
    """Giữ cùng chuẩn fingerprint của plan Fakeddit lịch sử."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

def read_json(file):
    return json.loads(path(file).read_text(encoding='utf-8'))

def write_json(file, value):
    p = path(file)
    p.parent.mkdir(parents=True, exist_ok=True)
    temporary = p.with_name(p.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(p)

def config(file='configs/project.yaml'):
    value = yaml.safe_load(path(file).read_text())
    if value.get('model') != MODEL:
        raise ValueError('Bản cuối chỉ chạy MetaCLIP2 ViT-H/14')
    if value.get('device') not in ['auto', 'cpu', 'cuda', 'mps']:
        raise ValueError('Device không hợp lệ')
    if type(value.get('batch_size')) is not int or not 1 <= value['batch_size'] <= 64:
        raise ValueError('Batch size không hợp lệ')
    spec = value.get('benchmark', {})
    if spec.get('methods') != ['clip', 'dclip', 'waffle', 'waffle_unit_ablation']:
        raise ValueError('Giữ bộ đối chứng benchmark đã khóa')
    if spec.get('seeds') != list(range(7)) or spec.get('waffle_pairs') != 15:
        raise ValueError('Benchmark dùng 7 seed, 15 cặp Waffle')
    if value.get('fakeddit', {}).get('samples_per_split') != 3000:
        raise ValueError('Giữ cohort Fakeddit v2 3000 mẫu')
    if value['fakeddit'].get('methods') != ['baseline_image', 'baseline_text', 'baseline_fusion', 'semantic_text_fusion', 'proposed', 'dclip_image', 'waffle_image']:
        raise ValueError('Giữ bộ đối chứng Fakeddit đã khóa')
    return value

def code_hashes():
    files = [ROOT / 'run.py', *sorted((ROOT / 'src/vlm_project').glob('*.py'))]
    return {str(p.relative_to(ROOT)): sha256(p) for p in files}

def protocol_hashes(settings):
    """Hash những phần ảnh hưởng suy luận; đường dẫn/máy chạy không làm đổi lock."""
    profile = settings['fakeddit']['prompt_file']
    inference_config = {
        'model': settings['model'],
        'batch_size': settings['batch_size'],
        'benchmark': settings['benchmark'],
        'fakeddit': {key: settings['fakeddit'][key] for key in ['samples_per_split', 'methods']},
    }
    libraries = ['numpy', 'scipy', 'scikit-learn', 'torch', 'torchvision',
                 'open_clip_torch', 'transformers', 'Pillow']
    return {
        'code': code_hashes(),
        'profile': sha256(profile),
        'config': digest(inference_config),
        'sources': sha256(ROOT / 'configs/research_sources.json'),
        'requirements': sha256(ROOT / 'requirements.txt'),
        'libraries': {name: importlib.metadata.version(name) for name in libraries},
        'fakeddit_descriptors': {
            task: sha256(ROOT / f'configs/descriptors/fakeddit_{task}.json')
            for task in ['2way', '6way']
        },
    }

def new_run(settings, dataset, split, smoke):
    p = path(settings['output_root']) / dataset / (split + '_' + ('smoke_' if smoke else '') + uuid.uuid4().hex[:12])
    p.mkdir(parents=True, exist_ok=False)
    return p

def input_roots():
    """Chỉ tìm mount gốc; không quét ảnh của các dataset khác."""
    base = Path('/kaggle/input')
    if not base.is_dir():
        return []
    values = [p for p in base.iterdir() if p.is_dir() and p.name != 'datasets']
    if (base / 'datasets').is_dir():
        for owner in (base / 'datasets').iterdir():
            if owner.is_dir():
                values.extend((p for p in owner.iterdir() if p.is_dir()))
    return values

def locate_data(kind, name, settings, override='auto'):
    """Ưu tiên đường dẫn tường minh, sau đó local hoặc Kaggle Input."""
    if override != 'auto':
        return path(override)
    if kind == 'benchmark':
        local = ROOT / 'data/benchmarks' / name
        candidates = [local]
        slug = settings['kaggle']['benchmark_slug']
        for mount in input_roots():
            if mount.name == slug:
                candidates.extend([mount / name, mount / 'public-benchmarks-eval-v1' / name])
        marker = 'manifest.tsv'
    else:
        candidates = [ROOT / 'data/processed/fakeddit_splits_v2' / name]
        slug = settings['kaggle']['fakeddit_slugs'][name]
        for mount in input_roots():
            if mount.name == slug:
                candidates.extend([mount, mount / name, mount / 'data/processed/fakeddit_splits_v2' / name])
        marker = 'fakeddit_sample.tsv'
    checked = [path(p) for p in candidates]
    found = list(dict.fromkeys(p for p in checked if path(p / marker).is_file()))
    if len(found) != 1:
        raise ValueError(f'Cần --data-root hoặc Add Input đúng dataset; tìm được {found}')
    return found[0]
