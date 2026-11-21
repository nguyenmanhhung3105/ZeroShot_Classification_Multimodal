"""Đọc các bộ đã chuẩn bị; không tải lại dữ liệu hoặc đọc raw."""
import csv
import hashlib
from itertools import islice
from pathlib import Path
from .io import path, read_json, sha256, digest, BENCHMARKS

def benchmark(root, name, smoke=False):
    root = path(root)
    provenance = read_json(root / 'provenance.json')
    classes = read_json(root / 'classes.json')
    count, size, split = BENCHMARKS[name]
    if provenance.get('dataset') != name or provenance.get('split') != split:
        raise ValueError('Sai nguồn dataset hoặc split')
    if not isinstance(classes, list) or len(classes) != count or any(not isinstance(v, str) or not v.strip() for v in classes) or len(set(classes)) != count:
        raise ValueError('Taxonomy không đúng benchmark')
    rows = []
    seen = set()
    with path(root / 'manifest.tsv').open(newline='') as stream:
        reader = csv.DictReader(stream, delimiter='\t')
        if not {'id', 'image', 'label', 'bytes', 'sha256'} <= set(reader.fieldnames or []):
            raise ValueError('Manifest thiếu schema')
        for raw in islice(reader, 5 if smoke else size + 1):
            image = path(root / 'images' / raw['image'])
            label = int(raw['label'])
            if not image.is_relative_to(path(root / 'images')) or raw['id'] in seen or (not 0 <= label < count):
                raise ValueError('ID/path/nhãn không hợp lệ')
            if image.stat().st_size != int(raw['bytes']) or sha256(image) != raw['sha256']:
                raise ValueError('Ảnh thiếu hoặc sai SHA256')
            seen.add(raw['id'])
            rows.append({'id': raw['id'], 'image': str(image), 'label': label, 'image_sha256': raw['sha256']})
    if not rows or (not smoke and len(rows) != size):
        raise ValueError('Không đủ số mẫu benchmark')
    identity = [{k: r[k] for k in ['id', 'label', 'image_sha256']} for r in rows]
    return (classes, rows, {'source': provenance, 'cohort_sha256': digest(identity), 'split': split})

def fakeddit(root, split, smoke=False, *, selected_indices=None):
    root = path(root)
    if smoke and split == 'test':
        raise ValueError('Smoke chỉ dùng train/dev; test phải chạy đủ cohort đã khóa')
    if selected_indices is not None:
        selected_indices = set(selected_indices)
        if (split != 'dev' or smoke or not selected_indices
                or any(type(i) is not int or not 0 <= i < 3000 for i in selected_indices)):
            raise ValueError('Subset chẩn đoán chỉ dùng dev và index trong [0, 3000)')
    manifest = read_json(root / 'manifest.json')
    source_split = {'train': 'train', 'dev': 'validate', 'test': 'test_public'}[split]
    table = root / 'fakeddit_sample.tsv'
    if manifest.get('status') != 'ready' or manifest.get('dataset') != 'fakeddit' or manifest.get('split') != split or (manifest.get('source_split') != source_split) or (manifest.get('selected', {}).get('samples') != 3000) or (sha256(table) != manifest.get('samples_sha256')):
        raise ValueError('Cohort Fakeddit v2 không khớp manifest/split/checksum')
    expected = manifest['samples']
    if len(expected) != 3000 or manifest['cohort_id'] != digest(expected)[:20]:
        raise ValueError('Manifest Fakeddit thiếu hoặc sai cohort_id')
    with table.open(newline='') as stream:
        raw_rows = list(islice(csv.DictReader(stream, delimiter='\t'), 3001))
    if len(raw_rows) != 3000 or len({r['id'] for r in raw_rows}) != 3000:
        raise ValueError('Cần đúng 3000 ID duy nhất')
    rows = []
    for i, (raw, entry) in enumerate(zip(raw_rows, expected)):
        two = raw['2_way_label']
        six = raw['6_way_label']
        title = raw['clean_title']
        if raw['id'] != entry['id'] or six != entry['y_true'] or two not in ['0', '1'] or (six not in list('012345')) or ((two == '1') != (six == '0')) or (raw['source_split'] != source_split) or (raw['split'] != split) or (hashlib.sha256(title.casefold().encode()).hexdigest() != entry['text_sha256']):
            raise ValueError('ID, taxonomy hoặc text khác cohort đã chuẩn bị')
        if smoke and i >= 5:
            continue
        if selected_indices is not None and i not in selected_indices:
            continue
        image = path(root / 'images' / Path(raw['image_path']).name)
        if not image.is_relative_to(path(root / 'images')) or sha256(image) != entry['image_sha256']:
            raise ValueError('Ảnh khác cohort hoặc nằm ngoài image_root')
        rows.append({'id': raw['id'], 'image': str(image), 'text': title, '2way': two, '6way': six, 'image_sha256': entry['image_sha256'], 'text_sha256': entry['text_sha256']})
    identity = [{k: r[k] for k in ['id', '2way', '6way', 'image_sha256', 'text_sha256']} for r in rows]
    return (rows, {'split': split, 'source_split': source_split, 'cohort_id': manifest['cohort_id'], 'cohort_sha256': digest(identity), 'manifest_sha256': sha256(root / 'manifest.json')})
