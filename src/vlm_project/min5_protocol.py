"""Chuẩn bị/niêm phong protocol few-shot Task 1 min5; không mở dataset/test."""
import csv
import importlib.metadata
import re

import numpy as np
import yaml

from .fewshot_support_thresholds import load_source, split_dev, choose_threshold, POLICIES
from .io import ROOT, code_hashes, digest, path, read_json, sha256


def checked_run(directory, names):
    directory = path(directory)
    receipt = read_json(directory / 'completed.json')
    if receipt.get('status') != 'complete' or receipt.get('run_sha256') != sha256(directory / 'run.json'):
        raise ValueError('Receipt không khớp run')
    run = read_json(directory / 'run.json')
    if run.get('status') != 'complete':
        raise ValueError('Run chưa hoàn tất')
    hashes = {}
    for name in names:
        file = path(directory / name)
        if not file.is_relative_to(directory):
            raise ValueError('Artifact trỏ ra ngoài source')
        actual = sha256(file)
        if run.get('output_sha256', {}).get(name) != actual:
            raise ValueError(f'Checksum artifact sai: {name}')
        hashes[name] = actual
    return run, hashes


def fingerprint(settings_file):
    """Code/config/dependency hiện tại; chỉ đọc source, không nạp backbone."""
    files = ['requirements.txt', 'scripts/prepare_min5_protocol.py', 'scripts/run_min5_test.py',
             'configs/fakeddit_fewshot_task1_min5.yaml', 'configs/fakeddit_fewshot_support_min5.yaml']
    libraries = ['numpy', 'scipy', 'scikit-learn', 'PyYAML', 'torch', 'torchvision',
                 'open_clip_torch', 'transformers', 'Pillow']
    versions = {}
    for name in libraries:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {'code': code_hashes(), 'files': {name: sha256(ROOT / name) for name in files},
            'settings_sha256': sha256(settings_file), 'libraries': versions}


def exclusive_write(destination, payload):
    destination = path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock = {'payload': payload, 'sha256': digest(payload)}
    import json
    text = json.dumps(lock, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    with destination.open('x', encoding='utf-8') as stream:
        stream.write(text)
    return lock


def identities(rows):
    result = []
    for row in rows:
        item = {key: row[key] for key in ['id', 'image_sha256', 'text_sha256']}
        if not isinstance(item['id'], str) or not item['id']:
            raise ValueError('ID không hợp lệ')
        if any(not isinstance(item[k], str) or not re.fullmatch(r'[0-9a-f]{64}', item[k])
               for k in ['image_sha256', 'text_sha256']):
            raise ValueError('Thiếu hash ảnh/text hợp lệ')
        result.append(item)
    if len(result) != 3000 or len({r['id'] for r in result}) != 3000:
        raise ValueError('Cần 3000 ID duy nhất')
    return result


def no_overlap(left, right):
    for key in ['id', 'image_sha256', 'text_sha256']:
        if {r[key] for r in left} & {r[key] for r in right}:
            raise ValueError(f'Train/dev/test overlap: {key}')


def validate_support(log, train):
    lookup = {r['id']: r for r in train}
    seen = set()
    for entry in log:
        key = (entry['policy'], entry['seed'])
        if key in seen or key[0] not in POLICIES or type(key[1]) is not int or key[1] not in range(5):
            raise ValueError('Support policy/seed thiếu, trùng hoặc sai')
        seen.add(key)
        ids, labels = entry['ids'], entry['labels_6way']
        if len(ids) != 384 or len(set(ids)) != 384 or len(labels) != 384 or not set(ids) <= lookup.keys():
            raise ValueError('Support phải có 384 ID duy nhất từ train')
        actual = [int(lookup[i]['6way']) for i in ids]
        quota = [64]*6 if key[0] == 'legacy_6way' else [192, 38, 38, 38, 38, 38]
        if key[0] == 'balanced_binary':
            for offset in range(2):
                quota[1 + (key[1]+offset) % 5] += 1
        if (actual != labels or entry['quotas_6way'] != quota
                or np.bincount(actual, minlength=6).tolist() != quota):
            raise ValueError('Support sai nhãn hoặc quota')
    if seen != {(p, s) for p in POLICIES for s in range(5)}:
        raise ValueError('Support cần đủ hai policy × năm seed')


def prepare(source, review, destination, settings_file='configs/project_min5.yaml'):
    source, review, settings_file = path(source), path(review), path(settings_file)
    destination = path(destination)
    if destination.is_relative_to(source) or destination.is_relative_to(review):
        raise ValueError('Không ghi protocol vào source/review')
    reviewed, review_hashes = checked_run(review, ['thresholds.tsv', 'split.json'])
    if (reviewed.get('kind') != 'fewshot_support_thresholds' or reviewed.get('split') != 'dev'
            or reviewed.get('mode') != 'full' or reviewed.get('test_opened') is not False
            or reviewed.get('source_run_sha256') != sha256(source / 'run.json')):
        raise ValueError('Review phải thuộc đúng source dev chưa mở test')
    spec = reviewed['specification']
    run, posterior_hashes, ids, six, scores = load_source(source, spec['modalities'])
    _, source_hashes = checked_run(source, ['support.json', 'train_cohort.json', 'cohort.json'])
    train, dev, support = (read_json(source / name) for name in ['train_cohort.json', 'cohort.json', 'support.json'])
    excluded_train, excluded_dev = identities(train), identities(dev)
    no_overlap(excluded_train, excluded_dev)
    if digest(support) != run.get('support_sha256'):
        raise ValueError('Support fingerprint không khớp source')
    validate_support(support, train)
    selection, audit = split_dev(ids, six, spec['selection_fraction'], spec['split_seed'])
    selected = set(selection.tolist())
    membership = [{'id': str(ids[i]), '6way': int(six[i]), 'partition': 'selection' if i in selected else 'audit'}
                  for i in range(len(ids))]
    if (membership != read_json(review / 'split.json') or digest(membership) != reviewed.get('split_sha256')
            or reviewed.get('samples') != len(audit)):
        raise ValueError('Partition review không khớp protocol')
    with (review / 'thresholds.tsv').open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream, delimiter='\t'))
    indexed = {}
    for row in rows:
        key = (row['policy'], row['modality'], int(row['seed']), float(row['target_recall_fake']))
        if key in indexed:
            raise ValueError('Ngưỡng review bị trùng')
        indexed[key] = float(row['threshold'])
    expected = {(p, m, s, t) for p in POLICIES for m in spec['modalities'] for s in range(5) for t in spec['targets']}
    if set(indexed) != expected:
        raise ValueError('Review thiếu/thừa ngưỡng')
    candidates = []
    for policy in POLICIES:
        for modality in spec['modalities']:
            for seed in range(5):
                base = {'policy': policy, 'modality': modality, 'seed': seed}
                candidates.append({**base, 'decision': 'default', 'rule': 'argmax_real_on_tie', 'threshold': None})
                for target in spec['targets']:
                    expected_threshold = choose_threshold((six[selection] != 0).astype(int),
                                                          scores[policy, modality, seed][selection, 1], target)
                    if indexed[policy, modality, seed, target] != expected_threshold:
                        raise ValueError('Ngưỡng không tái tạo từ selection; không dùng audit chọn lại')
                    candidates.append({**base, 'decision': f'target_{target!r}', 'rule': 'fake_if_score_ge_threshold',
                                       'target_recall_fake': target, 'threshold': expected_threshold})
    settings = yaml.safe_load(settings_file.read_text())
    if settings.get('model') != run['model'] or type(settings.get('batch_size')) is not int or settings['batch_size'] < 1:
        raise ValueError('Model/batch_size cấu hình không hợp lệ hoặc khác source')
    payload = {'version': 1, 'kind': 'min5_fewshot_task1_protocol', 'state': 'draft',
               'scope': 'Task 1 binary component only; not a completed cascade or a test runner',
               'model': run['model'], 'primary_metric': 'macro_f1', 'report_metrics': ['accuracy', 'macro_f1', 'balanced_accuracy', 'recall_real', 'recall_fake'],
               'test_opened': False, 'test_manifest_sha256': None, 'test_samples': 3000,
               'settings_file': str(settings_file), 'fingerprint': fingerprint(settings_file),
               'inference_settings': {'model': settings['model'], 'batch_size': settings['batch_size']},
               'source': {'run_sha256': sha256(source / 'run.json'), 'artifact_sha256': {**posterior_hashes, **source_hashes},
                          'specification': run['specification'], 'feature': run['feature'],
                          'model_runtime': run.get('model_runtime', {}), 'historical_code_sha256': run.get('code_sha256', {})},
               'review': {'run_sha256': sha256(review / 'run.json'), 'artifact_sha256': review_hashes,
                          'split_sha256': reviewed['split_sha256'], 'specification': spec},
               'support': support, 'candidates': candidates,
               'excluded_train': excluded_train, 'excluded_dev': excluded_dev,
               'limitations': 'Development-informed suite, no winner selected. Draft cannot authorize test. Test requires final scope, explicit test manifest hash, and a runner integrating validation before model loading.'}
    return exclusive_write(destination, payload)


def _validate_lock(lock, *, require_sealed=True, settings_file=None):
    if not isinstance(lock, dict) or not isinstance(lock.get('payload'), dict):
        raise ValueError('Protocol sai định dạng/checksum')
    payload = lock.get('payload', {})
    if (lock.get('sha256') != digest(payload) or payload.get('version') != 1
            or payload.get('kind') != 'min5_fewshot_task1_protocol' or payload.get('test_opened') is not False):
        raise ValueError('Protocol sai định dạng/checksum')
    if payload.get('state') not in ['draft', 'sealed_task1']:
        raise ValueError('Protocol sai trạng thái')
    if require_sealed and payload['state'] != 'sealed_task1':
        raise ValueError('Protocol còn draft; không cho phép test')
    test_hash = payload.get('test_manifest_sha256')
    if payload['state'] == 'sealed_task1' and (not isinstance(test_hash, str)
                                             or not re.fullmatch(r'[0-9a-f]{64}', test_hash)):
        raise ValueError('Thiếu hash manifest test đã khóa')
    if fingerprint(settings_file or payload['settings_file']) != payload.get('fingerprint'):
        raise ValueError('Code/cấu hình/dependency đã đổi sau khi tạo protocol')
    return lock


def validate(file, *, require_sealed=True, settings_file=None):
    return _validate_lock(read_json(file), require_sealed=require_sealed, settings_file=settings_file)


def seal(draft, test_manifest_sha256, destination):
    if not isinstance(test_manifest_sha256, str) or not re.fullmatch(r'[0-9a-f]{64}', test_manifest_sha256):
        raise ValueError('Cần SHA256 manifest test min5, không đọc test để chọn cấu hình')
    lock = validate(draft, require_sealed=False)
    if lock['payload']['state'] != 'draft':
        raise ValueError('Chỉ niêm phong từ draft')
    if test_manifest_sha256 in lock['payload']['source']['specification'].get('manifest_sha256', {}).values():
        raise ValueError('Không dùng manifest train/dev làm test')
    payload = {**lock['payload'], 'state': 'sealed_task1', 'test_manifest_sha256': test_manifest_sha256,
               'draft_sha256': lock['sha256']}
    return exclusive_write(destination, payload)


def check_test_cohort(lock, rows, metadata, *, settings_file=None):
    """Guard cho runner tương lai; nhận metadata đã load, không tự đọc dataset."""
    p = _validate_lock(lock, settings_file=settings_file)['payload']
    if (metadata.get('split') != 'test' or metadata.get('source_split') != 'test_public'
            or metadata.get('manifest_sha256') != p['test_manifest_sha256']):
        raise ValueError('Sai split/nguồn/hash test min5 đã khóa')
    test = identities(rows)
    no_overlap(test, p['excluded_train'])
    no_overlap(test, p['excluded_dev'])
