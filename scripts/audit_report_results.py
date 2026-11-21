"""Kiểm tra artifact/metric đã lưu, không đọc dataset hoặc nạp model.

Chỉ quét results/final, không quét results/_system (staging/cache).
Top-5 được đối chiếu giữa bảng seed và summary; không tái suy luận logits.
"""
import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from vlm_project.io import path, write_json
from vlm_project.protocol import completed_run


def rows(file):
    with file.open(newline='', encoding='utf-8') as stream:
        yield from csv.DictReader(stream, delimiter='\t')


def close(actual, expected, label):
    if not math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-10):
        raise ValueError(f'{label}: {actual} != {expected}')


def audit(directory):
    run = completed_run(directory)
    metrics = {}
    for row in rows(directory / 'metrics_by_seed.tsv'):
        key = (row['task'], row['method'], row.get('seed', ''))
        if key in metrics:
            raise ValueError(f'Metric trùng method/seed: {key}')
        metrics[key] = row
    counts = defaultdict(Counter)
    ids = defaultdict(set)
    cohorts = defaultdict(dict)
    for row in rows(directory / 'predictions.tsv'):
        key = (row['task'], row['method'], row.get('seed', ''))
        if key not in metrics or row['id'] in ids[key]:
            raise ValueError(f'Prediction thiếu metric hoặc trùng ID: {key}')
        task = key[0]
        if cohorts[task].setdefault(row['id'], row['truth']) != row['truth']:
            raise ValueError(f'Nhãn thật thay đổi giữa các đối chứng: {task}')
        ids[key].add(row['id'])
        counts[key][row['truth'], row['predicted']] += 1
    if set(counts) != set(metrics):
        raise ValueError('Thiếu dự đoán cho method/seed')
    for key, confusion in counts.items():
        row = metrics[key]
        size = sum(confusion.values())
        if size != int(row['samples']) or size != run['samples'] or ids[key] != set(cohorts[key[0]]):
            raise ValueError(f'Sai số mẫu/cohort: {key}')
        classes = run.get('classes')
        if isinstance(classes, dict):
            classes = classes['fine']
        count = len(classes) if classes else {'2way': 2, '6way': 6}[key[0]]
        labels = {str(i) for i in range(count)}
        if any(a not in labels or b not in labels for a, b in confusion):
            raise ValueError(f'Nhãn nằm ngoài taxonomy: {key}')
        support = Counter()
        predicted = Counter()
        for (a, b), n in confusion.items():
            support[a] += n
            predicted[b] += n
        close(row['accuracy'], sum(confusion[c, c] for c in labels) / size, f'{key} accuracy')
        f1 = [2 * confusion[c, c] / (support[c] + predicted[c])
              if support[c] + predicted[c] else 0. for c in labels]
        recall = [confusion[c, c] / support[c] for c in labels if support[c]]
        close(row['macro_f1'], statistics.mean(f1), f'{key} macro_f1')
        close(row['balanced_accuracy'], statistics.mean(recall), f'{key} balanced_accuracy')
    grouped = defaultdict(list)
    for key, row in metrics.items():
        grouped[key[:2]].append(row)
    summary = list(rows(directory / 'summary.tsv'))
    for table in [summary, run['summary']]:
        if len(table) != len(grouped) or len({(r['task'], r['method']) for r in table}) != len(grouped):
            raise ValueError('Summary thiếu hoặc trùng method')
        for row in table:
            group = grouped[row['task'], row['method']]
            if int(row['seeds']) != len(group) or int(row['samples']) != run['samples'] or int(row['std_ddof']) != 1:
                raise ValueError('Summary sai số seed/mẫu/ddof')
            for field in ['accuracy', 'macro_f1', 'balanced_accuracy', 'top5']:
                if field not in group[0]:
                    continue
                values = [float(r[field]) for r in group]
                close(row[field + '_mean'], statistics.mean(values), field + ' mean')
                close(row[field + '_std'], statistics.stdev(values) if len(values) > 1 else 0., field + ' std')
    return {'path': str(directory.relative_to(ROOT)), 'kind': run['kind'],
            'mode': run['mode'], 'samples': run['samples'], 'method_seed_groups': len(metrics),
            'prediction_rows': sum(sum(c.values()) for c in counts.values()),
            'artifacts_verified': len(run['output_sha256']), 'status': 'passed', 'summary': run['summary']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='docs/results_audit_20261116.json')
    args = parser.parse_args()
    results = []
    for file in sorted((ROOT / 'results/final').rglob('run.json')):
        directory = file.parent
        try:
            result = audit(directory)
        except (ValueError, KeyError, OSError, TypeError) as error:
            result = {'path': str(directory.relative_to(ROOT)), 'status': 'failed', 'error': str(error)}
        results.append(result)
        print(result['status'], result['path'], result.get('error', ''), flush=True)
    failures = sum(r['status'] != 'passed' for r in results)
    payload = {'status': 'failed' if failures or not results else 'passed', 'runs': len(results),
               'failures': failures, 'results': results,
               'scope': 'Checksum completion/output; streamed predictions: IDs/cohorts, accuracy/macro-F1/BA; seed mean/sample std. No model/dataset/feature replay. Top5 aggregation only. Historical archives: separate move checksums.'}
    write_json(path(args.output), payload)
    return 1 if payload['status'] != 'passed' else 0


if __name__ == '__main__':
    raise SystemExit(main())
