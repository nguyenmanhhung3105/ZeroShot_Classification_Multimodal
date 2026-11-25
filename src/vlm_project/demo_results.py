"""Offline export of audited results for the local report UI; stdlib only.

Reads explicit completed-run tables, never datasets, feature caches or model weights.
"""
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

METRICS = ('accuracy', 'macro_f1', 'balanced_accuracy', 'top5')
FAKEDDIT_SIX = ['Real', 'Satire/Parody', 'False Connection', 'Imposter Content',
                'Manipulated Content', 'Misleading Content']
OPTIONAL_TABLES = ('routing.tsv', 'gate_metrics.tsv', 'comparisons_summary.tsv',
                   'thresholds.tsv', 'precision_recall_selection.tsv', 'subtype_summary.tsv')


def sha(file):
    digest = hashlib.sha256()
    with Path(file).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def checked(file, expected):
    if not expected or sha(file) != expected:
        raise ValueError(f'Checksum không khớp: {file}')
    return Path(file)


def contained(root, relative):
    file = (root / relative).resolve()
    if not file.is_relative_to(root.resolve()):
        raise ValueError('Artifact nằm ngoài project root')
    return file


def read_json(file):
    return json.loads(Path(file).read_text())


def table(file):
    with Path(file).open(newline='') as stream:
        yield from csv.DictReader(stream, delimiter='\t')


def number(value):
    x = float(value)
    if not math.isfinite(x):
        raise ValueError('Metric không hữu hạn')
    return x


def stats(values):
    return {'mean': statistics.mean(values), 'std': statistics.stdev(values) if len(values) > 1 else 0.}


def close(actual, expected):
    if not math.isclose(number(actual), number(expected), rel_tol=1e-8, abs_tol=1e-9):
        raise ValueError(f'Metric không khớp: {actual} != {expected}')


def numeric_table(file):
    result = []
    for raw in table(file):
        row = {}
        for k, v in raw.items():
            if v is None or v == '': row[k] = None
            else:
                try: row[k] = number(v)
                except ValueError: row[k] = v
        result.append(row)
    return result


def display_labels(run, task, raw_labels):
    if run.get('dataset', '').startswith('fakeddit'):
        if task == '6way': return {str(i): label for i, label in enumerate(FAKEDDIT_SIX)}
        fewshot = 'fewshot' in run['kind']
        return {'0': 'Real' if fewshot else 'Fake', '1': 'Fake' if fewshot else 'Real'}
    classes = run.get('classes', [])
    if isinstance(classes, dict): classes = classes.get('fine', [])
    return {label: str(classes[int(label)]) if label.isdigit() and int(label) < len(classes) else label
            for label in raw_labels}


def variant_metrics(counts, labels):
    support, predicted = Counter(), Counter()
    for (a, b), n in counts.items(): support[a] += n; predicted[b] += n
    per_class = {}
    for label in labels:
        tp = counts[label, label]
        per_class[label] = {'precision': tp / predicted[label] if predicted[label] else 0.,
                            'recall': tp / support[label] if support[label] else 0.,
                            'f1': 2 * tp / (support[label] + predicted[label]) if support[label] + predicted[label] else 0.,
                            'support': support[label]}
    total = sum(counts.values())
    return {'accuracy': sum(counts[x, x] for x in labels) / total,
            'macro_f1': statistics.mean(x['f1'] for x in per_class.values()),
            'balanced_accuracy': statistics.mean(x['recall'] for x in per_class.values() if x['support'])}, per_class


def probability_curve(pairs):
    """Tie-aware ROC-AUC and average precision from all saved binary scores."""
    positive = sum(bool(truth) for _, truth in pairs)
    negative = len(pairs) - positive
    if not positive or not negative:
        raise ValueError('ROC/PR cần cả hai lớp thật')
    groups = defaultdict(lambda: [0, 0])
    for score, truth in pairs:
        if not 0 <= score <= 1: raise ValueError('p_fake nằm ngoài [0,1]')
        groups[score][0 if truth else 1] += 1
    points = [{'fpr': 0., 'tpr': 0., 'recall': 0., 'precision': 1., 'threshold': None}]
    tp = fp = 0
    area = ap = 0.
    for threshold, (p, n) in sorted(groups.items(), reverse=True):
        tp += p; fp += n
        point = {'fpr': fp / negative, 'tpr': tp / positive, 'recall': tp / positive,
                 'precision': tp / (tp + fp), 'threshold': threshold}
        previous = points[-1]
        area += (point['fpr'] - previous['fpr']) * (point['tpr'] + previous['tpr']) / 2
        ap += (point['recall'] - previous['recall']) * point['precision']
        points.append(point)
    count = len(points)
    indices = sorted({round(i * (count - 1) / min(200, count - 1)) for i in range(min(201, count))})
    return {'auc': area, 'average_precision': ap, 'source_points': count, 'points': [points[i] for i in indices]}


def compile_run(root, entry):
    source = contained(root, entry['source_run'])
    run_file = checked(source / 'run.json', entry['source_run_sha256'])
    run = read_json(run_file)
    completed = read_json(source / 'completed.json')
    if (completed.get('status') != 'complete' or completed.get('run_sha256') != sha(run_file)
            or run.get('status') != 'complete' or run.get('mode') != 'full'):
        raise ValueError('Chỉ xuất completed full run')
    published = checked(contained(root, entry['summary']), entry['summary_sha256'])

    def artifact(name):
        return checked(source / name, run.get('output_sha256', {}).get(name))

    summaries = list(table(artifact('summary.tsv')))
    if sha(published) != sha(source / 'summary.tsv'):
        raise ValueError('Published summary khác bảng nguồn')
    metrics = {(r['task'], r['method'], r.get('seed', '')): r for r in table(artifact('metrics_by_seed.tsv'))}
    counts, task_labels, probabilities = defaultdict(Counter), defaultdict(set), defaultdict(list)
    for row in table(artifact('predictions.tsv')):
        key = row['task'], row['method'], row.get('seed', '')
        if key not in metrics:
            raise ValueError('Prediction không có metric tương ứng')
        counts[key][row['truth'], row['predicted']] += 1
        task_labels[row['task']].update((row['truth'], row['predicted']))
        if row['task'] == '2way' and row.get('p_fake') not in ('', None):
            fake_code = '1' if 'fewshot' in run['kind'] else '0'
            probabilities[key].append((number(row['p_fake']), row['truth'] == fake_code))
    if set(counts) != set(metrics):
        raise ValueError('Thiếu prediction theo method/seed')
    for task in task_labels:
        classes = run.get('classes', [])
        if isinstance(classes, dict): classes = classes.get('fine', [])
        size = 2 if task == '2way' else 6 if task == '6way' else len(classes)
        task_labels[task].update(str(i) for i in range(size))
    groups = defaultdict(list)
    for key, cm in counts.items():
        metric = metrics[key]
        if sum(cm.values()) != int(metric['samples']) or int(metric['samples']) != run['samples']:
            raise ValueError('Sai số prediction/cohort')
        calculated, per_class = variant_metrics(cm, task_labels[key[0]])
        for name, value in calculated.items(): close(value, metric[name])
        groups[key[:2]].append((key[2], metric, cm, per_class))
    result = []
    for summary in summaries:
        key = summary['task'], summary['method']
        variants = groups[key]
        if int(summary['seeds']) != len(variants) or int(summary['std_ddof']) != 1:
            raise ValueError('Sai số seed hoặc quy ước std')
        raw_labels = sorted(task_labels[key[0]], key=lambda s: (0, int(s)) if s.isdigit() else (1, s))
        names = display_labels(run, key[0], raw_labels)
        order = sorted(raw_labels, key=lambda s: ['Real', 'Fake'].index(names[s])) if key[0] == '2way' else raw_labels
        row = {'key': f'{key[0]}:{key[1]}', 'task': key[0], 'method': key[1], 'samples': int(summary['samples']),
               'seeds': len(variants), 'labels': [names[x] for x in order], 'metrics': {}, 'seed_metrics': []}
        for name in METRICS:
            if name + '_mean' not in summary or summary[name + '_mean'] == '': continue
            values = [number(v[1][name]) for v in variants]
            agg = stats(values)
            close(agg['mean'], summary[name + '_mean']); close(agg['std'], summary[name + '_std'])
            row['metrics'][name] = agg
        for seed, metric, _, _ in variants:
            row['seed_metrics'].append({'seed': seed or 'deterministic',
                                       **{name: number(metric[name]) for name in METRICS if metric.get(name) not in (None, '')}})
        row['per_class'] = [{'label': names[label], 'support': statistics.mean(v[3][label]['support'] for v in variants),
                             **{name: stats([v[3][label][name] for v in variants]) for name in ('precision', 'recall', 'f1')}}
                            for label in order]
        combined = Counter()
        for _, _, cm, _ in variants: combined.update(cm)
        if len(order) <= 20:
            row['confusion'] = [[combined[a, b] / len(variants) for b in order] for a in order]
            row['seed_confusions'] = [{'seed': seed or 'deterministic',
                                       'matrix': [[cm[a, b] for b in order] for a in order]} for seed, _, cm, _ in variants]
        row['confusions'] = [{'truth': names[a], 'predicted': names[b], 'count': n / len(variants)}
                             for (a, b), n in combined.most_common() if a != b][:30]
        if all((key[0], key[1], seed) in probabilities for seed, _, _, _ in variants):
            curves = [{'seed': seed or 'deterministic', **probability_curve(probabilities[key[0], key[1], seed])}
                      for seed, _, _, _ in variants]
            row['roc'] = {'curves': curves, 'auc': stats([v['auc'] for v in curves]),
                          'average_precision': stats([v['average_precision'] for v in curves])}
        result.append(row)
    cohort = run.get('dev_cohort', run.get('cohort', {}))
    split = run.get('split') or cohort.get('split') or cohort.get('source', {}).get('split') or 'Theo metadata run'
    identifier = hashlib.sha256(entry['source_run'].encode()).hexdigest()[:16]
    data = {'id': identifier, 'kind': run['kind'], 'dataset': run.get('dataset', summaries[0]['task']),
            'model': run['model'], 'split': split, 'samples': run['samples'],
            'cohort': cohort.get('cohort_sha256') or cohort.get('cohort_id') or run.get('split_sha256') or entry['source_run_sha256'],
            'scope': run.get('scope', ''), 'source': entry['source_run'], 'source_run_sha256': sha(run_file),
            'sources': [{'run': entry['source_run'], 'sha256': sha(run_file)}], 'methods': result,
            'summary_sha256': sha(published), 'artifacts': {}, 'diagnostics': {}}
    for name in OPTIONAL_TABLES:
        if name not in run.get('output_sha256', {}): continue
        rows = numeric_table(artifact(name))
        if name == 'precision_recall_selection.tsv':
            curves = defaultdict(list)
            for row in rows: curves[row['policy'], row['modality'], row['seed']].append(row)
            data['artifacts'][name] = []
            for (policy, modality, seed), points in curves.items():
                points.sort(key=lambda p: p['recall_fake'])
                indices = sorted({round(i * (len(points)-1) / min(200, len(points)-1)) for i in range(min(201, len(points)))}) if len(points) > 1 else [0]
                data['artifacts'][name].append({'policy': policy, 'modality': modality, 'seed': seed,
                                                'source_points': len(points), 'points': [points[i] for i in indices]})
        else: data['artifacts'][name] = rows
    for name in ('branch_diagnostics.json', 'oracle_diagnostic.json'):
        if name in run.get('output_sha256', {}): data['diagnostics'][name] = read_json(artifact(name))
    if 'support.json' in run.get('output_sha256', {}):
        support = read_json(artifact('support.json'))
        budgets = []
        for item in support if isinstance(support, list) else []:
            labels = item.get('labels_6way', [])
            quotas = item.get('quotas_6way')
            if labels:
                counts = Counter(str(v) for v in labels)
                quotas = [counts[str(i)] for i in range(6)]
            if isinstance(quotas, list) and len(quotas) == 6:
                budgets.append({'policy': item.get('policy', 'legacy_6way'), 'seed': item['seed'],
                                'counts': quotas, 'labels': FAKEDDIT_SIX})
        if budgets: data['support_budget'] = budgets
    data['verified_artifacts'] = {name: value for name, value in run['output_sha256'].items()
                                  if name in ('summary.tsv', 'metrics_by_seed.tsv', 'predictions.tsv', *OPTIONAL_TABLES,
                                              'branch_diagnostics.json', 'oracle_diagnostic.json', 'support.json')}
    return data


def combine(runs):
    first = runs[0]
    if any(not first.get(k) or any(r.get(k) != first[k] for r in runs) for k in ('cohort', 'model', 'samples')):
        raise ValueError('Không gộp run khác cohort/model/số mẫu')
    data = {**first, 'id': 'fakeddit-binary-comparison', 'kind': 'fakeddit_comparison',
            'source': 'Các run cùng cohort — xem nguồn từng phương pháp', 'sources': [],
            'methods': [], 'artifacts': {}, 'diagnostics': {}}
    for run in runs:
        data['sources'].extend(run['sources'])
        if run.get('support_budget'): data['support_budget'] = run['support_budget']
        for row in run['methods']:
            data['methods'].append({**row, 'key': run['id'] + ':' + row['key'], 'source': run['source'],
                                    'regime': 'Few-shot' if 'fewshot' in run['kind'] else 'Zero-shot / thích nghi',
                                    'seed_type': 'Support seed' if 'fewshot' in run['kind'] else 'Grouping seed hoặc tất định'})
    data['scope'] = 'Đối chiếu cùng dev/cohort/backbone; zero-shot không dùng nhãn train, few-shot dùng support có nhãn. Std giữa seed không phải khoảng tin cậy.'
    data['verified_artifacts'] = {run['source']: run['verified_artifacts'] for run in runs}
    return data


def export(root, manifest_file, destination):
    root, destination = Path(root).resolve(), Path(destination)
    manifest = read_json(manifest_file)
    checked(contained(root, manifest['audit']), manifest['audit_sha256'])
    if read_json(contained(root, manifest['audit'])).get('status') != 'passed':
        raise ValueError('Audit nguồn chưa passed')
    runs = []
    for entry in manifest['runs']:
        if entry['mode'] != 'full': continue
        if entry.get('audit_status') != 'passed': raise ValueError('Run nguồn chưa passed audit')
        runs.append(compile_run(root, entry))
    support = next((r for r in runs if r['kind'] == 'fakeddit_fewshot_support'), None)
    baseline = next((r for r in runs if r['kind'] == 'fakeddit_task1' and support and r['cohort'] == support['cohort']), None)
    if support and baseline: runs.insert(0, combine([baseline, support]))
    directory = destination / 'experiments'; directory.mkdir(parents=True, exist_ok=True)
    index = {'version': 1, 'audit': manifest['audit'], 'audit_sha256': manifest['audit_sha256'],
             'manifest_sha256': sha(manifest_file), 'exported_from': manifest.get('exported_on'),
             'default': runs[0]['id'] if runs else None, 'runs': []}
    for run in runs:
        file = directory / (run['id'] + '.json')
        content = json.dumps(run, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
        file.write_text(content)
        index['runs'].append({k: run[k] for k in ('id', 'kind', 'dataset', 'model', 'split', 'samples', 'cohort', 'source')})
        index['runs'][-1].update(file='experiments/' + file.name, sha256=sha(file), methods=len(run['methods']))
    (destination / 'experiment-results.json').write_text(json.dumps(index, ensure_ascii=False, indent=2))
    return index
