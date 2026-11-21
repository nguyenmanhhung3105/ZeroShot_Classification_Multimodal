"""Metric và đầu ra thống nhất cho benchmark và Fakeddit."""
import csv
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from .io import write_json, sha256

def metric(truth, predicted, labels, scores=None):
    precision, recall, f1, support = precision_recall_fscore_support(truth, predicted, labels=labels, zero_division=0)
    result = {'accuracy': float(accuracy_score(truth, predicted)), 'macro_f1': float(f1.mean()), 'balanced_accuracy': float(recall[support > 0].mean()), 'per_class': {str(label): {'precision': float(precision[i]), 'recall': float(recall[i]), 'f1': float(f1[i]), 'support': int(support[i])} for i, label in enumerate(labels)}}
    if scores is not None:
        scores = np.asarray(scores)
        if scores.shape != (len(truth), len(labels)) or not np.isfinite(scores).all():
            raise ValueError('Score sai shape hoặc NaN/Inf')
        order = np.argsort(-scores, axis=1, kind='stable')[:, :min(5, len(labels))]
        result['top5'] = float((np.asarray(labels)[order] == np.asarray(truth)[:, None]).any(1).mean())
    return result

def summarize(records):
    """Dùng sample std (ddof=1), không chọn seed tốt nhất."""
    groups = {}
    for row in records:
        groups.setdefault((row['task'], row['method']), []).append(row)
    result = []
    for (task, method), values in groups.items():
        item = {'task': task, 'method': method, 'samples': values[0]['samples'], 'seeds': len(values), 'std_ddof': 1}
        for key in ['accuracy', 'macro_f1', 'balanced_accuracy', 'top5']:
            if key in values[0]:
                x = [r[key] for r in values]
                item[key + '_mean'] = float(np.mean(x))
                item[key + '_std'] = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
        result.append(item)
    return result

def table(file, rows):
    if not rows:
        raise ValueError('Không có kết quả')
    fields = list(dict.fromkeys((k for row in rows for k in row)))
    with file.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)

def publish(output, records, predictions, details, manifest):
    table(output / 'metrics_by_seed.tsv', records)
    table(output / 'predictions.tsv', predictions)
    aggregate = summarize(records)
    table(output / 'summary.tsv', aggregate)
    write_json(output / 'per_class.json', details)
    manifest['output_sha256'] = {p.name: sha256(p) for p in output.iterdir() if p.is_file() and p.name not in ['run.json', 'completed.json']}
    manifest.update(status='complete', summary=aggregate)
    write_json(output / 'run.json', manifest)
    write_json(output / 'completed.json', {'status': 'complete', 'run_sha256': sha256(output / 'run.json')})
    print('Kết quả:', output, flush=True)
    return output
