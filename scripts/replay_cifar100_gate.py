"""Exploratory uncertainty cascade on saved dev probabilities; no model or test."""
import argparse
from pathlib import Path
import sys
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from vlm_project.evaluate import metric, table
from vlm_project.io import path, sha256, write_json
from vlm_project.protocol import completed_run

THRESHOLDS = (0.5, 0.7, 0.9)


def candidates(names, descriptions):
    names, descriptions = np.asarray(names), np.asarray(descriptions)
    if names.ndim != 2 or names.shape != descriptions.shape or names.shape[1] != 100:
        raise ValueError('Expected aligned N x 100 probabilities')
    for scores in (names, descriptions):
        if not np.isfinite(scores).all() or np.any(scores < 0) or not np.allclose(scores.sum(1), 1):
            raise ValueError('Invalid probabilities')
    yield 'flat_names', names, np.zeros(len(names), dtype=bool)
    yield 'flat_descriptions', descriptions, np.ones(len(names), dtype=bool)
    yield 'equal_probability_fusion', (names + descriptions) / 2, np.ones(len(names), dtype=bool)
    for threshold in THRESHOLDS:
        # Ground truth is not accepted by this function or used by the gate.
        route = names.max(1) < threshold
        yield f'cascade_names_confidence_{threshold:g}', np.where(route[:, None], descriptions, names), route


def replay(directory, output_root='results/final/cifar100_gate_replay'):
    directory = path(directory)
    source = completed_run(directory)
    if source.get('kind') != 'cifar100_rerank' or source.get('split') != 'dev' or source.get('mode') != 'full':
        raise ValueError('Requires completed full rerank dev; no test/smoke replay')
    with np.load(directory / 'replay_scores.npz', allow_pickle=False) as archive:
        names = archive['flat_names']
        descriptions = archive['flat_descriptions']
        labels, ids = archive['labels'], archive['ids']
    if len(labels) != 10000 or len(np.unique(ids)) != 10000 or source['samples'] != 10000:
        raise ValueError('Expected the complete 10000-sample dev cohort')
    output = path(output_root) / uuid.uuid4().hex[:12]
    output.mkdir(parents=True)
    baseline = descriptions.argmax(1) == labels
    rows, predictions = [], []
    for name, scores, route in candidates(names, descriptions):
        predicted = scores.argmax(1)
        correct = predicted == labels
        values = metric(labels, predicted, list(range(100)), scores)
        values.pop('per_class')
        rows.append({'method': name, **values, 'description_branch_fraction': float(route.mean()),
                     'accuracy_delta_vs_flat_descriptions': float(correct.mean() - baseline.mean()),
                     'corrected_vs_flat_descriptions': int((correct & ~baseline).sum()),
                     'harmed_vs_flat_descriptions': int((~correct & baseline).sum())})
        predictions.extend({'id': int(i), 'truth': int(y), 'method': name, 'predicted': int(p),
                            'description_branch': bool(r)} for i, y, p, r in zip(ids, labels, predicted, route))
        print(f'{name}: accuracy={values["accuracy"]:.4%}, macro-F1={values["macro_f1"]:.4%}, descriptions={route.mean():.1%}')
    table(output / 'summary.tsv', rows)
    table(output / 'predictions.tsv', predictions)
    write_json(output / 'analysis.json', {'status': 'complete', 'scope': 'exploratory dev replay; no independent confirmation',
               'source_run_sha256': sha256(directory / 'run.json'), 'script_sha256': sha256(Path(__file__)),
               'thresholds': list(THRESHOLDS), 'samples': len(labels), 'summary': rows,
               'limitations': ['Confidence is uncalibrated at the original logit scale 100',
                               'No automatic winner selection or test configuration',
                               'Replay does not measure GPU savings; both branches already computed']})
    print('Output:', output)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--output-root', default='results/final/cifar100_gate_replay')
    args = parser.parse_args()
    replay(args.run, args.output_root)
