"""Audit saved artifacts only; optionally replay EM/StatA without image inference."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
import numpy as np
from scipy.special import softmax
from sklearn.metrics import confusion_matrix
from experiments.fakeddit_calip.prepare import load_reference, ASSET
from experiments.fakeddit_calip.core import hierarchy, majority
from experiments.waffle_lab.run import file_hash, write_json


def metrics(truth, pred, labels):
    cm = confusion_matrix(truth, pred, labels=labels)
    tp = cm.diagonal(); den = cm.sum(0)+cm.sum(1)
    f1 = np.divide(2*tp, den, out=np.zeros(len(labels), dtype=float), where=den != 0)
    return {'accuracy': float(tp.sum()/cm.sum()), 'macro_f1': float(f1.mean()),
            'f1': f1.tolist(), 'confusion': cm.tolist()}


def analyze(source, replay=False):
    source = Path(source)
    ref = load_reference(ROOT/ASSET); payload = ref['payload']
    ties = []; offline = {}; predictions = {}; errors = {}; checked = 0
    with zipfile.ZipFile(source/'run_artifacts.zip') as z:
        manifest = json.loads(z.read('run_manifest.json')); n = manifest['samples']
        assert manifest['status'] == 'complete' and n == 3000
        assert manifest['reference_sha256'] == ref['sha256']
        for name, expected in manifest['output_sha256'].items():
            assert hashlib.sha256(z.read(name)).hexdigest() == expected, name
        assert (source/'summary.tsv').read_bytes() == z.read('summary.tsv')
        for name, expected in manifest['code_sha256'].items():
            assert file_hash(ROOT/name) == expected, name
        assert json.loads(z.read('frozen_prompts.json')) == {t: p['banks'] for t, p in payload['tasks'].items()}
        cohort = json.loads(z.read('cohort.json'))
        assert len(cohort) == n and len({r['id'] for r in cohort}) == n
        extraction = json.loads(z.read('extraction.json'))
        assert len(extraction) == 375 and extraction[0]['global_reference_max_abs_error'] == 0.
        assert all(e['patches'] == 256 and e['embedding_dim'] == 1024 and e['same_encoder_forward']
                   and e['output_tokens_restored'] for e in extraction)
        analysis = json.loads(z.read('analysis.json'))
        summary = list(csv.DictReader(io.StringIO(z.read('summary.tsv').decode()), delimiter='\t'))
        with np.load(io.BytesIO(z.read('scores.npz')), allow_pickle=False) as pack:
            arrays = {k: pack[k] for k in pack.files}
        x = arrays['image_features']
        assert x.shape == (n, 1024) and np.isfinite(x).all()
        np.testing.assert_allclose(np.linalg.norm(x, axis=1), 1, atol=1e-5)

        def check_argmax(task, method, scores, saved):
            labels = payload['tasks'][task]['labels']
            saved_indices = np.array([labels.index(p) for p in saved])
            bad = np.flatnonzero(scores.argmax(1) != saved_indices)
            for i in bad:
                gap = float(scores[i].max()-scores[i, saved_indices[i]])
                # Float32 BLAS/platform differences may flip essentially tied scores.
                assert gap <= 1e-6, (task, method, i, gap)
                ties.append({'task': task, 'method': method, 'id': arrays['ids'][i].item(), 'score_gap': gap})

        for task, plan in payload['tasks'].items():
            rows = list(csv.DictReader(io.StringIO(z.read(f'predictions_{task}.tsv').decode()), delimiter='\t'))
            assert [r['id'] for r in rows] == [r['id'] for r in plan['rows']] == arrays['ids'].tolist() == [r['id'] for r in cohort]
            truth = [r['label'] for r in plan['rows']]
            assert [r['y_true'] for r in rows] == truth
            labels = plan['labels']
            pred = {name: np.array([r[name] for r in rows]) for name in rows[0] if name not in ['id', 'y_true']}
            predictions[task] = pred
            for name, values in pred.items():
                assert set(values) <= set(labels)
                report = metrics(truth, values, labels)
                saved = analysis[task][name]
                for key in ['accuracy', 'macro_f1']:
                    np.testing.assert_allclose(report[key], saved[key], atol=1e-14)
                np.testing.assert_allclose(report['f1'], [saved['per_class'][k]['f1'] for k in labels], atol=1e-14)
                checked += 1
            for bank in ['legacy', 'semantic', 'semantic_waffle']:
                means = arrays[task+'_'+bank+'_means']; raw = x@means.T
                unit = x@(means/np.linalg.norm(means, axis=1, keepdims=True)).T
                for suffix, scores in [('mean_cosine', raw), ('unit', unit)]:
                    check_argmax(task, bank+'_'+suffix, scores, pred[bank+'_'+suffix])
                if bank == 'legacy':
                    observed = softmax(plan['logit_scale']*raw, axis=1)
                    expected = np.array([r['baseline_image_prob'] for r in plan['rows']])
                    np.testing.assert_allclose(observed, expected, atol=2e-4, rtol=2e-4)
                    errors[task] = float(abs(observed-expected).max())
            check_argmax(task, 'calip_flat', arrays[task+'_calip'], pred['calip_flat'])
            np.testing.assert_array_equal(pred['cached_multimodal_baseline'], [r['baseline_pred'] for r in plan['rows']])
            assert file_hash(ROOT/plan['baseline_source']) == plan['baseline_zip_sha256']
            prefixes = ['majority'] if task == '2way' else ['majority_unit', 'majority_calip']
            for prefix in prefixes:
                for key in ['accuracy', 'macro_f1']:
                    values = [analysis[task][f'{prefix}_seed_{s}'][key] for s in [0, 1, 2]]
                    for operation in ['mean', 'std']:
                        np.testing.assert_allclose(getattr(np, operation)(values), analysis[task][prefix+'_aggregate'][key][operation], atol=1e-14)
            for row in [r for r in summary if r['task'] == task]:
                for key in ['accuracy', 'macro_f1']:
                    expected = (analysis[task][row['method']][key] if row['method'] in analysis[task]
                                else analysis[task][row['method'][:-5]+'_aggregate'][key]['mean'])
                    np.testing.assert_allclose(float(row[key]), expected, atol=1e-14)

        plan = payload['tasks']['6way']; truth = [r['label'] for r in plan['rows']]; labels = plan['labels']
        for seed in [0, 1, 2]:
            votes = arrays[f'votes_seed_{seed}']; groups = arrays[f'batch_ids_seed_{seed}']
            means = arrays['2way_semantic_means']; norm = x@(means/np.linalg.norm(means, axis=1, keepdims=True)).T
            np.testing.assert_array_equal(votes[:, 0], norm.argmax(1))
            order = np.random.default_rng(seed).permutation(n); expected = np.empty(n, dtype=int)
            for batch, start in enumerate(range(0, n, 75)): expected[order[start:start+75]] = batch
            np.testing.assert_array_equal(groups, expected)
            binary = np.array(payload['tasks']['2way']['labels'])[(votes.sum(1) >= 2).astype(int)]
            np.testing.assert_array_equal(binary, predictions['2way'][f'majority_seed_{seed}'])
            if replay:
                _, replayed = majority(x, means, seed)
                np.testing.assert_array_equal(replayed['votes'], votes)
                print('[REPLAY] EM/StatA seed', seed, 'passed', flush=True)
            for kind, scores in [('calip', arrays['6way_calip']), ('unit', x@(arrays['6way_semantic_waffle_means']/np.linalg.norm(arrays['6way_semantic_waffle_means'], axis=1, keepdims=True)).T)]:
                np.testing.assert_array_equal(hierarchy(binary, scores, labels), predictions['6way'][f'majority_{kind}_seed_{seed}'])
            # Diagnostic controls: same gate, alternative already-encoded leaf banks.
            # Exploratory follow-up after inspecting the original run, not pre-registered.
            for bank in ['legacy', 'semantic', 'semantic_waffle']:
                scores = x@arrays['6way_'+bank+'_means'].T
                pred = hierarchy(binary, scores, labels)
                report = metrics(truth, pred, labels)
                base = predictions['6way']['legacy_mean_cosine']; y = np.asarray(truth)
                report.update(seed=seed, fixed_vs_image_baseline=int(((base != y)&(pred == y)).sum()),
                              broken_vs_image_baseline=int(((base == y)&(pred != y)).sum()))
                offline.setdefault(bank, []).append(report)

    return {'status': 'passed', 'source': str(source), 'samples': n, 'tasks': ['2way', '6way'],
            'prediction_columns_metrics_checked': checked, 'output_hashes_verified': len(manifest['output_sha256']),
            'code_hashes_verified': len(manifest['code_sha256']), 'baseline_probability_max_abs_error': errors,
            'global_reference_max_abs_error': extraction[0]['global_reference_max_abs_error'],
            'em_stata_full_embedding_replay': replay, 'float32_near_ties': ties,
            'offline_exploratory_leaf_controls': offline,
            'offline_mean': {bank: {key: float(np.mean([r[key] for r in reports])) for key in ['accuracy', 'macro_f1']}
                             for bank, reports in offline.items()},
            'limitations': ['No raw/processed dataset or model weights accessed.',
                            'CALIP scores checked for hashes and predictions; patches were not saved for full attention replay.',
                            'Old baseline does not contain image-content hashes.',
                            'Offline leaf controls are exploratory after inspecting this validation result.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source'); parser.add_argument('--replay', action='store_true'); parser.add_argument('--output', required=True)
    args = parser.parse_args()
    report = analyze(args.source, args.replay)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True); write_json(output, report)
    print('[AUDIT] passed; offline controls:', report['offline_mean']); print('[SAVED]', output)
