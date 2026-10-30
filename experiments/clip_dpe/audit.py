"""Audit artifact DPE/fusion: code/hash, từng ảnh, cache, metric và reference."""
import argparse
import ast
import csv
import json
import math
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import softmax, log_softmax
from experiments.clip_tda.run import load_features
from experiments.clip_zlap.run import project_path, validate_majority, majority_gates, evaluate
from experiments.clip_dmn.run import load_companions
from experiments.clip_transductive.ensemble import humanitarian
from experiments.clip_transductive.run import sha256, write_json
from experiments.clip_dpe.run import average
from experiments.clip_dpe.fusion import FROLIC_COMMIT


def close(a, b):
    if a is None or b is None:
        assert a is b; return
    if isinstance(a, dict):
        assert set(a) == set(b)
        for k in a: close(a[k], b[k])
    else:
        np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-10)


def base(directory):
    m = validate_majority(directory)
    for name, digest in m['code_sha256'].items():
        assert sha256((ROOT/name).read_bytes()) == digest, name
    for name, digest in m['input_manifest_sha256'].items():
        path = project_path(m['settings'][name]) if name in m['settings'] else Path(name)
        assert sha256((path/'run_manifest.json').read_bytes()) == digest
    return m


def cache_audit(z, diag, prefix, classes):
    order = diag['order']; n = len(order)
    initial, final = z[prefix+'initial_logits'], z['informativeness' if prefix == 'info_' else 'humanitarian']
    np.testing.assert_array_equal(np.sort(order), np.arange(n))
    assert np.isfinite(initial).all() and np.isfinite(final).all()
    loss = -(softmax(initial.astype('float64'), axis=1) * log_softmax(initial.astype('float64'), axis=1)).sum(1)
    np.testing.assert_allclose(loss/math.log2(classes), z[prefix+'cache_entropy'], atol=1e-6, rtol=1e-5)
    post = -(softmax(final.astype('float64'), axis=1) * log_softmax(final.astype('float64'), axis=1)).sum(1)
    # avg_entropy tác giả trừ logsumexp ở FP32; so với entropy FP64 có sai
    # số triệt tiêu cỡ ULP của logits, kể cả khi entropy gần zero.
    entropy_bound = 2*np.finfo(np.float32).eps * max(1., float(abs(final).max()))
    assert float(abs(post-z[prefix+'post_entropy']).max()) <= entropy_bound
    updates = z[prefix+'post_entropy']/np.float32(math.log2(classes)) < .1
    np.testing.assert_array_equal(updates, z[prefix+'text_update'])
    assert int(updates.sum()) == diag['text_updates']
    memory = {c: [] for c in range(classes)}
    for i in order:
        c = int(z[prefix+'pseudo_label'][i]); q = float(z[prefix+'cache_entropy'][i])
        assert initial[i, c] == initial[i].max()
        accepted, replaced = False, -1
        if len(memory[c]) < 3:
            memory[c].append((q, i)); accepted = True
        elif q < memory[c][-1][0]:
            replaced = memory[c][-1][1]; memory[c][-1] = (q, i); accepted = True
        memory[c].sort(key=lambda v: v[0])
        assert accepted == bool(z[prefix+'memory_write'][i])
        assert replaced == int(z[prefix+'replaced_index'][i])
    assert [[i for q, i in memory[c]] for c in range(classes)] == diag['memory_indices']
    assert [len(memory[c]) for c in range(classes)] == diag['memory_counts']
    assert np.isfinite(diag['global_text']).all()


def audit_dpe(directory):
    m = base(directory); settings = m['settings']
    reference = json.loads((ROOT/'experiments/clip_dpe/reference_verification.json').read_text())
    assert reference['status'] == 'passed'
    assert reference['local_core_sha256'] == sha256((ROOT/'experiments/clip_dpe/core.py').read_bytes())
    actual = [v for v in reference['cases'] if v['scope'] in ['train', 'dev']]
    assert len(actual) == 12 and all(v['all_predictions_identical'] for v in actual)
    analysis = json.loads((directory/'analysis.json').read_text())
    md = project_path(settings['majority_run']); mm = validate_majority(md); checks = []
    for scope in ['train', 'dev']:
        source = load_features(project_path(settings[scope+'_run']))
        assert source['split'] == scope and source['feature_sha256'] == m['source_features'][scope]['sha256']
        ids = [r['id'] for r in source['rows']]; hl, il = source['labels'], source['info_labels']
        gates = majority_gates(source, scope, md, mm); dn, inmap = load_companions(source, scope, settings)
        for seed in settings['seeds']:
            folder = directory/scope/f'shuffle_{seed}'
            with (folder/'predictions.tsv').open() as f: rows = list(csv.DictReader(f, delimiter='\t'))
            assert [r['id'] for r in rows] == ids
            assert [r['y_true'] for r in rows] == [r['y_true'] for r in source['rows']]
            expected_truth = source['binary_truth'] if source['binary_truth'] is not None else ['']*len(ids)
            assert [r['binary_truth'] for r in rows] == expected_truth
            diagnostics = json.loads((folder/'diagnostics.json').read_text())
            with np.load(folder/'logits.npz', allow_pickle=False) as z:
                assert z['ids'].tolist() == ids
                cache_audit(z, diagnostics['informativeness'], 'info_', 2)
                cache_audit(z, diagnostics['humanitarian'], 'human_', 8)
                order = np.random.default_rng(seed).permutation(len(ids))
                assert diagnostics['informativeness']['order'] == order.tolist()
                assert diagnostics['humanitarian']['order'] == order.tolist()
                np.testing.assert_array_equal([int(r['stream_position']) for r in rows], np.argsort(order))
                root = np.asarray(il)[z['informativeness'].argmax(1)] == 'informative'
                for name, reports in analysis[scope]['by_method'].items():
                    gate = gates[seed] if name.startswith('majority') else inmap if name.startswith('inmap') else root
                    leaf = source['scores'][1] if name.endswith('raw_leaf') else dn if name.endswith('dn_leaf') or name.endswith('dn_control') else z['humanitarian']
                    pred = humanitarian(gate, leaf, hl)
                    assert pred == [r[name] for r in rows]
                    expected = evaluate(source, pred, np.where(gate, 'informative', 'not_informative').tolist())
                    close(expected, {k: v for k, v in reports[seed].items() if k != 'seed'})
            checks.append({'split': scope, 'seed': seed, 'images': len(ids), 'all_cache_events_predictions_metrics_verified': True})
        for name, reports in analysis[scope]['by_method'].items(): close(analysis[scope]['mean'][name], average(reports))
    result = {'status': 'passed', 'checks': checks, 'reference_actual_streams': 12,
              'reference_predictions_and_final_memory_identical': True, 'no_raw_processed_or_test_access': True}
    write_json(directory/'audit.json', result); return result


def frolic_functions(folder):
    folder = Path(folder)
    assert subprocess.check_output(['git', '-C', str(folder), 'rev-parse', 'HEAD'], text=True).strip() == FROLIC_COMMIT
    subprocess.run(['git', '-C', str(folder), 'diff', '--exit-code', 'HEAD', '--', 'main_zeroshot.py'], check=True, capture_output=True)
    tree = ast.parse((folder/'main_zeroshot.py').read_text())
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ['compute_confidence', 'search_T']]
    assert len(selected) == 2
    ns = {'F': F, 'torch': torch}
    exec(compile(ast.Module(body=selected, type_ignores=[]), '<Frolic confidence functions>', 'exec'), ns)
    return ns


def audit_fusion(directory, frolic_dir):
    m = base(directory); source_dir = Path(m['settings']['source_dpe']); dm = base(source_dir)
    settings = dm['settings']; analysis = json.loads((directory/'analysis.json').read_text())
    ref = frolic_functions(frolic_dir); checks = []
    md = project_path(settings['majority_run']); mm = validate_majority(md)
    previous = torch.get_num_threads(); torch.set_num_threads(1)
    try:
        for scope in ['train', 'dev']:
            source = load_features(project_path(settings[scope+'_run'])); assert source['split'] == scope
            ids = [r['id'] for r in source['rows']]; hl = source['labels']
            gates = majority_gates(source, scope, md, mm); dn, inmap = load_companions(source, scope, settings)
            for seed in [0, 1, 2]:
                with np.load(source_dir/scope/f'shuffle_{seed}/logits.npz') as z: dpe = z['humanitarian'].copy()
                anchor, other = torch.tensor(100.*dn), torch.tensor(dpe)
                target = ref['compute_confidence'](anchor, 1.)
                temperature = ref['search_T'](other, 1., target)
                diag = analysis[scope]['confidence_matching'][seed]
                assert temperature == diag['temperature']
                equal = (anchor+other).numpy(); mixed = (anchor+other/temperature).numpy()
                with np.load(directory/scope/f'logits_seed_{seed}.npz') as z:
                    assert z['ids'].tolist() == ids
                    np.testing.assert_array_equal(z['equal'], equal); np.testing.assert_array_equal(z['matched'], mixed)
                with (directory/scope/f'predictions_seed_{seed}.tsv').open() as f: rows = list(csv.DictReader(f, delimiter='\t'))
                assert [r['id'] for r in rows] == ids
                assert [r['y_true'] for r in rows] == [r['y_true'] for r in source['rows']]
                assert [r['binary_truth'] for r in rows] == (source['binary_truth'] if source['binary_truth'] is not None else ['']*len(ids))
                for name, reports in analysis[scope]['by_method'].items():
                    gate = gates[seed] if name.startswith('majority') else inmap
                    leaf = dn if name.endswith('control') else equal if name.endswith('equal') else mixed
                    pred = humanitarian(gate, leaf, hl); assert pred == [r[name] for r in rows]
                    close(evaluate(source, pred, np.where(gate, 'informative', 'not_informative').tolist()),
                          {k: v for k, v in reports[seed].items() if k != 'seed'})
                checks.append({'split': scope, 'seed': seed, 'images': len(ids), 'temperature': temperature,
                               'official_frolic_search_and_all_predictions_metrics_verified': True})
            for name, reports in analysis[scope]['by_method'].items(): close(analysis[scope]['mean'][name], average(reports))
        result = {'status': 'passed', 'checks': checks, 'frolic_reference_commit': FROLIC_COMMIT,
                  'scope': 'Confidence matching operator only; not full Frolic', 'no_raw_processed_or_test_access': True}
        write_json(directory/'audit.json', result); return result
    finally:
        torch.set_num_threads(previous)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--dpe-run', required=True)
    p.add_argument('--fusion-run'); p.add_argument('--frolic-dir'); a = p.parse_args()
    print('DPE', audit_dpe(project_path(a.dpe_run))['status'])
    if a.fusion_run:
        if not a.frolic_dir: p.error('--fusion-run cần --frolic-dir')
        print('Fusion', audit_fusion(project_path(a.fusion_run), a.frolic_dir)['status'])
