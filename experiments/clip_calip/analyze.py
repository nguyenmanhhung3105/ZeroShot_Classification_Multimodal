"""Kiểm tra CALIP local/ZIP Kaggle, tính lại từng prediction/metric; không ảnh/model."""
import argparse
import csv
import io
import json
from pathlib import Path
import sys
import zipfile
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
import numpy as np
from experiments.clip_two_stage.analyze import read_artifact
from experiments.clip_transductive.run import metric, sha256
from experiments.clip_transductive.ensemble import humanitarian
from experiments.clip_dpe.run import average
from experiments.clip_calip.core import score, PARAMETERS
from experiments.waffle_lab.run import digest


def binary_artifact(directory, name):
    if (directory/name).is_file(): return (directory/name).read_bytes()
    with zipfile.ZipFile(directory/'run_artifacts.zip') as z: return z.read(name)


def compare(a, b):
    if a is None or b is None: assert a is b
    elif isinstance(a, dict):
        assert set(a) == set(b)
        for key in a: compare(a[key], b[key])
    else: np.testing.assert_allclose(a, b, atol=1e-12, rtol=1e-12)


def analyze(directory, allow_synthetic=False):
    directory = Path(directory)
    js = lambda name: json.loads(read_artifact(directory, name))
    manifest = js('run_manifest.json')
    if (manifest['status'] != 'complete' or manifest['parameters'] != PARAMETERS or
            manifest.get('label_fit') is not False or manifest['source_split'] not in ['train', 'dev']):
        raise ValueError('CALIP artifact chưa complete hoặc sai protocol')
    if manifest.get('inference_backend') != 'metaclip2_real' and not allow_synthetic:
        raise ValueError('Không dùng synthetic/custom backend để báo chất lượng model thật')
    for name, expected in manifest['output_sha256'].items():
        if sha256(binary_artifact(directory, name)) != expected: raise ValueError('Output hash thay đổi: '+name)
    ref = js('reference.json'); payload = ref['payload']
    if digest(payload) != ref['sha256'] or ref['sha256'] != manifest['reference_fingerprint']:
        raise ValueError('Reference fingerprint thay đổi')
    n = manifest['samples']; source = payload['cohorts'][manifest['source_split']]
    il, hl = payload['info_labels'], payload['human_labels']
    analysis = js('analysis.json'); cohorts = [js(task+'/cohort.json')['rows'] for task in ['informativeness', 'humanitarian']]
    y = source['human_truth'][:n]; yb = source['binary_truth'][:n] if source['binary_truth'] is not None else None
    if [r['y_true'] for r in cohorts[1]] != y or [r['y_true'] for r in cohorts[0]] != (yb or [None]*n):
        raise ValueError('GT không khớp annotation nguồn')
    for rows in cohorts:
        assert [r['id'] for r in rows] == source['ids'][:n]
        assert [r['image_sha256'] for r in rows] == source['image_sha256'][:n]
    with np.load(io.BytesIO(binary_artifact(directory, 'scores.npz')), allow_pickle=False) as z:
        assert z['ids'].tolist() == source['ids'][:n]
        for task, labels, mean_key, raw_key in [('info', il, 'info_mean_prototypes', 'info_scores'),
                                               ('human', hl, 'human_mean_prototypes', 'human_scores')]:
            means = np.asarray(payload[mean_key], dtype='float32')
            np.testing.assert_array_equal(z[mean_key], means)
            np.testing.assert_allclose(z['image_features']@means.T, z[task+'_raw'], atol=2e-5, rtol=1e-4)
            np.testing.assert_allclose(z[task+'_raw'], source[raw_key][:n], atol=2e-5, rtol=1e-4)
            np.testing.assert_array_equal(z[task+'_raw'].argmax(1), np.asarray(source[raw_key][:n]).argmax(1))
            terms = z[task+'_terms']; combined = terms[:, 0] + 1.12*terms[:, 1] + .02*terms[:, 2]
            np.testing.assert_array_equal(combined, z[task+'_calip'])
            np.testing.assert_allclose(z[task+'_attention'].astype('float32').sum(1), 1., atol=1e-3, rtol=0)
        with np.load(io.BytesIO(binary_artifact(directory, 'verification_features.npz')), allow_pickle=False) as v:
            for task in ['info', 'human']:
                logits, terms, attention = score(v['global_features'], v['patches'], v[task+'_prototypes'])
                count = len(v['global_features'])
                np.testing.assert_allclose(logits.numpy(), z[task+'_calip'][:count], atol=3e-5, rtol=2e-5)
                np.testing.assert_array_equal(logits.argmax(1).numpy(), z[task+'_calip'][:count].argmax(1))
        raw_gate = np.asarray(il)[z['info_raw'].argmax(1)] == 'informative'
        calip_gate = np.asarray(il)[z['info_calip'].argmax(1)] == 'informative'
        means = np.asarray(payload['human_mean_prototypes'], dtype='float32')
        unit = means/np.linalg.norm(means, axis=1, keepdims=True)
        active = [i for i, label in enumerate(hl) if label != 'not_humanitarian']
        for seed in [0, 1, 2]:
            rows = list(csv.DictReader(io.StringIO(read_artifact(directory, f'predictions_seed_{seed}.tsv')), delimiter='\t'))
            assert [r['id'] for r in rows] == source['ids'][:n]
            assert [r['y_true'] for r in rows] == y
            assert [r['binary_truth'] for r in rows] == (yb or ['']*n)
            majority = np.asarray(source['majority_gates'][str(seed)][:n])
            for name, reports in analysis['by_method'].items():
                gate = raw_gate if name == 'two_stage_original' else calip_gate if name == 'calip_gate_dn_leaf' else majority
                leaf = (z['human_raw'] if name in ['two_stage_original', 'majority_raw_leaf'] else
                        z['image_features']@unit.T if name == 'majority_unit_leaf' else
                        z['human_calip'] if name == 'majority_calip_leaf' else np.asarray(source['dn_leaf'][:n]))
                # Dựng quy tắc độc lập thay vì gọi classify của runner.
                pred = np.asarray(hl)[np.asarray(active)[leaf[:, active].argmax(1)]]
                pred[~gate] = 'not_humanitarian'
                binary = np.where(gate, 'informative', 'not_informative').tolist()
                assert pred.tolist() == [r[name] for r in rows]
                assert binary == [r[name+'_binary'] for r in rows]
                expected = {'humanitarian': metric(y, pred, hl), 'binary': metric(yb, binary, il) if yb is not None else None}
                compare(expected, {k: v for k, v in reports[seed].items() if k != 'seed'})
    for name, reports in analysis['by_method'].items(): compare(analysis['mean'][name], average(reports))
    result = {'status': 'passed', 'samples': n, 'split': manifest['source_split'],
              'inference_backend': manifest['inference_backend'], 'mean': analysis['mean'],
              'scope': manifest['scope'], 'all_hashes_predictions_metrics_and_score_terms_verified': True}
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--run-dir', required=True); a = p.parse_args()
    report = analyze(a.run_dir)
    for name, tasks in report['mean'].items():
        print(name, 'human_macro_f1', round(100*tasks['humanitarian']['macro_f1'], 4),
              'binary_macro_f1', round(100*tasks['binary']['macro_f1'], 4) if tasks['binary'] else None)
    print('[AUDIT]', report['status'], report['scope'])
