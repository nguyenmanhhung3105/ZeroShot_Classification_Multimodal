"""Chẩn đoán lỗi cổng/tầng chi tiết từ artifact, không phải thuật toán dự đoán.

Các counterfactual dùng nhãn humanitarian thật chỉ để phân tích sau đánh giá;
không cung cấp nhãn informativeness dev hoặc kết quả model triển khai được.
"""
import argparse
import json
from pathlib import Path
import sys
import uuid
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
import numpy as np
from experiments.clip_tda.run import load_features
from experiments.clip_dmn.run import load_companions
from experiments.clip_zlap.run import project_path, validate_majority, majority_gates
from experiments.clip_transductive.ensemble import humanitarian
from experiments.clip_transductive.run import metric, sha256, write_json


def diagnose(truth, gate, leaf, labels):
    y, gate, leaf = np.asarray(truth), np.asarray(gate), np.asarray(leaf)
    if (len(set(labels)) != len(labels) or 'not_humanitarian' not in labels or
            gate.dtype != np.bool_ or gate.shape != (len(y),) or leaf.shape != (len(y), len(labels)) or
            not set(y).issubset(labels) or not np.isfinite(leaf).all()):
        raise ValueError('Sai đầu vào chẩn đoán')
    active = [i for i, c in enumerate(labels) if c != 'not_humanitarian']
    detail = np.asarray(labels)[np.asarray(active)[leaf[:, active].argmax(1)]]
    positive = y != 'not_humanitarian'
    pred = np.asarray(humanitarian(gate, leaf, labels))
    masks = {'target_negative_passed': ~positive & gate,
             'target_positive_blocked': positive & ~gate,
             'passed_positive_wrong_detail': positive & gate & (detail != y)}
    errors = {k: int(v.sum()) for k, v in masks.items()}
    assert sum(errors.values()) == int((pred != y).sum())
    combined = sum(v.astype('int64') for v in masks.values())
    np.testing.assert_array_equal(combined, (pred != y).astype('int64'))
    # Đây là oracle trên chính target humanitarian, KHÔNG phải binary GT.
    perfect_target_routing = np.where(positive, detail, 'not_humanitarian')
    perfect_detail = pred.copy(); perfect_detail[positive & gate] = y[positive & gate]
    by_class = {}
    for label in labels:
        if label == 'not_humanitarian': continue
        mask = y == label; count = int(mask.sum())
        by_class[label] = {'support': count, 'correct_without_gate': int((mask & (detail == y)).sum()),
                           'blocked': int((mask & ~gate).sum()),
                           'passed_wrong_detail': int((mask & gate & (detail != y)).sum())}
    assert sum(v['support'] for v in by_class.values()) == int(positive.sum())
    return {'actual': metric(y, pred, labels), 'errors': errors, 'total_errors': int((pred != y).sum()),
            'positive_support': int(positive.sum()), 'negative_support': int((~positive).sum()),
            'positive_detail_accuracy_without_gate': float((detail[positive] == y[positive]).mean()) if positive.any() else None,
            'by_positive_class': by_class,
            'counterfactual_only_not_deployable': {
                'perfect_humanitarian_target_routing': metric(y, perfect_target_routing, labels),
                'perfect_detail_for_passed_target_positive': metric(y, perfect_detail, labels)}}


def run(settings, output_dir=None):
    output = Path(output_dir) if output_dir else ROOT/'results/analysis/clip_bottleneck'/uuid.uuid4().hex[:12]
    output.mkdir(parents=True, exist_ok=False)
    directory = project_path(settings['majority_run']); mm = validate_majority(directory)
    manifest = {'status': 'running', 'purpose': 'posthoc diagnosis only, not a prediction candidate',
                'ground_truth_used_for_diagnosis': True, 'label_fit': False, 'test_access': False,
                'settings': settings, 'code_sha256': sha256(Path(__file__).read_bytes()),
                'source_features': {}, 'input_manifest_sha256': {
                    k: sha256((project_path(settings[k])/'run_manifest.json').read_bytes())
                    for k in ['majority_run', 'dn_run', 'inmap_run']}}
    write_json(output/'run_manifest.json', manifest)
    try:
        results = {}
        for scope in ['train', 'dev']:
            s = load_features(project_path(settings[scope+'_run']))
            if s['split'] != scope: raise ValueError('Không chẩn đoán ngoài train/dev')
            manifest['source_features'][scope] = {'directory': s['directory'], 'sha256': s['feature_sha256']}
            y = [r['y_true'] for r in s['rows']]; a, b = s['scores']; il, labels = s['info_labels'], s['labels']
            dn, inmap = load_companions(s, scope, settings)
            majority = majority_gates(s, scope, directory, mm)
            candidates = {'baseline': [(np.asarray(il)[a.argmax(1)] == 'informative', b)],
                          'inmap_dn': [(inmap, dn)],
                          'majority_dn': [(majority[k], dn) for k in [0, 1, 2]]}
            results[scope] = {}
            for name, trials in candidates.items():
                reports = [diagnose(y, g, leaf, labels) for g, leaf in trials]
                mean = {'actual_macro_f1': float(np.mean([r['actual']['macro_f1'] for r in reports])),
                        'errors': {k: float(np.mean([r['errors'][k] for r in reports])) for k in reports[0]['errors']},
                        'counterfactual_macro_f1': {k: float(np.mean([r['counterfactual_only_not_deployable'][k]['macro_f1'] for r in reports]))
                                                   for k in reports[0]['counterfactual_only_not_deployable']}}
                results[scope][name] = {'reports': reports, 'mean': mean}
                print(scope, name, mean, flush=True)
        results['limitations'] = [
            'Ground truth used only for posthoc error decomposition and counterfactual bounds; not new model results.',
            'Humanitarian target routing is not ground-truth informativeness; no binary dev metrics inferred.',
            'Counterfactual improvements are not forecasts, attainable guarantees or evidence that local features will fix errors.',
            'No images inspected in this diagnostic, no proof about visual cause of each error.',
            'Same reused development cohorts; no test, supervised fitting or changed predictions.']
        write_json(output/'analysis.json', results)
        manifest['status'] = 'complete'; manifest['analysis_sha256'] = sha256((output/'analysis.json').read_bytes())
        write_json(output/'run_manifest.json', manifest); print('[SAVED]', output); return output
    except Exception as error:
        manifest.update(status='failed', error=f'{type(error).__name__}: {error}')
        write_json(output/'run_manifest.json', manifest); raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default=str(ROOT/'experiments/clip_dpe/config.json')); p.add_argument('--output-dir')
    a = p.parse_args(); run(json.loads(Path(a.config).read_text()), a.output_dir)
