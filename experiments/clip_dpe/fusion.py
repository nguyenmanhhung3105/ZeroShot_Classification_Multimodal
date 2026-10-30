"""Kết hợp DPE + DN-unit bằng confidence matching từ core Frolic.

Đây là phép thử thành phần, không phải tái lập Frolic đầy đủ. Fit nhiệt độ
chỉ từ confidence không nhãn trên từng cohort; toàn fusion là transductive.
"""
import argparse
import csv
import json
from pathlib import Path
import sys
import uuid
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
import numpy as np
import torch
from experiments.clip_tda.run import load_features
from experiments.clip_dmn.run import load_companions
from experiments.clip_dpe.run import average
from experiments.clip_zlap.run import project_path, validate_majority, majority_gates, evaluate
from experiments.clip_transductive.ensemble import humanitarian
from experiments.clip_transductive.run import sha256, write_json

FROLIC_COMMIT = 'a42343948b7fa45dcf3a60602e5c894edaa956ff'


def confidence(logits, temperature=1.):
    return float((logits/temperature).softmax(-1).max(-1).values.mean())


def confidence_match(anchor_logits, other_logits):
    a = torch.as_tensor(anchor_logits, dtype=torch.float32)
    b = torch.as_tensor(other_logits, dtype=torch.float32)
    if (a.ndim != 2 or b.shape != a.shape or min(a.shape) < 1 or a.shape[1] < 2 or
            not torch.isfinite(a).all() or not torch.isfinite(b).all()):
        raise ValueError('Sai logits fusion')
    target = confidence(a); temperature = 1.; first = confidence(b)
    # Giữ bước 1.1 và tolerance 10% trong search_T tác giả; bounded để báo
    # lỗi nếu input suy biến, không âm thầm chọn nhiệt độ theo metric.
    for step in range(1000):
        current = confidence(b, temperature)
        if abs(target-current)/target < .1:
            mixed = a + b/temperature
            return mixed.numpy(), {'temperature': temperature, 'anchor_confidence': target,
                                   'other_confidence_before': first, 'other_confidence_after': current,
                                   'relative_error': abs(target-current)/target, 'steps': step,
                                   'tolerance': .1, 'label_fit': False}
        temperature = temperature/1.1 if target >= current else temperature*1.1
    raise RuntimeError('Confidence matching không hội tụ theo search_T tác giả')


def run(dpe_run, output_dir=None):
    directory = project_path(dpe_run); manifest = validate_majority(directory)
    settings = manifest['settings']
    # Khóa phép thử và đối chứng trước khi nạp feature hoặc tính metric.
    plan = {'source_dpe': str(directory), 'methods': ['majority_equal', 'inmap_equal', 'majority_matched', 'inmap_matched'],
            'dn_scale': 100., 'initial_temperature': 1., 'temperature_step': 1.1, 'relative_tolerance': .1,
            'fusion': 'DN logits + DPE logits / temperature, equal coefficients',
            'scope': 'unlabeled full-cohort confidence matching, no dev label fitting',
            'seeds': [0, 1, 2], 'label_fit': False, 'frolic_reference_commit': FROLIC_COMMIT}
    output = Path(output_dir) if output_dir else ROOT/'results/analysis/clip_dpe_fusion'/uuid.uuid4().hex[:12]
    output.mkdir(parents=True, exist_ok=False)
    meta = {'status': 'running', 'settings': plan, 'label_fit': False,
            'input_manifest_sha256': {str(directory): sha256((directory/'run_manifest.json').read_bytes())},
            'source_features': manifest['source_features'],
            'code_sha256': {p: sha256((ROOT/p).read_bytes()) for p in
                           ['experiments/clip_dpe/fusion.py', 'experiments/clip_dpe/run.py',
                            'experiments/clip_dmn/run.py', 'experiments/clip_zlap/run.py']}}
    write_json(output/'frozen_plan.json', meta); write_json(output/'run_manifest.json', meta)
    previous = torch.get_num_threads(); torch.set_num_threads(1)
    try:
        md = project_path(settings['majority_run']); mm = validate_majority(md); results = {}
        for scope in ['train', 'dev']:
            source = load_features(project_path(settings[scope+'_run']))
            if source['split'] != scope or source['feature_sha256'] != manifest['source_features'][scope]['sha256']:
                raise ValueError('Khác split/cohort DPE')
            ids = [r['id'] for r in source['rows']]; hl, il = source['labels'], source['info_labels']
            dn, inmap = load_companions(source, scope, settings)
            gates = majority_gates(source, scope, md, mm)
            by_method = {k: [] for k in plan['methods'] + ['majority_dn_control', 'inmap_dn_control']}
            matching = []
            folder = output/scope; folder.mkdir()
            for seed in plan['seeds']:
                with np.load(directory/scope/f'shuffle_{seed}/logits.npz', allow_pickle=False) as z:
                    if z['ids'].tolist() != ids: raise ValueError('Sai ID DPE')
                    dpe = z['humanitarian'].copy()
                mixed, diag = confidence_match(100.*dn, dpe)
                equal = 100.*dn + dpe
                matching.append({'seed': seed, **diag})
                choices = {name: humanitarian(gates[seed] if name.startswith('majority') else inmap,
                                              dn if name.endswith('control') else equal if name.endswith('equal') else mixed, hl)
                           for name in by_method}
                for name, prediction in choices.items():
                    gate = gates[seed] if name.startswith('majority') else inmap
                    by_method[name].append({'seed': seed, **evaluate(source, prediction,
                                            np.where(gate, 'informative', 'not_informative').tolist())})
                np.savez_compressed(folder/f'logits_seed_{seed}.npz', ids=np.asarray(ids), equal=equal, matched=mixed)
                with (folder/f'predictions_seed_{seed}.tsv').open('x', newline='') as f:
                    w = csv.writer(f, delimiter='\t'); w.writerow(['id', 'y_true', 'binary_truth', *choices])
                    w.writerows((r['id'], r['y_true'], source['binary_truth'][i] if source['binary_truth'] is not None else '',
                                 *[p[i] for p in choices.values()]) for i, r in enumerate(source['rows']))
            results[scope] = {'samples': len(ids), 'by_method': by_method, 'mean': {k: average(v) for k, v in by_method.items()},
                              'confidence_matching': matching}
            print(scope, {k: v['humanitarian']['macro_f1'] for k, v in results[scope]['mean'].items()}, flush=True)
        results['limitations'] = ['Only Frolic confidence matching component, not its Gaussian/debiasing/InMaP pipeline.',
                                 'Mean confidence matching is not calibration against true correctness.',
                                 'Two fixed gate choices and an equal-logit control; no weight or temperature metric sweep.',
                                 'Exploratory train/dev; no binary dev GT, no test.']
        write_json(output/'analysis.json', results)
        meta['status'] = 'complete'; meta['output_sha256'] = {str(p.relative_to(output)): sha256(p.read_bytes())
            for p in output.rglob('*') if p.is_file() and p.name != 'run_manifest.json'}
        write_json(output/'run_manifest.json', meta); print('[SAVED]', output, flush=True); return output
    except Exception as error:
        meta.update(status='failed', error=f'{type(error).__name__}: {error}'); write_json(output/'run_manifest.json', meta); raise
    finally:
        torch.set_num_threads(previous)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--dpe-run', required=True); p.add_argument('--output-dir')
    a = p.parse_args(); run(a.dpe_run, a.output_dir)
