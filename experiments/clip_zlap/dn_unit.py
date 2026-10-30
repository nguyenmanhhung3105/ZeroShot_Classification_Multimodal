"""DN-inspired trên embedding unit đã cache, mean ảnh chỉ lấy train.

Giữ coefficient 0.25 theo compute_zeroshot.py của repo DN NeurIPS 2023.
Repo tác giả dùng output encoder chưa chuẩn hóa; artifact dự án chỉ có unit
image vectors/mean prompt vectors nên đây là adaptation, không tái lập DN gốc.
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
from experiments.clip_zlap.core import unit_vectors
from experiments.clip_zlap.run import project_path, text_prototypes, validate_majority, majority_gates, evaluate
from experiments.clip_tda.run import load_features
from experiments.clip_transductive.ensemble import humanitarian
from experiments.clip_transductive.run import sha256, write_json

METHODS = ['dn_unit_gate', 'dn_unit_two_stage', 'majority_dn_unit_leaf']


def shifted_scores(features, prototypes, image_mean, coefficient=.25):
    if coefficient != .25:
        raise ValueError('Khóa hệ số 0.25 từ mã tác giả')
    x, t = unit_vectors(features), unit_vectors(prototypes)
    mu = np.asarray(image_mean, dtype=np.float32)
    if mu.shape != (x.shape[1],) or t.shape[1] != x.shape[1] or not np.isfinite(mu).all():
        raise ValueError('Sai mean/dimension')
    return unit_vectors(x-coefficient*mu) @ unit_vectors(t-coefficient*t.mean(0)).T


def run(settings, output_dir=None):
    if (settings.get('methods') != METHODS or settings.get('coefficient') != .25 or
            settings.get('image_mean_source') != 'train_only_unlabeled_features' or settings.get('label_fit') is not False):
        raise ValueError('Không đổi các lựa chọn đã khóa')
    output = Path(output_dir) if output_dir else ROOT/'results/analysis/clip_dn_unit'/uuid.uuid4().hex[:12]
    output.mkdir(parents=True, exist_ok=False)
    manifest = {'status': 'running', 'settings': settings, 'label_fit': False, 'support_samples': 0,
                'method': 'DN-inspired unit-feature adaptation, not full original DN',
                'image_mean': 'all unlabeled train unit embeddings, frozen for both train and dev',
                'code_sha256': {name: sha256((ROOT/name).read_bytes()) for name in [
                    'experiments/clip_zlap/dn_unit.py', 'experiments/clip_zlap/core.py',
                    'experiments/clip_zlap/run.py', 'experiments/clip_tda/run.py']}}
    write_json(output/'frozen_plan.json', manifest); write_json(output/'run_manifest.json', manifest)
    try:
        donor = load_features(project_path(settings['text_prototype_run']))
        train = load_features(project_path(settings['train_run']))
        if train['split'] != 'train': raise ValueError('Không lấy mean từ dev/test')
        mean = unit_vectors(train['features']).mean(0)
        directory = project_path(settings['majority_run']); majority_manifest = validate_majority(directory)
        manifest['train_feature_sha256'] = train['feature_sha256']
        manifest['text_feature_sha256'] = donor['feature_sha256']
        manifest['majority_manifest_sha256'] = sha256((directory/'run_manifest.json').read_bytes())
        results = {}
        for scope in ['train', 'dev']:
            source = train if scope == 'train' else load_features(project_path(settings['dev_run']))
            if source['split'] != scope: raise ValueError('Sai split')
            manifest.setdefault('source_feature_sha256', {})[scope] = source['feature_sha256']
            x, il, labels = source['features'], source['info_labels'], source['labels']; a, b = source['scores']
            t = text_prototypes(source, donor)
            root = shifted_scores(x, source['prototypes'], mean)
            leaf = shifted_scores(x, t, mean)
            gate = np.asarray(il)[root.argmax(1)] == 'informative'
            majority = majority_gates(source, scope, directory, majority_manifest)
            reports = {name: [] for name in METHODS}
            folder = output/scope; folder.mkdir()
            for seed in [0, 1, 2]:
                choices = {'dn_unit_gate': humanitarian(gate, b, labels),
                           'dn_unit_two_stage': humanitarian(gate, leaf, labels),
                           'majority_dn_unit_leaf': humanitarian(majority[seed], leaf, labels)}
                for name, prediction in choices.items():
                    used_gate = majority[seed] if name.startswith('majority') else gate
                    reports[name].append({'seed': seed, **evaluate(source, prediction, np.where(used_gate, 'informative', 'not_informative').tolist())})
                with (folder/f'predictions_seed_{seed}.tsv').open('x', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f, delimiter='\t')
                    writer.writerow(['id', 'y_true', 'binary_truth', 'dn_gate', 'majority_gate', *choices])
                    writer.writerows((r['id'], r['y_true'], source['binary_truth'][i] if source['binary_truth'] is not None else '',
                                      bool(gate[i]), bool(majority[seed][i]), *[v[i] for v in choices.values()]) for i, r in enumerate(source['rows']))
            results[scope] = {'samples': len(x), 'by_method': reports,
                'mean': {name: {task: {key: float(np.mean([r[task][key] for r in group])) for key in ['accuracy', 'macro_f1']}
                               if group[0][task] is not None else None for task in ['binary', 'humanitarian']} for name, group in reports.items()}}
            np.savez_compressed(folder/'scores.npz', ids=np.array([r['id'] for r in source['rows']]),
                                informativeness=root, humanitarian=leaf, image_mean=mean)
            for name, report in results[scope]['mean'].items(): print(scope, name, report, flush=True)
        results['limitations'] = ['Unit-feature adaptation, not raw pre-normalization encoder features used in original DN code.',
            'Train-only image mean and coefficient 0.25 fixed; no search over means/coefficient on dev.',
            'Train/dev exploratory, no test; no supervised support or fitting.',
            'Deterministic DN predictions repeated only to align with three majority grouping seeds, not independent runs.']
        write_json(output/'analysis.json', results)
        manifest['status'] = 'complete'
        manifest['output_sha256'] = {str(p.relative_to(output)): sha256(p.read_bytes()) for p in output.rglob('*') if p.is_file() and p.name != 'run_manifest.json'}
        write_json(output/'run_manifest.json', manifest); print('[SAVED]', output, flush=True)
        return output
    except Exception as error:
        manifest.update(status='failed', error=f'{type(error).__name__}: {error}')
        write_json(output/'run_manifest.json', manifest); raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT/'experiments/clip_zlap/config_dn_unit.json'))
    parser.add_argument('--output-dir'); args = parser.parse_args()
    run(json.loads(Path(args.config).read_text()), args.output_dir)
