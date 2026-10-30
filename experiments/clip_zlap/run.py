"""Đánh giá một cấu hình ZLaP trên artifact train/dev; không mở dataset/model."""
import argparse
import csv
import io
import json
from pathlib import Path
import sys
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
import numpy as np
from experiments.clip_zlap.core import predict, unit_vectors, PARAMETERS, REFERENCE_COMMIT
from experiments.clip_tda.run import load_features
from experiments.clip_two_stage.analyze import read_artifact
from experiments.clip_transductive.ensemble import humanitarian
from experiments.clip_transductive.run import metric, sha256, write_json

METHODS = ['zlap_gate', 'majority_zlap_leaf', 'zlap_two_stage', 'zlap_flat']


def project_path(value):
    p = Path(value)
    return p if p.is_absolute() else ROOT/p


def feature_bytes(source):
    manifest = source['manifest']; directory = Path(source['directory'])
    name = 'stata_features.npz' if manifest.get('stata_gate') else manifest.get('feature_file', 'clip_features.npz')
    if (directory/name).is_file():
        raw = (directory/name).read_bytes()
    else:
        with zipfile.ZipFile(directory/'run_artifacts.zip') as archive:
            raw = archive.read(name)
    if sha256(raw) != source['feature_sha256']:
        raise ValueError('Feature hash thay đổi')
    return raw


def text_prototypes(source, donor):
    """Pack dev cũ thiếu human prototypes; tái dùng text bank cùng checkpoint.

    Không lấy mẫu ảnh/nhãn train làm support. Kiểm tra bank và tái dựng mọi
    human cosine của cohort đích trước khi dùng vector text đã cache.
    """
    if source['manifest']['model'] != donor['manifest']['model']:
        raise ValueError('Text prototype khác backbone')
    plans = [json.loads(read_artifact(Path(item['directory']), 'humanitarian/frozen_plan.json')) for item in [source, donor]]
    if plans[0]['label_order'] != plans[1]['label_order'] or plans[0]['prompt_banks'] != plans[1]['prompt_banks']:
        raise ValueError('Không tái dùng text prototype giữa hai bank khác nhau')
    with np.load(io.BytesIO(feature_bytes(donor)), allow_pickle=False) as pack:
        means = pack['human_mean_prototypes'].copy()
    if means.shape != (len(source['labels']), source['features'].shape[1]):
        raise ValueError('Sai shape text prototype')
    if not np.allclose(source['features'] @ means.T, source['scores'][1], atol=2e-5, rtol=1e-4):
        raise ValueError('Text prototype không tái dựng được human cosine')
    return unit_vectors(means)


def validate_majority(directory):
    manifest = json.loads((directory/'run_manifest.json').read_text())
    if manifest.get('status') != 'complete' or manifest.get('label_fit') is not False:
        raise ValueError('Cần đối chứng majority đã hoàn tất, không fit nhãn')
    for name, digest in manifest['output_sha256'].items():
        if sha256((directory/name).read_bytes()) != digest:
            raise ValueError(f'Hash đối chứng majority thay đổi: {name}')
    return manifest


def majority_gates(source, scope, directory, manifest):
    if manifest['source_features'][scope]['sha256'] != source['feature_sha256']:
        raise ValueError('Majority dùng feature khác')
    gates = {}
    for seed in [0, 1, 2]:
        with (directory/scope/f'grouping_seed_{seed}'/'predictions.tsv').open(encoding='utf-8') as f:
            rows = list(csv.DictReader(f, delimiter='\t'))
        if [(r['id'], r['humanitarian_truth']) for r in rows] != [(r['id'], r['y_true']) for r in source['rows']]:
            raise ValueError('Majority khác ID, nhãn hoặc thứ tự')
        if any(r['majority_gate'] not in ['True', 'False'] for r in rows):
            raise ValueError('Gate không phải boolean')
        gate = np.array([r['majority_gate'] == 'True' for r in rows])
        if humanitarian(gate, source['scores'][1], source['labels']) != [r['majority_pred'] for r in rows]:
            raise ValueError('Majority prediction không khớp score/gate')
        gates[seed] = gate
    return gates


def evaluate(source, predicted, binary):
    labels, il = source['labels'], source['info_labels']
    return {'humanitarian': metric([r['y_true'] for r in source['rows']], predicted, labels),
            'binary': metric(source['binary_truth'], binary, il) if source['binary_truth'] is not None else None}


def run(settings, output_dir=None):
    if (settings.get('methods') != METHODS or settings.get('parameters') != PARAMETERS or
            settings.get('label_fit') is not False or
            settings.get('protocol') != 'separate_full_cohort_graphs_train_dev_no_labels'):
        raise ValueError('Chỉ chạy các quy tắc và tham số đã khóa, không search trên dev')
    output = Path(output_dir) if output_dir else ROOT/'results/analysis/clip_zlap'/uuid.uuid4().hex[:12]
    output.mkdir(parents=True, exist_ok=False)
    manifest = {'status': 'running', 'settings': settings, 'reference_commit': REFERENCE_COMMIT,
                'label_fit': False, 'support_samples': 0, 'probabilities_calibrated': False,
                'graph_scope': 'one independent full graph per split and task; no train/dev image mixing',
                'normalization': 'unit image and mean prompt vectors, as reference loader',
                'leaf_rule': 'positive classes selected from graph over all eight labels',
                'source_artifact_sha256': {},
                'code_sha256': {name: sha256((ROOT/name).read_bytes()) for name in [
                    'experiments/clip_zlap/core.py', 'experiments/clip_zlap/run.py',
                    'experiments/clip_tda/run.py', 'experiments/clip_two_stage/analyze.py',
                    'experiments/clip_transductive/ensemble.py', 'experiments/clip_transductive/run.py']}}
    write_json(output/'frozen_plan.json', manifest)
    write_json(output/'run_manifest.json', manifest)
    try:
        donor = load_features(project_path(settings['text_prototype_run']))
        majority_dir = project_path(settings['majority_run'])
        majority_manifest = validate_majority(majority_dir)
        manifest['majority_manifest_sha256'] = sha256((majority_dir/'run_manifest.json').read_bytes())
        manifest['text_prototype_feature_sha256'] = donor['feature_sha256']
        results = {}
        for scope in ['train', 'dev']:
            source_dir = project_path(settings[f'{scope}_run'])
            source = donor if str(source_dir) == donor['directory'] else load_features(source_dir)
            if source['split'] != scope:
                raise ValueError('Không đổi tên split đánh giá')
            source['human_prototypes'] = text_prototypes(source, donor)
            manifest['source_artifact_sha256'][scope] = {name: sha256(read_artifact(source_dir, name).encode('utf-8'))
                for name in ['run_manifest.json', 'predictions.tsv', 'informativeness/cohort.json',
                             'humanitarian/cohort.json', 'humanitarian/frozen_plan.json']}
            manifest.setdefault('source_features', {})[scope] = source['feature_sha256']
            x, il, labels = source['features'], source['info_labels'], source['labels']
            a, b = source['scores']; hp = source['human_prototypes']
            majority = majority_gates(source, scope, majority_dir, majority_manifest)
            print(f'[ZLAP] {scope}; samples={len(x)}; tasks=2,8; k=5; gamma=5; alpha=0.3', flush=True)
            root_scores, root_log = predict(x, source['prototypes'], **PARAMETERS)
            leaf_scores, leaf_log = predict(x, hp, **PARAMETERS)
            graph_gate = np.asarray(il)[root_scores.argmax(1)] == 'informative'
            raw_gate = np.asarray(il)[a.argmax(1)] == 'informative'
            normalized_gate = np.asarray(il)[(x @ source['prototypes'].T).argmax(1)] == 'informative'
            normalized_leaf = x @ hp.T
            binary_graph = np.where(graph_gate, 'informative', 'not_informative').tolist()
            binary_raw = np.where(raw_gate, 'informative', 'not_informative').tolist()
            binary_normalized = np.where(normalized_gate, 'informative', 'not_informative').tolist()
            summary = {'samples': len(x), 'binary_ground_truth_available': source['binary_truth'] is not None,
                       'controls': {
                           'baseline': evaluate(source, humanitarian(raw_gate, b, labels), binary_raw),
                           'single_stage': {'humanitarian': metric([r['y_true'] for r in source['rows']], np.asarray(labels)[b.argmax(1)].tolist(), labels), 'binary': None},
                           'normalized_gate_control': evaluate(source, humanitarian(normalized_gate, b, labels), binary_normalized),
                           'raw_gate_normalized_leaf': evaluate(source, humanitarian(raw_gate, normalized_leaf, labels), binary_raw),
                           'normalized_both': evaluate(source, humanitarian(normalized_gate, normalized_leaf, labels), binary_normalized)},
                       'by_method': {name: [] for name in METHODS + ['majority_control', 'majority_normalized_leaf']}}
            binary_summary = metric(source['binary_truth'], binary_graph, il) if source['binary_truth'] is not None else None
            for seed in [0, 1, 2]:
                choices = {
                    'zlap_gate': humanitarian(graph_gate, b, labels),
                    'majority_zlap_leaf': humanitarian(majority[seed], leaf_scores, labels),
                    'zlap_two_stage': humanitarian(graph_gate, leaf_scores, labels),
                    'zlap_flat': np.asarray(labels)[leaf_scores.argmax(1)].tolist(),
                    'majority_control': humanitarian(majority[seed], b, labels),
                    'majority_normalized_leaf': humanitarian(majority[seed], normalized_leaf, labels)}
                for name, prediction in choices.items():
                    if name == 'zlap_flat':
                        report = {'humanitarian': metric([r['y_true'] for r in source['rows']], prediction, labels), 'binary': None}
                    else:
                        gate = majority[seed] if name.startswith('majority') else graph_gate
                        report = evaluate(source, prediction, np.where(gate, 'informative', 'not_informative').tolist())
                    summary['by_method'][name].append({'seed': seed, **report})
                folder = output/scope/f'grouping_seed_{seed}'; folder.mkdir(parents=True)
                with (folder/'predictions.tsv').open('x', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f, delimiter='\t')
                    writer.writerow(['id', 'y_true', 'binary_truth', 'raw_gate', 'majority_gate', 'zlap_informative_gate', *choices])
                    writer.writerows((row['id'], row['y_true'], source['binary_truth'][i] if source['binary_truth'] is not None else '',
                        bool(raw_gate[i]), bool(majority[seed][i]), bool(graph_gate[i]),
                        *[prediction[i] for prediction in choices.values()]) for i, row in enumerate(source['rows']))
            summary['mean'] = {name: {task: {key: float(np.mean([r[task][key] for r in reports]))
                    for key in ['accuracy', 'macro_f1']} if reports[0][task] is not None else None
                    for task in ['binary', 'humanitarian']} for name, reports in summary['by_method'].items()}
            summary['zlap_binary'] = binary_summary
            summary['seed_note'] = 'ZLaP full graph is deterministic; seeds only index saved majority gates. Do not count repeated graph predictions as independent runs.'
            task_dir = output/scope
            np.savez_compressed(task_dir/'graph_scores.npz', ids=np.array([r['id'] for r in source['rows']]),
                                informativeness=root_scores, humanitarian=leaf_scores)
            write_json(task_dir/'diagnostics.json', {'informativeness': root_log, 'humanitarian': leaf_log})
            results[scope] = summary
            for name, tasks in summary['mean'].items():
                print(f'[RESULT] {scope}; {name}; human_f1={tasks["humanitarian"]["macro_f1"]:.6f}; binary_f1={tasks["binary"]["macro_f1"] if tasks["binary"] else None}', flush=True)
        results['limitations'] = [
            'One fixed configuration from author; no parameter/seed selection on dev.',
            'Full-cohort transductive graph needs unlabeled query images jointly; differs from 75-image EM/StatA groups.',
            'Train/dev already used for exploration; not independent test or proof of global optimum.',
            'CPU NumPy kNN replaces GPU FAISS, SciPy CG replaces CuPy CG; stable index tie handling recorded.',
            'Scores are geodesic similarities, not calibrated probabilities. No support labels, no pseudo-label balancing.',
            'Prototype normalization has separate controls; do not attribute every change to graph propagation.',
            'Binary dev metrics unavailable; flat humanitarian output is not an actual informativeness classifier.']
        write_json(output/'analysis.json', results)
        with (output/'summary.tsv').open('x', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter='\t'); writer.writerow(['split', 'task', 'method', 'samples', 'accuracy', 'macro_f1'])
            for scope in ['train', 'dev']:
                for name, tasks in {**results[scope]['controls'], **results[scope]['mean']}.items():
                    for task, values in tasks.items():
                        if values is not None:
                            writer.writerow([scope, task, name, results[scope]['samples'], values['accuracy'], values['macro_f1']])
        manifest['status'] = 'complete'
        manifest['output_sha256'] = {str(p.relative_to(output)): sha256(p.read_bytes()) for p in output.rglob('*')
                                    if p.is_file() and p.name != 'run_manifest.json'}
        write_json(output/'run_manifest.json', manifest)
        print(f'[SAVED] {output}', flush=True)
        return output
    except Exception as error:
        manifest.update(status='failed', error=f'{type(error).__name__}: {error}')
        write_json(output/'run_manifest.json', manifest)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT/'experiments/clip_zlap/config.json'))
    parser.add_argument('--output-dir')
    args = parser.parse_args()
    run(json.loads(Path(args.config).read_text()), args.output_dir)
