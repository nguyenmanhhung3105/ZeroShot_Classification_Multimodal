"""DPE single-view trên artifact, đối chứng đủ binary/humanitarian train/dev."""
import argparse
import csv
import json
from pathlib import Path
import sys
import uuid
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
import numpy as np
from experiments.clip_dpe.core import predict_stream, PARAMETERS, REFERENCE_COMMIT
from experiments.clip_tda.run import load_features
from experiments.clip_zlap.run import project_path, text_prototypes, validate_majority, majority_gates, evaluate
from experiments.clip_dmn.run import load_companions
from experiments.clip_transductive.ensemble import humanitarian
from experiments.clip_transductive.run import sha256, write_json
from experiments.clip_two_stage.analyze import read_artifact

METHODS = ['dpe_gate_raw_leaf', 'dpe_gate_dn_leaf', 'majority_dpe_leaf', 'inmap_dpe_leaf', 'dpe_both']
CODE = ['experiments/clip_dpe/core.py', 'experiments/clip_dpe/run.py', 'experiments/clip_dmn/core.py',
        'experiments/clip_dmn/run.py', 'experiments/clip_tda/run.py', 'experiments/clip_zlap/run.py',
        'experiments/clip_transductive/ensemble.py', 'experiments/clip_transductive/run.py']


def average(rows):
    result = {}
    for task in ['binary', 'humanitarian']:
        if rows[0][task] is None:
            result[task] = None; continue
        result[task] = {key: float(np.mean([r[task][key] for r in rows])) for key in ['accuracy', 'macro_f1']}
        result[task]['per_class'] = {
            c: {k: float(np.mean([r[task]['per_class'][c][k] for r in rows]))
                for k in rows[0][task]['per_class'][c]} for c in rows[0][task]['per_class']}
    return result


def run(settings, output_dir=None):
    if (settings.get('parameters') != PARAMETERS or settings.get('methods') != METHODS or
            settings.get('seeds') != [0, 1, 2] or settings.get('label_fit') is not False or
            settings.get('protocol') != 'dpe_single_view_cpu_fp32_residual_and_evolution_train_dev_no_search'):
        raise ValueError('Chỉ chạy protocol DPE đã khóa, không search tham số/seed')
    output = Path(output_dir) if output_dir else ROOT/'results/analysis/clip_dpe'/uuid.uuid4().hex[:12]
    output.mkdir(parents=True, exist_ok=False)
    manifest = {'status': 'running', 'settings': settings, 'reference_commit': REFERENCE_COMMIT,
                'label_fit': False, 'support_samples': 0, 'single_view_adaptation': True,
                'residual_optimization': True, 'backbone_frozen': True,
                'code_sha256': {p: sha256((ROOT/p).read_bytes()) for p in CODE},
                'input_manifest_sha256': {k: sha256((project_path(settings[k])/'run_manifest.json').read_bytes())
                                          for k in ['majority_run', 'dn_run', 'inmap_run']},
                'source_features': {}, 'source_artifact_sha256': {}}
    write_json(output/'frozen_plan.json', manifest); write_json(output/'run_manifest.json', manifest)
    try:
        donor = load_features(project_path(settings['text_prototype_run']))
        directory = project_path(settings['majority_run']); majority_manifest = validate_majority(directory)
        results = {}
        for scope in ['train', 'dev']:
            source = load_features(project_path(settings[scope+'_run']))
            if (source['split'] != scope or source['manifest']['model'] != 'metaclip2_vith14' or
                    source['manifest'].get('logit_scale') != 100.):
                raise ValueError('Sai split/checkpoint/scale MetaCLIP2')
            manifest['source_features'][scope] = {'directory': source['directory'], 'sha256': source['feature_sha256']}
            manifest['source_artifact_sha256'][scope] = {
                n: sha256(read_artifact(Path(source['directory']), n).encode()) for n in
                ['run_manifest.json', 'predictions.tsv', 'informativeness/cohort.json', 'humanitarian/cohort.json']}
            x, il, hl = source['features'], source['info_labels'], source['labels']; a, b = source['scores']
            hp = text_prototypes(source, donor)
            gates = majority_gates(source, scope, directory, majority_manifest)
            dn, inmap = load_companions(source, scope, settings)
            raw = np.asarray(il)[a.argmax(1)] == 'informative'
            normal = np.asarray(il)[(x @ source['prototypes'].T).argmax(1)] == 'informative'
            binary = lambda gate: np.where(gate, 'informative', 'not_informative').tolist()
            summary = {'samples': len(x), 'controls': {
                'baseline': evaluate(source, humanitarian(raw, b, hl), binary(raw)),
                'normalized_both': evaluate(source, humanitarian(normal, x @ hp.T, hl), binary(normal)),
                'inmap_dn': evaluate(source, humanitarian(inmap, dn, hl), binary(inmap))},
                'by_method': {name: [] for name in METHODS + ['majority_dn_control']}}
            for seed in settings['seeds']:
                print(f'[DPE] {scope}; shuffle={seed}; one residual step, cache/text evolution; root then leaf', flush=True)
                order = np.random.default_rng(seed).permutation(len(x))
                root, rt, rd = predict_stream(x, source['prototypes'], order, **PARAMETERS)
                leaf, lt, ld = predict_stream(x, hp, order, **PARAMETERS)
                gate = np.asarray(il)[root.argmax(1)] == 'informative'
                choices = {'dpe_gate_raw_leaf': humanitarian(gate, b, hl),
                           'dpe_gate_dn_leaf': humanitarian(gate, dn, hl),
                           'majority_dpe_leaf': humanitarian(gates[seed], leaf, hl),
                           'inmap_dpe_leaf': humanitarian(inmap, leaf, hl),
                           'dpe_both': humanitarian(gate, leaf, hl),
                           'majority_dn_control': humanitarian(gates[seed], dn, hl)}
                for name, pred in choices.items():
                    g = gates[seed] if name.startswith('majority') else inmap if name.startswith('inmap') else gate
                    report = {'seed': seed, **evaluate(source, pred, binary(g))}
                    summary['by_method'][name].append(report)
                    print('[RESULT]', scope, seed, name, 'human_f1', report['humanitarian']['macro_f1'],
                          'binary_f1', report['binary']['macro_f1'] if report['binary'] else None, flush=True)
                folder = output/scope/f'shuffle_{seed}'; folder.mkdir(parents=True)
                np.savez_compressed(folder/'logits.npz', ids=np.asarray([r['id'] for r in source['rows']]),
                                    informativeness=root, humanitarian=leaf,
                                    **{'info_'+k: v for k, v in rt.items()}, **{'human_'+k: v for k, v in lt.items()})
                write_json(folder/'diagnostics.json', {'informativeness': rd, 'humanitarian': ld})
                with (folder/'predictions.tsv').open('x', newline='') as f:
                    w = csv.writer(f, delimiter='\t')
                    w.writerow(['id', 'y_true', 'binary_truth', 'dpe_gate', 'majority_gate', 'inmap_gate', 'stream_position', *choices])
                    positions = np.argsort(order)
                    w.writerows((r['id'], r['y_true'], source['binary_truth'][i] if source['binary_truth'] is not None else '',
                                 bool(gate[i]), bool(gates[seed][i]), bool(inmap[i]), int(positions[i]),
                                 *[p[i] for p in choices.values()]) for i, r in enumerate(source['rows']))
            summary['mean'] = {name: average(reports) for name, reports in summary['by_method'].items()}
            summary['std_population'] = {
                name: {task: float(np.std([r[task]['macro_f1'] for r in reports])) if reports[0][task] else None
                       for task in ['binary', 'humanitarian']} for name, reports in summary['by_method'].items()}
            results[scope] = summary
        results['limitations'] = [
            'DPE single-view adaptation, CPU FP32, default ImageNet settings; not full 64-view CUDA benchmark.',
            'True residual optimization, including text/image alignment; not a cache-only method.',
            'No labeled support, hyperparameter sweep, event-based ordering or seed selection.',
            'Three shuffles measure stream-order sensitivity; pairing with majority grouping is not independent replication.',
            'Binary dev GT absent, remain null; train/dev reused for exploration, not independent test.']
        write_json(output/'analysis.json', results)
        with (output/'summary.tsv').open('x', newline='') as f:
            w = csv.writer(f, delimiter='\t'); w.writerow(['split', 'method', 'task', 'accuracy', 'macro_f1'])
            for scope in ['train', 'dev']:
                for name, tasks in results[scope]['mean'].items():
                    for task, values in tasks.items():
                        if values: w.writerow([scope, name, task, values['accuracy'], values['macro_f1']])
        manifest['status'] = 'complete'
        manifest['output_sha256'] = {str(p.relative_to(output)): sha256(p.read_bytes()) for p in output.rglob('*')
                                     if p.is_file() and p.name != 'run_manifest.json'}
        write_json(output/'run_manifest.json', manifest); print('[SAVED]', output, flush=True); return output
    except Exception as error:
        manifest.update(status='failed', error=f'{type(error).__name__}: {error}')
        write_json(output/'run_manifest.json', manifest); raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default=str(ROOT/'experiments/clip_dpe/config.json')); p.add_argument('--output-dir')
    a = p.parse_args(); run(json.loads(Path(a.config).read_text()), a.output_dir)
