"""Đánh giá đúng ba phương án đã khóa trên feature artifact test."""
import argparse
import csv
import json
from pathlib import Path
import sys
import uuid
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
import numpy as np
from experiments.clip_final.protocol import PLAN, load_plan
from experiments.clip_tda.run import load_features
from experiments.clip_two_stage.analyze import read_artifact
from experiments.clip_two_stage.validation import check_holdout
from experiments.clip_stata.core import gate_ablation
from experiments.clip_transductive.gate import predict_binary
from experiments.clip_transductive.ensemble import combine, humanitarian
from experiments.clip_transductive.run import metric, write_json, sha256
from experiments.clip_zlap.core import unit_vectors
from experiments.clip_zlap.dn_unit import shifted_scores


def predict_locked(features, info_scores, human_scores, info_means, protocol):
    """Không nhận ground truth; mean và text DN đều lấy từ frozen plan."""
    x = np.asarray(features, dtype=np.float32); a = np.asarray(info_scores); b = np.asarray(human_scores)
    il, hl = protocol['info_labels'], protocol['human_labels']
    frozen_info = np.asarray(protocol['info_mean_prototypes'], dtype=np.float32)
    frozen_human = np.asarray(protocol['human_mean_prototypes'], dtype=np.float32)
    if (not np.allclose(info_means, frozen_info, atol=2e-5, rtol=1e-4) or
            not np.allclose(x@frozen_human.T, b, atol=2e-5, rtol=1e-4)):
        raise ValueError('Prototype/score khác bank phát triển đã khóa')
    _, stata = gate_ablation(x, info_means, a, b, il, hl)
    norm = np.asarray(il)[(x@unit_vectors(info_means).T).argmax(1)] == 'informative'
    raw_gate = np.asarray(il)[a.argmax(1)] == 'informative'
    dn_leaf = shifted_scores(x, unit_vectors(frozen_human), np.asarray(protocol['image_mean'], dtype=np.float32))
    baseline = humanitarian(raw_gate, b, hl)
    trials = []
    for seed in protocol['rules']['grouping_seeds']:
        em, groups, _ = predict_binary(a, il, grouping_seed=seed, query_size=75, temperature=30., iterations=20, mm_iterations=1000)
        key = f'stata_gate_seed_{seed}'
        if groups.tolist() != stata['batch_ids'][key]: raise ValueError('EM và StatA khác nhóm')
        gate = combine(np.stack([norm, np.asarray(em)=='informative', np.asarray(stata['gate_labels'][key])=='informative'], 1))['majority']
        trials.append({'seed': seed, 'groups': groups, 'gate': gate,
                       'predictions': {'two_stage_original': baseline,
                           'majority': humanitarian(gate, b, hl),
                           'majority_dn_unit': humanitarian(gate, dn_leaf, hl)}})
    return trials, raw_gate


def run(directory, frozen_plan=ROOT/PLAN, output_dir=None):
    plan = load_plan(frozen_plan); protocol = plan['protocol']
    source = load_features(directory, allowed_splits=('test',))
    if source['manifest']['model'] != protocol['model'] or source['manifest']['seed'] != 0:
        raise ValueError('Test khác backbone/prompt seed đã khóa')
    if source['info_labels'] != protocol['info_labels'] or source['labels'] != protocol['human_labels']:
        raise ValueError('Test khác taxonomy')
    for task, variant in [('informativeness','semantic'), ('humanitarian','semantic_waffle')]:
        spec = json.loads(read_artifact(Path(directory), task+'/frozen_plan.json'))
        if spec['prompt_banks'][variant] != protocol['prompt_banks'][task]: raise ValueError('Prompt bank test thay đổi')
    if source['binary_truth'] is None: raise ValueError('Protocol test này yêu cầu hai bộ nhãn ảnh thật')
    if len(source['rows']) > protocol['rules']['max_test_samples']: raise ValueError('Test vượt trần đã khóa')
    disjoint = check_holdout(source['cohort'], protocol['exclusion'], 'test')
    # Dự đoán xong mới sử dụng ground truth để tính metric.
    trials, raw_gate = predict_locked(source['features'], *source['scores'], source['mean_prototypes'], protocol)
    output = Path(output_dir) if output_dir else ROOT/'results/analysis/clip_final'/uuid.uuid4().hex[:12]
    output.mkdir(parents=True, exist_ok=False)
    manifest = {'status': 'running', 'protocol_sha256': plan['protocol_sha256'], 'source_run': str(directory),
                'source_feature_sha256': source['feature_sha256'], 'disjoint': disjoint,
                'source_manifest_sha256': sha256(read_artifact(Path(directory),'run_manifest.json').encode()),
                'frozen_plan_sha256': sha256(Path(frozen_plan).read_bytes()), 'label_fit': False,
                'binary_ground_truth_available': True, 'selection_after_test': False}
    write_json(output/'run_manifest.json', manifest)
    try:
        truth = [r['y_true'] for r in source['rows']]; il, hl = source['info_labels'], source['labels']
        reports = {name: [] for name in protocol['rules']['methods']}
        for trial in trials:
            for name, predicted in trial['predictions'].items():
                gate = raw_gate if name == 'two_stage_original' else trial['gate']
                report = {'seed': trial['seed'], 'humanitarian': metric(truth, predicted, hl),
                    'binary': metric(source['binary_truth'], np.where(gate,'informative','not_informative').tolist(), il)}
                reports[name].append(report)
            with (output/f'predictions_seed_{trial["seed"]}.tsv').open('x', newline='') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerow(['id','y_true','binary_truth','majority_gate','batch_id',*protocol['rules']['methods']])
                writer.writerows((row['id'],row['y_true'],source['binary_truth'][i],bool(trial['gate'][i]),int(trial['groups'][i]),
                    *[trial['predictions'][name][i] for name in protocol['rules']['methods']]) for i,row in enumerate(source['rows']))
        analysis = {'samples': len(truth), 'source_split': 'test', 'by_method': reports,
            'mean': {name: {task: {key: float(np.mean([r[task][key] for r in values])) for key in ['accuracy','macro_f1']}
                for task in ['binary','humanitarian']} for name,values in reports.items()},
            'limitations': protocol['limits'], 'decision': 'Report all frozen methods; do not pick a new method/seed/threshold using test.'}
        write_json(output/'analysis.json', analysis)
        with (output/'final_dn_unit_summary.tsv').open('x', newline='') as f:
            writer = csv.writer(f, delimiter='\t');writer.writerow(['task','method','samples','accuracy','macro_f1'])
            for name,tasks in analysis['mean'].items():
                for task,values in tasks.items(): writer.writerow([task,name,len(truth),values['accuracy'],values['macro_f1']])
        manifest['status'] = 'complete'
        manifest['output_sha256'] = {str(p.relative_to(output)): sha256(p.read_bytes()) for p in output.rglob('*') if p.is_file() and p.name!='run_manifest.json'}
        write_json(output/'run_manifest.json', manifest)
        print('[FINAL TEST]', json.dumps(analysis['mean']), flush=True)
        return output
    except Exception as error:
        manifest.update(status='failed', error=f'{type(error).__name__}: {error}');write_json(output/'run_manifest.json', manifest);raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', required=True); parser.add_argument('--frozen-plan', default=str(ROOT/PLAN))
    parser.add_argument('--output-dir');args=parser.parse_args();run(args.source_run,args.frozen_plan,args.output_dir)
