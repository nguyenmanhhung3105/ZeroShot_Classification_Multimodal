"""One image forward for paired Fakeddit 2way/6way controls and majority+CALIP."""
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
import yaml
from scipy.special import softmax
from PIL import Image
from experiments.fakeddit_calip.prepare import load_reference, ASSET, SEMANTIC
from experiments.fakeddit_calip.core import majority, hierarchy
from experiments.clip_calip.core import score, PARAMETERS
from experiments.clip_calip.extract import encode
from experiments.clip_calip.run import load_backend
from experiments.clip_stata.core import encode_prototypes
from experiments.clip_transductive.run import metric
from experiments.waffle_lab.run import project_path, write_json, file_hash


def validate(job):
    if (job.get('kind') != 'standard' or job.get('dataset') != 'fakeddit' or job.get('task') != '6way'
            or job.get('models') != ['metaclip2_vith14'] or job.get('fakeddit_calip') is not True
            or job.get('max_samples') not in [5, 3000] or job.get('ensemble', {}).get('enabled')
            or job.get('image_text_relation', {}).get('enabled') is not False
            or job.get('result_method') != 'fakeddit_calip'):
        raise ValueError('Fakeddit CALIP: paired 2way/6way, MetaCLIP2, 5 or 3000, relation off')


def run(base, job, output, prepare_only=False):
    validate(job)
    ref = load_reference(ROOT/ASSET); payload = ref['payload']; n = job['max_samples']
    if file_hash(ROOT/SEMANTIC) != payload['semantic_sha256']:
        raise ValueError('Semantic prompts changed after freezing')
    output = project_path(output); output.mkdir(parents=True, exist_ok=False)
    manifest = {'status': 'prepared_no_inference', 'model': payload['model'], 'samples': n,
                'input': 'image_only', 'label_fit': False, 'evaluation_scope': 'validate_exploratory',
                'reference_sha256': ref['sha256'], 'parameters': PARAMETERS,
                'grouping_seeds': [0, 1, 2], 'prompt_seed': 0,
                'baseline_note': 'Cached multimodal baseline is contextual; legacy image branch is paired control.',
                'image_identity_limit': 'Old baseline records file size, not image content hash; image branch probabilities are replay-checked.',
                'code_sha256': {str(p.relative_to(ROOT)): file_hash(p) for directory in
                               ['experiments/fakeddit_calip', 'experiments/clip_calip', 'experiments/clip_stata',
                                'experiments/clip_transductive'] for p in (ROOT/directory).glob('*.py')}}
    write_json(output/'run_manifest.json', manifest)
    write_json(output/'frozen_prompts.json', {t: p['banks'] for t, p in payload['tasks'].items()})
    if prepare_only:
        print('[PREPARED]', output, 'no dataset/image/model access'); return output
    data = base['datasets']['fakeddit']
    # The original baseline snapshot already locks sample order and label IDs.
    # No processed TSV scan, no caption sent to any scorer, no label-derived prompt.
    if data.get('source_split') != payload['tasks']['6way']['source_split']:
        raise ValueError('Split differs from frozen baseline')
    backend = None
    try:
        manifest['status'] = 'running'; write_json(output/'run_manifest.json', manifest)
        rows = payload['tasks']['6way']['rows'][:n]
        root = project_path(data['image_dir']); cohort = []
        for row in rows:
            path = project_path(row['image_path'])
            if not path.is_relative_to(root) or path.stat().st_size != row['image_stat']['size']:
                raise ValueError('Image missing, outside processed root, or size changed: '+row['id'])
            cohort.append({'id': row['id'], 'image_path': row['image_path'], 'sha256': file_hash(path)})
        write_json(output/'cohort.json', cohort)
        backend = load_backend(None)
        prototypes = {}; calip = {t: [] for t in payload['tasks']}; vectors = []; extraction = []
        for task, plan in payload['tasks'].items():
            prototypes[task] = {name: encode_prototypes(backend, bank, plan['labels'], 8)
                                for name, bank in plan['banks'].items()}
        for start in range(0, n, 8):
            images = []
            try:
                for row in rows[start:start+8]:
                    with Image.open(project_path(row['image_path'])) as im: images.append(im.convert('RGB'))
                g, patches, details = encode(backend, images, verify_global=start == 0)
                vectors.append(g.cpu().numpy()); extraction.append(details)
                for task in payload['tasks']:
                    means = prototypes[task]['semantic_waffle']
                    unit = means/np.linalg.norm(means, axis=1, keepdims=True)
                    logits, _, _ = score(g, patches, torch.tensor(unit, device=backend.device))
                    calip[task].append(logits.cpu().numpy())
            finally:
                for im in images: im.close()
            print(f'[FAKEDDIT CALIP] {min(start+8,n)}/{n}', flush=True)
        backend.unload(); backend = None
        features = np.concatenate(vectors)
        calip = {t: np.concatenate(v) for t, v in calip.items()}
        predictions = {}; arrays = {'image_features': features}; errors = {}
        for task, plan in payload['tasks'].items():
            labels = np.asarray(plan['labels']); predictions[task] = {}
            for bank, means in prototypes[task].items():
                raw = features@means.T
                unit = features@(means/np.linalg.norm(means, axis=1, keepdims=True)).T
                arrays[task+'_'+bank+'_means'] = means
                predictions[task][bank+'_mean_cosine'] = labels[raw.argmax(1)]
                predictions[task][bank+'_unit'] = labels[unit.argmax(1)]
                if bank == 'legacy':
                    expected = np.asarray([r['baseline_image_prob'] for r in plan['rows'][:n]])
                    observed = softmax(plan['logit_scale']*raw, axis=1)
                    np.testing.assert_allclose(observed, expected, atol=2e-4, rtol=2e-4)
                    errors[task] = float(abs(observed-expected).max())
            predictions[task]['cached_multimodal_baseline'] = np.asarray([r['baseline_pred'] for r in plan['rows'][:n]])
            predictions[task]['calip_flat'] = labels[calip[task].argmax(1)]
            arrays[task+'_calip'] = calip[task]
        for seed in [0, 1, 2]:
            pred, diagnostics = majority(features, prototypes['2way']['semantic'], seed)
            binary = np.asarray(payload['tasks']['2way']['labels'])[pred]
            predictions['2way'][f'majority_seed_{seed}'] = binary
            predictions['6way'][f'majority_calip_seed_{seed}'] = hierarchy(binary, calip['6way'], payload['tasks']['6way']['labels'])
            means = prototypes['6way']['semantic_waffle']
            predictions['6way'][f'majority_unit_seed_{seed}'] = hierarchy(binary, features@(means/np.linalg.norm(means, axis=1, keepdims=True)).T, payload['tasks']['6way']['labels'])
            arrays[f'votes_seed_{seed}'] = diagnostics['votes']; arrays[f'batch_ids_seed_{seed}'] = diagnostics['batch_ids']
        reports = {}; summary = []
        for task, candidates in predictions.items():
            truth = [r['label'] for r in payload['tasks'][task]['rows'][:n]]
            reports[task] = {name: metric(truth, pred, payload['tasks'][task]['labels']) for name, pred in candidates.items()}
            with (output/f'predictions_{task}.tsv').open('x', newline='') as f:
                w = csv.writer(f, delimiter='\t'); w.writerow(['id', 'y_true', *candidates])
                w.writerows([row['id'], truth[i], *[p[i] for p in candidates.values()]] for i, row in enumerate(rows))
            for name, report in reports[task].items():
                summary.append([task, name, n, report['accuracy'], report['macro_f1']])
            for prefix in (['majority'] if task == '2way' else ['majority_unit', 'majority_calip']):
                values = [reports[task][f'{prefix}_seed_{s}'] for s in [0, 1, 2]]
                agg = {key: {'mean': float(np.mean([v[key] for v in values])),
                             'std': float(np.std([v[key] for v in values]))} for key in ['accuracy', 'macro_f1']}
                reports[task][prefix+'_aggregate'] = agg
                summary.append([task, prefix+'_mean', n, agg['accuracy']['mean'], agg['macro_f1']['mean']])
        arrays['ids'] = np.asarray([r['id'] for r in rows])
        np.savez_compressed(output/'scores.npz', **arrays)
        write_json(output/'analysis.json', reports); write_json(output/'extraction.json', extraction)
        with (output/'summary.tsv').open('x', newline='') as f:
            w = csv.writer(f, delimiter='\t'); w.writerow(['task', 'method', 'samples', 'accuracy', 'macro_f1']); w.writerows(summary)
        manifest.update(status='complete', baseline_probability_max_abs_error=errors)
        manifest['output_sha256'] = {p.name: file_hash(p) for p in output.iterdir() if p.is_file() and p.name != 'run_manifest.json'}
        write_json(output/'run_manifest.json', manifest)
        print('[SAVED]', output); return output
    except Exception as error:
        manifest.update(status='failed', error=f'{type(error).__name__}: {error}')
        write_json(output/'run_manifest.json', manifest); raise
    finally:
        if backend is not None: backend.unload()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/experiment_config.yaml')
    parser.add_argument('--job-config', default='experiments/fakeddit_calip/launcher.yaml')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    base = yaml.safe_load(project_path(args.config).read_text())
    job = yaml.safe_load(project_path(args.job_config).read_text())
    if 'jobs' in job: job = job['jobs']['fakeddit_calip']
    output = base['output']['results_dir']
    if output == 'auto': output = 'results/fakeddit_calip/fakeddit/paired/samples_'+str(job['max_samples'])+'/metaclip2_vith14/'+uuid.uuid4().hex[:12]
    run(base, job, output, args.prepare_only)
