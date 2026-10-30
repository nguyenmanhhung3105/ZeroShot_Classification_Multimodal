"""CALIP opt-in: một lượt encoder cho global/patch, đối chứng cached cùng cohort."""
import argparse
from copy import deepcopy
import csv
import importlib.metadata
import json
from pathlib import Path
import sys
import uuid
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
import numpy as np
import torch
import yaml
from experiments.clip_calip.core import score, PARAMETERS, REFERENCE_COMMIT
from experiments.clip_calip.extract import encode
from experiments.clip_calip.prepare import load_reference, ASSET
from experiments.waffle_lab import run as lab
from experiments.clip_stata.core import encode_prototypes
from experiments.clip_transductive.ensemble import humanitarian
from experiments.clip_transductive.run import metric
from results_layout import run_path

METHODS = ['two_stage_original', 'majority_raw_leaf', 'majority_unit_leaf', 'majority_dn_unit',
           'majority_calip_leaf', 'calip_gate_dn_leaf']
CODE = ['experiments/clip_calip/core.py', 'experiments/clip_calip/extract.py',
        'experiments/clip_calip/run.py', 'experiments/clip_calip/prepare.py',
        'experiments/clip_stata/core.py', 'experiments/waffle_lab/run.py', 'src/models/model_registry.py']


def classify(info_scores, human_scores, calip_info, calip_human, mean_human, features,
             controls, info_labels, labels):
    raw = np.asarray(info_labels)[info_scores.argmax(1)] == 'informative'
    calip = np.asarray(info_labels)[calip_info.argmax(1)] == 'informative'
    unit = mean_human/np.linalg.norm(mean_human, axis=1, keepdims=True)
    choices = {}
    for seed in [0, 1, 2]:
        majority = np.asarray(controls['majority_gates'][str(seed)], dtype=bool)
        candidates = {'two_stage_original': (raw, human_scores), 'majority_raw_leaf': (majority, human_scores),
                      'majority_unit_leaf': (majority, features@unit.T),
                      'majority_dn_unit': (majority, np.asarray(controls['dn_leaf'])),
                      'majority_calip_leaf': (majority, calip_human),
                      'calip_gate_dn_leaf': (calip, np.asarray(controls['dn_leaf']))}
        choices[seed] = {name: {'humanitarian': humanitarian(gate, leaf, labels),
                               'binary': np.where(gate, 'informative', 'not_informative').tolist()}
                         for name, (gate, leaf) in candidates.items()}
    return choices


def prefix_controls(reference, n):
    return {'dn_leaf': reference['dn_leaf'][:n],
            'majority_gates': {k: v[:n] for k, v in reference['majority_gates'].items()}}


def validate(settings, ref, base):
    lab.validate(settings)
    data = base['datasets']['crisismmd']; split = data.get('source_split')
    if (settings.get('calip') is not True or settings['backbone'] != 'metaclip2_vith14' or settings['seed'] != 0 or
            settings['task'] != 'humanitarian' or split not in ['train', 'dev'] or
            data.get('final_dn_unit_plan') or
            any(settings.get(k) for k in ['multimodal', 'full_frame_view', 'stata_gate', 'export_features', 'tda_evaluate'])):
        raise ValueError('CALIP riêng, MetaCLIP2 seed 0, train/dev, không dùng frozen test hoặc flags khác')
    if settings['max_samples'] not in [5, ref['payload']['cohorts'][split]['samples']]:
        raise ValueError('CALIP chỉ smoke 5 hoặc đủ cohort train/dev đã khóa')
    expected = 'exploratory' if split == 'train' else 'heldout'
    if settings.get('evaluation_scope', 'exploratory') != expected:
        raise ValueError('Scope không khớp train/dev')
    return split


def load_backend(device):
    from models.model_registry import load_model
    backend = load_model('metaclip2_vith14', device=device)
    backend.model.requires_grad_(False)
    return backend


def run(settings, prepare_only=False):
    ref = load_reference(lab.project_path(ASSET)); payload = ref['payload']
    base = yaml.safe_load(lab.project_path(settings['base_config']).read_text())
    split = validate(settings, ref, base); n = settings['max_samples']; reference = payload['cohorts'][split]
    output = (ROOT/run_path('clip_calip_heldout' if split == 'dev' else 'clip_calip', 'crisismmd',
                           'humanitarian', n, 'metaclip2_vith14', uuid.uuid4().hex[:12])
              if settings.get('output_dir', 'auto') == 'auto' else lab.project_path(settings['output_dir']))
    output.mkdir(parents=True, exist_ok=False)
    manifest = {'status': 'preparing', 'settings': settings, 'model': 'metaclip2_vith14',
                'source_split': split, 'samples': n, 'input': 'image_only', 'label_fit': False,
                'parameters': PARAMETERS, 'reference_commit': REFERENCE_COMMIT,
                'reference_fingerprint': ref['sha256'], 'methods': METHODS,
                'code_sha256': {p: lab.file_hash(ROOT/p) for p in CODE},
                'scope': 'smoke_prefix_of_fixed_controls' if n == 5 else 'full_cached_reference_cohort',
                'binary_ground_truth_available': reference['binary_truth'] is not None}
    lab.write_json(output/'run_manifest.json', manifest)
    lab.write_json(output/'frozen_plan.json', manifest)
    lab.write_json(output/'reference.json', ref)
    backend = None
    try:
        if prepare_only:
            manifest['status'] = 'prepared_no_inference'; lab.write_json(output/'run_manifest.json', manifest)
            print('[PREPARED]', output, 'no dataset/model access'); return output
        data = base['datasets']['crisismmd']; cohorts = []
        human_rows = lab.read_cohort(settings, base)
        if data.get('gate_labels_available', True):
            info_settings = {**settings, 'task': 'informativeness'}
            info_rows = lab.read_cohort(info_settings, base)
        else:
            info_rows = deepcopy(human_rows)
            for row in info_rows: row['y_true'] = None
        for task, rows in [('informativeness', info_rows), ('humanitarian', human_rows)]:
            if ([r['id'] for r in rows] != reference['ids'][:n] or
                    [r['image_sha256'] for r in rows] != reference['image_sha256'][:n]):
                raise ValueError('Cohort/ảnh/thứ tự khác đối chứng đã khóa')
            expected = reference['human_truth'][:n] if task == 'humanitarian' else (
                reference['binary_truth'][:n] if reference['binary_truth'] is not None else [None]*n)
            if [r['y_true'] for r in rows] != expected: raise ValueError('Nhãn/scope khác đối chứng')
            folder = output/task; folder.mkdir()
            lab.write_json(folder/'cohort.json', {'rows': rows, 'cohort_fingerprint': lab.digest(rows), 'samples': n})
            cohorts.append(rows)
        manifest['status'] = 'running'; lab.write_json(output/'run_manifest.json', manifest)
        device = settings.get('device', 'auto')
        backend = load_backend(None if device == 'auto' else device)
        manifest['inference_backend'] = ('metaclip2_real' if type(backend).__module__ == 'models.model_registry'
                                         else 'synthetic_or_custom')
        if backend.logit_scale != 100.: raise ValueError('Scale khác checkpoint đối chứng')
        il, hl = payload['info_labels'], payload['human_labels']
        means = [np.asarray(payload[k], dtype='float32') for k in ['info_mean_prototypes', 'human_mean_prototypes']]
        for task, labels, expected in zip(['informativeness', 'humanitarian'], [il, hl], means):
            actual = encode_prototypes(backend, payload['banks'][task], labels, settings['batch_size'])
            np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=1e-4)
        prototypes = [torch.tensor(m/np.linalg.norm(m, axis=1, keepdims=True), device=backend.device) for m in means]
        vectors, logits, terms, attentions = [], [[], []], [[], []], [[], []]
        verification = {}; extraction = []
        from PIL import Image
        for start in range(0, n, settings['batch_size']):
            images = []
            try:
                for row in human_rows[start:start+settings['batch_size']]:
                    with Image.open(lab.project_path(row['image_path'])) as image: images.append(image.convert('RGB'))
                g, p, details = encode(backend, images, verify_global=start == 0)
                vectors.append(g.cpu().numpy()); extraction.append(details)
                for k, t in enumerate(prototypes):
                    scores, pieces, attention = score(g, p, t)
                    logits[k].append(scores.cpu().numpy()); terms[k].append(pieces.cpu().numpy())
                    attentions[k].append(attention.cpu().numpy().astype('float16'))
                if start == 0:
                    verification = {'global_features': g[:5].cpu().numpy(), 'patches': p[:5].cpu().numpy(),
                                    'info_prototypes': prototypes[0].cpu().numpy(), 'human_prototypes': prototypes[1].cpu().numpy()}
            finally:
                for image in images: image.close()
            print(f'[CALIP] image {min(start+settings["batch_size"], n)}/{n}', flush=True)
        x = np.concatenate(vectors); x /= np.linalg.norm(x, axis=1, keepdims=True)
        raw = [x@m.T for m in means]; errors = []
        for observed, key in zip(raw, ['info_scores', 'human_scores']):
            expected = np.asarray(reference[key][:n], dtype='float32')
            np.testing.assert_allclose(observed, expected, atol=2e-5, rtol=1e-4)
            np.testing.assert_array_equal(observed.argmax(1), expected.argmax(1))
            errors.append(float(abs(observed-expected).max()))
        logits, terms, attentions = [[np.concatenate(v) for v in group] for group in [logits, terms, attentions]]
        for k in [0, 1]:
            np.testing.assert_allclose(terms[k][:, 0], 100.*x@(means[k]/np.linalg.norm(means[k], axis=1, keepdims=True)).T,
                                       atol=3e-5, rtol=2e-5)
        controls = prefix_controls(reference, n)
        predictions = classify(*raw, *logits, means[1], x, controls, il, hl)
        truth = [r['y_true'] for r in human_rows]
        binary_truth = reference['binary_truth'][:n] if reference['binary_truth'] is not None else None
        reports = {name: [] for name in METHODS}
        for seed, candidates in predictions.items():
            for name, pred in candidates.items():
                reports[name].append({'seed': seed, 'humanitarian': metric(truth, pred['humanitarian'], hl),
                                      'binary': metric(binary_truth, pred['binary'], il) if binary_truth is not None else None})
            with (output/f'predictions_seed_{seed}.tsv').open('x', newline='') as f:
                w = csv.writer(f, delimiter='\t'); w.writerow(['id', 'y_true', 'binary_truth', *METHODS,
                                                               *[name+'_binary' for name in METHODS]])
                w.writerows((row['id'], row['y_true'], binary_truth[i] if binary_truth is not None else '',
                             *[c[name]['humanitarian'][i] for name in METHODS], *[c[name]['binary'][i] for name in METHODS])
                            for i, (row, c) in enumerate((r, candidates) for r in human_rows))
        from experiments.clip_dpe.run import average
        analysis = {'by_method': reports, 'mean': {name: average(v) for name, v in reports.items()},
                    'limitations': ['Only current train/dev, no test, no hyperparameter search.',
                                    'CALIP root is deterministic; three repeats pair cached majority grouping seeds.',
                                    'Smoke uses prefix of full-cohort cached gates, no independent five-image adaptation.',
                                    'Attention is not a validated bounding box or segmentation.']}
        lab.write_json(output/'analysis.json', analysis)
        np.savez_compressed(output/'scores.npz', ids=np.asarray(reference['ids'][:n]), image_features=x,
                            info_mean_prototypes=means[0], human_mean_prototypes=means[1],
                            info_raw=raw[0], human_raw=raw[1], info_calip=logits[0], human_calip=logits[1],
                            info_terms=terms[0], human_terms=terms[1], info_attention=attentions[0], human_attention=attentions[1])
        np.savez_compressed(output/'verification_features.npz', **verification)
        lab.write_json(output/'extraction.json', extraction)
        with (output/'summary.tsv').open('x', newline='') as f:
            w = csv.writer(f, delimiter='\t'); w.writerow(['method', 'model', 'samples', 'accuracy', 'macro_f1', 'binary_macro_f1'])
            for name, report in analysis['mean'].items():
                h = report['humanitarian']; b = report['binary']
                w.writerow([name, 'metaclip2_vith14', n, h['accuracy'], h['macro_f1'], b['macro_f1'] if b else ''])
        manifest.update(status='complete', raw_score_max_abs_error=errors, extraction=extraction[0],
                        torch_version=torch.__version__)
        try: manifest['open_clip_version'] = importlib.metadata.version('open_clip_torch')
        except importlib.metadata.PackageNotFoundError: manifest['open_clip_version'] = 'synthetic_backend'
        manifest['output_sha256'] = {str(p.relative_to(output)): lab.file_hash(p) for p in output.rglob('*')
                                     if p.is_file() and p.name != 'run_manifest.json'}
        lab.write_json(output/'run_manifest.json', manifest)
        from experiments.clip_calip.analyze import analyze
        audit = analyze(output, allow_synthetic=manifest['inference_backend'] != 'metaclip2_real')
        lab.write_json(output/'audit.json', audit)
        manifest['output_sha256']['audit.json'] = lab.file_hash(output/'audit.json')
        lab.write_json(output/'run_manifest.json', manifest)
        print('[AUDIT] passed'); print('[SAVED]', output); return output
    except Exception as error:
        manifest.update(status='failed', error=f'{type(error).__name__}: {error}')
        lab.write_json(output/'run_manifest.json', manifest); raise
    finally:
        if backend is not None: backend.unload()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--config', required=True)
    p.add_argument('--prepare-only', action='store_true'); a = p.parse_args()
    run(yaml.safe_load(Path(a.config).read_text()), a.prepare_only)
