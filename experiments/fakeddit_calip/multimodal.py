"""Paired image+title fusion from verified saved scores; no model/dataset access."""
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
from scipy.special import softmax
from experiments.fakeddit_calip.prepare import ASSET, load_reference
from experiments.fakeddit_calip.analyze import analyze, metrics
from experiments.fakeddit_calip.core import hierarchy
from experiments.waffle_lab.run import file_hash, write_json, digest

DEFAULT_SOURCE = 'results/fakeddit_calip/fakeddit/6way/samples_3000/metaclip2_vith14/6way_a3f7a8a3a561'


def probabilities(values):
    p = np.asarray(values, dtype=np.float64)
    if (p.ndim != 2 or p.shape[1] < 2 or not np.isfinite(p).all() or np.any(p < 0)
            or not np.allclose(p.sum(1), 1, atol=2e-6, rtol=0)):
        raise ValueError('Expected finite normalized class probabilities')
    return p


def fuse(image, text, weight):
    image, text = probabilities(image), probabilities(text)
    if image.shape != text.shape or isinstance(weight, bool) or not np.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError('Fusion requires aligned probabilities and weight in [0,1]')
    return weight*image+(1-weight)*text


def replace_clip_vote(votes, multimodal_clip_prob):
    """Only the normalized CLIP voter sees title; EM/StatA remain image-only."""
    votes = np.asarray(votes)
    p = probabilities(multimodal_clip_prob)
    if (votes.shape != (len(p), 3) or p.shape[1] != 2 or not np.isin(votes, [0, 1]).all()):
        raise ValueError('Expected three binary votes per image')
    updated = votes.copy(); updated[:, 0] = p.argmax(1)
    return (updated.sum(1) >= 2).astype(int), updated


def decide(arrays, text, image_weights, labels):
    """Inference only. No ground truth or text contents are accepted."""
    x = arrays['image_features']
    image = {}; mixed = {}; pred = {'2way': {}, '6way': {}}
    for task in ['2way', '6way']:
        means = arrays[task+'_legacy_means']
        semantic = arrays[task+'_semantic_means']
        semantic /= np.linalg.norm(semantic, axis=1, keepdims=True)
        image[task] = {'legacy': softmax(100*(x@means.T), axis=1),
                       'semantic': softmax(100*(x@semantic.T), axis=1),
                       'calip': softmax(arrays[task+'_calip'], axis=1)}
        mixed[task] = {k: fuse(p, text[task], image_weights[task]) for k, p in image[task].items()}
        ll = np.asarray(labels[task])
        pred[task]['text_only_legacy'] = ll[np.asarray(text[task]).argmax(1)]
        for bank in image[task]:
            pred[task][bank+'_image_flat'] = ll[image[task][bank].argmax(1)]
            pred[task][bank+'_multimodal_flat'] = ll[mixed[task][bank].argmax(1)]
    gate_diagnostics = {}
    for seed in [0, 1, 2]:
        votes = arrays[f'votes_seed_{seed}']
        image_gate = np.asarray(labels['2way'])[(votes.sum(1) >= 2).astype(int)]
        mm_indices, mm_votes = replace_clip_vote(votes, mixed['2way']['semantic'])
        mm_gate = np.asarray(labels['2way'])[mm_indices]
        gate_diagnostics[str(seed)] = {'clip_vote_changes': int((votes[:, 0] != mm_votes[:, 0]).sum()),
                                      'majority_changes': int((image_gate != mm_gate).sum())}
        pred['2way'][f'majority_image_seed_{seed}'] = image_gate
        pred['2way'][f'majority_clip_text_seed_{seed}'] = mm_gate
        # Factorial controls isolate adding title to gate vs adding title to leaf.
        for gate_name, gate in [('image_gate', image_gate), ('clip_text_gate', mm_gate)]:
            for leaf in ['legacy', 'calip']:
                for mode, scores in [('image_leaf', image['6way'][leaf]), ('multimodal_leaf', mixed['6way'][leaf])]:
                    name = f'{gate_name}_{leaf}_{mode}_seed_{seed}'
                    pred['6way'][name] = hierarchy(gate, scores, labels['6way'])
    return pred, image, mixed, gate_diagnostics


def run(source, output):
    source = Path(source)
    audit = analyze(source, replay=False)
    ref = load_reference(ROOT/ASSET); payload = ref['payload']
    with zipfile.ZipFile(source/'run_artifacts.zip') as z:
        original_manifest = json.loads(z.read('run_manifest.json'))
        with np.load(io.BytesIO(z.read('scores.npz')), allow_pickle=False) as pack:
            arrays = {k: pack[k].copy() for k in pack.files}
    text = {}; baseline = {}; baseline_image = {}; weights = {}; metadata = {}
    n = len(arrays['ids'])
    for task, plan in payload['tasks'].items():
        path = ROOT/plan['baseline_source']
        if file_hash(path) != plan['baseline_zip_sha256']: raise ValueError('Baseline archive changed')
        with zipfile.ZipFile(path) as z:
            def read(suffix):
                names = [name for name in z.namelist() if name.endswith('/'+suffix)]
                if len(names) != 1: raise ValueError('Ambiguous baseline artifact')
                return z.read(names[0]).decode()
            manifest = json.loads(read('manifest.json')); snap = json.loads(read('evaluation_snapshot.json'))
            rows = list(csv.DictReader(io.StringIO(read('raw_predictions.tsv')), delimiter='\t'))
        if (manifest['model'] != 'metaclip2_vith14' or manifest['logit_scale'] != 100.
                or manifest['text_logit_scale_effective'] != 25 or manifest['relation_enabled']
                or manifest['image_weight'] != .75 or manifest['sample_set_id'] != plan['sample_set_id']
                or manifest['rendered_prompts'] != plan['banks']['legacy']):
            raise ValueError('Unexpected cached text/fusion protocol')
        if [r['id'] for r in rows] != arrays['ids'].tolist(): raise ValueError('Text/image order mismatch')
        if [r['y_true'] for r in rows] != [r['label'] for r in plan['rows']]: raise ValueError('Label mismatch')
        if snap['cohort']['samples'] != [{k: v for k, v in r.items() if k not in ['baseline_pred', 'baseline_image_prob']} for r in plan['rows']]:
            raise ValueError('Baseline text hash/cohort differs from frozen reference')
        if any(r['_text_encoded'] != 'True' or r['_text_was_empty'] != 'False' for r in rows):
            raise ValueError('This cohort expects a nonempty encoded title for every image')
        labels = plan['labels']
        text[task] = probabilities([[float(r['score_text_'+k]) for k in labels] for r in rows])
        baseline_image[task] = probabilities([[float(r['score_image_'+k]) for k in labels] for r in rows])
        baseline[task] = np.asarray([r['y_pred'] for r in rows])
        np.testing.assert_array_equal(baseline[task], [r['baseline_pred'] for r in plan['rows']])
        expected = np.asarray(labels)[fuse(baseline_image[task], text[task], .75).argmax(1)]
        np.testing.assert_array_equal(expected, baseline[task])
        weights[task] = .75
        metadata[task] = {'baseline_zip_sha256': plan['baseline_zip_sha256'], 'sample_set_id': plan['sample_set_id'],
                          'text_scale': 25, 'image_weight': .75, 'prompt_bank': 'legacy',
                          'title_representation': 'cached_MetaCLIP2_title_probabilities',
                          'truncation': manifest.get('truncation')}
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    frozen = {'source': str(source), 'source_manifest_sha256': digest(original_manifest),
              'reference_sha256': ref['sha256'], 'samples': n, 'tasks': ['2way', '6way'],
              'input': 'image_and_title_cached', 'label_fit': False, 'evaluation_scope': 'validate_exploratory',
              'parameters': metadata, 'grouping_seeds': [0, 1, 2],
              'majority_rule': 'Replace only CLIP vote with fused semantic-image/legacy-title probabilities; EM/StatA unchanged.',
              'direct_image_title_similarity': False, 'weight_search': False,
              'code_sha256': {name: file_hash(ROOT/name) for name in ['experiments/fakeddit_calip/multimodal.py',
                                                                  'experiments/fakeddit_calip/core.py']}}
    write_json(output/'frozen_plan.json', frozen)
    manifest = {**frozen, 'status': 'running'}; write_json(output/'run_manifest.json', manifest)
    try:
        labels = {t: p['labels'] for t, p in payload['tasks'].items()}
        # Copies prevent normalized semantic prototypes from mutating source arrays.
        pred, image, mixed, diagnostics = decide({k: v.copy() for k, v in arrays.items()}, text, weights, labels)
        reports = {}; summary = []; saved = {'ids': arrays['ids']}
        for task, candidates in pred.items():
            candidates['cached_multimodal_baseline'] = baseline[task]
            np.testing.assert_array_equal(candidates['legacy_multimodal_flat'], baseline[task])
            truth = np.asarray([r['label'] for r in payload['tasks'][task]['rows']])
            reports[task] = {}
            for name, values in candidates.items():
                report = metrics(truth, values, labels[task])
                report['fixed_vs_multimodal_baseline'] = int(((baseline[task] != truth)&(values == truth)).sum())
                report['broken_vs_multimodal_baseline'] = int(((baseline[task] == truth)&(values != truth)).sum())
                reports[task][name] = report
                summary.append([task, name, n, report['accuracy'], report['macro_f1'], '', ''])
            prefixes = sorted({name.rsplit('_seed_', 1)[0] for name in candidates if '_seed_' in name})
            for prefix in prefixes:
                group = [reports[task][f'{prefix}_seed_{seed}'] for seed in [0, 1, 2]]
                mean = {k: float(np.mean([r[k] for r in group])) for k in ['accuracy', 'macro_f1']}
                std = {k: float(np.std([r[k] for r in group])) for k in ['accuracy', 'macro_f1']}
                reports[task][prefix+'_aggregate'] = {'mean': mean, 'std': std}
                summary.append([task, prefix+'_mean', n, mean['accuracy'], mean['macro_f1'], std['accuracy'], std['macro_f1']])
            with (output/f'predictions_{task}.tsv').open('x', newline='') as f:
                w = csv.writer(f, delimiter='\t'); w.writerow(['id', 'y_true', *candidates])
                w.writerows([key, truth[i], *[v[i] for v in candidates.values()]] for i, key in enumerate(arrays['ids']))
            saved[task+'_text_prob'] = text[task]
            for name in image[task]:
                saved[task+'_'+name+'_image_prob'] = image[task][name]
                saved[task+'_'+name+'_multimodal_prob'] = mixed[task][name]
        np.savez_compressed(output/'probabilities.npz', **saved)
        write_json(output/'analysis.json', reports); write_json(output/'gate_diagnostics.json', diagnostics)
        write_json(output/'source_audit.json', audit)
        with (output/'summary.tsv').open('x', newline='') as f:
            w = csv.writer(f, delimiter='\t'); w.writerow(['task', 'method', 'samples', 'accuracy', 'macro_f1', 'accuracy_std', 'macro_f1_std']); w.writerows(summary)
        manifest['status'] = 'complete'
        manifest['output_sha256'] = {p.name: file_hash(p) for p in output.iterdir() if p.name != 'run_manifest.json'}
        write_json(output/'run_manifest.json', manifest)
        print('[SAVED]', output); return output
    except Exception as error:
        manifest.update(status='failed', error=f'{type(error).__name__}: {error}')
        write_json(output/'run_manifest.json', manifest); raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default=DEFAULT_SOURCE); parser.add_argument('--output')
    args = parser.parse_args()
    output = args.output or str(ROOT/'results/fakeddit_multimodal/fakeddit/paired/samples_3000/metaclip2_vith14'/uuid.uuid4().hex[:12])
    run(args.source, output)
