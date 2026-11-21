"""Thí nghiệm dev 2way riêng: StatA và fusion ảnh–tiêu đề một tầng."""
import time

import numpy as np
from scipy.special import softmax
import yaml

from . import data
from .backbone import Backbone, encode_cohort
from .diagnostic import infer_groups, probability
from .evaluate import metric, publish, table
from .fakeddit import profile
from .io import ROOT, code_hashes, new_run, sha256, write_json
from .methods import fuse

SPEC_FILE = 'configs/fakeddit_task1.yaml'


def specification():
    spec = yaml.safe_load((ROOT / SPEC_FILE).read_text())
    fixed = {'version': 1, 'samples': 3000, 'grouping_seeds': [0, 1, 2],
             'adaptation_batch_size': 75, 'cpu_threads': 1,
             'image_logit_scale': 100., 'text_logit_scale': 25.,
             'clip_weight': .5, 'image_weight': .75, 'timeout_seconds': 7200}
    if spec != fixed:
        raise ValueError('Giữ cấu hình Task 1 đã định trước: 3000 dev, seed 0/1/2, fusion 1/2 và 3/4')
    return spec


def unit(values):
    values = np.asarray(values)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.isfinite(values).all() or (norms <= 1e-12).any():
        raise ValueError('Embedding/prototype rỗng hoặc không hữu hạn')
    return values / norms


def predict(features, titles, image_prototypes, text_prototypes, spec):
    """Không nhận nhãn. Trả posterior và group ID theo thứ tự mẫu gốc."""
    features, titles = unit(features), unit(titles)
    image_prototypes, text_prototypes = unit(image_prototypes), unit(text_prototypes)
    if (features.shape != titles.shape or image_prototypes.shape != text_prototypes.shape
            or image_prototypes.shape != (2, features.shape[1])):
        raise ValueError('Task 1 cần embedding cùng chiều và hai prototype theo thứ tự fake/real')
    clip = probability(softmax(spec['image_logit_scale'] * features @ image_prototypes.T, axis=1))
    text = probability(softmax(spec['text_logit_scale'] * titles @ text_prototypes.T, axis=1))
    baseline = fuse(clip, text, spec['image_weight'])
    results = [('clip_image', None, clip), ('clip_title', None, text),
               ('clip_image_title', None, baseline)]
    groups = {}
    for seed in spec['grouping_seeds']:
        stata, group = infer_groups(features, image_prototypes, clip,
                                   {**spec, 'grouping_seed': seed, 'temperatures': {'stata': 1.}})
        image = fuse(clip, stata, spec['clip_weight'])
        results.extend([('stata_image', seed, stata), ('clip_stata_image', seed, image),
                        ('clip_stata_image_title', seed, fuse(image, text, spec['image_weight']))])
        groups[f'groups_s{seed}'] = group
    return results, groups


def run(settings, split, root, smoke=False, backend_factory=Backbone):
    if split != 'dev':
        raise ValueError('Thí nghiệm Task 1 chỉ dùng dev; không chạy train/test hoặc khóa test')
    spec = specification()
    prompts = profile(settings)
    rows, cohort = data.fakeddit(root, split, smoke)
    expected = 5 if smoke else spec['samples']
    if len(rows) != expected:
        raise ValueError(f'Task 1 cần đúng {expected} mẫu')
    output = new_run(settings, 'fakeddit_task1', split, smoke)
    image_bank = prompts['image_prompts']['2way']['semantic_v1']
    text_bank = prompts['text_prompts']['2way']
    plan = {'kind': 'fakeddit_task1', 'dataset': 'fakeddit_2way', 'split': split,
            'mode': 'smoke' if smoke else 'full', 'samples': len(rows), 'stages': 1,
            'model': settings['model'], 'classes': ['fake', 'real'], 'cohort': cohort,
            'specification': spec, 'specification_sha256': sha256(ROOT / SPEC_FILE),
            'code_sha256': code_hashes(), 'profile_sha256': sha256(settings['fakeddit']['prompt_file']),
            'requirements_sha256': sha256(ROOT / 'requirements.txt'),
            'image_prompts': image_bank, 'text_prompts': text_bank,
            'batch_size': settings['batch_size'],
            'prototype_normalization': 'unit mean of unit prompt embeddings for both image and title banks',
            'title_tokenization': 'existing backbone tokenizer/context; titles may be truncated, prompts strict',
            'scope': 'Development experiment, includes earlier pilot samples; no independent confirmation or test lock.',
            'decision': 'one argmax on two class probabilities, no binary gate or hierarchy',
            'transductive': 'StatA on unlabeled image groups of 75, seeds 0/1/2; no model weight updates'}
    write_json(output / 'frozen_plan.json', plan)
    write_json(output / 'cohort.json', rows)
    write_json(output / 'run.json', {**plan, 'status': 'running'})
    start = time.perf_counter()
    backend = backend_factory(settings['device'])
    try:
        image_prototypes = unit(backend.prototypes(image_bank))
        text_prototypes = unit(backend.prototypes(text_bank))
        features, titles = encode_cohort(backend, rows, settings['batch_size'], with_titles=True)
        plan['model_runtime'] = getattr(backend, 'metadata', {'backend_class': type(backend).__name__})
    finally:
        backend.unload()
    encoding_seconds = time.perf_counter() - start
    results, groups = predict(features, titles, image_prototypes, text_prototypes, spec)
    # Nhãn chỉ được lấy để đánh giá sau khi mọi posterior đã tạo xong.
    truth = np.asarray([int(row['2way']) for row in rows])
    clip_pred = results[0][2].argmax(1)
    baseline_pred = results[2][2].argmax(1)
    records, predictions, diagnostics, comparisons, details, arrays = [], [], [], [], {}, {}
    for method, seed, posterior in results:
        posterior = probability(posterior)
        predicted = posterior.argmax(1)
        values = metric(truth, predicted, [0, 1])  # Top-5 vô nghĩa trên 2 lớp.
        key = method if seed is None else f'{method}_s{seed}'
        details[key] = values.pop('per_class')
        records.append({'task': '2way', 'method': method, 'seed': seed, 'samples': len(rows), **values})
        comparison = {'method': method, 'seed': seed, **values}
        for name, reference in [('clip', clip_pred), ('image_title', baseline_pred)]:
            comparison.update({f'{name}_wrong_to_correct': int(((reference != truth) & (predicted == truth)).sum()),
                               f'{name}_correct_to_wrong': int(((reference == truth) & (predicted != truth)).sum()),
                               f'delta_{name}_accuracy_pp': float(100 * ((predicted == truth).mean() - (reference == truth).mean()))})
        comparisons.append(comparison)
        diagnostics.append({'method': method, 'seed': seed,
                            'mean_confidence': float(posterior.max(1).mean()),
                            'mean_normalized_entropy': float(-(posterior * np.log(np.maximum(posterior, 1e-300))).sum(1).mean() / np.log(2)),
                            'nll': float(-np.log(np.maximum(posterior[np.arange(len(rows)), truth], 1e-300)).mean())})
        predictions.extend({'task': '2way', 'method': method, 'seed': seed, 'id': row['id'],
                            'truth': int(label), 'predicted': int(pred), 'p_fake': float(p[0]), 'p_real': float(p[1])}
                           for row, label, pred, p in zip(rows, truth, predicted, posterior))
        arrays['p_' + key] = posterior
        print(f'{method} seed={seed}: accuracy={values["accuracy"]:.4f}, macro_f1={values["macro_f1"]:.4f}', flush=True)
    table(output / 'comparisons.tsv', comparisons)
    write_json(output / 'branch_diagnostics.json', diagnostics)
    np.savez_compressed(output / 'replay_scores.npz', labels=truth, ids=np.asarray([r['id'] for r in rows]),
                        class_names=np.asarray(['fake', 'real']), global_features=features, title_features=titles,
                        image_prototypes=image_prototypes, text_prototypes=text_prototypes, **groups, **arrays)
    plan.update(encoding_seconds=encoding_seconds, elapsed_seconds=time.perf_counter() - start,
                label_usage='evaluation only after all predictions; fixed weights, no grid search')
    return publish(output, records, predictions, details, plan)
