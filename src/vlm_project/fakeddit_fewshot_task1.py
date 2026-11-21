"""Few-shot thật/giả trên min5: ba modality, shared support, frozen MetaCLIP2."""
import importlib.metadata
import re
import time
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
import yaml

from . import data
from .backbone import Backbone, encode_cohort
from .evaluate import metric, publish, table
from .io import ROOT, code_hashes, digest, new_run, path, sha256, write_json

SPEC_FILE = 'configs/fakeddit_fewshot_task1_min5.yaml'


def specification():
    spec = yaml.safe_load((ROOT / SPEC_FILE).read_text())
    fixed = {'version': 1, 'train_samples': 3000, 'dev_samples': 3000,
             'shots_per_6way_class': 64, 'seeds': list(range(5)),
             'methods': ['image', 'title', 'image_title'], 'C': 1.,
             'class_weight': 'balanced', 'solver': 'lbfgs', 'max_iter': 2000,
             'tol': 1e-6, 'timeout_seconds': 7200,
             'dataset_slugs': {'train': 'vlm-fakeddit-train-min5-v1', 'dev': 'vlm-fakeddit-dev-min5-v1'}}
    if {k: v for k, v in spec.items() if k != 'manifest_sha256'} != fixed:
        raise ValueError('Giữ protocol few-shot min5: 64/lớp 6way, 5 seed và ba modality đã định trước')
    hashes = spec.get('manifest_sha256', {})
    if set(hashes) != {'train', 'dev'} or any(not re.fullmatch(r'[0-9a-f]{64}', h) for h in hashes.values()):
        raise ValueError('Cần hash manifest của đúng train/dev min5')
    return spec


def select_support(y6, shots, seed):
    """Giữ đúng thứ tự RNG và cách lấy mẫu của experiment few-shot cũ."""
    y6 = np.asarray(y6)
    if y6.ndim != 1 or not np.isin(y6, range(6)).all() or type(shots) is not int or shots < 1:
        raise ValueError('Nhãn support phải là 6way và shots dương')
    rng = np.random.default_rng(seed)
    selected = []
    for label in range(6):
        indices = np.flatnonzero(y6 == label)
        if len(indices) < shots:
            raise ValueError(f'Train class {label}: {len(indices)} < {shots}-shot')
        selected.extend(rng.permutation(indices)[:shots])
    return np.asarray(selected, dtype=np.int64)


def feature_banks(image, title):
    image, title = np.asarray(image), np.asarray(title)
    if (image.ndim != 2 or image.shape != title.shape or not len(image)
            or not np.isfinite(image).all() or not np.isfinite(title).all()
            or not np.allclose(np.linalg.norm(image, axis=1), 1., atol=1e-5)
            or not np.allclose(np.linalg.norm(title, axis=1), 1., atol=1e-5)):
        raise ValueError('Feature ảnh/tiêu đề phải unit, hữu hạn và cùng shape')
    return {'image': image, 'title': title,
            'image_title': np.concatenate([image, title], axis=1) / np.sqrt(2)}


def probe(x, y, spec):
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        return LogisticRegression(C=spec['C'], class_weight=spec['class_weight'],
                                  solver=spec['solver'], max_iter=spec['max_iter'], tol=spec['tol']).fit(x, y)


def predict(train_image, train_title, train_y6, dev_image, dev_title, supports, spec, expected_quotas=None):
    """Chỉ nhận nhãn TRAIN; không có nhãn dev hoặc bước tuning ngưỡng."""
    train = feature_banks(train_image, train_title)
    dev = feature_banks(dev_image, dev_title)
    y = (np.asarray(train_y6) != 0).astype(int)  # Giữ mapping gate cũ: 0=Real, 1=Fake.
    if len(y) != len(train_image) or train_image.shape[1] != dev_image.shape[1]:
        raise ValueError('Nhãn train hoặc chiều feature không khớp')
    results, coefficients = [], {}
    for seed in spec['seeds']:
        indices = np.asarray(supports[seed], dtype=np.int64)
        if len(set(indices.tolist())) != len(indices) or (indices < 0).any() or (indices >= len(y)).any():
            raise ValueError('Support trùng hoặc nằm ngoài train features')
        target = expected_quotas[seed] if expected_quotas is not None else np.repeat(spec['shots_per_6way_class'], 6)
        if not np.array_equal(np.bincount(np.asarray(train_y6)[indices], minlength=6), target):
            raise ValueError('Support không đúng ngân sách từng lớp 6way')
        for method in spec['methods']:
            print(f'Fit {method}, seed={seed}, support={len(indices)}', flush=True)
            classifier = probe(train[method][indices], y[indices], spec)
            if classifier.classes_.tolist() != [0, 1]:
                raise ValueError('Classifier sai thứ tự Real/Fake')
            posterior = classifier.predict_proba(dev[method])
            results.append((method, seed, posterior))
            coefficients[f'{method}_s{seed}_coef'] = classifier.coef_
            coefficients[f'{method}_s{seed}_intercept'] = classifier.intercept_
    return results, coefficients


def run(settings, train_root, dev_root, smoke=False, backend_factory=Backbone, *, balance_ablation=False):
    if balance_ablation:
        from .fakeddit_fewshot_support import specification as support_specification, SPEC_FILE as spec_file, select_balanced_support, quotas
        spec = support_specification()
    else:
        spec = specification()
        spec_file = SPEC_FILE
    for split, root in [('train', train_root), ('dev', dev_root)]:
        if sha256(path(root) / 'manifest.json') != spec['manifest_sha256'][split]:
            raise ValueError(f'Sai cohort {split} min5; không dùng dataset v2 cũ hoặc test')
    train, train_meta = data.fakeddit(train_root, 'train')
    dev, dev_meta = data.fakeddit(dev_root, 'dev', smoke)
    if len(train) != spec['train_samples'] or len(dev) != (5 if smoke else spec['dev_samples']):
        raise ValueError('Sai số mẫu train/dev')
    for key in ['id', 'image_sha256', 'text_sha256']:
        if {r[key] for r in train} & {r[key] for r in dev}:
            raise ValueError(f'Train/dev overlap: {key}')
    y6 = np.asarray([int(r['6way']) for r in train])
    full_sets = {'legacy_6way': {seed: select_support(y6, spec['shots_per_6way_class'], seed) for seed in spec['seeds']}}
    if balance_ablation:
        full_sets['balanced_binary'] = {seed: select_balanced_support(y6, spec, seed) for seed in spec['seeds']}
    union = np.unique(np.concatenate([ix for group in full_sets.values() for ix in group.values()]))
    lookup = {int(original): local for local, original in enumerate(union)}
    support_sets = {policy: {seed: np.asarray([lookup[int(i)] for i in indices]) for seed, indices in group.items()}
                    for policy, group in full_sets.items()}
    train_selected = [train[i] for i in union]
    support_log = []
    for policy, group in full_sets.items():
        for seed, indices in group.items():
            entry = {'seed': seed, 'ids': [train[i]['id'] for i in indices], 'labels_6way': y6[indices].tolist()}
            if balance_ablation:
                entry.update(policy=policy, quotas_6way=np.bincount(y6[indices], minlength=6).tolist())
            else:
                entry['shots_per_6way_class'] = spec['shots_per_6way_class']
            support_log.append(entry)
    kind = 'fakeddit_fewshot_support' if balance_ablation else 'fakeddit_fewshot_task1'
    output = new_run(settings, kind + '_min5', 'dev', smoke)
    identity = lambda rows: [{k: r[k] for k in ['id', '6way', 'image_sha256', 'text_sha256']} for r in rows]
    plan = {'kind': kind, 'dataset': 'fakeddit_2way', 'split': 'dev',
            'mode': 'smoke' if smoke else 'full', 'samples': len(dev), 'model': settings['model'],
            'regime': 'supervised_fewshot_frozen_backbone', 'stages': 1, 'test_opened': False,
            'specification': spec, 'specification_sha256': sha256(ROOT / spec_file),
            'code_sha256': code_hashes(), 'requirements_sha256': sha256(ROOT / 'requirements.txt'),
            'sklearn_version': importlib.metadata.version('scikit-learn'),
            'train_cohort': train_meta, 'dev_cohort': dev_meta,
            'support_sha256': digest(support_log), 'train_features_samples': len(union),
            'support_budget': '64 per 6way class = 384 total per seed; 64 Real and 320 Fake, not 64 per binary class',
            'binary_labels': {'0': 'Real', '1': 'Fake'},
            'source_label_mapping': 'binary=(6way!=0); source Fakeddit 2way uses inverse coding',
            'feature': 'unit image; unit title; concatenated unit image/title divided by sqrt(2)',
            'scope': 'Development min5 experiment; fixed classifier and support across modalities; no threshold search/test lock'}
    if balance_ablation:
        plan['support_budget'] = '384/seed/policy: 64 Real + 320 Fake versus 192 Real + 192 Fake; fake subtypes 38/39 rotating remainder'
        plan['interpretation'] = 'Support allocation changes Real diversity, Fake coverage and effective balanced sample weights; not isolated evidence for one cause.'
    write_json(output / 'frozen_plan.json', plan)
    write_json(output / 'support.json', support_log)
    write_json(output / 'cohort.json', identity(dev))
    write_json(output / 'train_cohort.json', identity(train))
    write_json(output / 'run.json', {**plan, 'status': 'running'})
    start = time.perf_counter()
    backend = backend_factory(settings['device'])
    try:
        train_image, train_title = encode_cohort(backend, train_selected, settings['batch_size'], with_titles=True)
        dev_image, dev_title = encode_cohort(backend, dev, settings['batch_size'], with_titles=True)
        plan['model_runtime'] = getattr(backend, 'metadata', {})
    finally:
        backend.unload()
    plan['encoding_seconds'] = time.perf_counter() - start
    # Lưu feature sớm để xử lý offline nếu fit chưa hội tụ, không phải nạp backbone lại.
    cache = {'train_image': train_image, 'train_title': train_title, 'dev_image': dev_image, 'dev_title': dev_title,
             'train_y6': y6[union], 'train_ids': np.asarray([r['id'] for r in train_selected]),
             'dev_ids': np.asarray([r['id'] for r in dev]),
             **{(f'support_{policy}_s{seed}' if balance_ablation else f'support_s{seed}'): indices
                for policy, group in support_sets.items() for seed, indices in group.items()}}
    np.savez_compressed(output / 'features.npz', **cache)
    results, coefficients = [], {}
    for policy, supports in support_sets.items():
        target = {seed: quotas(spec, seed) for seed in spec['seeds']} if policy == 'balanced_binary' else None
        inferred, models = predict(train_image, train_title, y6[union], dev_image, dev_title, supports, spec, target)
        results.extend((f'{policy}_{method}' if balance_ablation else method, seed, p) for method, seed, p in inferred)
        coefficients.update({f'{policy}_{key}' if balance_ablation else key: value for key, value in models.items()})
    # Chỉ dùng nhãn dev để đánh giá sau khi đã fit/dự đoán mọi method/seed.
    truth = (np.asarray([int(r['6way']) for r in dev]) != 0).astype(int)
    records, predictions, details, comparison, scores = [], [], {}, [], {}
    references = {seed: p.argmax(1) for method, seed, p in results if method == ('legacy_6way_image' if balance_ablation else 'image')}
    for method, seed, p in results:
        predicted = p.argmax(1)
        values = metric(truth, predicted, [0, 1])
        per_class = values.pop('per_class')
        key = f'{method}_s{seed}'
        details[key] = per_class
        records.append({'task': '2way', 'method': method, 'seed': seed, 'samples': len(dev), **values,
                        'recall_real': per_class['0']['recall'], 'recall_fake': per_class['1']['recall']})
        base = references[seed]
        comparison.append({'method': method, 'seed': seed, **values,
                           'recall_real': per_class['0']['recall'], 'recall_fake': per_class['1']['recall'],
                           'fake_as_real': int(((truth == 1) & (predicted == 0)).sum()),
                           'real_as_fake': int(((truth == 0) & (predicted == 1)).sum()),
                           'image_wrong_to_correct': int(((base != truth) & (predicted == truth)).sum()),
                           'image_correct_to_wrong': int(((base == truth) & (predicted != truth)).sum())})
        predictions.extend({'task': '2way', 'method': method, 'seed': seed, 'id': row['id'],
                            'truth': int(y), 'predicted': int(pred), 'p_real': float(prob[0]), 'p_fake': float(prob[1])}
                           for row, y, pred, prob in zip(dev, truth, predicted, p))
        scores['p_' + key] = p
    table(output / 'comparisons.tsv', comparison)
    class_summary = []
    methods = [f'{policy}_{method}' for policy in support_sets for method in spec['methods']] if balance_ablation else spec['methods']
    for method in methods:
        selected = [row for row in comparison if row['method'] == method]
        row = {'method': method, 'seeds': len(selected)}
        for name in ['recall_real', 'recall_fake', 'fake_as_real', 'real_as_fake']:
            values = [item[name] for item in selected]
            row.update({name + '_mean': float(np.mean(values)), name + '_std': float(np.std(values, ddof=1))})
        class_summary.append(row)
    table(output / 'class_summary.tsv', class_summary)
    if balance_ablation:
        paired = []
        lookup_results = {(method, seed): p for method, seed, p in results}
        for method in spec['methods']:
            for seed in spec['seeds']:
                base = lookup_results[f'legacy_6way_{method}', seed].argmax(1)
                pred = lookup_results[f'balanced_binary_{method}', seed].argmax(1)
                a = metric(truth, base, [0, 1]); b = metric(truth, pred, [0, 1])
                paired.append({'modality': method, 'seed': seed,
                               **{f'delta_{key}_pp': 100*(b[key]-a[key]) for key in ['accuracy','macro_f1','balanced_accuracy']},
                               'delta_recall_real_pp': 100*(b['per_class']['0']['recall']-a['per_class']['0']['recall']),
                               'delta_recall_fake_pp': 100*(b['per_class']['1']['recall']-a['per_class']['1']['recall']),
                               'legacy_wrong_to_correct': int(((base!=truth)&(pred==truth)).sum()),
                               'legacy_correct_to_wrong': int(((base==truth)&(pred!=truth)).sum())})
        table(output / 'support_comparisons.tsv', paired)
    np.savez_compressed(output / 'replay_scores.npz', ids=cache['dev_ids'], labels=truth,
                        class_names=np.asarray(['Real', 'Fake']), **scores)
    np.savez_compressed(output / 'linear_models.npz', **coefficients)
    plan.update(elapsed_seconds=time.perf_counter() - start, label_usage='train labels for fitting; dev labels for evaluation only')
    return publish(output, records, predictions, details, plan)
