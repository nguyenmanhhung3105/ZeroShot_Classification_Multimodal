"""Phân tích binary few-shot và chọn C bằng train-CV; chỉ dùng cache đã xác minh."""
import time

import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from .evaluate import metric, publish, table
from .fakeddit_fewshot_task1 import feature_banks, probe
from .io import code_hashes, digest, new_run, path, read_json, sha256, write_json
from .protocol import completed_run

C_GRID = [.01, .1, 1., 10., 100.]
NAMES = ['Real', 'Satire/Parody', 'False Connection', 'Imposter', 'Manipulated', 'Misleading']


def choose_c(x, y6, indices, spec, seed):
    """Không nhận feature/nhãn dev. CV chỉ trên 384 support của seed hiện tại."""
    x = x[indices]
    six = np.asarray(y6)[indices]
    y = (six != 0).astype(int)
    splits = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=seed).split(x, six))
    rows = []
    means = {}
    for c in C_GRID:
        values = []
        for fold, (train, validation) in enumerate(splits):
            classifier = probe(x[train], y[train], {**spec, 'C': c})
            value = float(balanced_accuracy_score(y[validation], classifier.predict(x[validation])))
            rows.append({'seed': seed, 'C': c, 'fold': fold, 'balanced_accuracy': value,
                         'fit_samples': len(train), 'validation_samples': len(validation)})
            values.append(value)
        means[c] = float(np.mean(values))
    best = max(means.values())
    tied = [c for c in C_GRID if np.isclose(means[c], best, rtol=0, atol=1e-12)]
    selected = min(tied, key=lambda c: (abs(np.log10(c)), c))
    classifier = probe(x, y, {**spec, 'C': selected})
    return selected, classifier, rows


def diagnostics(y6, predictions, ids):
    y6 = np.asarray(y6)
    truth = (y6 != 0).astype(int)
    subtype, persistent, changes = [], [], []
    for method, values in predictions.items():
        for label, name in enumerate(NAMES):
            selected = y6 == label
            if selected.any():
                recalls = (values[:, selected] == truth[selected]).mean(1)
                subtype.append({'method': method, 'subtype': name, 'samples': int(selected.sum()),
                                'binary_detection_recall_mean': float(recalls.mean()),
                                'binary_detection_recall_std': float(recalls.std(ddof=1))})
    joint = predictions['image_title']
    errors = (joint != truth).sum(0)
    for index, count in enumerate(errors):
        if count:
            persistent.append({'id': str(ids[index]), 'subtype': NAMES[y6[index]],
                               'truth': int(truth[index]), 'wrong_seeds': int(count),
                               'image_wrong_seeds': int((predictions['image'][:, index] != truth[index]).sum()),
                               'title_wrong_seeds': int((predictions['title'][:, index] != truth[index]).sum())})
    for label, name in [(0, 'Real'), (1, 'Fake')]:
        selected = truth == label
        for seed in range(len(joint)):
            base = predictions['image'][seed]
            values = joint[seed]
            changes.append({'seed': seed, 'class': name,
                            'image_wrong_to_correct': int(((base != truth) & (values == truth) & selected).sum()),
                            'image_correct_to_wrong': int(((base == truth) & (values != truth) & selected).sum())})
    return subtype, persistent, changes


def analyze(source, output_root='results/final/fewshot_task1_analysis'):
    source = path(source)
    run = completed_run(source)
    if (run.get('kind') != 'fakeddit_fewshot_task1' or run.get('mode') != 'full'
            or run.get('samples') != 3000 or run.get('test_opened') is not False):
        raise ValueError('Cần run few-shot tầng 1 min5 full dev đã hoàn tất')
    spec = run['specification']
    seeds = spec['seeds']
    if seeds != list(range(5)) or spec['C'] != 1. or spec['shots_per_6way_class'] != 64:
        raise ValueError('Cần baseline C=1, 64-shot/lớp 6way, 5 seed')
    output = new_run({'output_root': output_root}, 'train_cv', 'dev', False)
    plan = {'kind': 'fewshot_task1_train_cv', 'model': run['model'], 'dataset': 'fakeddit_2way',
            'mode': 'full', 'samples': 3000, 'split': 'dev', 'test_opened': False,
            'source': str(source), 'source_run_sha256': sha256(source/'run.json'),
            'code_sha256': code_hashes(), 'specification': spec,
            'selection': {'C_grid': C_GRID, 'folds': 3, 'stratification': '6way train support only',
                          'scoring': 'mean fold balanced accuracy, binary labels',
                          'ties': 'prefer closest C to 1 in log10 space, then smaller C'},
            'scope': 'Exploratory dev analysis; designs informed by prior dev results, not independent test.',
            'fixed': 'same cached features, support, class_weight, solver; only joint classifier C changes',
            'binary_labels': {'0': 'Real', '1': 'Fake'}}
    write_json(output/'frozen_plan.json', plan)
    write_json(output/'run.json', {**plan, 'status': 'running'})
    with np.load(source/'features.npz', allow_pickle=False) as cache:
        train = feature_banks(cache['train_image'], cache['train_title'])['image_title']
        dev = feature_banks(cache['dev_image'], cache['dev_title'])['image_title']
        y6_train = cache['train_y6']
        supports = {seed: cache[f'support_s{seed}'] for seed in seeds}
        ids = cache['dev_ids']
        train_ids = cache['train_ids']
    support_log = read_json(source/'support.json')
    if digest(support_log) != run['support_sha256']:
        raise ValueError('Support log đã đổi')
    for entry in support_log:
        indices = supports[entry['seed']]
        if train_ids[indices].tolist() != entry['ids']:
            raise ValueError('Support ID không khớp cache')
        np.testing.assert_array_equal(y6_train[indices], entry['labels_6way'])
        np.testing.assert_array_equal(np.bincount(y6_train[indices], minlength=6), [64]*6)
    results, cv_rows, selections, models = [], [], [], {}
    started = time.perf_counter()
    with threadpool_limits(limits=1):
        for seed in seeds:
            c, classifier, folds = choose_c(train, y6_train, supports[seed], spec, seed)
            results.append(('image_title_train_cv', seed, classifier.predict_proba(dev)))
            models[f'coef_s{seed}'] = classifier.coef_
            models[f'intercept_s{seed}'] = classifier.intercept_
            cv_rows.extend(folds)
            selections.append({'seed': seed, 'selected_C': c})
            print(f'Train-CV seed {seed}: selected C={c}', flush=True)
    # Nhãn dev chỉ dùng sau khi CV đã chọn C và dự đoán xong.
    cohort = read_json(source/'cohort.json')
    if [row['id'] for row in cohort] != ids.tolist():
        raise ValueError('Thứ tự dev không khớp')
    y6 = np.asarray([int(row['6way']) for row in cohort])
    truth = (y6 != 0).astype(int)
    with np.load(source/'replay_scores.npz', allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['labels'], truth)
        np.testing.assert_array_equal(saved['ids'], ids)
        for method in ['image', 'title', 'image_title']:
            for seed in seeds:
                results.append((method, seed, saved[f'p_{method}_s{seed}']))
    records, prediction_rows, details, scores, comparisons = [], [], {}, {}, []
    reference = {seed: p.argmax(1) for method, seed, p in results if method == 'image_title'}
    for method, seed, p in results:
        predicted = p.argmax(1)
        values = metric(truth, predicted, [0, 1])
        key = f'{method}_s{seed}'
        details[key] = values.pop('per_class')
        records.append({'task': '2way', 'method': method, 'seed': seed, 'samples': len(ids), **values})
        scores['p_' + key] = p
        prediction_rows.extend({'task': '2way', 'method': method, 'seed': seed, 'id': str(id_),
                                'truth': int(y), 'predicted': int(pred)} for id_, y, pred in zip(ids, truth, predicted))
        if method == 'image_title_train_cv':
            base = reference[seed]
            comparisons.append({'seed': seed, **values,
                                'delta_accuracy_pp': float(100*((predicted == truth).mean()-(base == truth).mean())),
                                'baseline_wrong_to_correct': int(((base != truth)&(predicted == truth)).sum()),
                                'baseline_correct_to_wrong': int(((base == truth)&(predicted != truth)).sum()),
                                'recall_real': details[key]['0']['recall'], 'recall_fake': details[key]['1']['recall']})
    all_predictions = {method: np.asarray([p.argmax(1) for m, seed, p in sorted(results, key=lambda row: row[1]) if m == method])
                       for method in ['image', 'title', 'image_title', 'image_title_train_cv']}
    subtype, persistent, changes = diagnostics(y6, all_predictions, ids)
    table(output/'subtype_detection.tsv', subtype)
    if persistent:
        table(output/'persistent_errors.tsv', persistent)
    table(output/'modality_changes.tsv', changes)
    table(output/'train_cv.tsv', cv_rows)
    table(output/'selected_C.tsv', selections)
    table(output/'comparisons.tsv', comparisons)
    np.savez_compressed(output/'replay_scores.npz', ids=ids, labels=truth, **scores)
    np.savez_compressed(output/'linear_models.npz', **models)
    plan['elapsed_seconds'] = time.perf_counter()-started
    return publish(output, records, prediction_rows, details, plan)
