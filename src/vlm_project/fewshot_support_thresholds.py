"""So support policy từ posterior dev: selection chọn ngưỡng, audit chỉ đánh giá."""
import math

import numpy as np

from .evaluate import metric, publish, table
from .io import code_hashes, digest, new_run, path, read_json, sha256, write_json

POLICIES = ['legacy_6way', 'balanced_binary']
MODALITIES = ['image', 'title', 'image_title']
NAMES = ['Real', 'Satire/Parody', 'False Connection', 'Imposter', 'Manipulated', 'Misleading']


def split_dev(ids, six, selection_fraction=.5, split_seed=42):
    """Chia theo subtype, sắp ID trước RNG để không phụ thuộc thứ tự source."""
    ids, six = np.asarray(ids), np.asarray(six)
    if (not 0 < selection_fraction < 1 or ids.ndim != 1 or six.shape != ids.shape
            or len(set(ids.tolist())) != len(ids) or not np.isin(six, range(6)).all()):
        raise ValueError('Cần ID duy nhất, nhãn 6way hợp lệ và 0 < selection_fraction < 1')
    rng = np.random.default_rng(split_seed)
    selection, audit = [], []
    for label in range(6):
        indices = np.flatnonzero(six == label)
        if len(indices) < 2:
            raise ValueError('Mỗi subtype cần ít nhất hai mẫu để chia selection/audit')
        ordered = indices[np.argsort(ids[indices], kind='stable')]
        shuffled = rng.permutation(ordered)
        count = max(1, min(len(indices) - 1, math.floor(len(indices) * selection_fraction)))
        selection.extend(shuffled[:count])
        audit.extend(shuffled[count:])
    return np.sort(selection), np.sort(audit)


def choose_threshold(truth, p_fake, target):
    """Ngưỡng cao nhất đạt recall Fake >= target; ties score đều nhận Fake."""
    truth, p_fake = np.asarray(truth), np.asarray(p_fake)
    if (not np.isfinite(target) or not 0 < target <= 1 or truth.ndim != 1
            or p_fake.shape != truth.shape or not np.isin(truth, [0, 1]).all()
            or not np.isfinite(p_fake).all() or ((p_fake < 0) | (p_fake > 1)).any()):
        raise ValueError('Target cần trong (0, 1], score hữu hạn trong [0, 1], nhãn binary')
    positive = np.sort(p_fake[truth == 1])[::-1]
    if not len(positive):
        raise ValueError('Selection không có Fake')
    required = math.ceil(target * len(positive))
    return float(positive[required - 1])


def precision_recall_rows(truth, p_fake):
    """Mỗi score khác nhau là một ngưỡng >=; xử lý tied scores cùng lúc."""
    order = np.argsort(-p_fake, kind='stable')
    scores, y = p_fake[order], truth[order]
    ends = np.r_[np.flatnonzero(scores[:-1] != scores[1:]), len(scores) - 1]
    tp = np.cumsum(y)[ends]
    fp = ends + 1 - tp
    positives = int(y.sum())
    # Điểm đầu không dự đoán Fake; precision=0 theo zero_division=0.
    rows = [{'threshold': float(np.nextafter(scores[0], np.inf)), 'precision_fake': 0.,
             'recall_fake': 0., 'recall_real': 1., 'tp': 0, 'fp': 0}]
    rows.extend({'threshold': float(scores[end]), 'precision_fake': float(t / (t + f)),
                 'recall_fake': float(t / positives), 'recall_real': float(1 - f / (len(y) - positives)),
                 'tp': int(t), 'fp': int(f)} for end, t, f in zip(ends, tp, fp))
    return rows


def load_source(source, modalities):
    """Chỉ xác minh/đọc receipt, run, cohort và posterior; bỏ feature/model cache."""
    receipt = read_json(source / 'completed.json')
    if receipt.get('status') != 'complete' or receipt.get('run_sha256') != sha256(source / 'run.json'):
        raise ValueError('Receipt không khớp run.json')
    run = read_json(source / 'run.json')
    if (run.get('status') != 'complete' or run.get('kind') != 'fakeddit_fewshot_support'
            or run.get('split') != 'dev' or run.get('mode') != 'full' or run.get('samples') != 3000
            or run.get('test_opened') is not False or run.get('binary_labels') != {'0': 'Real', '1': 'Fake'}):
        raise ValueError('Cần run support min5 full dev, 3000 mẫu, 0=Real/1=Fake, chưa mở test')
    spec = run.get('specification', {})
    if spec.get('seeds') != list(range(5)) or spec.get('support_policies') != POLICIES:
        raise ValueError('Cần đủ hai policy và năm seed 0–4')
    verified = {}
    for name in ['cohort.json', 'replay_scores.npz']:
        file = path(source / name)
        if not file.is_relative_to(source):
            raise ValueError('Artifact không được trỏ ra ngoài run')
        expected = run.get('output_sha256', {}).get(name)
        actual = sha256(file)
        if not expected or expected != actual:
            raise ValueError(f'Checksum artifact không khớp: {name}')
        verified[name] = actual
    cohort = read_json(source / 'cohort.json')
    if len(cohort) != 3000:
        raise ValueError('Cohort không đủ 3000 mẫu')
    ids = np.asarray([str(row['id']) for row in cohort])
    # Chỉ nhãn 6way và ID từ artifact; không đọc ảnh/text hoặc dataset.
    if any(str(row['6way']) not in list('012345') for row in cohort):
        raise ValueError('Nhãn cohort phải là 0..5')
    six = np.asarray([int(row['6way']) for row in cohort])
    truth = (six != 0).astype(int)
    scores = {}
    with np.load(source / 'replay_scores.npz', allow_pickle=False) as saved:
        if (not np.array_equal(saved['ids'], ids) or not np.array_equal(saved['labels'], truth)
                or not np.array_equal(saved['class_names'], ['Real', 'Fake'])):
            raise ValueError('ID, nhãn hoặc thứ tự lớp posterior không khớp cohort')
        for modality in modalities:
            for policy in POLICIES:
                for seed in spec['seeds']:
                    key = f'p_{policy}_{modality}_s{seed}'
                    if key not in saved:
                        raise ValueError(f'Thiếu posterior {key}')
                    p = saved[key]
                    if (p.shape != (len(ids), 2) or not np.isfinite(p).all()
                            or ((p < 0) | (p > 1)).any()
                            or not np.allclose(p.sum(1), 1., rtol=0, atol=1e-6)):
                        raise ValueError(f'Posterior không hợp lệ: {key}')
                    scores[policy, modality, seed] = p.copy()
    return run, verified, ids, six, scores


def measurements(truth, predicted):
    values = metric(truth, predicted, [0, 1])
    details = values.pop('per_class')
    values.update(recall_real=details['0']['recall'], recall_fake=details['1']['recall'],
                  precision_fake=details['1']['precision'],
                  fake_as_real=int(((truth == 1) & (predicted == 0)).sum()),
                  real_as_fake=int(((truth == 0) & (predicted == 1)).sum()))
    return values, details


def aggregate(rows, groups, fields):
    collected = {}
    for row in rows:
        collected.setdefault(tuple(row[k] for k in groups), []).append(row)
    result = []
    for key, values in collected.items():
        entry = {**dict(zip(groups, key)), 'seeds': len(values), 'std_ddof': 1}
        for field in fields:
            x = np.asarray([r[field] for r in values])
            entry.update({field + '_mean': float(x.mean()),
                          field + '_std': float(x.std(ddof=1)) if len(x) > 1 else 0.})
        result.append(entry)
    return result


def analyze(source, output_root='results/final/fewshot_support_thresholds', targets=None,
            modalities=None, selection_fraction=.5, split_seed=42):
    targets = [.87] if targets is None else list(targets)
    modalities = ['image_title'] if modalities is None else list(modalities)
    if (not targets or len(set(targets)) != len(targets)
            or any(not np.isfinite(t) or not 0 < t <= 1 for t in targets)):
        raise ValueError('Targets phải khác nhau, hữu hạn và trong (0, 1]')
    if not modalities or len(set(modalities)) != len(modalities) or not set(modalities) <= set(MODALITIES):
        raise ValueError('Modality phải khác nhau và thuộc image/title/image_title')
    source, output_root = path(source), path(output_root)
    if output_root.is_relative_to(source):
        raise ValueError('Output không được nằm trong source run')
    run, verified, ids, six, scores = load_source(source, modalities)
    selection, audit = split_dev(ids, six, selection_fraction, split_seed)
    truth = (six != 0).astype(int)
    plan = {'kind': 'fewshot_support_thresholds', 'model': run['model'], 'dataset': 'fakeddit_2way',
            'mode': 'full', 'split': 'dev', 'samples': len(audit), 'source_samples': len(ids),
            'test_opened': False, 'source': str(source), 'source_run_sha256': sha256(source / 'run.json'),
            'verified_input_sha256': verified, 'verification_scope': 'receipt/run and consumed cohort/posterior only; no features or weights read',
            'code_sha256': code_hashes(), 'binary_labels': {'0': 'Real', '1': 'Fake'},
            'specification': {'targets': targets, 'modalities': modalities, 'seeds': list(range(5)),
                              'policies': POLICIES, 'split_seed': split_seed, 'selection_fraction': selection_fraction},
            'selection': 'Subtype-stratified ID-sorted seeded split; floor fraction per subtype, clamped to [1,n-1]. Highest threshold meeting target on selection, per policy/modality/seed; score >= threshold is Fake.',
            'default_decision': 'argmax [Real,Fake], ties choose Real; distinct from tuned >= threshold',
            'scope': 'Exploratory development analysis. Full dev previously inspected; audit is not independent test. Audit never selects thresholds. Target recall is not guaranteed on audit.',
            'uncertainty': 'Sample std across overlapping support seeds on same audit IDs; not a confidence interval.'}
    thresholds, curves, records, predictions, details, subtype, comparisons = [], [], [], [], {}, [], []
    outcomes = {}
    for modality in modalities:
        for policy in POLICIES:
            for seed in range(5):
                p = scores[policy, modality, seed]
                identity = {'policy': policy, 'modality': modality, 'seed': seed}
                curves.extend({**identity, **row} for row in precision_recall_rows(truth[selection], p[selection, 1]))
                decisions = [('default', None, p.argmax(1))]
                for target in targets:
                    threshold = choose_threshold(truth[selection], p[selection, 1], target)
                    pred = (p[:, 1] >= threshold).astype(int)
                    selected_values, _ = measurements(truth[selection], pred[selection])
                    thresholds.append({**identity, 'target_recall_fake': target, 'threshold': threshold,
                                       'selection_samples': len(selection),
                                       **{'selection_' + k: v for k, v in selected_values.items()}})
                    decisions.append((f'target_{target!r}', threshold, pred))
                for decision, threshold, pred in decisions:
                    method = f'{policy}_{modality}_{decision}'
                    values, per_class = measurements(truth[audit], pred[audit])
                    records.append({'task': '2way', 'method': method, **identity, 'decision': decision,
                                    'samples': len(audit), **values})
                    details[f'{method}_s{seed}'] = per_class
                    outcomes[policy, modality, seed, decision] = (pred[audit], values)
                    predictions.extend({'task': '2way', 'method': method, 'seed': seed, 'id': str(ids[i]),
                                        'truth': int(truth[i]), 'predicted': int(pred[i]), 'p_fake': float(p[i, 1]),
                                        'threshold': '' if threshold is None else threshold} for i in audit)
                    for label, name in enumerate(NAMES):
                        ix = audit[six[audit] == label]
                        subtype.append({**identity, 'decision': decision, 'subtype': name, 'samples': len(ix),
                                        'binary_detection_recall': float((pred[ix] == truth[ix]).mean())})
        for seed in range(5):
            for decision in ['default', *[f'target_{t!r}' for t in targets]]:
                base, a = outcomes[POLICIES[0], modality, seed, decision]
                pred, b = outcomes[POLICIES[1], modality, seed, decision]
                comparisons.append({'modality': modality, 'seed': seed, 'decision': decision,
                                    **{'delta_' + k + '_pp': 100*(b[k]-a[k]) for k in
                                       ['accuracy', 'macro_f1', 'balanced_accuracy', 'recall_real', 'recall_fake']},
                                    'legacy_wrong_to_correct': int(((base != truth[audit]) & (pred == truth[audit])).sum()),
                                    'legacy_correct_to_wrong': int(((base == truth[audit]) & (pred != truth[audit])).sum())})
    selection_set = set(selection.tolist())
    membership = [{'id': str(ids[i]), '6way': int(six[i]), 'partition': 'selection' if i in selection_set else 'audit'}
                  for i in range(len(ids))]
    plan['split_sha256'] = digest(membership)
    output = new_run({'output_root': output_root}, 'recall_tradeoff', 'dev', False)
    write_json(output / 'frozen_plan.json', plan)
    write_json(output / 'split.json', membership)
    table(output / 'thresholds.tsv', thresholds)
    table(output / 'precision_recall_selection.tsv', curves)
    table(output / 'subtype_detection.tsv', subtype)
    table(output / 'comparisons.tsv', comparisons)
    table(output / 'comparisons_summary.tsv', aggregate(comparisons, ['modality', 'decision'],
          [k for k in comparisons[0] if k.startswith('delta_')]))
    table(output / 'class_summary.tsv', aggregate(records, ['method'],
          ['recall_real', 'recall_fake', 'precision_fake', 'fake_as_real', 'real_as_fake']))
    table(output / 'subtype_summary.tsv', aggregate(subtype, ['policy', 'modality', 'decision', 'subtype'],
          ['binary_detection_recall']))
    return publish(output, records, predictions, details, plan)
