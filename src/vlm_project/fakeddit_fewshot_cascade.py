"""Few-shot min5 dev: Real/Fake → năm loại Fake và flat 6way cùng ngân sách."""
import tempfile
import time
from pathlib import Path

import numpy as np

from . import data, min5_protocol
from .backbone import Backbone, encode_cohort
from .evaluate import metric, publish, summarize, table
from .fakeddit_fewshot_task1 import feature_banks, probe
from .fewshot_support_thresholds import POLICIES, measurements, aggregate
from .io import ROOT, code_hashes, digest, new_run, path, read_json, sha256, write_json
from .min5_test_runner import FEATURE, RUNTIME_KEYS

SPEC_FILE = 'configs/fakeddit_fewshot_cascade_min5.json'


def export_specification(source, review, destination, settings_file='configs/project_min5.yaml'):
    """Snapshot dev portable; không phải lock test, không đọc dataset/feature/weights."""
    destination, source, review = path(destination), path(source), path(review)
    if destination.is_relative_to(source) or destination.is_relative_to(review):
        raise ValueError('Không ghi snapshot vào source/review')
    with tempfile.TemporaryDirectory() as temp:
        lock = min5_protocol.prepare(source, review, Path(temp) / 'draft.json', settings_file)
    p = lock['payload']
    if p['review']['specification']['modalities'] != ['image_title']:
        raise ValueError('Cascade hiện giữ đối chứng cùng modality ảnh–tiêu đề')
    membership = read_json(review / 'split.json')
    spec = {**p['source']['specification'], 'version': 1, 'kind': 'fakeddit_fewshot_cascade',
            'model': p['model'], 'split': 'dev', 'development_only': True, 'stages': 2,
            'methods': ['image_title'], 'feature': p['source']['feature'],
            'head_training': 'C=1 balanced lbfgs for binary, five-fake and flat-six heads; no added support',
            'support_snapshot': p['support'], 'gate_candidates': p['candidates'],
            'source_run_sha256': p['source']['run_sha256'], 'review_run_sha256': p['review']['run_sha256'],
            'train_identity_sha256': digest(p['excluded_train']), 'dev_identity_sha256': digest(p['excluded_dev']),
            'model_runtime_reference': p['source']['model_runtime'],
            'audit_ids': [r['id'] for r in membership if r['partition'] == 'audit'],
            'selection_ids': [r['id'] for r in membership if r['partition'] == 'selection'],
            'scope': 'Development only. Fixed Task 1 gates inherited from selection dev; full dev and audit subset are exploratory, not independent test. Oracle diagnostic only.'}
    # Các fixture tổng hợp cũng dùng cùng schema; kiểm trước khi ghi snapshot bất biến.
    _check_specification(spec)
    return min5_protocol.exclusive_write(destination, spec)


def _check_specification(spec):
    fixed = {'version': 1, 'kind': 'fakeddit_fewshot_cascade', 'split': 'dev', 'development_only': True,
             'stages': 2, 'seeds': list(range(5)), 'methods': ['image_title'], 'train_samples': 3000,
             'dev_samples': 3000, 'C': 1., 'class_weight': 'balanced', 'solver': 'lbfgs',
             'max_iter': 2000, 'tol': 1e-6, 'shots_per_6way_class': 64, 'support_policies': POLICIES,
             'feature': FEATURE}
    if any(spec.get(k) != v for k, v in fixed.items()):
        raise ValueError('Giữ giao thức cascade min5 dev, hai policy và classifier đã định trước')
    if set(spec.get('manifest_sha256', {})) != {'train', 'dev'}:
        raise ValueError('Chỉ nhận manifest train/dev, không test')
    if not set(RUNTIME_KEYS) <= spec.get('model_runtime_reference', {}).keys():
        raise ValueError('Thiếu metadata backbone nguồn')
    selection, audit = spec['selection_ids'], spec['audit_ids']
    if (not selection or not audit or len(set(selection)) != len(selection) or len(set(audit)) != len(audit)
            or set(selection) & set(audit) or len(selection) + len(audit) != 3000):
        raise ValueError('Selection/audit phải chia đủ dev, không trùng ID')
    expected, found = {(p, s) for p in POLICIES for s in range(5)}, set()
    defaults = set()
    for c in spec['gate_candidates']:
        key = (c['policy'], c['seed'])
        if key not in expected or c['modality'] != 'image_title':
            raise ValueError('Gate policy/modality/seed không hợp lệ')
        identity = (*key, c['decision'])
        if identity in found:
            raise ValueError('Gate bị trùng')
        found.add(identity)
        if c['decision'] == 'default':
            if c['rule'] != 'argmax_real_on_tie' or c['threshold'] is not None:
                raise ValueError('Default gate phải dùng argmax, tie Real')
            defaults.add(key)
        elif (c['rule'] != 'fake_if_score_ge_threshold' or not np.isfinite(c['threshold'])
              or not 0 <= c['threshold'] <= 1):
            raise ValueError('Ngưỡng gate sai')
    if defaults != expected:
        raise ValueError('Cần đủ default gate của hai policy/năm seed')


def specification(file=SPEC_FILE):
    snapshot = read_json(file)
    if snapshot.get('sha256') != digest(snapshot.get('payload')):
        raise ValueError('Snapshot cascade sai checksum')
    spec = snapshot['payload']
    _check_specification(spec)
    return spec


def route(gate, child, threshold=None):
    """Hard gate: Real→0; Fake→argmax năm subtype + 1, không argmax joint scores."""
    gate, child = np.asarray(gate), np.asarray(child)
    if (gate.ndim != 2 or gate.shape[1] != 2 or child.shape != (len(gate), 5)
            or not len(gate) or not np.isfinite(gate).all() or not np.isfinite(child).all()
            or (gate < 0).any() or (child < 0).any()
            or not np.allclose(gate.sum(1), 1.) or not np.allclose(child.sum(1), 1.)):
        raise ValueError('Gate/child cần posterior 2/5 lớp hữu hạn, cùng số mẫu')
    if threshold is not None and (not np.isfinite(threshold) or not 0 <= threshold <= 1):
        raise ValueError('Ngưỡng cần hữu hạn trong [0,1]')
    fake = gate.argmax(1) == 1 if threshold is None else gate[:, 1] >= threshold
    return np.where(fake, child.argmax(1) + 1, 0)


def infer(train_image, train_title, train_y6, query_image, query_title, train_ids, spec):
    """Không nhận nhãn query. Mỗi head chỉ dùng cùng support train đã snapshot."""
    train = feature_banks(train_image, train_title)['image_title']
    query = feature_banks(query_image, query_title)['image_title']
    y6 = np.asarray(train_y6)
    if (y6.shape != (len(train),) or not np.isin(y6, range(6)).all()
            or len(train_ids) != len(train) or len(set(train_ids)) != len(train_ids)):
        raise ValueError('Train feature/ID/label không hợp lệ')
    min5_protocol.validate_support(spec['support_snapshot'], [{'id': id_, '6way': int(y)} for id_, y in zip(train_ids, y6)])
    lookup = {id_: i for i, id_ in enumerate(train_ids)}
    scores, models = {}, {}
    for entry in spec['support_snapshot']:
        policy, seed = entry['policy'], entry['seed']
        indices = np.asarray([lookup[i] for i in entry['ids']])
        fake_indices = indices[y6[indices] != 0]
        banks = {}
        for name, ix, labels in [('gate', indices, (y6[indices] != 0).astype(int)),
                                  ('child', fake_indices, y6[fake_indices]), ('flat', indices, y6[indices])]:
            model = probe(train[ix], labels, spec)
            expected = [0,1] if name == 'gate' else list(range(1,6)) if name == 'child' else list(range(6))
            if model.classes_.tolist() != expected:
                raise ValueError('Thứ tự lớp của head không đúng taxonomy')
            banks[name] = model.predict_proba(query)
            key = f'{name}_{policy}_s{seed}'
            models[f'{key}_coef'] = model.coef_
            models[f'{key}_intercept'] = model.intercept_
            models[f'{key}_classes'] = model.classes_
        scores[policy, seed] = banks
    return scores, models


def evaluate(truth, ids, scores, spec):
    """Chỉ đưa nhãn dev vào sau inference; oracle nằm ngoài bảng phương pháp chính."""
    truth, ids = np.asarray(truth), np.asarray(ids)
    binary = (truth != 0).astype(int)
    audit = np.flatnonzero(np.isin(ids, spec['audit_ids']))
    records, predictions, details, comparisons, routing, subsets, gates, subtypes = [], [], {}, [], [], [], [], []
    oracle, arrays = {}, {}
    flat_predictions = {}
    def record(method, seed, predicted):
        values = metric(truth, predicted, list(range(6)))
        details[f'{method}_s{seed}'] = values.pop('per_class')
        records.append({'task': '6way', 'method': method, 'seed': seed, 'samples': len(ids), **values})
        predictions.extend({'task': '6way', 'method': method, 'seed': seed, 'id': str(i),
                            'truth': int(y), 'predicted': int(p)} for i, y, p in zip(ids, truth, predicted))
        arrays[f'prediction_{method}_s{seed}'] = predicted
        if len(audit):
            subset = metric(truth[audit], predicted[audit], list(range(6))); subset.pop('per_class')
            subsets.append({'task': '6way', 'method': method, 'seed': seed, 'samples': len(audit), **subset})
        return values
    for (policy, seed), bank in scores.items():
        flat = bank['flat'].argmax(1)
        flat_predictions[policy, seed] = flat
        record(f'{policy}_flat', seed, flat)
        for name, p in bank.items(): arrays[f'p_{name}_{policy}_s{seed}'] = p
        oracle_prediction = np.where(binary == 0, 0, bank['child'].argmax(1) + 1)
        oracle[f'{policy}_s{seed}'] = metric(truth, oracle_prediction, list(range(6)))
    for candidate in spec['gate_candidates']:
        policy, seed, decision = (candidate[k] for k in ['policy','seed','decision'])
        bank = scores[policy, seed]
        predicted = route(bank['gate'], bank['child'], candidate['threshold'])
        method = f'{policy}_hierarchy_{decision}'
        values = record(method, seed, predicted)
        fake_gate = (predicted != 0).astype(int)
        child = bank['child'].argmax(1) + 1
        detected_fake = (truth != 0) & (fake_gate == 1)
        all_fake = truth != 0
        child_metric = metric(truth[all_fake], child[all_fake], list(range(1,6))) if all_fake.any() else None
        gate_values, _ = measurements(binary, fake_gate)
        gates.append({'policy': policy, 'seed': seed, 'decision': decision, 'samples': len(ids), **gate_values})
        routing.append({'policy': policy, 'seed': seed, 'decision': decision,
                        'real_routed_to_fake': int(((truth == 0) & (fake_gate == 1)).sum()),
                        'fake_routed_to_real': int(((truth != 0) & (fake_gate == 0)).sum()),
                        'subtype_errors_after_correct_fake_gate': int((detected_fake & (child != truth)).sum()),
                        'true_fake_samples': int(all_fake.sum()), 'correct_fake_gate_samples': int(detected_fake.sum()),
                        'child_accuracy_all_true_fake': child_metric['accuracy'] if child_metric else '',
                        'child_macro_f1_all_true_fake': child_metric['macro_f1'] if child_metric else '',
                        'child_accuracy_given_correct_fake_gate': float((child[detected_fake] == truth[detected_fake]).mean()) if detected_fake.any() else ''})
        for label in range(6):
            selected = truth == label
            if selected.any():
                subtypes.append({'policy': policy, 'seed': seed, 'decision': decision, 'class_6way': label,
                                 'samples': int(selected.sum()),
                                 'binary_gate_recall': float((fake_gate[selected] == int(label != 0)).mean()),
                                 'child_recall_all_true_subtype': float((child[selected] == label).mean()) if label else '',
                                 'end_to_end_recall': float((predicted[selected] == label).mean())})
        flat = flat_predictions[policy, seed]; baseline = metric(truth, flat, list(range(6)))
        comparisons.append({'policy': policy, 'seed': seed, 'decision': decision,
                            **{f'delta_{k}_pp': 100*(values[k]-baseline[k]) for k in ['accuracy','macro_f1','balanced_accuracy']},
                            'flat_wrong_to_correct': int(((flat != truth) & (predicted == truth)).sum()),
                            'flat_correct_to_wrong': int(((flat == truth) & (predicted != truth)).sum())})
    return records, predictions, details, comparisons, routing, gates, subsets, oracle, arrays, subtypes


def run(settings, train_root, dev_root, smoke=False, backend_factory=Backbone, *, spec_file=SPEC_FILE):
    spec = specification(spec_file)
    if settings['model'] != spec['model']:
        raise ValueError('Model khác snapshot cascade')
    train_root, dev_root, output_root = path(train_root), path(dev_root), path(settings['output_root'])
    if output_root.is_relative_to(train_root) or output_root.is_relative_to(dev_root):
        raise ValueError('Output phải tách dataset')
    for split, root in [('train', train_root), ('dev', dev_root)]:
        if sha256(root / 'manifest.json') != spec['manifest_sha256'][split]:
            raise ValueError(f'Sai manifest {split} min5; không dùng test/cohort v2')
    train, train_meta = data.fakeddit(train_root, 'train')
    dev, dev_meta = data.fakeddit(dev_root, 'dev', smoke)
    if len(train) != 3000 or len(dev) != (5 if smoke else 3000):
        raise ValueError('Sai số mẫu train/dev')
    if digest(min5_protocol.identities(train)) != spec['train_identity_sha256']:
        raise ValueError('Train khác snapshot nguồn')
    if not smoke and digest(min5_protocol.identities(dev)) != spec['dev_identity_sha256']:
        raise ValueError('Dev khác snapshot nguồn')
    min5_protocol.no_overlap(train, dev)
    min5_protocol.validate_support(spec['support_snapshot'], train)
    if not smoke and set(r['id'] for r in dev) != set(spec['selection_ids']) | set(spec['audit_ids']):
        raise ValueError('Dev khác partition nguồn')
    union = {i for entry in spec['support_snapshot'] for i in entry['ids']}
    selected_train = [r for r in train if r['id'] in union]
    output = new_run(settings, 'fakeddit_fewshot_cascade_min5', 'dev', smoke)
    manifest = {'kind': 'fakeddit_fewshot_cascade', 'model': spec['model'], 'dataset': 'fakeddit_6way',
                'mode': 'smoke' if smoke else 'full', 'split': 'dev', 'samples': len(dev), 'stages': 2,
                'test_opened': False, 'specification': spec, 'specification_sha256': sha256(spec_file),
                'code_sha256': code_hashes(), 'train_cohort': train_meta, 'dev_cohort': dev_meta,
                'train_features_samples': len(selected_train), 'support_budget_per_seed': 384,
                'scope': spec['scope'], 'label_usage': 'Train-only heads; dev labels after inference, oracle diagnostic only.'}
    write_json(output / 'frozen_plan.json', manifest)
    write_json(output / 'support.json', spec['support_snapshot'])
    write_json(output / 'cohort.json', [{k:r[k] for k in ['id','6way','image_sha256','text_sha256']} for r in dev])
    write_json(output / 'run.json', {**manifest, 'status': 'running'})
    start = time.perf_counter(); backend = backend_factory(settings['device'])
    try:
        runtime = getattr(backend, 'metadata', {})
        if any(digest(runtime.get(k)) != digest(spec['model_runtime_reference'][k]) for k in RUNTIME_KEYS):
            raise ValueError('Backbone khác metadata snapshot')
        image, title = encode_cohort(backend, selected_train, settings['batch_size'], with_titles=True)
        query_image, query_title = encode_cohort(backend, dev, settings['batch_size'], with_titles=True)
        manifest['model_runtime'] = runtime
    finally:
        backend.unload()
    manifest['encoding_seconds'] = time.perf_counter() - start
    scores, models = infer(image, title, np.asarray([int(r['6way']) for r in selected_train]),
                            query_image, query_title, [r['id'] for r in selected_train], spec)
    truth = np.asarray([int(r['6way']) for r in dev]); ids = np.asarray([r['id'] for r in dev])
    records, predictions, details, comparisons, routing, gates, subsets, oracle, arrays, subtypes = evaluate(truth, ids, scores, spec)
    table(output / 'comparisons.tsv', comparisons)
    table(output / 'comparisons_summary.tsv', aggregate(comparisons, ['policy','decision'],
          ['delta_accuracy_pp','delta_macro_f1_pp','delta_balanced_accuracy_pp']))
    table(output / 'routing.tsv', routing); table(output / 'gate_metrics.tsv', gates)
    table(output / 'subtype_detection.tsv', subtypes)
    table(output / 'subtype_summary.tsv', aggregate(subtypes, ['policy','decision','class_6way'],
          ['binary_gate_recall','end_to_end_recall']))
    if subsets:
        table(output / 'audit_metrics_by_seed.tsv', subsets); table(output / 'audit_summary.tsv', summarize(subsets))
    write_json(output / 'oracle_diagnostic.json', {'uses_true_binary_labels': True,
               'scope': 'Oracle gate diagnostic; accuracy ceiling for fixed child predictions, not a bound on all metrics; excluded from ranking', 'metrics': oracle})
    np.savez_compressed(output / 'replay_scores.npz', ids=ids, labels_6way=truth, **arrays)
    np.savez_compressed(output / 'linear_models.npz', **models)
    manifest['elapsed_seconds'] = time.perf_counter() - start
    return publish(output, records, predictions, details, manifest)
