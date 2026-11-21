"""Few-shot Task 1 test min5: khóa trước dữ liệu, không chọn lại support/ngưỡng."""
import time

import numpy as np
import yaml

from . import data, min5_protocol
from .backbone import Backbone, encode_cohort
from .evaluate import publish, table
from .fakeddit_fewshot_task1 import predict
from .fewshot_support_thresholds import POLICIES, MODALITIES, measurements, aggregate
from .io import path, digest, sha256, new_run, write_json

FEATURE = 'unit image; unit title; concatenated unit image/title divided by sqrt(2)'
RUNTIME_KEYS = ['architecture', 'pretrained', 'tokenizer', 'context_length', 'dtype', 'image_size']


def inspect(protocol_file, settings_file=None, *, require_sealed=False):
    lock = min5_protocol.validate(protocol_file, require_sealed=require_sealed, settings_file=settings_file)
    p = lock['payload']
    config_file = path(settings_file or p['settings_file'])
    settings = yaml.safe_load(config_file.read_text())
    if {k: settings.get(k) for k in ['model', 'batch_size']} != p['inference_settings']:
        raise ValueError('Model/batch_size không khớp protocol')
    if settings.get('device', 'auto') not in ['auto', 'cpu', 'cuda', 'mps']:
        raise ValueError('Device không hợp lệ')
    spec = p['source']['specification']
    fixed = {'C': 1., 'class_weight': 'balanced', 'solver': 'lbfgs', 'max_iter': 2000,
             'tol': 1e-6, 'shots_per_6way_class': 64, 'seeds': list(range(5)),
             'train_samples': 3000, 'dev_samples': 3000, 'support_policies': POLICIES}
    if any(spec.get(k) != v for k, v in fixed.items()) or p['source']['feature'] != FEATURE:
        raise ValueError('Runner chỉ hỗ trợ classifier/feature few-shot min5 đã định trước')
    if (set(spec.get('manifest_sha256', {})) != {'train', 'dev'}
            or not set(RUNTIME_KEYS) <= p['source'].get('model_runtime', {}).keys()):
        raise ValueError('Thiếu hash train/dev hoặc metadata backbone nguồn')
    review = p['review']['specification']
    modalities, targets = review['modalities'], review['targets']
    if (not modalities or len(set(modalities)) != len(modalities) or not set(modalities) <= set(MODALITIES)
            or not targets or len(set(targets)) != len(targets)
            or any(not np.isfinite(t) or not 0 < t <= 1 for t in targets)):
        raise ValueError('Bộ modality/target không hợp lệ')
    expected = {(policy, modality, seed, decision) for policy in POLICIES for modality in modalities
                for seed in range(5) for decision in ['default', *[f'target_{t!r}' for t in targets]]}
    found = set()
    for c in p['candidates']:
        key = tuple(c[k] for k in ['policy', 'modality', 'seed', 'decision'])
        if key in found:
            raise ValueError('Ứng viên bị trùng')
        found.add(key)
        if c['decision'] == 'default':
            valid = c['rule'] == 'argmax_real_on_tie' and c['threshold'] is None
        else:
            t = c.get('threshold')
            valid = (c['rule'] == 'fake_if_score_ge_threshold' and isinstance(t, (float, int))
                     and np.isfinite(t) and 0 <= t <= 1
                     and c['decision'] == f"target_{c.get('target_recall_fake')!r}")
        if not valid:
            raise ValueError('Quy tắc/ngưỡng ứng viên không hợp lệ')
    if found != expected:
        raise ValueError('Bộ ứng viên thiếu/thừa so với review đã khóa')
    return lock, settings, config_file


def plan(protocol_file, settings_file=None):
    lock, settings, _ = inspect(protocol_file, settings_file)
    p = lock['payload']
    return {'status': 'plan_only', 'protocol_state': p['state'], 'protocol_sha256': lock['sha256'],
            'split': 'test', 'samples': p['test_samples'], 'candidate_seed_groups': len(p['candidates']),
            'modalities': p['review']['specification']['modalities'],
            'test_opened': False, 'model': settings['model'],
            'executable': p['state'] == 'sealed_task1',
            'scope': 'Task 1 binary only; prepare-only does not read dataset roots or load model'}


def run(protocol_file, train_root, test_root, output_root='results/final/fakeddit_fewshot_test_min5',
        settings_file=None, *, backend_factory=Backbone):
    # Quan trọng: draft/code drift dừng trước mọi truy cập dataset hoặc constructor model.
    lock, settings, config_file = inspect(protocol_file, settings_file, require_sealed=True)
    p, spec = lock['payload'], lock['payload']['source']['specification']
    train_root, test_root, output_root = path(train_root), path(test_root), path(output_root)
    if (output_root.is_relative_to(train_root) or output_root.is_relative_to(test_root)
            or path(protocol_file).is_relative_to(output_root)):
        raise ValueError('Output phải tách khỏi dataset và protocol')
    for root, expected in [(train_root, spec['manifest_sha256']['train']), (test_root, p['test_manifest_sha256'])]:
        if sha256(root / 'manifest.json') != expected:
            raise ValueError('Sai hash manifest train/test min5 đã khóa; không dùng cohort v2')
    train, train_meta = data.fakeddit(train_root, 'train')
    if (train_meta.get('split') != 'train' or train_meta.get('source_split') != 'train'
            or train_meta.get('manifest_sha256') != spec['manifest_sha256']['train']
            or min5_protocol.identities(train) != p['excluded_train']):
        raise ValueError('Train không khớp cohort nguồn đã khóa')
    min5_protocol.validate_support(p['support'], train)
    test, test_meta = data.fakeddit(test_root, 'test')
    min5_protocol.check_test_cohort(lock, test, test_meta, settings_file=config_file)
    # Giữ nguyên ID và thứ tự trong từng support; chỉ encode hợp support một lần.
    union = {id_ for entry in p['support'] for id_ in entry['ids']}
    selected_train = [r for r in train if r['id'] in union]
    lookup = {r['id']: i for i, r in enumerate(selected_train)}
    train_y6 = np.asarray([int(r['6way']) for r in selected_train])
    modalities = p['review']['specification']['modalities']
    local_spec = {**spec, 'methods': modalities}
    manifest = {'kind': 'fakeddit_fewshot_test_min5', 'model': p['model'], 'dataset': 'fakeddit_2way',
                'split': 'test', 'mode': 'full', 'samples': len(test), 'test_opened': True, 'stages': 1,
                'regime': 'supervised_fewshot_frozen_backbone', 'protocol_sha256': lock['sha256'],
                'protocol_file_sha256': sha256(protocol_file), 'protocol_fingerprint': p['fingerprint'],
                'specification': spec, 'candidates': p['candidates'], 'train_cohort': train_meta,
                'test_cohort': test_meta, 'train_features_samples': len(selected_train),
                'binary_labels': {'0': 'Real', '1': 'Fake'}, 'feature': FEATURE,
                'label_usage': 'Fit uses train labels only; test labels used only after all scores predicted.',
                'scope': 'Fixed candidate suite on independent test; no threshold search, winner/seed selection or cascade.',
                'uncertainty': 'Sample std across support seeds on same test IDs, not a confidence interval.'}
    output = new_run({'output_root': output_root}, 'task1', 'test', False)
    write_json(output / 'frozen_plan.json', manifest)
    write_json(output / 'support.json', p['support'])
    write_json(output / 'cohort.json', [{k: r[k] for k in ['id','6way','image_sha256','text_sha256']} for r in test])
    write_json(output / 'run.json', {**manifest, 'status': 'running'})
    started = time.perf_counter()
    backend = backend_factory(settings.get('device', 'auto'))
    try:
        reference = p['source']['model_runtime']
        runtime = getattr(backend, 'metadata', {})
        # JSON chuyển tuple image_size thành list; so theo biểu diễn chuẩn của artifact.
        if any(digest(runtime.get(k)) != digest(reference[k]) for k in RUNTIME_KEYS):
            raise ValueError('Backbone runtime khác architecture/pretrained/tokenizer/context/dtype/image_size đã khóa')
        manifest['model_runtime'] = runtime
        train_image, train_title = encode_cohort(backend, selected_train, settings['batch_size'], with_titles=True)
        test_image, test_title = encode_cohort(backend, test, settings['batch_size'], with_titles=True)
    finally:
        backend.unload()
    manifest['encoding_seconds'] = time.perf_counter() - started
    posterior, models = {}, {}
    for policy in POLICIES:
        entries = [e for e in p['support'] if e['policy'] == policy]
        supports = {e['seed']: np.asarray([lookup[i] for i in e['ids']]) for e in entries}
        quotas = {e['seed']: np.asarray(e['quotas_6way']) for e in entries}
        results, coefficients = predict(train_image, train_title, train_y6, test_image, test_title,
                                        supports, local_spec, quotas)
        posterior.update({(policy, method, seed): score for method, seed, score in results})
        models.update({f'{policy}_{key}': value for key, value in coefficients.items()})
    # Test labels không xuất hiện trong fit hoặc chọn ngưỡng.
    truth = (np.asarray([int(r['6way']) for r in test]) != 0).astype(int)
    records, predictions, details = [], [], {}
    for candidate in p['candidates']:
        policy, modality, seed, decision = (candidate[k] for k in ['policy', 'modality', 'seed', 'decision'])
        score = posterior[policy, modality, seed]
        predicted = score.argmax(1) if decision == 'default' else (score[:, 1] >= candidate['threshold']).astype(int)
        method = f'{policy}_{modality}_{decision}'
        values, per_class = measurements(truth, predicted)
        records.append({'task': '2way', 'method': method, 'seed': seed, 'samples': len(test), **values})
        details[f'{method}_s{seed}'] = per_class
        predictions.extend({'task': '2way', 'method': method, 'seed': seed, 'id': row['id'],
                            'truth': int(y), 'predicted': int(pred), 'p_fake': float(prob[1]),
                            'threshold': '' if decision == 'default' else candidate['threshold']}
                           for row, y, pred, prob in zip(test, truth, predicted, score))
    table(output / 'class_summary.tsv', aggregate(records, ['method'],
          ['recall_real', 'recall_fake', 'precision_fake', 'fake_as_real', 'real_as_fake']))
    np.savez_compressed(output / 'replay_scores.npz', ids=np.asarray([r['id'] for r in test]), labels=truth,
                        class_names=np.asarray(['Real', 'Fake']),
                        **{f'p_{policy}_{modality}_s{seed}': score for (policy, modality, seed), score in posterior.items()})
    np.savez_compressed(output / 'linear_models.npz', **models)
    manifest['elapsed_seconds'] = time.perf_counter() - started
    return publish(output, records, predictions, details, manifest)
