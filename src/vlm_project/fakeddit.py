"""Fakeddit v2: giữ hai cấu hình đã chọn trên dev, kèm đối chứng modality."""
import time
import numpy as np
from scipy.special import softmax
from . import data
from .backbone import Backbone, encode_cohort
from .benchmark import prompt_trials
from .evaluate import metric, publish
from .io import ROOT, path, read_json, write_json, new_run, protocol_hashes
from .methods import majority, hierarchy, fuse, replace_clip_vote
from .paper import verify_sources, author_api
SELECTED = {'2way': 'majority_v1_text', '6way': 'majority_v1_legacy_mean_mm'}

def profile(settings):
    value = read_json(settings['fakeddit']['prompt_file'])
    if value['model'] != settings['model'] or value['selected_methods'] != SELECTED or value['grouping_seeds'] != [0, 1, 2] or (value['rules'] != {'image_logit_scale': 100.0, 'text_logit_scale': 25.0, 'image_weight': 0.75}):
        raise ValueError('Profile khác cấu hình đã chọn trên dev')
    for task, size in [('2way', 2), ('6way', 6)]:
        labels = [str(k) for k in range(size)]
        for bank in [*value['image_prompts'][task].values(), value['text_prompts'][task]]:
            if list(bank) != labels or any((not isinstance(v, list) or not v for v in bank.values())):
                raise ValueError('Prompt Fakeddit thiếu lớp hoặc sai thứ tự')
    return value

def predict(features, titles, means, text_means, spec):
    """Không nhận nhãn thật. Trả từng method/seed và log phiếu Majority."""
    results = []
    diagnostics = {}
    rules = spec['rules']
    weight = rules['image_weight']
    text_prob = {}
    image_prob = {}
    semantic_prob = {}
    for task in ['2way', '6way']:
        image_prob[task] = softmax(rules['image_logit_scale'] * (features @ means[task]['legacy'].T), axis=1)
        legacy_text = softmax(rules['text_logit_scale'] * (titles @ means[task]['legacy'].T), axis=1)
        text_prob[task] = softmax(rules['text_logit_scale'] * (titles @ text_means[task].T), axis=1)
        semantic_prob[task] = fuse(image_prob[task], text_prob[task], weight)
        scores = {'baseline_image': image_prob[task], 'baseline_text': legacy_text, 'baseline_fusion': fuse(image_prob[task], legacy_text, weight), 'semantic_text_fusion': semantic_prob[task]}
        for method, values in scores.items():
            results.append((task, method, None, values.argmax(1).astype(str)))
    binary = means['2way']['semantic_v1']
    unit = binary / np.linalg.norm(binary, axis=1, keepdims=True)
    clip_mm = fuse(softmax(100 * (features @ unit.T), axis=1), text_prob['2way'], weight)
    for seed in spec['grouping_seeds']:
        gate, detail = majority(features, binary, seed)
        mm_gate, mm_votes = replace_clip_vote(detail['votes'], clip_mm)
        results.append(('2way', 'proposed', seed, mm_gate.astype(str)))
        results.append(('6way', 'proposed', seed, hierarchy(gate.astype(str), semantic_prob['6way'], list('012345'))))
        diagnostics[f'votes_{seed}'] = detail['votes']
        diagnostics[f'mm_votes_{seed}'] = mm_votes
        diagnostics[f'groups_{seed}'] = detail['batch_ids']
    return (results, diagnostics)

def run(settings, split, root, smoke=False, backend_factory=Backbone):
    from .protocol import validate_lock, check_test_cohort
    spec = profile(settings)
    lock = validate_lock(settings) if split == 'test' else None
    sources = verify_sources()
    rows, cohort = data.fakeddit(root, split, smoke)
    if lock:
        check_test_cohort(lock, rows)
    output = new_run(settings, 'fakeddit', split, smoke)
    api = author_api()
    trials = {}
    for task in ['2way', '6way']:
        names = list(spec['names'][task].values())
        labeled = read_json(ROOT / f'configs/descriptors/fakeddit_{task}.json')['descriptors']
        if set(labeled) != set(spec['names'][task]):
            raise ValueError('Descriptor thiếu ID lớp Fakeddit')
        source = {name: labeled[label] for label, name in spec['names'][task].items()}
        if set(names) != set(source):
            raise ValueError('Descriptor Fakeddit sai taxonomy')
        converted = output / f'descriptors_{task}.json'
        write_json(converted, {n: source[n] for n in names})
        trials[task] = [t for t in prompt_trials(api, names, converted, settings) if t['method'] != 'clip']
    plan = {'kind': 'fakeddit', 'model': settings['model'], 'split': split, 'mode': 'smoke' if smoke else 'full', 'samples': len(rows), 'cohort': cohort, 'settings': settings, 'profile': spec, 'protocol': protocol_hashes(settings), 'sources': sources, 'trials': trials, 'test_lock_sha256': lock['sha256'] if lock else None, 'selected_methods': SELECTED, 'comparison_scope': 'DCLIP/Waffle image-only adaptations use manually authored Fakeddit taxonomy descriptors; not original paper results.', 'transductive': 'Groups of 75, seeds 0/1/2, unlabeled query features; EM/StatA never receive labels.', 'controls': 'Image-only, title-only, late fusion, semantic title fusion share the same cohort and backbone.'}
    write_json(output / 'frozen_plan.json', plan)
    write_json(output / 'cohort.json', rows)
    write_json(output / 'run.json', {**plan, 'status': 'running'})
    backend = backend_factory(settings['device'])
    plan['model_runtime'] = getattr(backend, 'metadata', {'backend_class': type(backend).__name__})
    started = time.perf_counter()
    records = []
    predictions = []
    details = {}
    try:
        means = {task: {bank: backend.prototypes(prompts) for bank, prompts in banks.items()} for task, banks in spec['image_prompts'].items()}
        text_means = {task: backend.prototypes(bank) for task, bank in spec['text_prompts'].items()}
        features, titles = encode_cohort(backend, rows, settings['batch_size'], with_titles=True)
        inference, diagnostics = predict(features, titles, means, text_means, spec)
        np.savez_compressed(output / 'majority_diagnostics.npz', **diagnostics)
        for task, bank in trials.items():
            for trial in bank:
                vectors = backend.prototypes(trial['prompts'])
                predicted = (features @ vectors.T).argmax(1).astype(str)
                inference.append((task, trial['method'] + '_image', trial['seed'], predicted))
        for task, method, seed, predicted in inference:
            labels = list('01' if task == '2way' else '012345')
            truth = [r[task] for r in rows]
            values = metric(truth, predicted, labels)
            details[f'{task}:{method}:{seed}'] = values.pop('per_class')
            records.append({'task': task, 'method': method, 'seed': seed, 'samples': len(rows), **values})
            for row, pred in zip(rows, predicted):
                predictions.append({'task': task, 'method': method, 'seed': seed, 'id': row['id'], 'truth': row[task], 'predicted': str(pred)})
            print(f"{task} {method} seed={seed}: macro_f1={values['macro_f1']:.4f}", flush=True)
    finally:
        backend.unload()
    plan.update(elapsed_seconds=time.perf_counter() - started)
    return publish(output, records, predictions, details, plan)
