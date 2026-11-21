"""Đối chiếu prompt của CLIP, DCLIP và Waffle trên cùng một backbone."""
import time
import numpy as np
from . import data
from .backbone import Backbone, encode_cohort
from .evaluate import metric, publish
from .io import ROOT, BENCHMARKS, new_run, path, read_json, sha256, write_json, protocol_hashes
from .paper import verify_sources, author_api, generate_prompts

def prompt_trials(api, names, descriptor, settings, smoke=False):
    """Một bank Waffle/seed được dùng chung cho cả hai cách aggregation."""
    spec = settings['benchmark']
    trials = []
    for method in ['clip'] if smoke else ['clip', 'dclip', 'waffle']:
        for seed in spec['seeds'] if method == 'waffle' else [None]:
            print(f'Chuẩn bị prompt: {method} seed={seed}', flush=True)
            prompts, _ = generate_prompts(api, method, names, descriptor, spec['waffle_pairs'], seed or 0)
            if list(prompts) != list(names) or any((not v for v in prompts.values())):
                raise ValueError('Prompt thiếu lớp hoặc thay đổi thứ tự lớp')
            for texts in prompts.values():
                api.clip.tokenize(texts)
            trials.append({'method': method, 'seed': seed, 'prompts': prompts})
    return trials

def run(settings, name, root, smoke=False, backend_factory=Backbone, backend=None):
    print(f'[{"SMOKE" if smoke else "FULL"}] {name}: '
          f'{"5 ảnh, CLIP, đủ lớp" if smoke else "đủ split, tất cả phương pháp và 7 Waffle seed"}', flush=True)
    sources = verify_sources()
    names, rows, cohort = data.benchmark(root, name, smoke)
    descriptor = ROOT / 'third_party/WaffleCLIP/descriptors' / f"descriptors_{('imagenet' if name == 'imagenetv2' else name)}.json"
    values = read_json(descriptor)
    if set(values) != set(names):
        raise ValueError('Descriptor khác taxonomy đã xuất')
    output = new_run(settings, name, 'test', smoke)
    converted = output / 'descriptors.json'
    write_json(converted, {n: values[n] for n in names})
    trials = prompt_trials(author_api(), names, converted, settings, smoke=smoke)
    plan = {'kind': 'benchmark', 'model': settings['model'], 'dataset': name, 'mode': 'smoke' if smoke else 'full', 'samples': len(rows), 'split': BENCHMARKS[name][2], 'settings': settings, 'protocol': protocol_hashes(settings), 'sources': sources, 'descriptor_sha256': sha256(descriptor), 'classes': names, 'trials': trials, 'cohort': cohort, 'comparison_scope': 'Methods rerun on MetaCLIP2. Published paper numbers on other backbones are contextual only.', 'ablation': 'waffle_unit_ablation normalizes class means; a control, not a claimed new method.'}
    plan['smoke_scope'] = 'clip_only_all_classes_5_images' if smoke else None
    write_json(output / 'frozen_plan.json', plan)
    write_json(output / 'cohort.json', rows)
    write_json(output / 'run.json', {**plan, 'status': 'running'})
    owns_backend = backend is None
    if owns_backend:
        backend = backend_factory(settings['device'])
    plan['model_runtime'] = getattr(backend, 'metadata', {'backend_class': type(backend).__name__})
    records = []
    predictions = []
    details = {}
    started = time.perf_counter()
    try:
        features, _ = encode_cohort(backend, rows, settings['batch_size'])
        image_seconds = time.perf_counter() - started
        truth = np.asarray([r['label'] for r in rows])
        labels = list(range(len(names)))
        for trial in trials:
            print(f'Bắt đầu {trial["method"]} seed={trial["seed"]}: '
                  f'{sum(map(len,trial["prompts"].values()))} prompt', flush=True)
            means = backend.prototypes(trial['prompts'])
            variants = [(trial['method'], means)]
            if trial['method'] == 'waffle':
                variants.append(('waffle_unit_ablation', means / np.maximum(np.linalg.norm(means, axis=1, keepdims=True), 1e-12)))
            for method, prototypes in variants:
                scores = features @ prototypes.T
                predicted = scores.argmax(1)
                values = metric(truth, predicted, labels, scores)
                tag = f"{method}:{trial['seed']}"
                details[tag] = values.pop('per_class')
                records.append({'task': name, 'method': method, 'seed': trial['seed'], 'samples': len(rows), **values})
                for row, pred in zip(rows, predicted):
                    predictions.append({'task': name, 'method': method, 'seed': trial['seed'], 'id': row['id'], 'truth': row['label'], 'predicted': int(pred)})
                print(f"{method} seed={trial['seed']}: accuracy={values['accuracy']:.4f}", flush=True)
    finally:
        if owns_backend:
            backend.unload()
    plan.update(elapsed_seconds=time.perf_counter() - started, image_encode_seconds=image_seconds)
    return publish(output, records, predictions, details, plan)
