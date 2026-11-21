"""Đối chứng một tầng: prompt prototype làm prior cho StatA, không fusion posterior."""
import time

import numpy as np
from scipy.special import softmax
import yaml

from .backbone import Backbone, encode_cohort
from .diagnostic import infer_groups, probability, subset
from .evaluate import metric, publish, table
from .io import ROOT, BENCHMARKS, code_hashes, new_run, sha256, read_json, write_json
from . import data
from .paper import author_api, generate_prompts, verify_sources

SPEC_FILE = 'configs/waffle_stata.yaml'
DATASETS = ['dtd', 'eurosat']


def specification():
    spec = yaml.safe_load((ROOT / SPEC_FILE).read_text())
    if spec != {'version': 1, 'samples': 256, 'selection_seed': 0,
                'prompt_seeds': list(range(7)), 'grouping_seeds': [0, 1, 2],
                'waffle_pairs': 15, 'adaptation_batch_size': 75,
                'cpu_threads': 1, 'timeout_seconds': 7200}:
        raise ValueError('Giữ pilot 256 ảnh, 7 prompt seed, 3 grouping seed đã định trước')
    return spec


def unit(means):
    norms = np.linalg.norm(means, axis=1, keepdims=True)
    if not np.isfinite(means).all() or (norms <= 1e-12).any():
        raise ValueError('Prototype phải hữu hạn, khác zero')
    return means / norms


def adapt(features, prototypes, initial, spec, seed):
    return infer_groups(features, prototypes, initial,
                        {**spec, 'grouping_seed': seed, 'temperatures': {'stata': 1.}})


def run(settings, name, root, limit=None, backend_factory=Backbone, *, prompt_method='waffle', full_dtd=False, semantic=False):
    if semantic:
        if not full_dtd or prompt_method != 'waffle':
            raise ValueError('Semantic chỉ dùng Waffle full benchmark')
        from .waffle_semantic_stata import specification as semantic_specification, SPEC_FILES
        spec = semantic_specification(name)
        spec_file = SPEC_FILES[name]
    elif full_dtd:
        if prompt_method != 'waffle' or name != 'dtd':
            raise ValueError('Full Waffle–StatA chỉ dùng DTD')
        from .waffle_stata_full import specification as full_specification, SPEC_FILE as spec_file
        spec = full_specification()
    elif prompt_method == 'dclip':
        from .dclip_stata import specification as dclip_specification, SPEC_FILE as spec_file
        spec = dclip_specification()
    elif prompt_method == 'waffle':
        spec = specification()
        spec_file = SPEC_FILE
    else:
        raise ValueError('Chỉ hỗ trợ prompt Waffle/DCLIP')
    if name not in DATASETS or limit not in [None, 5, spec['samples']]:
        raise ValueError('Sai dataset hoặc số mẫu đã khai báo trong cấu hình')
    limit = spec['samples'] if limit is None else limit
    full = limit == BENCHMARKS[name][1]
    if full:
        names, rows, cohort = data.benchmark(root, name)
    else:
        names, rows, cohort = subset(root, name, limit, spec['selection_seed'])
    sources = verify_sources()
    api = author_api()
    descriptor = ROOT / f'third_party/WaffleCLIP/descriptors/descriptors_{name}.json'
    kind = 'waffle_semantic_stata' if semantic else prompt_method + '_stata'
    output = new_run(settings, kind, name, limit == 5)
    prompt_descriptor = descriptor
    if prompt_method == 'dclip' or semantic:
        descriptors = read_json(descriptor)
        if set(descriptors) != set(names):
            raise ValueError('DCLIP descriptor khác taxonomy benchmark')
        prompt_descriptor = output / 'descriptors.json'
        write_json(prompt_descriptor, {n: descriptors[n] for n in names})
    trials = []
    for method, seeds in [('clip', [None]), (prompt_method, spec['prompt_seeds'])]:
        for seed in seeds:
            bank, _ = generate_prompts(api, method, names, prompt_descriptor, spec['waffle_pairs'], seed or 0)
            if list(bank) != names or any(not texts for texts in bank.values()):
                raise ValueError('Prompt thay đổi taxonomy/thứ tự lớp')
            trials.append({'method': method, 'seed': seed, 'prompts': bank})
    semantic_bank = None
    if semantic:
        semantic_bank, _ = generate_prompts(api, 'dclip', names, prompt_descriptor, spec['waffle_pairs'], 0)
        if list(semantic_bank) != names or any(not texts for texts in semantic_bank.values()):
            raise ValueError('Semantic descriptor sai thứ tự/taxonomy')
    plan = {'kind': kind, 'model': settings['model'], 'dataset': name,
            'mode': 'smoke' if limit == 5 else 'full' if full else 'pilot', 'samples': len(rows),
            'classes': names, 'cohort': cohort, 'sources': sources, 'trials': trials,
            'code_sha256': code_hashes(), 'specification': spec,
            'specification_sha256': sha256(ROOT / spec_file), 'descriptor_sha256': sha256(descriptor),
            'decision': f'single argmax; StatA receives unit {prompt_method} prototypes, not posterior averaging',
            'scope': 'Development evaluation on public benchmark; includes previously inspected samples, not independent confirmation.',
            'normalization_control': f'{prompt_method}_unit isolates class-mean normalization required by StatA',
            'seed_reporting': f'{len(spec["prompt_seeds"])} prompt bank(s) x 3 grouping seeds; no best seed selection, std not CI'}
    if semantic:
        plan['semantic'] = {'prompts': semantic_bank, 'source': f'pinned {name} descriptors via original DCLIP generator',
                            'formula': 'unit((1-w)*unit(Waffle mean)+w*unit(semantic mean)); w=0.5',
                            'usage': 'class descriptions, not sentiment or sample titles; fixed before evaluation'}
    write_json(output / 'frozen_plan.json', plan)
    write_json(output / 'cohort.json', rows)
    write_json(output / 'run.json', {**plan, 'status': 'running'})
    start = time.perf_counter()
    backend = backend_factory(settings['device'])
    try:
        features, _ = encode_cohort(backend, rows, settings['batch_size'])
        means = [backend.prototypes(trial['prompts']) for trial in trials]
        semantic_prototypes = unit(backend.prototypes(semantic_bank)) if semantic else None
        plan['model_runtime'] = getattr(backend, 'metadata', {})
    finally:
        backend.unload()
    results, arrays, groups, references = [], {'global_features': features}, {}, {}
    clip = probability(softmax(100 * features @ means[0].T, axis=1))
    results.append(('clip', None, clip))
    arrays['prototypes_clip'] = means[0]
    for grouping in spec['grouping_seeds']:
        prior = softmax(100 * features @ unit(means[0]).T, axis=1)
        posterior, group = adapt(features, unit(means[0]), prior, spec, grouping)
        results.append(('clip_stata', f'g{grouping}', posterior))
        groups[f'groups_g{grouping}'] = group
        references[grouping] = posterior
    semantic_references = {}
    if semantic:
        arrays['prototypes_semantic'] = semantic_prototypes
        semantic_prior = probability(softmax(100 * features @ semantic_prototypes.T, axis=1))
        results.append(('semantic_unit', None, semantic_prior))
        for grouping in spec['grouping_seeds']:
            posterior, group = adapt(features, semantic_prototypes, semantic_prior, spec, grouping)
            if not np.array_equal(group, groups[f'groups_g{grouping}']):
                raise ValueError('Semantic phải dùng cùng nhóm ảnh')
            results.append(('semantic_stata', f'g{grouping}', posterior))
            semantic_references[grouping] = posterior
    comparisons = []
    semantic_comparisons = []
    # Tất cả posterior được tạo trước khi lấy nhãn để đánh giá.
    for trial, mean in zip(trials[1:], means[1:]):
        seed = trial['seed']
        raw = probability(softmax(100 * features @ mean.T, axis=1))
        normalized = unit(mean)
        prior = probability(softmax(100 * features @ normalized.T, axis=1))
        results.extend([(prompt_method, f'p{seed}', raw), (prompt_method + '_unit', f'p{seed}', prior)])
        arrays[f'prototypes_{prompt_method}_p{seed}'] = mean
        if semantic:
            blended = unit((1-spec['semantic_weight'])*normalized + spec['semantic_weight']*semantic_prototypes)
            blended_prior = probability(softmax(100 * features @ blended.T, axis=1))
            results.append(('waffle_semantic', f'p{seed}', blended_prior))
            arrays[f'prototypes_waffle_semantic_p{seed}'] = blended
        for grouping in spec['grouping_seeds']:
            posterior, group = adapt(features, normalized, prior, spec, grouping)
            if not np.array_equal(group, groups[f'groups_g{grouping}']):
                raise ValueError('StatA phải dùng cùng nhóm ảnh giữa các phương pháp')
            results.append((prompt_method + '_stata', f'p{seed}_g{grouping}', posterior))
            comparisons.append((seed, grouping, posterior, raw, prior, references[grouping]))
            if semantic:
                improved, semantic_group = adapt(features, blended, blended_prior, spec, grouping)
                if not np.array_equal(group, semantic_group):
                    raise ValueError('Waffle–semantic phải dùng cùng nhóm ảnh')
                results.append(('waffle_semantic_stata', f'p{seed}_g{grouping}', improved))
                semantic_comparisons.append((seed, grouping, improved, blended_prior, posterior, semantic_references[grouping]))
    truth = np.asarray([row['label'] for row in rows])
    records, predictions, details, paired = [], [], {}, []
    for method, seed, posterior in results:
        posterior = probability(posterior)
        predicted = posterior.argmax(1)
        values = metric(truth, predicted, list(range(len(names))), posterior)
        key = method + ('' if seed is None else '_' + seed)
        details[key] = values.pop('per_class')
        records.append({'task': name, 'method': method, 'seed': seed, 'samples': len(rows), **values})
        predictions.extend({'task': name, 'method': method, 'seed': seed, 'id': row['id'],
                            'truth': row['label'], 'predicted': int(pred)} for row, pred in zip(rows, predicted))
        arrays['p_' + key] = posterior
        print(f'{method} {seed}: accuracy={values["accuracy"]:.4f}', flush=True)
    for seed, grouping, posterior, raw, prior, reference in comparisons:
        pred = posterior.argmax(1)
        row = {'prompt_seed': seed, 'grouping_seed': grouping}
        for name_ref, p in [(prompt_method, raw), (prompt_method + '_unit', prior), ('clip_stata', reference)]:
            base = p.argmax(1)
            row.update({f'delta_{name_ref}_accuracy_pp': float(100 * ((pred == truth).mean() - (base == truth).mean())),
                        f'{name_ref}_wrong_to_correct': int(((base != truth) & (pred == truth)).sum()),
                        f'{name_ref}_correct_to_wrong': int(((base == truth) & (pred != truth)).sum())})
        paired.append(row)
    for seed, grouping, posterior, blended_prior, waffle_adapted, semantic_adapted in semantic_comparisons:
        predicted = posterior.argmax(1)
        row = {'prompt_seed': seed, 'grouping_seed': grouping, 'method': 'waffle_semantic_stata'}
        for name_ref, p in [('waffle_semantic', blended_prior), ('waffle_stata', waffle_adapted), ('semantic_stata', semantic_adapted)]:
            baseline = p.argmax(1)
            row.update({f'delta_{name_ref}_accuracy_pp': float(100*((predicted==truth).mean()-(baseline==truth).mean())),
                        f'{name_ref}_wrong_to_correct': int(((baseline!=truth)&(predicted==truth)).sum()),
                        f'{name_ref}_correct_to_wrong': int(((baseline==truth)&(predicted!=truth)).sum())})
        paired.append(row)
    if semantic:
        for row in paired:
            row.setdefault('method', 'waffle_stata')
    table(output / 'comparisons.tsv', paired)
    np.savez_compressed(output / 'replay_scores.npz', ids=np.asarray([r['id'] for r in rows]),
                        labels=truth, class_names=np.asarray(names), **arrays, **groups)
    plan['elapsed_seconds'] = time.perf_counter() - start
    return publish(output, records, predictions, details, plan)
