"""Kiểm chứng benchmark paper bằng OpenAI ViT-L/14, tách model chính."""
from datetime import datetime, timezone
import importlib.metadata
import time
import uuid
import numpy as np

from . import data
from .evaluate import metric, publish
from .io import ROOT, path, read_json, write_json, sha256, code_hashes
from .paper import verify_sources, author_api, generate_prompts, load_functions

MODEL = 'openai_clip_vitl14'
ARCHITECTURE = 'ViT-L/14'
BATCH_SIZE = 64
REFERENCE_DATASETS = {
    'dtd': {'descriptor': 'dtd', 'split': 'test-partition-1'},
    'imagenetv2': {'descriptor': 'imagenet', 'split': 'matched-frequency'},
    'imagenet': {'descriptor': 'imagenet', 'split': 'val'},
    'cub': {'descriptor': 'cub', 'split': 'test'},
    'food101': {'descriptor': 'food101', 'split': 'test'},
    'pets': {'descriptor': 'pets', 'split': 'test'},
    'eurosat': {'descriptor': 'eurosat', 'split': 'all-images-as-waffle-upstream'},
    'places365': {'descriptor': 'places365', 'split': 'val-small-256'},
}


def reference_trials(api, names, descriptor, dataset='dtd', label_order=None):
    """Giao thức paper được đóng băng trước khi encode ảnh."""
    labels = names if label_order is None else label_order
    if len(set(names)) != len(names) or len(labels) != len(names) or set(labels) != set(names):
        raise ValueError('Taxonomy prompt và dataset phải khớp chính xác từng tên lớp')
    trials = []
    for method, mode, seeds in [
        ('clip_photo', 'clip', [None]),
        ('dclip_photo', 'gpt_descriptions', [None]),
        ('waffle', 'waffle', list(range(7))),
        ('clip_labels_original', 'dclip_labels', [None]),
        ('dclip_original', 'dclip', [None]),
    ]:
        for seed in seeds:
            prompts, _ = generate_prompts(api, mode, names, descriptor, 15, seed or 0)
            # DCLIP load.py sets after_text = label_after_text = '.' for ImageNet.
            if dataset == 'imagenet' and method in ['clip_labels_original', 'dclip_original']:
                prompts = {name: [value + '.' for value in values]
                           for name, values in prompts.items()}
            if list(prompts) != names:
                raise ValueError('Prompt tham chiếu đổi thứ tự lớp')
            for values in prompts.values():
                api.clip.tokenize(values)
            # Sinh RNG theo thứ tự tác giả rồi mới đổi cột sang ID dataset.
            # Pets lưu theo ID annotation; JSON tác giả nhóm mèo trước chó.
            trials.append({'method': method, 'seed': seed,
                           'prompts': {name: prompts[name] for name in labels}})
    return trials


def reference_targets(dataset='dtd'):
    """Đọc mốc đã lưu offline; giữ riêng số paper và CSV tác giả."""
    registry_file = ROOT / 'configs/paper_results/references.json'
    registry = read_json(registry_file)
    if registry.get('schema_version') != 1 or registry.get('backbone') != {
            'provider':'OpenAI CLIP', 'architecture':ARCHITECTURE, 'image_size':224}:
        raise ValueError('Bảng tham chiếu sai phiên bản hoặc backbone')
    specification = registry['datasets'][dataset]
    provenance = {'reference_registry_file': str(registry_file.relative_to(ROOT)),
                  'reference_registry_sha256': sha256(registry_file)}
    targets = {}
    for method in ['clip_photo', 'dclip_photo', 'waffle']:
        value = specification['waffle']['author_csv'][method]
        snapshot = registry['author_csv_snapshots'][value['source_file']]
        if sha256(ROOT / snapshot['file']) != snapshot['sha256']:
            raise ValueError('CSV tham chiếu đã lưu bị đổi checksum')
        targets[method] = {
            'paper_accuracy_percent': specification['waffle']['paper_accuracy_percent'][method],
            'author_csv_accuracy_percent': value['accuracy_percent'],
            'author_csv_std_percent_ddof0': value['std_percent_ddof0'],
            'source_file': value['source_file'], 'source_sha256': snapshot['sha256'],
            'source_snapshot_file': snapshot['file'],
            'paper': registry['publications']['waffle']['url'], 'table': '10', **provenance,
        }
    for method, value in specification['dclip']['paper_accuracy_percent'].items():
        targets[method] = {'paper_accuracy_percent': value,
                           'paper': registry['publications']['dclip']['url'], 'table': '1', **provenance}
    return targets


class OpenAIReference:
    def __init__(self, device, api):
        if device != 'cuda' or not api.torch.cuda.is_available():
            raise ValueError('Lượt tham chiếu yêu cầu CUDA, giống code tác giả')
        self.api, self.device = api, device
        self.model, _ = api.clip.load(ARCHITECTURE, device=device, jit=False)
        self.model.eval().requires_grad_(False)
        from PIL import Image
        import torchvision.transforms as transforms
        original = load_functions(ROOT / 'third_party/WaffleCLIP/datasets.py', ['_transform'],
                                  {'Image': Image, 'transforms': transforms})
        self.preprocess = original._transform(224)
        url = api.clip.load.__globals__['_MODELS'][ARCHITECTURE]
        self.metadata = {
            'architecture': ARCHITECTURE, 'pretrained': 'OpenAI original release',
            'checkpoint_url': url, 'checkpoint_sha256_expected': url.split('/')[-2],
            'tokenizer': 'OpenAI CLIP BPE', 'context_length': 77,
            'image_size': 224, 'device': device,
            'dtype': str(self.model.dtype),
            'first_parameter_dtype': str(next(self.model.parameters()).dtype),
            'preprocess': 'WaffleCLIP datasets.py::_transform(224), exact pinned function',
            'aggregation': 'mean of per-descriptor cosine similarities, native torch dtype',
            'decision': 'torch.topk on softmax scores; original class-name CLIP uses raw scores as in DCLIP main.py',
        }
        api.torch.backends.cudnn.deterministic = True
        api.torch.backends.cudnn.benchmark = True
        print(f'Model tham chiếu {ARCHITECTURE}: {self.metadata}', flush=True)

    def encode_images(self, rows):
        from PIL import Image
        blocks = []
        with self.api.torch.no_grad():
            for start in range(0, len(rows), BATCH_SIZE):
                tensors = []
                for row in rows[start:start + BATCH_SIZE]:
                    with Image.open(row['image']) as image:
                        tensors.append(self.preprocess(image))
                images = self.api.torch.stack(tensors).to(self.device)
                blocks.append(self.api.F.normalize(self.model.encode_image(images)).cpu())
                print(f'Reference encode: {min(start+BATCH_SIZE,len(rows))}/{len(rows)}', flush=True)
        return self.api.torch.cat(blocks)

    def classify(self, features, trial):
        torch = self.api.torch
        with torch.no_grad():
            if trial['method'] == 'clip_labels_original':
                texts = [values[0] for values in trial['prompts'].values()]
                vectors = self.api.F.normalize(self.model.encode_text(self.api.clip.tokenize(texts).to(self.device)))
                banks = [vectors[i:i+1] for i in range(len(vectors))]
            else:
                banks = [self.api.F.normalize(self.model.encode_text(self.api.clip.tokenize(values).to(self.device)))
                         for values in trial['prompts'].values()]
            probabilities, predictions, top5 = [], [], []
            for block in features.split(BATCH_SIZE):
                images = block.to(self.device)
                if trial['method'] == 'clip_labels_original':
                    scores = images @ vectors.T
                else:
                    scores = torch.stack([self.api.waffle.aggregate_similarity(images @ vectors.T, 'mean')
                                          for vectors in banks], dim=1)
                values = scores.softmax(dim=-1)
                decision = scores if trial['method'] == 'clip_labels_original' else values
                probabilities.append(values.cpu().numpy())
                predictions.append(decision.topk(1, dim=1).indices[:, 0].cpu().numpy())
                top5.append(decision.topk(5, dim=1).indices.cpu().numpy())
        return np.concatenate(predictions), np.concatenate(top5), np.concatenate(probabilities)

    def unload(self):
        del self.model
        self.api.torch.cuda.empty_cache()


def comparison(records, targets):
    result = []
    for method, target in targets.items():
        values = [r['accuracy'] * 100 for r in records if r['method'] == method]
        observed = float(np.mean(values))
        reference = target.get('author_csv_accuracy_percent', target['paper_accuracy_percent'])
        result.append({'method': method, 'observed_accuracy_percent': observed,
                       'observed_std_percent_ddof0': float(np.std(values)),
                       'reference_accuracy_percent': reference, 'delta_pp': observed-reference,
                       'approx_match_within_0_10pp': abs(observed-reference) <= .10,
                       'tolerance_pp_predeclared': .10, **target})
    return result


def run(settings, name, root, smoke=False, backend_factory=OpenAIReference):
    if name not in REFERENCE_DATASETS:
        raise ValueError('Lượt kiểm chứng chỉ dành cho 8 benchmark đã khai báo (gồm DTD) và OpenAI ViT-L/14')
    specification = REFERENCE_DATASETS[name]
    sources = verify_sources()
    names, rows, cohort = data.benchmark(root, name, smoke)
    descriptor = ROOT / f'third_party/WaffleCLIP/descriptors/descriptors_{specification["descriptor"]}.json'
    source = read_json(descriptor)
    if name == 'pets':
        if len(source) != len(names) or set(source) != set(names):
            raise ValueError('Taxonomy Pets khác tên lớp trong repo tác giả')
    elif list(source) != names:
        raise ValueError(f'Thứ tự {name} khác repo tác giả')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = path(f'results/final/paper_reference/{name}') / MODEL / (
        f'test_{"smoke" if smoke else "full"}_n{len(rows)}_{stamp}_{uuid.uuid4().hex[:8]}')
    output.mkdir(parents=True)
    write_json(output / 'descriptors.json', source)
    api = author_api()
    trials = reference_trials(api, list(source), output/'descriptors.json', dataset=name,
                              label_order=names)
    targets = reference_targets(name)
    write_json(output/'paper_targets.json', targets)
    plan = {
        'kind': 'paper_reference', 'model': MODEL, 'dataset': name,
        'split': specification['split'], 'mode': 'smoke' if smoke else 'full', 'samples': len(rows),
        'classes': names, 'cohort': cohort, 'sources': sources, 'code_sha256': code_hashes(),
        'author_class_order': list(source),
        'author_to_dataset_label': [names.index(name) for name in source],
        'batch_size': BATCH_SIZE, 'author_default_batch_size': 640,
        'libraries': {name: importlib.metadata.version(name) for name in ['numpy','torch','torchvision','Pillow']},
        'trials': trials,
        'settings': {'model': MODEL, 'architecture': ARCHITECTURE, 'device': 'cuda',
                     'batch_size': BATCH_SIZE, 'waffle_pairs': 15, 'waffle_seeds': list(range(7)),
                     'methods': list(dict.fromkeys(t['method'] for t in trials))},
        'project_config_sha256': sha256(ROOT/'configs/project.yaml'), 'paper_targets': targets,
        'comparison_tolerance_pp_predeclared': .10,
        'scope': 'One OpenAI ViT-L/14 checkpoint; author photo prompts plus original DCLIP controls. Main MetaCLIP2 config unchanged.',
    }
    write_json(output/'frozen_plan.json', plan)
    write_json(output/'cohort.json', rows)
    write_json(output/'run.json', {**plan, 'status': 'running'})
    backend = backend_factory('cuda', api)
    plan['model_runtime'] = backend.metadata
    started = time.perf_counter()
    records, predictions, details, arrays = [], [], {}, {}
    truth = np.asarray([r['label'] for r in rows])
    try:
        features = backend.encode_images(rows)
        for trial in trials:
            predicted, top5, probabilities = backend.classify(features, trial)
            values = metric(truth, predicted, list(range(len(names))))
            values['top5'] = float((top5 == truth[:, None]).any(1).mean())
            tag = f'{trial["method"]}_seed{trial["seed"]}'
            details[tag] = values.pop('per_class')
            arrays[tag+'_probabilities'] = probabilities
            arrays[tag+'_top1'] = predicted
            arrays[tag+'_top5'] = top5
            records.append({'task': name, 'method': trial['method'], 'seed': trial['seed'],
                            'samples': len(rows), **values})
            predictions.extend({'task': name, 'method': trial['method'], 'seed': trial['seed'],
                                'id': row['id'], 'truth': row['label'], 'predicted': int(pred)}
                               for row, pred in zip(rows, predicted))
            print(f'Reference {tag}: accuracy={values["accuracy"]:.6f}', flush=True)
    finally:
        backend.unload()
    np.savez_compressed(output/'replay_scores.npz', labels=truth, ids=np.asarray([r['id'] for r in rows]), **arrays)
    if not smoke:
        write_json(output/'paper_comparison.json', comparison(records, targets))
    plan['elapsed_seconds'] = time.perf_counter()-started
    return publish(output, records, predictions, details, plan)
