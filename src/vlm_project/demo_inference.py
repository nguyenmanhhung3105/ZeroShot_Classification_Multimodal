"""Kaggle-only adapters for arbitrary uploads; research runners remain unchanged."""
import base64
from collections import OrderedDict
import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np
from scipy.special import expit, softmax

from .demo_service import METHODS

ROOT = Path(__file__).resolve().parents[2]


def label_names(task, fewshot=False):
    if task == '2way':
        return ['Real', 'Fake'] if fewshot else ['Fake', 'Real']
    return ['Real', 'Satire/Parody', 'False Connection', 'Imposter Content',
            'Manipulated Content', 'Misleading Content']


def linear_probability(x, coef, intercept):
    x, coef, intercept = map(np.asarray, (x, coef, intercept))
    if (x.ndim != 2 or coef.ndim != 2 or x.shape[1] != coef.shape[1]
            or intercept.shape != (len(coef),)
            or any(not np.isfinite(a).all() for a in (x, coef, intercept))):
        raise ValueError('Head khác chiều backbone hoặc tham số không hợp lệ')
    logits = x @ coef.T + intercept
    if len(coef) == 1:
        p = expit(logits[:, 0])
        return np.stack([1 - p, p], axis=1)
    return softmax(logits, axis=1)


def cascade_decision(gate, child, threshold=None):
    from .fakeddit_fewshot_cascade import route
    predicted = int(route(gate, child, threshold)[0])
    joint = np.concatenate([gate[:, :1], gate[:, 1:] * child], axis=1)[0]
    return predicted, joint


def create_backend(name):
    from .backbone import Backbone
    if name == 'metaclip2':
        return Backbone('cuda')
    # OpenCLIP adapter for demo, deliberately distinct from paper_reference.py.
    import open_clip
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Demo worker yêu cầu GPU CUDA; không fallback về CPU local')

    class OpenAIAdapter(Backbone):
        def __init__(self):
            self.device, self.torch = 'cuda', torch
            self.name = 'openai_clip'
            self.model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
            self.model.to('cuda').eval().requires_grad_(False)
            self.tokenizer = open_clip.get_tokenizer('ViT-L-14')
            self._prototype_cache, self._text_batch_size = {}, 64
            self.metadata = {'architecture': 'ViT-L-14', 'pretrained': 'openai',
                             'scope': 'OpenCLIP demo adapter; not exact paper reference runtime'}

        def check_prompts(self, texts):
            if any(len(self.tokenizer.encode(t)) + 2 > self.model.context_length for t in texts):
                raise ValueError('Prompt vượt context của OpenAI CLIP')

    return OpenAIAdapter()


class Engine:
    """One resident backbone by default; bounded image cache and per-model prototype cache."""
    def __init__(self, backbones, assets=None, backend_factory=create_backend):
        self.enabled = list(backbones)
        self.factory = backend_factory
        self.backend, self.current = None, None
        self.image_cache = OrderedDict()
        self.prompt_banks, self.prompt_error = {}, ''
        self.prediction_cache = {}
        self.profile = json.loads((ROOT / 'configs/fakeddit_prompts.json').read_text())
        self.task1 = {'image_logit_scale': 100., 'text_logit_scale': 25., 'image_weight': .75,
                      'clip_weight': .5, 'grouping_seeds': [0], 'adaptation_batch_size': 75,
                      'cpu_threads': 1}
        self.assets_root = Path(assets).resolve() if assets else None
        self.assets = json.loads((self.assets_root / 'manifest.json').read_text()) if assets else {}

    def prepare_prompts(self):
        from .paper import install_sources, author_api, generate_prompts
        import tempfile
        try:
            install_sources()
            api = author_api()
            for task in ('2way', '6way'):
                mapping = self.profile['names'][task]
                descriptors = json.loads((ROOT / f'configs/descriptors/fakeddit_{task}.json').read_text())['descriptors']
                with tempfile.TemporaryDirectory() as tmp:
                    converted = Path(tmp) / 'descriptors.json'
                    converted.write_text(json.dumps({name: descriptors[k] for k, name in mapping.items()}))
                    for method in ('dclip', 'waffle'):
                        bank, _ = generate_prompts(api, method, list(mapping.values()), converted, 15, 0)
                        self.prompt_banks[task, method] = {k: bank[name] for k, name in mapping.items()}
        except Exception as exc:
            self.prompt_error = f'Không chuẩn bị được prompt tác giả ({type(exc).__name__})'

    def load(self, name):
        if name not in self.enabled:
            raise ValueError('Backbone chưa bật trên Kaggle worker')
        if self.current == name:
            return self.backend
        if self.backend is not None:
            self.backend.unload()
            self.backend, self.current = None, None
        self.image_cache.clear()
        self.backend = self.factory(name)
        self.current = name
        return self.backend

    def asset(self, backbone, key):
        entry = self.assets.get(backbone, {}).get(key)
        if not entry:
            raise ValueError(f'Chưa có {key} đúng backbone; xuất assets từ artifact train/dev trước demo')
        file = (self.assets_root / entry['file']).resolve()
        if not file.is_relative_to(self.assets_root):
            raise ValueError('Đường dẫn asset không hợp lệ')
        if hashlib.sha256(file.read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError('Asset sai checksum')
        return np.load(file, allow_pickle=False)

    def head(self, archive, key, x, expected_classes):
        if key + '_classes' in archive:
            if archive[key + '_classes'].tolist() != expected_classes:
                raise ValueError('Head sai thứ tự lớp')
        return linear_probability(x, archive[key + '_coef'], archive[key + '_intercept'])

    def features(self, backend, payload):
        from PIL import Image
        started = time.perf_counter()
        raw = base64.b64decode(payload['image'], validate=True)
        identity = hashlib.sha256(raw).hexdigest()
        if identity in self.image_cache:
            image_feature = self.image_cache.pop(identity)
        else:
            with Image.open(io.BytesIO(raw)) as im:
                image_feature = backend.encode_images([im.convert('RGB')])
        self.image_cache[identity] = image_feature
        while len(self.image_cache) > 8:
            self.image_cache.popitem(last=False)
        title = backend.encode_texts([payload['title']])
        return image_feature, title, time.perf_counter() - started

    def run(self, payload, emit):
        rows, encoding = [], {}
        started = time.perf_counter()
        for backbone in payload['backbones']:
            self.prediction_cache = {}
            emit({'rows': rows, 'message': f'Đang chuẩn bị {backbone}'})
            try:
                backend = self.load(backbone)
                image, title, seconds = self.features(backend, payload)
                encoding[backbone] = seconds
                features_error = None
            except Exception as exc:
                features_error = f'{type(exc).__name__}: {exc}'
            for method in payload['methods']:
                name, modality, pipeline = METHODS[method]
                row = {'backbone': backbone, 'method': method, 'name': name, 'modality': modality,
                       'pipeline': pipeline, 'status': 'unavailable', 'task': payload['task']}
                mark = time.perf_counter()
                try:
                    if payload['pipeline'] != 'all' and payload['pipeline'] != pipeline:
                        raise ValueError('Ngoài pipeline được chọn')
                    if pipeline == 'cascade' and payload['task'] != '6way':
                        raise ValueError('Hai tầng trả nhãn 6 lớp; chọn bài toán 6 lớp')
                    if features_error:
                        raise ValueError(features_error)
                    row.update(self.predict_method(method, backbone, backend, image, title, payload))
                    row['status'] = 'complete'
                except (ValueError, KeyError, OSError) as exc:
                    row['message'] = str(exc)
                except Exception as exc:
                    row.update(status='failed', message=f'{type(exc).__name__}: {exc}')
                row['seconds'] = time.perf_counter() - mark
                rows.append(row)
                emit({'rows': rows, 'encoding_seconds': encoding, 'message': f'Đã xử lý {name}'})
        return {'rows': rows, 'encoding_seconds': encoding,
                'elapsed_seconds': time.perf_counter() - started,
                'message': 'Hoàn tất; xem trạng thái từng phương pháp', 'status': 'complete'}

    @staticmethod
    def result(prob, task, *, fewshot=False, predicted=None, **detail):
        prob = np.asarray(prob).reshape(-1)
        labels = label_names(task, fewshot)
        if len(prob) != len(labels) or not np.isfinite(prob).all() or not np.isclose(prob.sum(), 1):
            raise ValueError('Đầu ra phương pháp không phải phân phối lớp hợp lệ')
        index = int(prob.argmax()) if predicted is None else int(predicted)
        return {'predicted': labels[index], 'score': float(prob[index]),
                'scores': {k: float(v) for k, v in zip(labels, prob)}, **detail}

    def predict_method(self, method, backbone, backend, image, title, payload):
        from .fakeddit_task1 import unit
        from .methods import fuse
        task = payload['task']
        prompts = self.profile['image_prompts'][task]['semantic_v1' if task == '2way' else 'legacy']
        means = backend.prototypes(prompts)
        if task == '2way':
            means = unit(means)
        text_means = backend.prototypes(self.profile['text_prompts'][task])
        if task == '2way':
            text_means = unit(text_means)
        clip = softmax(100 * image @ means.T, axis=1)
        text = softmax(25 * title @ text_means.T, axis=1)
        if method.startswith('legacy_'):
            raw = backend.prototypes(self.profile['image_prompts'][task]['legacy'])
            pi = softmax(100 * image @ raw.T, axis=1)
            pt = softmax(25 * title @ raw.T, axis=1)
            return self.result({'legacy_image': pi, 'legacy_title': pt,
                                'legacy_fusion': fuse(pi, pt, .75)}[method], task)
        if method in ('clip_image', 'clip_title', 'clip_image_title'):
            prob = {'clip_image': clip, 'clip_title': text,
                    'clip_image_title': fuse(clip, text, .75)}[method]
            return self.result(prob, task)
        if method in ('dclip', 'waffle', 'waffle_unit'):
            key = 'waffle' if method.startswith('waffle') else method
            if (task, key) not in self.prompt_banks:
                raise ValueError(self.prompt_error or 'Chưa chuẩn bị prompt tác giả')
            vectors = backend.prototypes(self.prompt_banks[task, key])
            if method == 'waffle_unit':
                vectors = unit(vectors)
            return self.result(softmax(100 * image @ vectors.T, axis=1), task, seed=0,
                               note='Prompt Fakeddit, seed demo cố định 0; không phải kết quả paper')
        if method == 'calip':
            # A CALIP pass needs projected patches; global-only cached features do not suffice.
            from PIL import Image
            from .calip import encode, score
            with Image.open(io.BytesIO(base64.b64decode(payload['image']))) as im:
                g, p, _ = encode(backend, [im.convert('RGB')])
            logits, _ = score(g.cpu(), p.cpu(), unit(means))
            return self.result(softmax(logits.numpy(), axis=1), task)
        if method == 'hierarchy_clip':
            from .methods import hierarchy
            binary = unit(backend.prototypes(self.profile['image_prompts']['2way']['semantic_v1']))
            six = unit(backend.prototypes(self.profile['image_prompts']['6way']['legacy']))
            gate = softmax(100 * image @ binary.T, axis=1)
            scores = softmax(100 * image @ six.T, axis=1)
            pred = int(hierarchy(gate.argmax(1).astype(str), scores, list('012345'))[0])
            return {'predicted': label_names('6way')[pred], 'score': None, 'scores': {},
                    'route': 'Real' if pred == 0 else 'Fake → ' + label_names('6way')[pred]}
        if method in ('stata_image', 'clip_stata_image', 'clip_stata_image_title', 'em_image',
                      'hierarchy', 'hierarchy_em', 'hierarchy_majority', 'flat_stata', 'flat_majority', 'majority'):
            return self.transductive(method, backbone, backend, image, title, task, means, text)
        return self.fewshot(method, backbone, image, title, task)

    def transductive(self, method, backbone, backend, image, title, task, means, text):
        from .fakeddit_task1 import unit, predict
        from .adaptation import probability_features, em_dirichlet
        with self.asset(backbone, 'reference') as ref:
            context = ref['image']
            if context.ndim != 2 or not 3 <= len(context) <= 74 or context.shape[1] != image.shape[1]:
                raise ValueError('Cần 3–74 đặc trưng ảnh tham chiếu cùng backbone')
            features = unit(np.concatenate([context, image]))
            if method.startswith('clip_stata') or method == 'stata_image':
                if task != '2way':
                    raise ValueError('Đối chứng CLIP–StatA Task 1 chỉ dùng Real/Fake')
                titles = np.concatenate([ref['title'], title])
                t = backend.prototypes(self.profile['text_prompts']['2way'])
                if 'task1' not in self.prediction_cache:
                    self.prediction_cache['task1'] = predict(features, titles, means, t, self.task1)[0]
                results = self.prediction_cache['task1']
                prob = next(p[-1:] for m, _, p in results if m == method)
                return self.result(prob, task, seed=0, note='Demo với nhóm tham chiếu train cố định; khác protocol dev')
            if method == 'majority':
                if task != '2way':
                    raise ValueError('Majority ảnh–tiêu đề hiện dùng Real/Fake')
                from .methods import majority, replace_clip_vote, fuse
                binary = backend.prototypes(self.profile['image_prompts']['2way']['semantic_v1'])
                _, info = majority(features, binary, 0)
                clip = softmax(100 * features @ unit(binary).T, axis=1)
                titles = np.concatenate([ref['title'], title])
                text_means = backend.prototypes(self.profile['text_prompts']['2way'])
                text_prob = softmax(25 * titles @ text_means.T, axis=1)
                predicted, _ = replace_clip_vote(info['votes'], fuse(clip, text_prob, .75))
                return {'predicted': label_names(task)[int(predicted[-1])], 'score': None, 'scores': {},
                        'note': 'Majority ba phiếu trên nhóm tham chiếu train; chỉ phiếu CLIP dùng tiêu đề'}
            if method in ('hierarchy', 'hierarchy_em', 'hierarchy_majority', 'flat_stata', 'flat_majority'):
                if task != '6way':
                    raise ValueError('Nhánh hierarchy/phẳng này dùng 6 lớp')
                from .fakeddit_hierarchy import predict as infer, specification
                binary = backend.prototypes(self.profile['image_prompts']['2way']['semantic_v1'])
                six = backend.prototypes(self.profile['image_prompts']['6way']['legacy'])
                six_text = backend.prototypes(self.profile['text_prompts']['6way'])
                titles = np.concatenate([ref['title'], title])
                spec = {**specification(), 'grouping_seeds': [0]}
                if 'hierarchy' not in self.prediction_cache:
                    self.prediction_cache['hierarchy'] = infer(features, titles, binary, six, six_text, spec)[0]
                result = self.prediction_cache['hierarchy']
                # Hierarchy exposes labels; do not invent a calibrated posterior.
                target = {'hierarchy': 'hierarchy_stata_image', 'hierarchy_em': 'hierarchy_em_image',
                          'hierarchy_majority': 'hierarchy_majority_image',
                          'flat_stata': 'flat_stata_image', 'flat_majority': 'flat_majority_image'}[method]
                selected = [r for r in result if r[0] == target]
                if not selected:
                    raise ValueError('Không tìm thấy nhánh hierarchy đúng schema')
                pred = int(selected[0][-1][-1])
                return {'predicted': label_names('6way')[pred], 'score': None, 'scores': {},
                        'note': 'Hierarchy theo nhóm tham chiếu; chỉ nhãn, không có posterior hợp nhất'}
            query = probability_features(features @ means.T)
            labels, _, _ = em_dirichlet(query[None])
            pred = int(labels[0, -1])
            return {'predicted': label_names(task)[pred], 'score': None, 'scores': {},
                    'note': 'EM dùng nhóm tham chiếu train cố định; nhãn đã ghép cụm, không hiển thị confidence giả'}

    def fewshot(self, method, backbone, image, title, task):
        combined = np.concatenate([image, title], axis=1) / np.sqrt(2)
        rows = []
        if method in ('fewshot_flat', 'fewshot_cascade', 'fewshot_cascade_r87'):
            if task != '6way':
                raise ValueError('Head phẳng/cascade này dùng bộ nhãn 6 lớp')
            with self.asset(backbone, 'cascade') as heads:
                for policy in ('legacy_6way', 'balanced_binary'):
                    for seed in range(5):
                        key = f'{policy}_s{seed}'
                        if method == 'fewshot_flat':
                            p = self.head(heads, 'flat_' + key, combined, list(range(6)))
                            rows.append(self.result(p, task, policy=policy, seed=seed))
                        else:
                            gate = self.head(heads, 'gate_' + key, combined, [0, 1])
                            child = self.head(heads, 'child_' + key, combined, list(range(1, 6)))
                            threshold = None
                            if method.endswith('r87'):
                                threshold = self.assets[backbone]['thresholds'][key]
                            pred, joint = cascade_decision(gate, child, threshold)
                            rows.append(self.result(joint, task, predicted=pred, policy=policy, seed=seed,
                                                    gate={'Real': float(gate[0, 0]), 'Fake': float(gate[0, 1])},
                                                    threshold=threshold,
                                                    route='Real' if pred == 0 else 'Fake → ' + label_names(task)[pred]))
        else:
            if task != '2way':
                raise ValueError('Head ảnh/tiêu đề few-shot này dùng Real/Fake')
            modality = method.removeprefix('fewshot_')
            x = {'image': image, 'title': title, 'image_title': combined}[modality]
            with self.asset(backbone, 'binary') as heads:
                for policy in ('legacy_6way', 'balanced_binary'):
                    for seed in range(5):
                        key = f'{policy}_{modality}_s{seed}'
                        if key + '_coef' not in heads:
                            if policy == 'legacy_6way': key = f'{modality}_s{seed}'
                            else: continue
                        p = self.head(heads, key, x, [0, 1])
                        rows.append(self.result(p, task, fewshot=True, policy=policy, seed=seed))
        if not rows:
            raise ValueError('Không tìm thấy head tương ứng')
        # Display one transparent variant, retain every seed/policy prediction in details.
        chosen = next((r for r in rows if r['policy'] == 'balanced_binary' and r['seed'] == 0), rows[0])
        return {**chosen, 'variants': rows,
                'note': 'Dòng chính: policy B seed 0 nếu có, nếu không policy L seed 0; mọi biến thể trong chi tiết'}
