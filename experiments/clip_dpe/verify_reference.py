"""Chạy vòng DPE gốc qua AST với encoder giả trả cached feature; không dataset.

Chuyển CUDA/half sang CPU/FP32 và tắt logging/accuracy; giữ optimizer và flow.
InfoNCE thay bằng công thức normalize-dot / .01 và cross entropy mặc định.
"""
import argparse
import ast
from contextlib import nullcontext
import hashlib
import json
import math
import operator
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from experiments.clip_dpe.core import predict_stream, REFERENCE_COMMIT


class Cpu(ast.NodeTransformer):
    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'cuda':
            return self.visit(node.func.value)
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'half':
            node.func.attr = 'float'
        if ast.unparse(node.func) == 'torch.cuda.amp.autocast':
            return ast.copy_location(ast.Call(func=ast.Name(id='nullcontext', ctx=ast.Load()), args=[], keywords=[]), node)
        return self.generic_visit(node)


def reference(folder):
    folder = Path(folder)
    pin = subprocess.check_output(['git', '-C', str(folder), 'rev-parse', 'HEAD'], text=True).strip()
    if pin != REFERENCE_COMMIT:
        raise ValueError('Sai commit DPE')
    subprocess.run(['git', '-C', str(folder), 'diff', '--exit-code', 'HEAD', '--',
                    'main_dpe.py', 'utils.py', 'configs/imagenet.yaml'], check=True, capture_output=True)
    picked = []
    groups = {'main_dpe.py': {'update_cache', 'cache_key_value', 'compute_cache_logits',
                             'TextResidue', 'PositiveCacheResidue', 'run_test_dpe'},
              'utils.py': {'get_clip_logits', 'get_entropy', 'avg_entropy', 'softmax_entropy'}}
    for name, wanted in groups.items():
        parsed = ast.parse((folder/name).read_text())
        nodes = [v for v in parsed.body if isinstance(v, (ast.FunctionDef, ast.ClassDef)) and v.name in wanted]
        if {v.name for v in nodes} != wanted:
            raise ValueError('Thiếu operator DPE')
        if name == 'main_dpe.py':
            loop = next(v for v in nodes if v.name == 'run_test_dpe')
            # Chỉ thay return cuối để quan sát trạng thái; không đổi bước suy luận.
            loop.body[-1] = ast.parse('return pos_cache, clip_weights_global, num_avg').body[0]
        picked.extend(nodes)
    def infonce(a, b):
        # Default InfoNCE(query, positive_key), negative_keys=None, temperature=.01.
        a, b = F.normalize(a, dim=-1), F.normalize(b, dim=-1)
        return F.cross_entropy((a @ b.T)/.01, torch.arange(len(a)))
    ns = {'torch': torch, 'F': F, 'nn': nn, 'np': np, 'math': math, 'operator': operator,
          'nullcontext': nullcontext, 'InfoNCELoss': infonce,
          'tqdm': lambda values, **kwargs: values, 'print': lambda *a, **kw: None,
          'wandb': type('Log', (), {'log': staticmethod(lambda *a, **kw: None)})}
    module = ast.fix_missing_locations(Cpu().visit(ast.Module(body=picked, type_ignores=[])))
    exec(compile(module, '<DPE official AST adapted CPU>', 'exec'), ns)
    return ns, {name: hashlib.sha256((folder/name).read_bytes()).hexdigest() for name in groups}


def compare(ns, x, t, order):
    previous = torch.get_num_threads(); torch.set_num_threads(1)
    try:
        observed = []
        ns['cls_acc'] = lambda logits, target: (observed.append(logits.detach().clone()), 0.)[1]
        model = type('FeatureEncoder', (), {'encode_image': staticmethod(lambda data: data.reshape(-1, x.shape[1]))})()
        loader = [(torch.tensor(x[index]).reshape(1, 1, 1, -1), torch.zeros(1, dtype=torch.long)) for index in order]
        cache, global_text, count = ns['run_test_dpe'](
            {'enabled': True, 'shot_capacity': 3, 'alpha': 6., 'beta': 5.},
            {'text': .0006, 'image': .0006, 'align': .5}, loader, model, torch.tensor(t.T.copy()), 'cached-single-view')
        expected = np.empty((len(x), len(t)), dtype=np.float32)
        expected[order] = torch.cat(observed).numpy()
        actual, trace, diag = predict_stream(x, t, order)
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=3e-5)
        np.testing.assert_array_equal(actual.argmax(1), expected.argmax(1))
        np.testing.assert_allclose(np.array(diag['global_text']), global_text.T.detach().numpy(), rtol=2e-5, atol=2e-6)
        assert count == diag['text_updates']
        assert diag['memory_counts'] == [len(cache.get(c, [])) for c in range(len(t))]
        normalized = F.normalize(torch.tensor(x), dim=-1).numpy()
        for c in range(len(t)):
            for position, index in enumerate(diag['memory_indices'][c]):
                np.testing.assert_array_equal(normalized[index], cache[c][position][0].numpy()[0])
                np.testing.assert_allclose(trace['cache_entropy'][index], cache[c][position][1], rtol=2e-5, atol=1e-6)
        return {'images': len(x), 'classes': len(t), 'max_logit_abs_error': float(abs(actual-expected).max()),
                'all_predictions_identical': True, 'final_memory_identical': True, 'text_updates_identical': True}
    finally:
        torch.set_num_threads(previous)


def verify(folder, output, settings=None):
    ns, hashes = reference(folder); cases = []
    from experiments.clip_zlap.core import unit_vectors
    rng = np.random.default_rng(92)
    for c in [2, 8]:
        x = unit_vectors(rng.normal(size=(43, 16))); t = unit_vectors(rng.normal(size=(c, 16)))
        case = compare(ns, x, t, rng.permutation(len(x))); case['scope'] = 'synthetic'; cases.append(case)
    if settings:
        from experiments.clip_tda.run import load_features
        from experiments.clip_zlap.run import project_path, text_prototypes
        donor = load_features(project_path(settings['text_prototype_run']))
        for scope in ['train', 'dev']:
            source = load_features(project_path(settings[scope+'_run']))
            for task, t in [('informativeness', source['prototypes']), ('humanitarian', text_prototypes(source, donor))]:
                for seed in settings['seeds']:
                    case = compare(ns, source['features'], t, np.random.default_rng(seed).permutation(len(source['features'])))
                    case.update(scope=scope, task=task, seed=seed); cases.append(case); print(case, flush=True)
    report = {'status': 'passed', 'reference_commit': REFERENCE_COMMIT, 'source_sha256': hashes,
              'local_core_sha256': hashlib.sha256((ROOT/'experiments/clip_dpe/core.py').read_bytes()).hexdigest(),
              'cases': cases, 'scope': 'Official single-view branch, CPU FP32; no datasets/model weights, no actual GT.',
              'adaptations': ['CUDA/half/autocast removed; logging and accuracy replaced by output observer.',
                              'InfoNCE default expressed as normalized dot-product / .01 and cross entropy.',
                              'Cached features through identity encoder; no full multi-view benchmark reproduction.']}
    Path(output).write_text(json.dumps(report, indent=2)+'\n'); return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--reference-dir', required=True)
    p.add_argument('--output', required=True); p.add_argument('--config')
    a = p.parse_args(); verify(a.reference_dir, a.output, json.loads(Path(a.config).read_text()) if a.config else None)
