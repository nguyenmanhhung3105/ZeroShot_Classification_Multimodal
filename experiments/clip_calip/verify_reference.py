"""Đối chiếu get_logits CALIP bằng AST, không chạy ImageNet/search/weights."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
import numpy as np
import torch
import torch.nn.functional as F
from experiments.clip_calip.core import score, REFERENCE_COMMIT


def functions(folder):
    folder = Path(folder)
    if subprocess.check_output(['git', '-C', str(folder), 'rev-parse', 'HEAD'], text=True).strip() != REFERENCE_COMMIT:
        raise ValueError('Sai commit CALIP')
    subprocess.run(['git', '-C', str(folder), 'diff', '--exit-code', 'HEAD', '--', 'run_imagenet.py'], check=True, capture_output=True)
    tree = ast.parse((folder/'run_imagenet.py').read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    method = next(n for n in main.body if isinstance(n, ast.FunctionDef) and n.name == 'get_logits')
    return method, hashlib.sha256((folder/'run_imagenet.py').read_bytes()).hexdigest()


def compare(method, g, p, t):
    ns = {'torch': torch, 'F': F, 'tqdm': lambda v: v,
          'img_spatial_feat': torch.tensor(p).permute(0, 2, 1),
          'img_global_feat': torch.tensor(g), 'feat_t': torch.tensor(t.T.copy())}
    exec(compile(ast.Module(body=[method], type_ignores=[]), '<official CALIP get_logits>', 'exec'), ns)
    l1, l2 = ns['get_logits'](); l0 = 100.*ns['img_global_feat']@ns['feat_t']
    expected = l0 + 1.12*l1 + .02*l2
    actual, terms, attention = score(g, p, t)
    for a, b in [(terms[:, 0], l0), (terms[:, 1], l1), (terms[:, 2], l2), (actual, expected)]:
        torch.testing.assert_close(a, b, atol=3e-5, rtol=2e-5)
    np.testing.assert_array_equal(actual.argmax(1), expected.argmax(1))
    return {'images': len(g), 'patches': p.shape[1], 'classes': len(t), 'embedding_dim': t.shape[1],
            'max_logit_abs_error': float(abs(actual-expected).max()), 'all_predictions_identical': True}


def verify(folder, output, features=None):
    method, source_hash = functions(folder); cases = []; rng = np.random.default_rng(131)
    previous = torch.get_num_threads(); torch.set_num_threads(1)
    try:
        for classes, dim in [(2, 16), (8, 16), (2, 1024), (8, 1024)]:
            norm = lambda v: F.normalize(torch.tensor(v, dtype=torch.float32), dim=-1).numpy()
            g = norm(rng.normal(size=(5, dim))); p = norm(rng.normal(size=(5, 9, dim))); t = norm(rng.normal(size=(classes, dim)))
            case = compare(method, g, p, t); case['scope'] = 'synthetic'; cases.append(case)
        if features:
            with np.load(features, allow_pickle=False) as z:
                for key in ['info_prototypes', 'human_prototypes']:
                    case = compare(method, z['global_features'], z['patches'], z[key])
                    case.update(scope='provided_verification_features', task=key); cases.append(case)
    finally:
        torch.set_num_threads(previous)
    result = {'status': 'passed', 'reference_commit': REFERENCE_COMMIT, 'source_sha256': source_hash,
              'core_sha256': hashlib.sha256((ROOT/'experiments/clip_calip/core.py').read_bytes()).hexdigest(),
              'cases': cases, 'scope': 'get_logits only; no labels/search/dataset/model or original benchmark reproduction'}
    Path(output).write_text(json.dumps(result, indent=2)+'\n'); return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--reference-dir', required=True)
    p.add_argument('--output', required=True); p.add_argument('--features'); a = p.parse_args()
    print(verify(a.reference_dir, a.output, a.features)['status'])
