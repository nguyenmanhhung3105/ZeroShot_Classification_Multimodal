"""Đối chiếu số với class EM_DIRICHLET gốc đã tải, không nạp dataset/model.

Chỉ thực thi AST của class thuật toán và các hàm matching/one-hot đã đọc;
không import môi trường benchmark, CLIP, dataset hoặc source utils toàn bộ.
"""
import argparse
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from experiments.clip_transductive.core import em_dirichlet, probability_features
from experiments.clip_transductive.run import load_source, REFERENCE, write_json


class QuietLogger:
    def __init__(self, *args): pass
    def info(self, *args): pass
    def del_logger(self): pass


class QuietProgress:
    def __init__(self, sequence): self.sequence = sequence
    def __iter__(self): return iter(self.sequence)
    def set_description(self, *args): pass


def compile_selected(path, names, namespace):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    body = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    if {node.name for node in body} != set(names): raise ValueError('Thiếu định nghĩa tham chiếu')
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), 'exec'), namespace)


def verify(reference_dir, source_dir=None, output=None, all_groupings=False):
    reference_dir = Path(reference_dir)
    commit = subprocess.check_output(['git', '-C', str(reference_dir), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != REFERENCE: raise ValueError('Sai commit tham chiếu')
    names = ['src/methods/zero_shot/em_dirichlet.py', 'src/utils.py']
    subprocess.run(['git', '-C', str(reference_dir), 'diff', '--exit-code', 'HEAD', '--', *names],
                   check=True, capture_output=True)
    hashes = {name: hashlib.sha256((reference_dir / name).read_bytes()).hexdigest() for name in names}
    namespace = dict(torch=torch, np=np, deepcopy=deepcopy, time=time, tqdm=QuietProgress,
                     Logger=QuietLogger, linear_sum_assignment=linear_sum_assignment)
    compile_selected(reference_dir / names[1], ['get_one_hot', 'compute_graph_matching', 'compute_basic_matching'], namespace)
    compile_selected(reference_dir / names[0], ['BASE', 'EM_DIRICHLET'], namespace)
    # Chỉ đổi precision để kiểm chứng các phương trình; không gradient hoặc trọng số.
    old_dtype = torch.get_default_dtype()
    old_threads = torch.get_num_threads()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    reports = []
    try:
        rng = np.random.default_rng(10)
        cases = {'synthetic': rng.dirichlet(np.ones(7), size=(2, 75))}
        if source_dir:
            source = load_source(source_dir)
            info, human = source['scores']
            gate = np.asarray(source['info_labels'])[info.argmax(1)] == 'informative'
            positive = [i for i, label in enumerate(source['labels']) if label != 'not_humanitarian']
            features = probability_features(human[:, positive])
            cases['saved_dev_scores'] = features[np.flatnonzero(gate)[:75]][None]
            if all_groupings:
                settings = json.loads((ROOT / 'experiments/clip_transductive/config.json').read_text())
                if settings['query_size'] != 75 or settings['grouping_seeds'] != [0, 1, 2]:
                    raise ValueError('Audit này khóa query_size=75 và grouping seeds 0,1,2')
                for seed in settings['grouping_seeds']:
                    order = np.random.default_rng(seed).permutation(np.flatnonzero(gate))
                    batches = [order[start:start + 75] for start in range(0, len(order), 75)]
                    for size in sorted({len(batch) for batch in batches}):
                        cases[f'dev_grouping_{seed}_size_{size}'] = features[np.stack([batch for batch in batches if len(batch) == size])]
        for name, query in cases.items():
            args = SimpleNamespace(iter=20, iter_mm=1000, num_classes_test=7, n_class=7,
                                   n_query=query.shape[1], T=30, graph_matching=True, use_softmax_feature=True)
            model = namespace['EM_DIRICHLET'](model=None, device='cpu', log_file=None, args=args)
            model.run_method(torch.from_numpy(query), torch.zeros(query.shape[:2], dtype=torch.long))
            expected_assignments = model.u.detach().numpy()
            # Matching tham chiếu gốc; y_q dummy chỉ dùng trong bước ghi accuracy.
            clusters = model.u.argmax(2)
            one_hot = namespace['get_one_hot'](clusters, 7)
            prototypes = (one_hot[..., None] * torch.from_numpy(query)[:, :, None, :]).sum(1)
            prototypes /= one_hot.sum(1).clamp(min=1e-15)[..., None]
            expected = namespace['compute_graph_matching'](clusters, prototypes, args).numpy()
            actual, log, assignments = em_dirichlet(query)
            np.testing.assert_allclose(assignments, expected_assignments, atol=1e-7, rtol=1e-6)
            np.testing.assert_array_equal(actual, expected)
            reports.append({'case': name, 'shape': list(query.shape), 'predictions_identical': True,
                            'max_assignment_abs_error': float(np.max(np.abs(assignments - expected_assignments))),
                            'objective_nonincreasing': True})
            print(f'[REFERENCE PASS] {name}: max_abs_error={reports[-1]["max_assignment_abs_error"]:.3g}', flush=True)
    finally:
        torch.set_default_dtype(old_dtype)
        torch.set_num_threads(old_threads)
    report = {'status': 'passed', 'reference_commit': commit, 'reference_source_sha256': hashes,
              'precision': 'float64 on both implementations; original explicit float casts retained',
              'cases': reports, 'reference_repo': 'https://github.com/SegoleneMartin/transductive-CLIP'}
    if output: write_json(Path(output), report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir', required=True)
    parser.add_argument('--source-run')
    parser.add_argument('--output')
    parser.add_argument('--all-groupings', action='store_true', help='Đối chiếu mọi nhóm dev của cả ba seed đã khóa')
    args = parser.parse_args()
    verify(args.reference_dir, args.source_run, args.output, args.all_groupings)
