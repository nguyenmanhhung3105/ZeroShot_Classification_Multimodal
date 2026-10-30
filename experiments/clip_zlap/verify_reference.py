"""Đối chiếu hàm ZLaP tác giả, chỉ thay backend GPU bằng CPU."""
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
from scipy import sparse
from scipy.sparse import linalg
from experiments.clip_zlap.core import predict, unit_vectors, PARAMETERS, REFERENCE_COMMIT
from experiments.clip_zlap.run import text_prototypes, project_path
from experiments.clip_tda.run import load_features


class Cpu(ast.NodeTransformer):
    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Attribute):
            if node.func.attr == 'get' and not node.args and not node.keywords:
                return node.func.value
            if isinstance(node.func.value, ast.Name) and node.func.value.id == 'cp' and node.func.attr == 'asnumpy':
                return node.args[0]
            if node.func.attr == 'cg':
                for keyword in node.keywords:
                    if keyword.arg == 'tol': keyword.arg = 'rtol'
        return node


def load_functions(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes} != set(names):
        raise ValueError('Thiếu hàm reference')
    tree = Cpu().visit(ast.Module(body=nodes, type_ignores=[])); ast.fix_missing_locations(tree)
    exec(compile(tree, str(path), 'exec'), namespace)


def reference_namespace(folder):
    if subprocess.check_output(['git', '-C', str(folder), 'rev-parse', 'HEAD'], text=True).strip() != REFERENCE_COMMIT:
        raise ValueError('Sai commit ZLaP')
    subprocess.run(['git', '-C', str(folder), 'diff', '--exit-code', 'HEAD', '--', 'zlap.py', 'utils.py'], check=True, capture_output=True)
    def search_cpu(reference, query, k):
        # Adapter cho FAISS inner-product; không dùng kNN của core địa phương.
        orders = []; values = []
        for start in range(0, len(query), 256):
            similarity = query[start:start+256] @ reference.T
            order = np.argsort(-similarity, axis=1, kind='stable')[:, :k]
            orders.append(order); values.append(np.take_along_axis(similarity, order, axis=1))
        return np.concatenate(orders), np.concatenate(values)
    ns = dict(cp=np, np=np, csr_matrix=sparse.csr_matrix, diags=sparse.diags,
              eye=sparse.eye, s_linalg=linalg, search_faiss=search_cpu)
    load_functions(folder/'utils.py', ['normalize_connection_graph', 'knn2laplacian', 'dfs_search'], ns)
    load_functions(folder/'zlap.py', ['combine_separate_knns', 'create_separate_graph', 'do_transductive_lp'], ns)
    return ns


def verify(folder, output, settings=None):
    ns = reference_namespace(Path(folder)); cases = []
    def compare(name, features, prototypes):
        x, t = unit_vectors(features), unit_vectors(prototypes)
        expected = ns['do_transductive_lp'](x, t, 5, 5., .3)
        actual, diagnostic = predict(features, prototypes, **PARAMETERS)
        np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=1e-5)
        np.testing.assert_array_equal(actual.argmax(1), expected.argmax(1))
        result = {'case': name, 'images': len(x), 'classes': len(t), 'predictions_identical': True,
                  'max_score_abs_error': float(abs(actual-expected).max()),
                  'all_cg_converged': diagnostic['all_cg_converged']}
        cases.append(result); print(result, flush=True)
    rng = np.random.default_rng(321)
    for classes in [2, 8]:
        compare(f'synthetic_{classes}', rng.uniform(.1, 1., (37, 16)), rng.uniform(.1, 1., (classes, 16)))
    if settings is not None:
        donor = load_features(project_path(settings['text_prototype_run']))
        for scope in ['train', 'dev']:
            source = load_features(project_path(settings[f'{scope}_run']))
            compare(scope+'_informativeness', source['features'], source['prototypes'])
            compare(scope+'_humanitarian', source['features'], text_prototypes(source, donor))
    report = {'status': 'passed', 'reference_commit': REFERENCE_COMMIT, 'cases': cases,
              'adaptations': ['Extract only listed pure functions; do not import dataset/model functions.',
                  'cp aliases NumPy, CuPy sparse/CG aliases SciPy, array.get/asnumpy removed, tol renamed rtol.',
                  'FAISS GPU search replaced by exact CPU dot products in blocks of 256 with stable index tie handling. Both paths use the same numeric backend; full-batch FP32 GEMM may resolve near-tied neighbors differently.',
                  'This checks algorithm/control flow with a CPU backend; not a CUDA numeric reproduction.'],
              'reference_sha256': {name: hashlib.sha256((Path(folder)/name).read_bytes()).hexdigest() for name in ['zlap.py', 'utils.py']},
              'local_core_sha256': hashlib.sha256((ROOT/'experiments/clip_zlap/core.py').read_bytes()).hexdigest()}
    Path(output).write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir', required=True); parser.add_argument('--output', required=True)
    parser.add_argument('--config')
    args = parser.parse_args()
    verify(args.reference_dir, args.output, json.loads(Path(args.config).read_text()) if args.config else None)
