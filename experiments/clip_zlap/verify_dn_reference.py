"""Chỉ chạy bốn phép hiệu chỉnh DN từ AST; không import loader/model tác giả."""
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
from experiments.clip_zlap.core import unit_vectors
from experiments.clip_zlap.dn_unit import shifted_scores
from experiments.clip_zlap.run import project_path, text_prototypes
from experiments.clip_tda.run import load_features

REFERENCE_COMMIT = 'be4bdb6a158871a15ed60e4e4852b9de6db28da1'


def verify(folder, output, settings):
    folder = Path(folder); file = folder/'compute_zeroshot.py'
    if subprocess.check_output(['git', '-C', str(folder), 'rev-parse', 'HEAD'], text=True).strip() != REFERENCE_COMMIT:
        raise ValueError('Sai commit DN')
    subprocess.run(['git', '-C', str(folder), 'diff', '--exit-code', 'HEAD', '--', file.name], check=True, capture_output=True)
    tree = ast.parse(file.read_text())
    coefficient = [n for n in tree.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'LAMBDA']
    if len(coefficient) != 1 or ast.literal_eval(coefficient[0].value) != .25:
        raise ValueError('Hệ số reference thay đổi')
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'zeroshot_prediction')
    operations = sorted([n for n in ast.walk(function) if isinstance(n, ast.Assign) and
                         isinstance(n.targets[0], ast.Name) and n.targets[0].id in ['_image_embeddings', '_caption_embeddings'] and
                         (isinstance(n.value, ast.BinOp) or isinstance(n.value, ast.Call))], key=lambda n: n.lineno)
    if len(operations) != 4: raise ValueError('Reference không còn đúng bốn phép DN đã đọc')
    formula = compile(ast.fix_missing_locations(ast.Module(body=operations, type_ignores=[])), str(file), 'exec')
    donor = load_features(project_path(settings['text_prototype_run']))
    train = load_features(project_path(settings['train_run']))
    if train['split'] != 'train': raise ValueError('Không lấy mean từ dev/test')
    train_vectors = unit_vectors(train['features']); mean = train_vectors.mean(0)
    cases = []
    for scope in ['train', 'dev']:
        source = train if scope == 'train' else load_features(project_path(settings['dev_run']))
        for task, prototypes in [('informativeness', source['prototypes']), ('humanitarian', text_prototypes(source, donor))]:
            x, t = unit_vectors(source['features']), unit_vectors(prototypes)
            namespace = dict(torch=torch, F=F, LAMBDA=.25, image_embeddings=torch.tensor(x),
                             caption_embeddings=torch.tensor(t), val_embeddings=torch.tensor(train_vectors))
            exec(formula, namespace)
            expected = (namespace['_image_embeddings'] @ namespace['_caption_embeddings'].T).numpy()
            actual = shifted_scores(source['features'], prototypes, mean)
            np.testing.assert_allclose(actual, expected, atol=5e-7, rtol=1e-5)
            np.testing.assert_array_equal(actual.argmax(1), expected.argmax(1))
            case = {'split': scope, 'task': task, 'samples': len(x), 'predictions_identical': True,
                    'max_cosine_abs_error': float(abs(actual-expected).max())}
            print(case, flush=True); cases.append(case)
    report = {'status': 'passed', 'reference_commit': REFERENCE_COMMIT,
              'reference_sha256': hashlib.sha256(file.read_bytes()).hexdigest(),
              'local_source_sha256': hashlib.sha256((ROOT/'experiments/clip_zlap/dn_unit.py').read_bytes()).hexdigest(),
              'cases': cases, 'limitation': 'Formula equivalence for chosen unit-vector inputs; original DN uses raw pre-normalization encoder outputs, so this is not full benchmark reproduction.'}
    Path(output).write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir', required=True); parser.add_argument('--output', required=True)
    parser.add_argument('--config', default=str(ROOT/'experiments/clip_zlap/config_dn_unit.json'))
    args = parser.parse_args()
    verify(args.reference_dir, args.output, json.loads(Path(args.config).read_text()))
