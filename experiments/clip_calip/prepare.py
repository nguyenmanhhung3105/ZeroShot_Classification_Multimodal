"""Đóng băng prompt/prototype và đối chứng từ result train/dev, không đọc ảnh."""
import io
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
import numpy as np
from experiments.clip_tda.run import load_features
from experiments.clip_zlap.run import project_path, feature_bytes, validate_majority, majority_gates
from experiments.clip_dmn.run import load_companions
from experiments.clip_two_stage.analyze import read_artifact
from experiments.waffle_lab.run import digest, write_json

ASSET = 'configs/prompts/crisismmd_calip_reference.json'


def load_reference(path):
    ref = json.loads(Path(path).read_text()); payload = ref['payload']
    if ref['sha256'] != digest(payload) or payload['model'] != 'metaclip2_vith14' or payload['prompt_seed'] != 0:
        raise ValueError('Reference CALIP sai fingerprint/model/seed')
    if set(payload['cohorts']) != {'train', 'dev'}:
        raise ValueError('Reference CALIP chỉ train/dev')
    return ref


def prepare(path=None):
    path = Path(path) if path else ROOT/ASSET
    if path.exists(): return load_reference(path)
    settings = json.loads((ROOT/'experiments/clip_dpe/config.json').read_text())
    train = load_features(project_path(settings['train_run']))
    with np.load(io.BytesIO(feature_bytes(train)), allow_pickle=False) as pack:
        human = pack['human_mean_prototypes'].copy()
    banks = {task: json.loads(read_artifact(Path(train['directory']), task+'/frozen_plan.json'))['prompt_banks'][variant]
             for task, variant in [('informativeness', 'semantic'), ('humanitarian', 'semantic_waffle')]}
    md = project_path(settings['majority_run']); mm = validate_majority(md)
    payload = {'model': 'metaclip2_vith14', 'prompt_seed': 0, 'banks': banks,
               'info_labels': train['info_labels'], 'human_labels': train['labels'],
               'info_mean_prototypes': train['mean_prototypes'].tolist(), 'human_mean_prototypes': human.tolist(),
               'cohorts': {}, 'label_fit': False, 'scope': 'cached controls and prompts, no supervised support'}
    for scope in ['train', 'dev']:
        source = load_features(project_path(settings[scope+'_run']))
        if source['split'] != scope: raise ValueError('Sai split reference')
        if source['labels'] != payload['human_labels'] or source['info_labels'] != payload['info_labels']:
            raise ValueError('Khác taxonomy')
        for task, variant in [('informativeness', 'semantic'), ('humanitarian', 'semantic_waffle')]:
            plan = json.loads(read_artifact(Path(source['directory']), task+'/frozen_plan.json'))
            if plan['prompt_banks'][variant] != banks[task]: raise ValueError('Khác prompt bank')
        np.testing.assert_allclose(source['features']@human.T, source['scores'][1], atol=2e-5, rtol=1e-4)
        dn, _ = load_companions(source, scope, settings)
        gates = majority_gates(source, scope, md, mm)
        payload['cohorts'][scope] = {'samples': len(source['rows']), 'feature_sha256': source['feature_sha256'],
            'ids': [r['id'] for r in source['rows']], 'image_sha256': [r['image_sha256'] for r in source['cohort']],
            'human_truth': [r['y_true'] for r in source['rows']], 'binary_truth': source['binary_truth'],
            'info_scores': source['scores'][0].tolist(), 'human_scores': source['scores'][1].tolist(),
            'dn_leaf': dn.tolist(), 'majority_gates': {str(k): g.tolist() for k, g in gates.items()},
            'source_run': str(Path(source['directory']).relative_to(ROOT))}
    ref = {'payload': payload, 'sha256': digest(payload)}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f: json.dump(ref, f, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    load_reference(path); print('[PREPARED]', path, 'train/dev only, no image/model access'); return ref


if __name__ == '__main__': prepare()
