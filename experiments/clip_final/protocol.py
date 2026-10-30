"""Khóa ứng viên và nguồn phát triển trước khi đánh giá test."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
import numpy as np
from experiments.waffle_lab.run import digest, file_hash, write_json

PLAN = 'configs/prompts/crisismmd_final_dn_unit.json'
RULES = {'coefficient': .25, 'prompt_seed': 0, 'grouping_seeds': [0, 1, 2],
         'query_size': 75, 'em_temperature': 30., 'em_iterations': 20, 'mm_iterations': 1000,
         'max_test_samples': 3000, 'selection': 'common_test_ids_hash_order_no_label_stratification',
         'methods': ['two_stage_original', 'majority', 'majority_dn_unit'],
         'primary_metric': 'humanitarian_macro_f1', 'image_mean_source': 'train_only_unlabeled_unit_features'}
CODE = ['experiments/clip_final/protocol.py', 'experiments/clip_final/evaluate.py',
        'experiments/clip_two_stage/run.py', 'experiments/clip_tda/run.py',
        'experiments/clip_zlap/dn_unit.py', 'experiments/clip_zlap/core.py',
        'experiments/clip_stata/core.py', 'experiments/clip_transductive/core.py',
        'experiments/clip_transductive/gate.py', 'experiments/clip_transductive/ensemble.py']


def load_plan(path, root=ROOT, verify_code=True):
    plan = json.loads(Path(path).read_text())
    protocol = plan['protocol']
    if digest(protocol) != plan['protocol_sha256'] or protocol['rules'] != RULES:
        raise ValueError('Protocol bị thay đổi hoặc khác cấu hình đã khóa')
    if protocol['model'] != 'metaclip2_vith14' or protocol['label_fit'] is not False:
        raise ValueError('Chỉ khóa MetaCLIP2 image-only không fit nhãn')
    if set(protocol['code_sha256']) != set(CODE):
        raise ValueError('Thiếu hash code trong protocol đã khóa')
    if verify_code:
        for name, expected in protocol['code_sha256'].items():
            if file_hash(Path(root)/name) != expected:
                raise ValueError(f'Code khác bản đã khóa: {name}; không sửa sau khi xem test')
    il, hl = protocol['info_labels'], protocol['human_labels']
    info = np.asarray(protocol['info_mean_prototypes'], dtype=np.float32)
    human = np.asarray(protocol['human_mean_prototypes'], dtype=np.float32)
    mean = np.asarray(protocol['image_mean'], dtype=np.float32)
    if (set(il) != {'informative', 'not_informative'} or len(il) != 2 or len(hl) != 8 or
            len(set(hl)) != 8 or 'not_humanitarian' not in hl or info.ndim != 2 or
            human.shape != (8, info.shape[1]) or info.shape[0] != 2 or mean.shape != (info.shape[1],) or
            not all(np.isfinite(x).all() for x in [info, human, mean]) or
            np.any(np.linalg.norm(info, axis=1) <= 1e-12) or np.any(np.linalg.norm(human, axis=1) <= 1e-12)):
        raise ValueError('Sai vector hoặc taxonomy trong frozen plan')
    return plan


def freeze(root=ROOT):
    """Chỉ đọc artifact train/dev, không đọc processed/raw hoặc tải model."""
    from experiments.clip_tda.run import load_features
    from experiments.clip_zlap.run import feature_bytes
    from experiments.clip_two_stage.analyze import read_artifact
    import io
    root = Path(root)
    target = root/PLAN
    if target.exists():
        return load_plan(target, root)
    train_dir = root/'results/clip_two_stage_features/crisismmd/humanitarian/samples_3000/metaclip2_vith14/humanitarian_471dc4a07814'
    dev_dir = root/'results/clip_two_stage_stata_heldout/crisismmd/humanitarian/samples_2219/metaclip2_vith14/humanitarian_ca2d96c17e89'
    train, dev = [load_features(p) for p in [train_dir, dev_dir]]
    if [train['split'], dev['split']] != ['train', 'dev']:
        raise ValueError('Không lấy mean/prototype từ test')
    with np.load(io.BytesIO(feature_bytes(train)), allow_pickle=False) as pack:
        human = pack['human_mean_prototypes'].copy()
    images = train['cohort'] + dev['cohort']
    exclusion = {'image_ids': sorted({r['id'] for r in images}),
                 'tweet_ids': sorted({r['id'].rsplit('_', 1)[0] for r in images}),
                 'image_sha256': sorted({r['image_sha256'] for r in images})}
    from experiments.clip_zlap.core import unit_vectors
    banks = {}
    for task, variant in [('informativeness', 'semantic'), ('humanitarian', 'semantic_waffle')]:
        plans = [json.loads(read_artifact(p, task+'/frozen_plan.json')) for p in [train_dir, dev_dir]]
        if plans[0]['prompt_banks'][variant] != plans[1]['prompt_banks'][variant]:
            raise ValueError('Train/dev khác bank prompt')
        banks[task] = plans[0]['prompt_banks'][variant]
    protocol = {'model': 'metaclip2_vith14', 'label_fit': False, 'rules': RULES,
                'info_labels': train['info_labels'], 'human_labels': train['labels'], 'prompt_banks': banks,
                'info_mean_prototypes': train['mean_prototypes'].tolist(), 'human_mean_prototypes': human.tolist(),
                'image_mean': unit_vectors(train['features']).mean(0).tolist(), 'exclusion': exclusion,
                'development_sources': [{'run': str(p.relative_to(root)), 'feature_sha256': s['feature_sha256'],
                    'cohort_sha256': file_hash(p/'humanitarian/cohort.json') if (p/'humanitarian/cohort.json').is_file()
                    else digest(s['cohort']), 'samples': len(s['rows'])} for p, s in [(train_dir, train), (dev_dir, dev)]],
                'code_sha256': {name: file_hash(root/name) for name in CODE},
                'limits': ['DN-unit on cached unit vectors, not original raw-vector DN benchmark.',
                           'Test restricted to intersection of both official task test splits.',
                           'No best-seed selection, parameter tuning, or new method selection after test.']}
    plan = {'protocol': protocol, 'protocol_sha256': digest(protocol), 'status': 'frozen_before_test'}
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x') as f: json.dump(plan, f, ensure_ascii=False, indent=2, allow_nan=False)
    load_plan(target, root)
    print(json.dumps({'frozen_plan': str(target), 'protocol_sha256': plan['protocol_sha256'],
                      'development_images': len(exclusion['image_ids']), 'test_read': False}, indent=2))
    return plan


if __name__ == '__main__': freeze()
