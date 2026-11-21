"""Đối chứng ngân sách support binary, giữ 384 mẫu và classifier cố định."""
import numpy as np
import yaml
from .backbone import Backbone
from .io import ROOT
from .fakeddit_fewshot_task1 import specification as base_specification, run as base_run

SPEC_FILE = 'configs/fakeddit_fewshot_support_min5.yaml'
POLICIES = ['legacy_6way', 'balanced_binary']


def specification():
    spec = yaml.safe_load((ROOT / SPEC_FILE).read_text())
    expected = {**base_specification(), 'support_policies': POLICIES,
                'binary_samples_per_class': 192, 'fake_allocation': 'equal_rotating_remainder'}
    if spec != expected:
        raise ValueError('Giữ tổng 384 support: legacy 64/320 và balanced 192/192; cùng 5 seed/C=1')
    return spec


def quotas(spec, seed):
    """192 fake / 5 subtype: 38 mỗi lớp, hai vị trí nhận thêm 1, xoay theo seed."""
    count = spec['binary_samples_per_class']
    base, extra = divmod(count, 5)
    result = np.asarray([count, *([base] * 5)], dtype=np.int64)
    for offset in range(extra):
        result[1 + (seed + offset) % 5] += 1
    return result


def select_balanced_support(y6, spec, seed):
    y6 = np.asarray(y6)
    if y6.ndim != 1 or not np.isin(y6, range(6)).all():
        raise ValueError('Support cần nhãn train 6way hợp lệ')
    rng = np.random.default_rng(seed)
    selected = []
    for label, count in enumerate(quotas(spec, seed)):
        indices = np.flatnonzero(y6 == label)
        if len(indices) < count:
            raise ValueError(f'Train class {label}: {len(indices)} < quota {count}')
        # Cùng permutation với sampler legacy; chỉ thay số lượng lấy từng lớp.
        selected.extend(rng.permutation(indices)[:count])
    return np.asarray(selected, dtype=np.int64)


def run(settings, train_root, dev_root, smoke=False, backend_factory=Backbone):
    return base_run(settings, train_root, dev_root, smoke, backend_factory, balance_ablation=True)
