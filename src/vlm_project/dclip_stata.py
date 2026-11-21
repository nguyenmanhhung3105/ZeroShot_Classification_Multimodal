"""DTD đủ split: DCLIP prototype → StatA, dùng lại runner đối chứng prompt."""
import yaml
from .backbone import Backbone
from .io import ROOT
from .waffle_stata import run as run_prototype

SPEC_FILE = 'configs/dclip_stata.yaml'
DATASETS = ['dtd']


def specification():
    spec = yaml.safe_load((ROOT / SPEC_FILE).read_text())
    if spec != {'version': 1, 'samples': 1880, 'selection_seed': 0, 'prompt_seeds': [0],
                'grouping_seeds': [0, 1, 2], 'waffle_pairs': 15,
                'adaptation_batch_size': 75, 'cpu_threads': 1, 'timeout_seconds': 7200}:
        raise ValueError('DCLIP–StatA giữ DTD 1880 ảnh và grouping seed 0/1/2')
    return spec


def run(settings, name, root, limit=None, backend_factory=Backbone):
    if name not in DATASETS:
        raise ValueError('Job DCLIP–StatA hiện chỉ hỗ trợ DTD')
    return run_prototype(settings, name, root, limit, backend_factory, prompt_method='dclip')
