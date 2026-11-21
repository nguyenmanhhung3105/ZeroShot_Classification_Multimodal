"""Full benchmark: Waffle + semantic descriptor prototype → StatA."""
import yaml
from .backbone import Backbone
from .io import ROOT
from .waffle_stata_full import specification as full_specification
from .waffle_stata import run as run_prototype

SPEC_FILE = 'configs/waffle_semantic_stata.yaml'
SPEC_FILES = {'dtd': SPEC_FILE, 'eurosat': 'configs/waffle_semantic_stata_eurosat.yaml'}
DATASETS = list(SPEC_FILES)


def specification(name='dtd'):
    if name not in SPEC_FILES:
        raise ValueError('Semantic chỉ hỗ trợ DTD/EuroSAT')
    spec = yaml.safe_load((ROOT / SPEC_FILES[name]).read_text())
    expected = {**full_specification(), 'semantic_weight': .5}
    if name == 'eurosat':
        expected.update(samples=27000, timeout_seconds=14400)
    if spec != expected:
        raise ValueError('Giữ full split, 7x3 seed và semantic weight=0.5 đã định trước')
    return spec


def run(settings, name, root, limit=None, backend_factory=Backbone):
    if name not in DATASETS:
        raise ValueError('Waffle–semantic–StatA hiện chỉ hỗ trợ DTD/EuroSAT')
    return run_prototype(settings, name, root, limit, backend_factory, full_dtd=True, semantic=True)
