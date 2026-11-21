"""Waffle→StatA đủ DTD, giữ riêng job/config pilot 256 ảnh."""
import yaml
from .backbone import Backbone
from .io import ROOT
from .waffle_stata import run as run_prototype, specification as pilot_specification

SPEC_FILE = 'configs/waffle_stata_full.yaml'
DATASETS = ['dtd']


def specification():
    spec = yaml.safe_load((ROOT / SPEC_FILE).read_text())
    if spec != {**pilot_specification(), 'samples': 1880}:
        raise ValueError('Waffle–StatA full giữ 1880 ảnh, 7 prompt seed và 3 grouping seed')
    return spec


def run(settings, name, root, limit=None, backend_factory=Backbone):
    if name not in DATASETS:
        raise ValueError('Waffle–StatA full hiện chỉ hỗ trợ DTD')
    return run_prototype(settings, name, root, limit, backend_factory, full_dtd=True)
