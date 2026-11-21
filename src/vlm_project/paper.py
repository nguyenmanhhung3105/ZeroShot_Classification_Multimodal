"""Chỉ dùng bộ sinh prompt tác giả; backbone chạy duy nhất MetaCLIP2."""
import argparse, ast, importlib, json, os, random, subprocess, sys
from pathlib import Path
from collections import OrderedDict
from contextlib import contextmanager
from types import SimpleNamespace
import numpy as np
ROOT = Path(__file__).resolve().parents[2]

def install_sources():
    """Clone đúng commit; không reset hay ghi đè repo có sẵn."""
    sources = json.loads((ROOT / 'configs/research_sources.json').read_text())
    (ROOT / 'third_party').mkdir(exist_ok=True)
    for name, spec in sources.items():
        directory = ROOT / 'third_party' / name
        if not directory.exists():
            subprocess.run(['git', 'clone', '--no-checkout', spec['url'], str(directory)], check=True)
            subprocess.run(['git', '-C', str(directory), 'checkout', '--detach', spec['commit']], check=True)
    return verify_sources()

def verify_sources():
    sources = json.loads((ROOT / 'configs/research_sources.json').read_text())
    for name, settings in sources.items():
        directory = ROOT / 'third_party' / name
        commit = subprocess.check_output(['git', '-C', str(directory), 'rev-parse', 'HEAD'], text=True).strip()
        status = subprocess.check_output(['git', '-C', str(directory), 'status', '--porcelain', '--untracked-files=all'], text=True)
        status = [line for line in status.splitlines() if not (line.startswith('?? ') and (line[3:] == '.DS_Store' or line[3:].endswith('/.DS_Store')))]
        if commit != settings['commit'] or status:
            raise ValueError(f'Repo {name} phải sạch và đúng commit trong sources.json')
    return sources

def load_functions(path, names, namespace):
    """Nạp thân hàm nguyên bản mà không thực thi phần tải dataset ở đầu module."""
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes} != set(names):
        raise ValueError(f'Thiếu hàm trong repo tác giả: {path}')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return SimpleNamespace(**{name: namespace[name] for name in names})

@contextmanager
def source_directory(path):
    """Repo gốc mở từ điển theo cwd; khôi phục cwd sau khi sinh prompt."""
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)

def author_api():
    import torch
    import torch.nn.functional as F
    import tqdm
    clip_root = (ROOT / 'third_party/OpenAI_CLIP').resolve()
    sys.path.insert(0, str(clip_root))
    try:
        clip = importlib.import_module('clip')
    finally:
        sys.path.remove(str(clip_root))
    if not Path(clip.__file__).resolve().is_relative_to(clip_root):
        raise ValueError('Module clip đã được import từ nguồn khác; chạy trong tiến trình Python mới')
    common = {'argparse': argparse, 'OrderedDict': OrderedDict, 'Any': object, 'List': list, 'np': np, 'json': json, 'torch': torch, 'F': F, 'tqdm': tqdm, 'clip': clip}
    helpers = ['load_json', 'wordify', 'make_descriptor_sentence', 'modify_descriptor', 'load_gpt_descriptions']
    waffle_ns = dict(common)
    waffle = load_functions(ROOT / 'third_party/WaffleCLIP/waffle_tools.py', helpers + ['compute_description_encodings', 'aggregate_similarity'], waffle_ns)
    dclip_ns = dict(common)
    dclip_helpers = load_functions(ROOT / 'third_party/DCLIP/loading_helpers.py', helpers, dclip_ns)
    dclip = load_functions(ROOT / 'third_party/DCLIP/load.py', ['compute_description_encodings', 'compute_label_encodings', 'aggregate_similarity'], dclip_ns)
    return SimpleNamespace(clip=clip, torch=torch, F=F, waffle=waffle, dclip=dclip, dclip_helpers=dclip_helpers, waffle_ns=waffle_ns, dclip_ns=dclip_ns)

def generate_prompts(api, method, names, descriptor_path, pairs, seed):
    """Dùng bộ sinh của tác giả, lưu/khôi phục trạng thái RNG ngoài hàm."""
    numpy_state, python_state = (np.random.get_state(), random.getstate())
    np.random.seed(seed)
    random.seed(seed)
    options = argparse.Namespace(dataset='custom', model_size='ViT-B/32', mode=method, classes_to_load=None, label_to_classname=list(names), n_classes=len(names), descriptor_fname=str(Path(descriptor_path).resolve()), waffle_count=pairs, label_before_text='A photo of a ', label_after_text='.', pre_descriptor_text='', descriptor_separator=', ', apply_descriptor_modification=True)
    try:
        if method in ('dclip', 'dclip_labels'):
            hparams = dict(descriptor_fname=options.descriptor_fname, category_name_inclusion='prepend', apply_descriptor_modification=True, before_text='', between_text=', ', after_text='', label_before_text='', label_after_text='')
            if method == 'dclip_labels':
                return ({name: [api.dclip_helpers.wordify(name)] for name in names}, hparams)
            prompts, _ = api.dclip_helpers.load_gpt_descriptions(hparams)
            return (prompts, hparams)
        with source_directory(ROOT / 'third_party/WaffleCLIP'):
            prompts = api.waffle.load_gpt_descriptions(options, mode=method)
        prompts = {name: [value] if isinstance(value, str) else value for name, value in prompts.items()}
        return (prompts, options)
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)
