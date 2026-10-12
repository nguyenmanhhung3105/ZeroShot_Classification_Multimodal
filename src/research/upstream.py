"""Gọi trực tiếp các hàm của repo tác giả, giữ checkpoint và tokenizer OpenAI CLIP.

Chỉ nạp định nghĩa hàm từ code gốc để thay loader dataset chạy ngay khi import.
Thân hàm sinh prompt, mã hóa mô tả và tổng hợp similarity được giữ nguyên.
"""
import argparse
import ast
from collections import OrderedDict
from contextlib import contextmanager
import importlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def verify_sources():
    sources = json.loads((ROOT / "configs/research/sources.json").read_text())
    for name, settings in sources.items():
        directory = ROOT / "third_party" / name
        commit = subprocess.check_output(["git", "-C", str(directory), "rev-parse", "HEAD"], text=True).strip()
        status = subprocess.check_output(["git", "-C", str(directory), "status", "--porcelain"], text=True).strip()
        if commit != settings["commit"] or status:
            raise ValueError(f"Repo {name} phải sạch và đúng commit trong sources.json")
    return sources


def load_functions(path, names, namespace):
    """Nạp thân hàm nguyên bản mà không thực thi phần tải dataset ở đầu module."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes} != set(names):
        raise ValueError(f"Thiếu hàm trong repo tác giả: {path}")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
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
    clip_root = ROOT / "third_party/OpenAI_CLIP"
    sys.path.insert(0, str(clip_root))
    try:
        clip = importlib.import_module("clip")
    finally:
        sys.path.remove(str(clip_root))
    if not Path(clip.__file__).resolve().is_relative_to(clip_root):
        raise ValueError("Module clip đã được import từ nguồn khác; chạy trong tiến trình Python mới")
    common = {"argparse": argparse, "OrderedDict": OrderedDict, "Any": object,
              "List": list, "np": np, "json": json, "torch": torch, "F": F,
              "tqdm": tqdm, "clip": clip}
    helpers = ["load_json", "wordify", "make_descriptor_sentence", "modify_descriptor", "load_gpt_descriptions"]
    waffle_ns = dict(common)
    waffle = load_functions(ROOT / "third_party/WaffleCLIP/waffle_tools.py",
                            helpers + ["compute_description_encodings", "aggregate_similarity"], waffle_ns)
    dclip_ns = dict(common)
    dclip_helpers = load_functions(ROOT / "third_party/DCLIP/loading_helpers.py", helpers, dclip_ns)
    dclip = load_functions(ROOT / "third_party/DCLIP/load.py",
                          ["compute_description_encodings", "compute_label_encodings", "aggregate_similarity"], dclip_ns)
    return SimpleNamespace(clip=clip, torch=torch, F=F, waffle=waffle, dclip=dclip,
                           dclip_helpers=dclip_helpers, waffle_ns=waffle_ns, dclip_ns=dclip_ns)


def generate_prompts(api, method, names, descriptor_path, pairs, seed):
    """Dùng bộ sinh của tác giả, lưu/khôi phục trạng thái RNG ngoài hàm."""
    numpy_state, python_state = np.random.get_state(), random.getstate()
    np.random.seed(seed)
    random.seed(seed)
    options = argparse.Namespace(dataset="custom", model_size="ViT-B/32", mode=method,
        classes_to_load=None, label_to_classname=list(names), n_classes=len(names),
        descriptor_fname=str(Path(descriptor_path).resolve()), waffle_count=pairs,
        label_before_text="A photo of a ", label_after_text=".", pre_descriptor_text="",
        descriptor_separator=", ", apply_descriptor_modification=True)
    try:
        if method in ("dclip", "dclip_labels"):
            hparams = dict(descriptor_fname=options.descriptor_fname, category_name_inclusion="prepend",
                           apply_descriptor_modification=True, before_text="", between_text=", ", after_text="",
                           label_before_text="", label_after_text="")
            if method == "dclip_labels":
                return {name: [api.dclip_helpers.wordify(name)] for name in names}, hparams
            prompts, _ = api.dclip_helpers.load_gpt_descriptions(hparams)
            return prompts, hparams
        with source_directory(ROOT / "third_party/WaffleCLIP"):
            prompts = api.waffle.load_gpt_descriptions(options, mode=method)
        prompts = {name: [value] if isinstance(value, str) else value for name, value in prompts.items()}
        return prompts, options
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)


def description_encodings(api, model, method, prompts, options, device):
    """Hàm tác giả mã hóa đúng prompt đã cố định; không sinh lại từ dữ liệu đánh giá."""
    with api.torch.no_grad():
        if method in ("dclip", "dclip_labels"):
            api.dclip_ns.update(gpt_descriptions=prompts, hparams={**options, "device": device})
            if method == "dclip_labels":
                api.dclip_ns["label_to_classname"] = list(prompts)
                values = api.dclip.compute_label_encodings(model)
                return OrderedDict((name, value[None]) for name, value in zip(prompts, values))
            return api.dclip.compute_description_encodings(model)
        # compute_description_encodings gọi load_gpt_descriptions trong namespace gốc.
        # Đưa đúng prompt đã cố định vào lượt mã hóa để không thay seed sau khi khóa plan.
        original = api.waffle_ns["load_gpt_descriptions"]
        api.waffle_ns["load_gpt_descriptions"] = lambda *args, **kwargs: prompts
        options.device = device
        try:
            return api.waffle.compute_description_encodings(options, model, mode=method)
        finally:
            api.waffle_ns["load_gpt_descriptions"] = original


def image_scores(api, image_embeddings, descriptions, method):
    """Cùng phép tính ở main tác giả; gọi trực tiếp hàm tổng hợp gốc."""
    aggregate = api.dclip.aggregate_similarity if method.startswith("dclip") else api.waffle.aggregate_similarity
    return api.torch.stack([aggregate(image_embeddings @ values.T, "mean")
                            for values in descriptions.values()], dim=1)


class OpenAIClipBackend:
    """Model/preprocess/tokenizer của repo OpenAI, tương thích báo cáo và chunking."""
    def __init__(self, api, backbone, device):
        self.api, self.device, self.backbone = api, device, backbone
        self.model_name = "openai_clip_" + backbone.lower().replace("/", "_").replace("-", "_")
        self.model, self.preprocess = api.clip.load(backbone, device=device, jit=False)
        if device != "cuda":
            self.model.float()
        self.model.eval()
        self.model.requires_grad_(False)
        self.tokenizer = api.clip.simple_tokenizer.SimpleTokenizer()
        self.logit_scale = float(self.model.logit_scale.exp().item())
        self.logit_bias = 0.0

    def encode_images(self, images):
        with self.api.torch.no_grad():
            batch = self.api.torch.stack([self.preprocess(image) for image in images]).to(self.device)
            return self.api.F.normalize(self.model.encode_image(batch))

    def encode_texts(self, texts):
        with self.api.torch.no_grad():
            tokens = self.api.clip.tokenize(texts, context_length=self.model.context_length, truncate=True).to(self.device)
            return self.api.F.normalize(self.model.encode_text(tokens))

    def unload(self):
        del self.model
        if self.device == "cuda":
            self.api.torch.cuda.empty_cache()
        elif self.device == "mps":
            self.api.torch.mps.empty_cache()


def choose_device(torch, choice):
    if choice == "auto":
        return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if choice not in ("cpu", "cuda", "mps"):
        raise ValueError("device phải là auto, cpu, cuda hoặc mps")
    if choice == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA không khả dụng")
    if choice == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS không khả dụng")
    return choice
