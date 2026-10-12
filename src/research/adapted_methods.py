"""
Bản thử WaffleCLIP/DCLIP trên nhánh ảnh với backbone hiện có.

Dùng từ điển gốc, cặp từ và ký tự ngẫu nhiên chia sẻ giữa các lớp, trung bình
cosine; dùng bộ sinh số cục bộ và bảng chữ cái cố định. Kết quả khác bản gốc
về seed và cách sinh ký tự. Không đọc dữ liệu hoặc tải model tại module này.
"""
import hashlib
import json
from pathlib import Path
import pickle
import string

import numpy as np

from inference import build_class_embeddings

ROOT = Path(__file__).resolve().parents[2]


class _StringsOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Từ điển gốc chứa np.str_; chỉ cho phép dựng giá trị Unicode,
        # không cho phép dựng đối tượng có mã thực thi.
        if module == "numpy" and name == "dtype":
            return np.dtype
        if module in ("numpy.core.multiarray", "numpy._core.multiarray") and name == "scalar":
            return _unicode_scalar
        raise pickle.UnpicklingError("Executable pickle objects are forbidden")

    def persistent_load(self, pid):
        raise pickle.UnpicklingError("Persistent pickle objects are forbidden")


def _unicode_scalar(dtype, data):
    if dtype.kind != "U" or not isinstance(data, bytes) or len(data) % 4:
        raise pickle.UnpicklingError("Only Unicode scalar data allowed")
    if dtype.itemsize == 0 and not data:
        return ""
    if dtype.itemsize == 0:
        dtype = np.dtype(f"{dtype.byteorder}U{len(data) // 4}")
    if not data or len(data) != dtype.itemsize:
        raise pickle.UnpicklingError("Invalid Unicode scalar length")
    return str(np.frombuffer(data, dtype=dtype, count=1)[0])


def official_words():
    path = ROOT / "third_party/WaffleCLIP/word_list.pkl"
    with path.open("rb") as stream:
        words = _StringsOnlyUnpickler(stream).load()
    if not isinstance(words, list) or not words or any(not isinstance(w, str) for w in words):
        raise ValueError("Expected official nonempty list[str] vocabulary")
    return words, hashlib.sha256(path.read_bytes()).hexdigest()


def descriptor_sentence(value):
    # Ghép mô tả theo mẫu câu của WaffleCLIP và DCLIP gốc.
    if value.startswith(("a", "an", "used")):
        return f"which is {value}"
    if value.startswith(("has", "often", "typically", "may", "can")):
        return f"which {value}"
    return f"which has {value}"


def render_descriptor(name, value):
    return f"A photo of a {name}, {descriptor_sentence(value)}."


def waffle_prompts(names, words, pairs=15, seed=0):
    if type(pairs) is not int or not 1 <= pairs <= 256:
        raise ValueError("waffle_pairs must be an integer in [1, 256]")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if not names or not words or any(not isinstance(w, str) for w in words):
        raise ValueError("Nonempty names and vocabulary required")
    values = [str(name).replace("_", " ").strip() for name in names.values()]
    if any(not value for value in values):
        raise ValueError("Empty class name")
    n_words = max(int(np.round(np.mean([len(x.split()) for x in values]))), 1)
    word_length = max(int(np.round(np.mean([np.mean([len(w) for w in x.split()]) for x in values]))), 1)
    n_groups = int(np.round(np.mean([x.count(" ") for x in values]))) + 1
    n_chars = int(np.ceil(np.mean([max(map(len, x.split())) for x in values])))
    n_chars += n_groups - n_chars % n_groups
    rng = np.random.RandomState(seed)  # không thay bộ sinh số ngẫu nhiên toàn cục
    vocabulary = [w[:word_length] for w in words]
    alphabet = list(string.ascii_lowercase)
    suffixes = []
    for _ in range(pairs):
        suffixes.append(" ".join(rng.choice(vocabulary, n_words)))
        suffixes.append(", ".join("".join(rng.choice(alphabet, n_chars // n_groups)) for _ in range(n_groups)))
    return {key: [render_descriptor(name, suffix) for suffix in suffixes]
            for key, name in zip(names, values)}


def load_descriptors(path, names):
    """Đọc mô tả cố định có nguồn; không suy mô tả thiếu từ dữ liệu đánh giá."""
    raw = Path(path).read_bytes()
    spec = json.loads(raw)
    if not isinstance(spec.get("provenance"), str) or not spec["provenance"].strip():
        raise ValueError("Descriptor file requires nonempty provenance")
    values = spec.get("descriptors", {})
    if set(values) != {str(key) for key in names}:
        raise ValueError("Descriptor keys must exactly match dataset labels")
    prompts = {}
    for key, name in names.items():
        descriptors = values[str(key)]
        if not isinstance(descriptors, list) or not descriptors or any(
                not isinstance(d, str) or not d.strip() for d in descriptors):
            raise ValueError(f"Missing/invalid descriptors for {key}")
        prompts[key] = [render_descriptor(name.replace("_", " "), d.strip()) for d in descriptors]
    return prompts, {"descriptor_sha256": hashlib.sha256(raw).hexdigest(),
                     "descriptor_provenance": spec["provenance"]}


def research_embeddings(vlm, task_config, baseline_prompts, label_order):
    """Điểm nối cho runner nghiên cứu; cấu hình trống giữ nguyên cách chạy nền."""
    prompts = task_config.get("_research_image_prompts")
    metadata = task_config.get("_research_metadata", {})
    if prompts is None:
        return None, metadata, baseline_prompts
    if list(prompts) != label_order:
        raise ValueError("Research image prompt label order differs from text labels")
    if any(not isinstance(v, list) or not v or any(not isinstance(p, str) or not p.strip() for p in v)
           for v in prompts.values()):
        raise ValueError("Research prompts must be nonempty lists of strings")
    image_classes, _ = build_class_embeddings(vlm, prompts, aggregation="mean_sim")
    return image_classes, metadata, {"image": prompts, "text": baseline_prompts}
