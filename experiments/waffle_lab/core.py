"""Các phép tính chỉ nhận điểm/embedding, không nhận nhãn thật hoặc caption."""
import numpy as np

VARIANTS = ("clip", "waffle", "waffle_unit", "semantic", "semantic_waffle",
            "anchor_mix", "stable_anchor")
REQUIRED_BANKS = {
    "clip": ("clip",), "waffle": ("waffle",), "waffle_unit": ("waffle",),
    "semantic": ("semantic",), "semantic_waffle": ("semantic_waffle",),
    "anchor_mix": ("semantic", "semantic_waffle"),
    "stable_anchor": ("semantic", "semantic_waffle"),
}


def _scores(value):
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 3 or not array.shape[0] or not array.shape[1] or array.shape[2] < 2:
        raise ValueError("Điểm phải có shape [samples, prompts, classes], không rỗng")
    if not np.isfinite(array).all():
        raise ValueError("Điểm có NaN/Inf")
    return array


def combine(variant, banks, noise_cap=0.5, prototype_norm=None):
    """Giữ cosine gốc; λ thích nghi dùng phiếu của từng cặp word/character.

    Agreement là độ đồng thuận, không phải xác suất dự đoán đúng.
    Không fit hoặc chọn trọng số theo nhãn của tập đánh giá.
    """
    if variant not in VARIANTS:
        raise ValueError(f"Biến thể không hợp lệ: {variant}")
    if (isinstance(noise_cap, bool) or not isinstance(noise_cap, (int, float, np.number))
            or not np.isfinite(noise_cap) or not 0 <= noise_cap <= 1):
        raise ValueError("noise_cap phải trong [0, 1]")
    arrays = {key: _scores(banks[key]) for key in REQUIRED_BANKS[variant]}
    if len({(value.shape[0], value.shape[2]) for value in arrays.values()}) != 1:
        raise ValueError("Các nhánh khác số mẫu hoặc thứ tự/kích thước lớp")
    first = next(iter(arrays.values()))
    count, _, classes = first.shape
    weights = np.zeros(count)
    agreement = np.full(count, np.nan)
    if variant in ("clip", "waffle", "semantic", "semantic_waffle"):
        result = first.mean(axis=1)
        if variant in ("waffle", "semantic_waffle"):
            weights[:] = 1.
    elif variant == "waffle_unit":
        norm = np.asarray(prototype_norm, dtype=np.float64)
        if norm.shape != (classes,) or not np.isfinite(norm).all() or (norm <= 1e-12).any():
            raise ValueError("Thiếu norm prototype Waffle hợp lệ")
        result = first.mean(axis=1) / norm
        weights[:] = 1.
    else:
        noisy = arrays["semantic_waffle"]
        if noisy.shape[1] % 2:
            raise ValueError("Waffle cần đủ cặp word/character")
        noisy_mean = noisy.mean(axis=1)
        anchor = arrays["semantic"].mean(axis=1)
        pair_scores = (noisy[:, 0::2, :] + noisy[:, 1::2, :]) / 2
        pair_top = pair_scores.argmax(axis=2)
        # Không đếm các phiếu hòa là bằng chứng đồng thuận cho lớp đứng đầu.
        pair_sorted = np.sort(pair_scores, axis=2)
        unique_vote = pair_sorted[:, :, -1] > pair_sorted[:, :, -2] + 1e-12
        noisy_top = noisy_mean.argmax(axis=1)
        agreement = ((pair_top == noisy_top[:, None]) & unique_vote).mean(axis=1)
        if variant == "stable_anchor":
            adjusted = np.clip((agreement - 1 / classes) / (1 - 1 / classes), 0, 1)
            weights = noise_cap * adjusted
        else:
            weights[:] = noise_cap
        result = (1 - weights[:, None]) * anchor + weights[:, None] * noisy_mean
    return {"scores": result, "noise_weight": weights, "prompt_agreement": agreement}


def attach_semantics(names, descriptors, author_prompts, pairs):
    """Ghép neo cố định vào chính hậu tố tác giả, cân bằng theo cặp.

    Không gọi LLM, không lấy câu từ dataset. Mỗi lớp có cùng số neo; mỗi neo
    nhận cùng số cặp word/character. Không sinh lại RNG hoặc khử prompt trùng.
    """
    if list(author_prompts) != list(names.values()) or set(descriptors) != set(names):
        raise ValueError("Tên/thứ tự lớp hoặc descriptor không khớp")
    counts = {len(items) for items in descriptors.values()}
    if len(counts) != 1 or not counts or next(iter(counts)) < 1:
        raise ValueError("Mỗi lớp cần cùng số descriptor, không rỗng")
    anchors = next(iter(counts))
    if type(pairs) is not int or pairs < 1 or pairs % anchors:
        raise ValueError("waffle_pairs phải chia hết cho số neo/lớp")
    clean, noisy, suffix_reference = {}, {}, None
    for key, name in names.items():
        items = descriptors[key]
        if any(not isinstance(s, str) or not s.strip() for s in items):
            raise ValueError("Descriptor phải là chuỗi không rỗng")
        clean[key] = [f"{name}: {s.strip().rstrip('.')}." for s in items]
        prefix = f"A photo of a {name.replace('_', ' ')}, "
        original = author_prompts[name]
        if len(original) != 2 * pairs or any(not s.startswith(prefix) or not s.endswith('.') for s in original):
            raise ValueError("Cấu trúc prompt tác giả không khớp; không cắt hậu tố theo suy đoán")
        suffixes = [s[len(prefix):-1] for s in original]
        if suffix_reference is not None and suffixes != suffix_reference:
            raise ValueError("Hậu tố Waffle không được chia sẻ giữa các lớp")
        suffix_reference = suffixes
        noisy[key] = [clean[key][(j // 2) % anchors].rstrip('.') + ", " + suffix + "."
                      for j, suffix in enumerate(suffixes)]
    return clean, noisy
