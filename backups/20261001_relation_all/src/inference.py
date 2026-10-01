"""
Core zero-shot inference — multimodal (image + text) fusion, CLIP-style.

Cơ chế (SCORE-LEVEL FUSION — KHÔNG fuse embedding):
    Image  -> image embedding  (normalize) -> sim_image = image_embed @ class_embeds.T
    Text   -> text embedding   (normalize) -> sim_text  = text_embed  @ class_embeds.T
    Mỗi nhánh được calibrate riêng bằng logit_scale/logit_bias CỦA CHÍNH MODEL đó
    (KHÔNG cộng cosine thô của 2 nhánh — thang đo cross-modal và text-text khác nhau
    hoàn toàn do modality gap của CLIP/SigLIP).
    p_img, p_text -> fuse ở tầng PROBABILITY: p = w * p_img + (1-w) * p_text

Single-label (Fakeddit, CrisisMMD): softmax mỗi nhánh theo lớp -> fuse -> ARGMAX
Multi-label  (MM-IMDb)            : sigmoid mỗi nhánh theo lớp -> fuse -> THRESHOLD

LƯU Ý: nhánh text dùng logit_scale của model dù model đó được calibrate cho
cặp (ảnh, text) chứ chưa từng được calibrate cho (text sample, text prompt).
Đây là giả định hợp lý nhất hiện có, nhưng nên coi là điểm cần validate/tune
riêng (xem tham số text_logit_scale) chứ không mặc định đúng.

PROGRESS BAR: encode_multimodal_embeddings (và do đó predict_single_label /
predict_multi_label) hiện % tiến trình qua các batch bằng tqdm, nếu có cài
đặt. Tắt bằng show_progress=False khi chạy trong pipeline không muốn in ra
console (vd: ghi log file, job chấm điểm tự động).
"""

import numpy as np
from text_scale import validate_text_scale

try:
    from tqdm.auto import tqdm as _tqdm
    _HAS_TQDM = True
except ImportError:  # tqdm không bắt buộc — vẫn chạy được, chỉ mất progress bar
    _HAS_TQDM = False

_warned_missing_tqdm = False


def _warn_missing_tqdm() -> None:
    global _warned_missing_tqdm
    if not _warned_missing_tqdm:
        print("[inference] Không tìm thấy 'tqdm' — bỏ qua progress bar. "
              "Cài bằng: pip install tqdm")
        _warned_missing_tqdm = True


def _progress_iter(iterable, enabled: bool = True, desc: str = "", unit: str = "batch"):
    """Bọc iterable bằng tqdm để hiện % tiến trình (nếu có tqdm và enabled=True).
    Không có tqdm hoặc enabled=False -> trả về iterable gốc, không crash.
    """
    if not enabled:
        return iterable
    if not _HAS_TQDM:
        _warn_missing_tqdm()
        return iterable
    return _tqdm(iterable, desc=desc, unit=unit)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def _l2_normalize(x: np.ndarray, axis: int = -1) -> np.ndarray:
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (norm + 1e-12)


def _to_numpy(tensor) -> np.ndarray:
    """Chuyển torch.Tensor (có thể trên GPU) về numpy an toàn."""
    return tensor.detach().cpu().numpy()


_warned_missing_calibration = set()


def _calibrated_logits(vlm, sims: np.ndarray) -> np.ndarray:
    """Áp logit_scale/logit_bias THẬT của model lên cosine similarity thô.
    Bắt buộc phải làm bước này trước khi so sánh/kết hợp điểm giữa các nhánh,
    vì cosine thô của các model/nhánh khác nhau không cùng thang đo.

    Fallback 100.0 là giá trị trần logit_scale kinh điển của CLIP (ln(100) là
    cap được dùng khi train) — không dùng 1.0 như code cũ vì 1.0 sẽ làm
    softmax/sigmoid gần như phẳng, mọi lớp gần như đồng đều bất kể model
    thật sự tự tin đến đâu.
    """
    if not hasattr(vlm, "logit_scale") and vlm.model_name not in _warned_missing_calibration:
        print(f"[inference] CẢNH BÁO: '{vlm.model_name}' không có logit_scale/logit_bias "
              f"(model_registry.py chưa cập nhật?). Dùng fallback 100.0/0.0 — "
              f"kết quả fusion/threshold có thể không đáng tin cậy.")
        _warned_missing_calibration.add(vlm.model_name)

    logit_scale = getattr(vlm, "logit_scale", 100.0)
    logit_bias = getattr(vlm, "logit_bias", 0.0)
    return sims * logit_scale + logit_bias


def _weighted_fuse(p_img: np.ndarray, p_text: np.ndarray, image_weight: float,
                    text_empty_mask: np.ndarray = None) -> np.ndarray:
    """Fuse 2 nhánh Ở TẦNG PROBABILITY (sau softmax hoặc sigmoid), không phải
    ở tầng embedding hay cosine thô.

    text_empty_mask: True tại các sample có text rỗng/NaN -> ép image_weight=1.0
    CHỈ cho sample đó, tránh việc embedding của chuỗi rỗng (một vector cố định,
    vô nghĩa) âm thầm kéo lệch kết quả.
    """
    w = np.full(p_img.shape[0], float(image_weight), dtype=float)
    if text_empty_mask is not None:
        w[text_empty_mask] = 1.0
    w = w[:, None]
    return w * p_img + (1.0 - w) * p_text


def build_class_embeddings(vlm, prompt_set: dict) -> tuple:
    """
    Tạo embedding đại diện cho từng class bằng prompt ensembling.
    (Không đổi — class prototype vẫn là text-only, đây là chuẩn zero-shot CLIP.)
    """
    label_order = list(prompt_set.keys())
    class_embeds = []

    for label in label_order:
        prompts = prompt_set[label]
        if isinstance(prompts, str):
            prompts = [prompts]

        text_embeds = _to_numpy(vlm.encode_texts(prompts))
        mean_embed = text_embeds.mean(axis=0)
        mean_embed = mean_embed / (np.linalg.norm(mean_embed) + 1e-12)
        class_embeds.append(mean_embed)

    return np.stack(class_embeds, axis=0), label_order


def _clean_texts(texts: list) -> list:
    """Thay text rỗng/NaN bằng chuỗi rỗng để tokenizer không crash."""
    cleaned = []
    for t in texts:
        if isinstance(t, str) and t.strip():
            cleaned.append(t.strip())
        else:
            cleaned.append("")
    return cleaned


def encode_multimodal_embeddings(vlm, images: list, texts: list, batch_size: int = 16,
                                  show_progress: bool = True, desc: str = "Encoding",
                                  image_weight: float = None, chunk_long_text=False,
                                  diagnostics=None, max_sim_classes=None, chunk_scores=None) -> tuple:
    """Encode theo batch, bỏ nhánh có trọng số 0; giữ fallback ảnh khi text rỗng."""
    if len(images) != len(texts) or not texts:
        raise ValueError("Ảnh và text phải cùng số mẫu và không rỗng")
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size phải là số nguyên dương")
    if image_weight is not None and not 0 <= image_weight <= 1:
        raise ValueError("image_weight phải nằm trong [0, 1]")
    texts = _clean_texts(texts)
    text_empty_mask = np.array([t == "" for t in texts])
    chunk_counts = [0] * len(texts)
    if max_sim_classes is not None:
        if not chunk_long_text or chunk_scores is None:
            raise ValueError("max_sim requires chunking and a score output buffer")
        chunk_scores["text"] = np.zeros((len(texts), len(max_sim_classes)))
    image_embeds = text_embeds = None
    iterator = _progress_iter(range(0, len(images), batch_size), enabled=show_progress,
                              desc=desc, unit="batch")
    for start in iterator:
        end = min(start + batch_size, len(images))
        image_indices = [i for i in range(start, end)
                         if image_weight is None or image_weight > 0 or text_empty_mask[i]]
        text_indices = [i for i in range(start, end)
                        if (image_weight is None or image_weight < 1) and not text_empty_mask[i]]
        encoded_image = encoded_text = None
        if image_indices:
            batch = [images[i] for i in image_indices]
            try:
                encoded_image = _to_numpy(vlm.encode_images(batch))
            finally:
                # Ảnh do sequence mở thuộc batch này, không đóng ảnh PIL do caller sở hữu.
                if hasattr(images, "paths"):
                    for img in batch:
                        img.close()
        if text_indices:
            if max_sim_classes is not None:
                from text_chunking import encode_chunked_max_sim
                scores, counts = encode_chunked_max_sim(
                    vlm, [texts[i] for i in text_indices], batch_size, max_sim_classes)
                chunk_scores["text"][text_indices] = scores
                # No single text vector represents per-class maximum scores.
                encoded_text = np.zeros((len(text_indices), max_sim_classes.shape[1]), dtype=scores.dtype)
                for i, count in zip(text_indices, counts):
                    chunk_counts[i] = count
            elif chunk_long_text:
                from text_chunking import encode_chunked_texts
                encoded_text, counts = encode_chunked_texts(
                    vlm, [texts[i] for i in text_indices], batch_size)
                for i, count in zip(text_indices, counts):
                    chunk_counts[i] = count
            else:
                encoded_text = _to_numpy(vlm.encode_texts([texts[i] for i in text_indices]))
        if image_embeds is None:
            prototype = encoded_image if encoded_image is not None else encoded_text
            image_embeds = np.zeros((len(images), prototype.shape[1]), dtype=prototype.dtype)
            text_embeds = np.zeros_like(image_embeds)
        if encoded_image is not None:
            image_embeds[image_indices] = _l2_normalize(encoded_image)
        if encoded_text is not None:
            text_embeds[text_indices] = _l2_normalize(encoded_text)
    if chunk_long_text and diagnostics is not None:
        diagnostics["chunking"] = {"enabled": True, "aggregation": "max_sim" if max_sim_classes is not None else "mean_embed",
                                  "counts": chunk_counts, "total_chunks": sum(chunk_counts),
                                  "n_chunked_texts": sum(n > 1 for n in chunk_counts)}
    return image_embeds, text_embeds, text_empty_mask


def predict_single_label(vlm, images: list, texts: list, class_embeds: np.ndarray, label_order: list,
                          batch_size: int = 16, image_weight: float = 0.5,
                          text_logit_scale: float = None, show_progress: bool = True,
                          diagnostics=None, image_text_relation_enabled=False,
                          relation_scores=None) -> tuple:
    """
    Single-label multimodal inference cho Fakeddit / CrisisMMD.

    text_logit_scale: cho phép override scale riêng cho nhánh text (mặc định
    dùng chung logit_scale của model — xem lưu ý ở docstring đầu file).
    show_progress   : hiện % tiến trình trong lúc encode ảnh/text.

    Returns:
        predictions    : list[str]
        p              : np.ndarray (N, n_classes) — probability đã fuse, DÙNG ĐỂ EVALUATE
        p_img          : np.ndarray (N, n_classes) — probability riêng nhánh ảnh, DÙNG ĐỂ LOG/DEBUG
        p_text         : np.ndarray (N, n_classes) — probability riêng nhánh text, DÙNG ĐỂ LOG/DEBUG
        text_empty_mask: np.ndarray[bool] (N,)
    """
    validate_text_scale(text_logit_scale)
    if not isinstance(image_text_relation_enabled, bool):
        raise ValueError('image_text_relation_enabled must be boolean')
    if not 0 <= image_weight <= 1:
        raise ValueError('image_weight must be in [0, 1]')
    if image_text_relation_enabled and relation_scores is None:
        raise ValueError('Relation diagnostics require a relation_scores output dict')
    image_embeds, text_embeds, text_empty_mask = encode_multimodal_embeddings(
        vlm, images, texts, batch_size=batch_size,
        show_progress=show_progress, desc="Single-label inference",
        image_weight=None if image_text_relation_enabled else image_weight,
    )

    if image_text_relation_enabled:
        relation_scores['image_text_similarity'] = [
            None if empty else float(np.dot(i, t))
            for i, t, empty in zip(image_embeds, text_embeds, text_empty_mask)]
        # Extra encodings serve diagnostics only; keep baseline unused-branch scores.
        if image_weight == 0:
            image_embeds[~text_empty_mask] = 0
        if image_weight == 1:
            text_embeds[:] = 0

    sim_image = image_embeds @ class_embeds.T
    sim_text = text_embeds @ class_embeds.T

    _record_cosine(diagnostics, sim_image, sim_text, image_weight, text_empty_mask)

    logits_image = _calibrated_logits(vlm, sim_image)
    scale_text = text_logit_scale if text_logit_scale is not None else getattr(vlm, "logit_scale", 100.0)
    logits_text = sim_text * scale_text + getattr(vlm, "logit_bias", 0.0)

    p_img = _softmax(logits_image, axis=-1)
    p_text = _softmax(logits_text, axis=-1)

    p = _weighted_fuse(p_img, p_text, image_weight, text_empty_mask)

    pred_indices = np.argmax(p, axis=1)
    predictions = [label_order[i] for i in pred_indices]

    return predictions, p, p_img, p_text, text_empty_mask


def predict_multi_label(vlm, images: list, texts: list, class_embeds: np.ndarray, label_order: list,
                         threshold: float = 0.22, fallback_top1: bool = True,
                         batch_size: int = 16, image_weight: float = 0.5,
                         text_logit_scale: float = None, show_progress: bool = True,
                         decision_mode="fixed_threshold",
                         relative_z_threshold=1.0, diagnostics=None,
                         chunk_long_text=False, chunk_aggregation="mean_embed") -> tuple:
    """
    Multi-label multimodal inference cho MM-IMDb.

    Mỗi lớp được đánh giá ĐỘC LẬP qua sigmoid ở TỪNG NHÁNH riêng, rồi mới fuse —
    không sigmoid trên cosine đã bị trộn 2 modality (khác biệt cốt lõi so với bản cũ).

    show_progress: hiện % tiến trình trong lúc encode ảnh/text.

    decision_mode="relative_z": bypass sigmoid/model calibration and threshold;
    return relative scores in the existing probs/p_img/p_text tuple positions.
    Select scores > relative_z_threshold. These outputs are NOT probabilities.
    Main MM-IMDb runners explicitly select this mode through their config.

    Returns:
        predictions    : list[list[str]]
        probs          : np.ndarray (N, n_classes) — đã fuse, DÙNG ĐỂ EVALUATE
        p_img          : np.ndarray (N, n_classes) — DÙNG ĐỂ LOG/DEBUG
        p_text         : np.ndarray (N, n_classes) — DÙNG ĐỂ LOG/DEBUG
        used_fallback  : np.ndarray[bool] (N,) — True nếu sample này không có
                          lớp nào vượt threshold, phải fallback về top-1
        text_empty_mask: np.ndarray[bool] (N,)
    """
    if decision_mode not in ("fixed_threshold", "relative_z"):
        raise ValueError("Unknown decision_mode")
    validate_text_scale(text_logit_scale)
    if chunk_aggregation not in ("mean_embed", "max_sim"):
        raise ValueError("chunk_aggregation must be mean_embed or max_sim")
    relative = decision_mode == "relative_z"
    if relative and (not np.isfinite(relative_z_threshold) or relative_z_threshold < 0):
        raise ValueError("relative_z_threshold must be finite and nonnegative")
    if not relative and not 0 <= threshold <= 1:
        raise ValueError("threshold phải nằm trong [0, 1]")
    use_max = chunk_long_text and chunk_aggregation == "max_sim"
    chunk_scores = {}
    image_embeds, text_embeds, text_empty_mask = encode_multimodal_embeddings(
        vlm, images, texts, batch_size=batch_size,
        show_progress=show_progress, desc="Multi-label inference", image_weight=image_weight,
        chunk_long_text=chunk_long_text, diagnostics=diagnostics,
        max_sim_classes=class_embeds if use_max else None, chunk_scores=chunk_scores,
    )

    sim_image = image_embeds @ class_embeds.T
    sim_text = chunk_scores["text"] if use_max else text_embeds @ class_embeds.T

    if relative:
        probs, p_img, p_text = relative_multilabel_scores(sim_image, sim_text, image_weight, text_empty_mask)
    else:
        logits_image = _calibrated_logits(vlm, sim_image)
        scale_text = text_logit_scale if text_logit_scale is not None else getattr(vlm, "logit_scale", 100.0)
        logits_text = sim_text * scale_text + getattr(vlm, "logit_bias", 0.0)
        p_img = _sigmoid(logits_image)
        p_text = _sigmoid(logits_text)
        probs = _weighted_fuse(p_img, p_text, image_weight, text_empty_mask)

    _record_cosine(diagnostics, sim_image, sim_text, image_weight, text_empty_mask)
    predictions = []
    used_fallback = []
    for row in probs:
        labels = [label_order[i] for i, prob in enumerate(row)
                  if (prob > relative_z_threshold if relative else prob >= threshold)]
        if not labels and fallback_top1:
            labels = [label_order[int(np.argmax(row))]]
            used_fallback.append(True)
        else:
            used_fallback.append(False)
        predictions.append(labels)

    return predictions, probs, p_img, p_text, np.array(used_fallback), text_empty_mask


def relative_multilabel_scores(sim_image, sim_text, image_weight=0.5, text_empty_mask=None):
    """Per-sample, across-label z scores; no labels or cross-sample statistics.

    Standardize each modality separately, then linearly fuse. These are scores,
    NOT probabilities; learned model scale/bias and sigmoid are not applied.
    Constant rows map to zeros, leaving the caller's explicit top-1 fallback.
    """
    image = np.asarray(sim_image, dtype=np.float64)
    text = np.asarray(sim_text, dtype=np.float64)
    if (image.ndim != 2 or image.shape != text.shape or image.shape[1] < 2
            or not np.isfinite(image).all() or not np.isfinite(text).all()
            or not 0 <= image_weight <= 1):
        raise ValueError("Invalid cosine matrices or fusion weight")
    if text_empty_mask is not None:
        text_empty_mask = np.asarray(text_empty_mask)
        if text_empty_mask.shape != (len(image),) or text_empty_mask.dtype != np.bool_:
            raise ValueError("Invalid text empty mask")
    def standardize(scores):
        centered = scores - scores.mean(axis=1, keepdims=True)
        std = scores.std(axis=1, keepdims=True)
        return np.divide(centered, std, out=np.zeros_like(scores), where=std > 1e-8)
    zi, zt = standardize(image), standardize(text)
    return _weighted_fuse(zi, zt, image_weight, text_empty_mask), zi, zt


def multilabel_settings(task_config, threshold):
    """Main MM-IMDb runners default to relative_z; legacy API remains opt-in compatible."""
    mode = task_config.get("decision_mode", "relative_z")
    cutoff = task_config.get("relative_z_threshold", 1.0)
    chunking = task_config.get("chunk_long_text", False)
    if not isinstance(chunking, bool):
        raise ValueError("chunk_long_text must be a YAML boolean")
    aggregation = task_config.get("chunk_aggregation", "mean_embed")
    if aggregation not in ("mean_embed", "max_sim"):
        raise ValueError("chunk_aggregation must be mean_embed or max_sim")
    options = {"decision_mode": mode, "relative_z_threshold": cutoff,
               "chunk_long_text": chunking, "chunk_aggregation": aggregation}
    relative = mode == "relative_z"
    metadata = {"decision_mode": mode, "score_type": "relative_z" if relative else "probability",
                "threshold": None if relative else threshold,
                "relative_z_threshold": cutoff if relative else None,
                "model_scale_applied": not relative, "apply_bias_to_text": not relative,
                "chunk_long_text": chunking, "chunk_aggregation": aggregation if chunking else None}
    return options, metadata


def _record_cosine(diagnostics, image, text, weight, empty):
    if diagnostics is None:
        return
    diagnostics["cosine"] = {}
    for name, scores, mask in (("image", image, np.ones(len(image), bool) if weight > 0 else empty),
                               ("text", text, ~empty & (weight < 1))):
        if np.any(mask):
            diagnostics["cosine"][name] = {"cosine_mean": float(scores[mask].mean()),
                                           "cosine_std": float(scores[mask].std())}
