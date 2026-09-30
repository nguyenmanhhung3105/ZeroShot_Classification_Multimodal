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
                                  image_weight: float = None) -> tuple:
    """Encode theo batch, bỏ nhánh có trọng số 0; giữ fallback ảnh khi text rỗng."""
    if len(images) != len(texts) or not texts:
        raise ValueError("Ảnh và text phải cùng số mẫu và không rỗng")
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size phải là số nguyên dương")
    if image_weight is not None and not 0 <= image_weight <= 1:
        raise ValueError("image_weight phải nằm trong [0, 1]")
    texts = _clean_texts(texts)
    text_empty_mask = np.array([t == "" for t in texts])
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
            encoded_text = _to_numpy(vlm.encode_texts([texts[i] for i in text_indices]))
        if image_embeds is None:
            prototype = encoded_image if encoded_image is not None else encoded_text
            image_embeds = np.zeros((len(images), prototype.shape[1]), dtype=prototype.dtype)
            text_embeds = np.zeros_like(image_embeds)
        if encoded_image is not None:
            image_embeds[image_indices] = _l2_normalize(encoded_image)
        if encoded_text is not None:
            text_embeds[text_indices] = _l2_normalize(encoded_text)
    return image_embeds, text_embeds, text_empty_mask


def predict_single_label(vlm, images: list, texts: list, class_embeds: np.ndarray, label_order: list,
                          batch_size: int = 16, image_weight: float = 0.5,
                          text_logit_scale: float = None, show_progress: bool = True,
                          img_class_embeds=None, txt_class_embeds=None,
                          encoded_embeddings=None) -> tuple:
    """
    Single-label multimodal inference cho Fakeddit / CrisisMMD.

    text_logit_scale: cho phép override scale riêng cho nhánh text (mặc định
    dùng chung logit_scale của model — xem lưu ý ở docstring đầu file).
    show_progress   : hiện % tiến trình trong lúc encode ảnh/text.

    img_class_embeds/txt_class_embeds: optional unit-norm prototypes, same
    shape and label order as class_embeds. None keeps shared legacy prompts.
    encoded_embeddings: optional (image, text, empty_mask) from the exact same
    ordered samples/model with both branches encoded; None encodes normally.

    Returns:
        predictions    : list[str]
        p              : np.ndarray (N, n_classes) — probability đã fuse, DÙNG ĐỂ EVALUATE
        p_img          : np.ndarray (N, n_classes) — probability riêng nhánh ảnh, DÙNG ĐỂ LOG/DEBUG
        p_text         : np.ndarray (N, n_classes) — probability riêng nhánh text, DÙNG ĐỂ LOG/DEBUG
        text_empty_mask: np.ndarray[bool] (N,)
    """
    image_embeds, text_embeds, text_empty_mask = _prediction_embeddings(
        vlm, images, texts, batch_size=batch_size,
        show_progress=show_progress, desc="Single-label inference", image_weight=image_weight,
        encoded_embeddings=encoded_embeddings,
    )

    sim_image = image_embeds @ _class_matrix(class_embeds, img_class_embeds).T
    sim_text = text_embeds @ _class_matrix(class_embeds, txt_class_embeds).T

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
                         img_class_embeds=None, txt_class_embeds=None,
                         encoded_embeddings=None) -> tuple:
    """
    Multi-label multimodal inference cho MM-IMDb.

    Mỗi lớp được đánh giá ĐỘC LẬP qua sigmoid ở TỪNG NHÁNH riêng, rồi mới fuse —
    không sigmoid trên cosine đã bị trộn 2 modality (khác biệt cốt lõi so với bản cũ).

    show_progress: hiện % tiến trình trong lúc encode ảnh/text.

    Optional branch prototypes and encoded_embeddings follow the same contract
    as predict_single_label; omitted arguments preserve legacy behavior.

    Returns:
        predictions    : list[list[str]]
        probs          : np.ndarray (N, n_classes) — đã fuse, DÙNG ĐỂ EVALUATE
        p_img          : np.ndarray (N, n_classes) — DÙNG ĐỂ LOG/DEBUG
        p_text         : np.ndarray (N, n_classes) — DÙNG ĐỂ LOG/DEBUG
        used_fallback  : np.ndarray[bool] (N,) — True nếu sample này không có
                          lớp nào vượt threshold, phải fallback về top-1
        text_empty_mask: np.ndarray[bool] (N,)
    """
    if not 0 <= threshold <= 1:
        raise ValueError("threshold phải nằm trong [0, 1]")
    image_embeds, text_embeds, text_empty_mask = _prediction_embeddings(
        vlm, images, texts, batch_size=batch_size,
        show_progress=show_progress, desc="Multi-label inference", image_weight=image_weight,
        encoded_embeddings=encoded_embeddings,
    )

    sim_image = image_embeds @ _class_matrix(class_embeds, img_class_embeds).T
    sim_text = text_embeds @ _class_matrix(class_embeds, txt_class_embeds).T

    logits_image = _calibrated_logits(vlm, sim_image)
    scale_text = text_logit_scale if text_logit_scale is not None else getattr(vlm, "logit_scale", 100.0)
    logits_text = sim_text * scale_text + getattr(vlm, "logit_bias", 0.0)

    p_img = _sigmoid(logits_image)
    p_text = _sigmoid(logits_text)

    probs = _weighted_fuse(p_img, p_text, image_weight, text_empty_mask)

    predictions = []
    used_fallback = []
    for row in probs:
        labels = [label_order[i] for i, prob in enumerate(row) if prob >= threshold]
        if not labels and fallback_top1:
            labels = [label_order[int(np.argmax(row))]]
            used_fallback.append(True)
        else:
            used_fallback.append(False)
        predictions.append(labels)

    return predictions, probs, p_img, p_text, np.array(used_fallback), text_empty_mask


def _class_matrix(legacy, branch):
    """Optional prototypes use exactly the caller's legacy label order."""
    if branch is None:
        return legacy
    branch = np.asarray(branch)
    if branch.shape != legacy.shape or not np.isfinite(branch).all():
        raise ValueError("Branch prototypes must match legacy shape and be finite")
    if not np.allclose(np.linalg.norm(branch, axis=1), 1, atol=1e-5):
        raise ValueError("Branch prototypes must have unit L2 norm")
    return branch


def _prediction_embeddings(vlm, images, texts, encoded_embeddings=None, **kwargs):
    """Cache is opt-in, for the exact same ordered samples/model, both branches encoded.

    The caller owns cache provenance. Default delegates to the unchanged encoder.
    """
    if encoded_embeddings is None:
        return encode_multimodal_embeddings(vlm, images, texts, **kwargs)
    if not 0 <= kwargs["image_weight"] <= 1:
        raise ValueError("image_weight phải nằm trong [0, 1]")
    image, text, empty = map(np.asarray, encoded_embeddings)
    if (not texts or len(images) != len(texts) or image.ndim != 2
            or image.shape != text.shape or image.shape[0] != len(texts)
            or empty.shape != (len(texts),) or empty.dtype != np.bool_
            or not np.isfinite(image).all() or not np.isfinite(text).all()
            or not np.array_equal(empty, [t == "" for t in _clean_texts(texts)])):
        raise ValueError("Invalid cached embeddings or sample alignment")
    return image, text, empty
