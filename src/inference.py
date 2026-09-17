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
                                  show_progress: bool = True, desc: str = "Encoding") -> tuple:
    """
    Encode ảnh và text của TỪNG SAMPLE thành 2 ma trận RIÊNG BIỆT — KHÔNG fuse
    ở đây. Đây là điểm khác biệt cốt lõi so với encode_fused_embeddings cũ
    (đã bỏ hoàn toàn, xem giải thích ở đầu file).

    show_progress: hiện thanh tiến trình (%) qua các batch trong lúc encode.
    desc         : nhãn hiển thị trên thanh tiến trình.

    Returns:
        image_embeds   : np.ndarray (N, D), đã L2-normalize
        text_embeds    : np.ndarray (N, D), đã L2-normalize
        text_empty_mask: np.ndarray[bool] (N,) — True nếu text gốc rỗng/NaN
    """
    if len(images) != len(texts):
        raise ValueError(f"Số ảnh ({len(images)}) và số text ({len(texts)}) không khớp nhau.")

    texts = _clean_texts(texts)
    text_empty_mask = np.array([t == "" for t in texts])

    image_chunks, text_chunks = [], []

    batch_starts = range(0, len(images), batch_size)
    iterator = _progress_iter(batch_starts, enabled=show_progress, desc=desc, unit="batch")

    for start in iterator:
        end = start + batch_size
        img_batch = images[start:end]
        txt_batch = texts[start:end]

        image_chunks.append(_l2_normalize(_to_numpy(vlm.encode_images(img_batch))))
        text_chunks.append(_l2_normalize(_to_numpy(vlm.encode_texts(txt_batch))))

    return (np.concatenate(image_chunks, axis=0),
            np.concatenate(text_chunks, axis=0),
            text_empty_mask)


def predict_single_label(vlm, images: list, texts: list, class_embeds: np.ndarray, label_order: list,
                          batch_size: int = 16, image_weight: float = 0.5,
                          text_logit_scale: float = None, show_progress: bool = True) -> tuple:
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
    image_embeds, text_embeds, text_empty_mask = encode_multimodal_embeddings(
        vlm, images, texts, batch_size=batch_size,
        show_progress=show_progress, desc="Single-label inference"
    )

    sim_image = image_embeds @ class_embeds.T
    sim_text = text_embeds @ class_embeds.T

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
                         text_logit_scale: float = None, show_progress: bool = True) -> tuple:
    """
    Multi-label multimodal inference cho MM-IMDb.

    Mỗi lớp được đánh giá ĐỘC LẬP qua sigmoid ở TỪNG NHÁNH riêng, rồi mới fuse —
    không sigmoid trên cosine đã bị trộn 2 modality (khác biệt cốt lõi so với bản cũ).

    show_progress: hiện % tiến trình trong lúc encode ảnh/text.

    Returns:
        predictions    : list[list[str]]
        probs          : np.ndarray (N, n_classes) — đã fuse, DÙNG ĐỂ EVALUATE
        p_img          : np.ndarray (N, n_classes) — DÙNG ĐỂ LOG/DEBUG
        p_text         : np.ndarray (N, n_classes) — DÙNG ĐỂ LOG/DEBUG
        used_fallback  : np.ndarray[bool] (N,) — True nếu sample này không có
                          lớp nào vượt threshold, phải fallback về top-1
        text_empty_mask: np.ndarray[bool] (N,)
    """
    image_embeds, text_embeds, text_empty_mask = encode_multimodal_embeddings(
        vlm, images, texts, batch_size=batch_size,
        show_progress=show_progress, desc="Multi-label inference"
    )

    sim_image = image_embeds @ class_embeds.T
    sim_text = text_embeds @ class_embeds.T

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