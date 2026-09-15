"""
Core zero-shot inference.

Cơ chế:
    Image -> image embedding
    Prompt -> text embedding
    Cosine similarity giữa image và class embeddings

Single-label:
    Fakeddit, CrisisMMD -> ARGMAX

Multi-label:
    MM-IMDb -> SIGMOID + THRESHOLD
"""

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def build_class_embeddings(vlm, prompt_set: dict) -> tuple:
    """
    Tạo embedding đại diện cho từng class bằng prompt ensembling.

    prompt_set:
        {
            "class_a": ["prompt 1", "prompt 2"],
            "class_b": ["prompt 1", "prompt 2"]
        }

    Mỗi prompt được encode -> lấy mean embedding -> normalize lại.

    Returns:
        class_embeds: np.ndarray (n_classes, D)
        label_order : list[str]
    """
    label_order = list(prompt_set.keys())
    class_embeds = []

    for label in label_order:
        prompts = prompt_set[label]
        if isinstance(prompts, str): prompts = [prompts]

        text_embeds = vlm.encode_text(prompts)
        mean_embed = text_embeds.mean(axis=0)
        mean_embed = mean_embed / (np.linalg.norm(mean_embed) + 1e-12)
        class_embeds.append(mean_embed)

    return np.stack(class_embeds, axis=0), label_order


def predict_single_label(vlm, images: list, class_embeds: np.ndarray, label_order: list) -> tuple:
    """
    Single-label inference cho Fakeddit / CrisisMMD.

    Mỗi ảnh nhận đúng 1 class có cosine similarity cao nhất.

    Returns:
        predictions: list[str]
        sims       : np.ndarray (N, n_classes)
    """
    image_embeds = vlm.encode_image(images)
    sims = image_embeds @ class_embeds.T
    pred_indices = np.argmax(sims, axis=1)
    predictions = [label_order[i] for i in pred_indices]

    return predictions, sims


def predict_multi_label(vlm, images: list, class_embeds: np.ndarray, label_order: list, threshold: float = 0.22, fallback_top1: bool = True) -> tuple:
    """
    Multi-label inference cho MM-IMDb.

    Mỗi class được đánh giá độc lập bằng sigmoid.
    Class có probability > threshold sẽ được chọn.

    Args:
        threshold    : ngưỡng multi-label, cần tune trên validation set.
        fallback_top1: nếu không class nào vượt threshold thì lấy class có điểm cao nhất.

    Returns:
        predictions: list[list[str]]
        probs      : np.ndarray (N, n_classes)
    """
    image_embeds = vlm.encode_image(images)
    sims = image_embeds @ class_embeds.T

    logit_scale = getattr(vlm, "logit_scale", 1.0)
    logit_bias = getattr(vlm, "logit_bias", None)
    if logit_bias is None: logit_bias = 0.0

    probs = _sigmoid(logit_scale * sims + logit_bias)

    predictions = []
    for row in probs:
        labels = [label_order[i] for i, prob in enumerate(row) if prob >= threshold]
        if not labels and fallback_top1: labels = [label_order[int(np.argmax(row))]]
        predictions.append(labels)

    return predictions, probs