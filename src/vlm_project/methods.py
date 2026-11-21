"""Majority hai nhánh và ánh xạ Fakeddit, giữ nguyên công thức cũ."""
import numpy as np
import torch
from .adaptation import probability_features, em_dirichlet, stata

def majority(features, means, seed):
    unit = means / np.linalg.norm(means, axis=1, keepdims=True)
    normalized = (features @ unit.T).argmax(1)
    em = normalized.copy()
    st = normalized.copy()
    groups = np.full(len(features), -1, dtype=int)
    order = np.random.default_rng(seed).permutation(len(features))
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for batch, start in enumerate(range(0, len(order), 75)):
            ids = order[start:start + 75]
            groups[ids] = batch
            query = probability_features(features[ids] @ means.T)
            em[ids] = em_dirichlet(query[None])[0][0]
            if len(ids) > 3:
                st[ids] = stata(features[ids], unit)[1].argmax(1)
    finally:
        torch.set_num_threads(old)
    votes = np.stack([normalized, em, st], axis=1)
    if means.shape[0] != 2:
        raise ValueError('Binary majority only')
    return ((votes.sum(1) >= 2).astype(int), {'votes': votes, 'batch_ids': groups})

def hierarchy(binary_labels, six_scores, six_labels):
    """2way real=1 maps to 6way true=0; fake selects among 1..5."""
    binary = np.asarray(binary_labels)
    scores = np.asarray(six_scores)
    if set(binary.tolist()) - {'0', '1'} or scores.shape != (len(binary), 6):
        raise ValueError('Invalid Fakeddit hierarchy inputs')
    if list(six_labels) != [str(k) for k in range(6)] or not np.isfinite(scores).all():
        raise ValueError('Invalid six-way label order/scores')
    result = np.asarray(six_labels)[1 + scores[:, 1:].argmax(1)]
    result[binary == '1'] = '0'
    return result

def probabilities(values):
    p = np.asarray(values, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] < 2 or (not np.isfinite(p).all()) or np.any(p < 0) or (not np.allclose(p.sum(1), 1, atol=2e-06, rtol=0)):
        raise ValueError('Expected finite normalized class probabilities')
    return p

def fuse(image, text, weight):
    image, text = (probabilities(image), probabilities(text))
    if image.shape != text.shape or isinstance(weight, bool) or (not np.isfinite(weight)) or (not 0 <= weight <= 1):
        raise ValueError('Fusion requires aligned probabilities and weight in [0,1]')
    return weight * image + (1 - weight) * text

def replace_clip_vote(votes, multimodal_clip_prob):
    """Only the normalized CLIP voter sees title; EM/StatA remain image-only."""
    votes = np.asarray(votes)
    p = probabilities(multimodal_clip_prob)
    if votes.shape != (len(p), 3) or p.shape[1] != 2 or (not np.isin(votes, [0, 1]).all()):
        raise ValueError('Expected three binary votes per image')
    updated = votes.copy()
    updated[:, 0] = p.argmax(1)
    return ((updated.sum(1) >= 2).astype(int), updated)
