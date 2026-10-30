"""Label-free binary majority and Fakeddit hierarchy."""
import numpy as np
import torch
from experiments.clip_transductive.core import probability_features, em_dirichlet
from experiments.clip_stata.core import stata


def majority(features, means, seed):
    unit = means / np.linalg.norm(means, axis=1, keepdims=True)
    normalized = (features @ unit.T).argmax(1)
    em = normalized.copy(); st = normalized.copy()
    groups = np.full(len(features), -1, dtype=int)
    order = np.random.default_rng(seed).permutation(len(features))
    old = torch.get_num_threads(); torch.set_num_threads(1)
    try:
        for batch, start in enumerate(range(0, len(order), 75)):
            ids = order[start:start+75]; groups[ids] = batch
            query = probability_features(features[ids] @ means.T)
            em[ids] = em_dirichlet(query[None])[0][0]
            if len(ids) > 3:
                st[ids] = stata(features[ids], unit)[1].argmax(1)
            # Groups <=3 retain normalized CLIP for StatA (no sufficient kNN).
    finally:
        torch.set_num_threads(old)
    votes = np.stack([normalized, em, st], axis=1)
    if means.shape[0] != 2: raise ValueError('Binary majority only')
    return (votes.sum(1) >= 2).astype(int), {'votes': votes, 'batch_ids': groups}


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
