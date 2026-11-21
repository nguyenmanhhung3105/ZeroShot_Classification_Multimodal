"""Frozen regression reference; extracted verbatim from legacy few-shot runner."""
import numpy as np
from sklearn.linear_model import LogisticRegression


def select_support(y, shots, seed):
    rng = np.random.default_rng(seed)
    chosen = []
    for label in range(6):
        indices = np.flatnonzero(y == label)
        if len(indices) < shots:
            raise ValueError(f'Train class {label}: {len(indices)} samples < {shots}-shot')
        chosen.extend(rng.permutation(indices)[:shots])
    return np.asarray(chosen)


def probe(x, y):
    # Fixed hyperparameters: no tuning on dev. Same C and class weighting throughout.
    return LogisticRegression(C=1.0, class_weight='balanced', solver='lbfgs',
                              max_iter=2000, tol=1e-6).fit(x, y)
