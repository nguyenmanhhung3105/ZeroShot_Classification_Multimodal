"""EM-Dirichlet zero-shot trên điểm CLIP; không nhận nhãn ground truth.

Viết từ phương trình (6), (9), (15) và Algorithm 1 của arXiv:2405.18437.
Đối chiếu implementation chính thức commit 9066ad371757c794e4b8255e852f837ac9a1d3b1.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import digamma, gammaln, polygamma, softmax, xlogy

EPS = 1e-15


def probability_features(scores, temperature=30.):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] < 2 or not np.isfinite(scores).all():
        raise ValueError('Điểm phải là ma trận hữu hạn với ít nhất hai lớp')
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('Temperature phải dương')
    if np.any(np.abs(scores) > 1 + 1e-6):
        raise ValueError('Đầu vào phải là điểm cosine, không phải logits đã scale')
    return softmax(temperature * scores, axis=-1)


def _log_density(log_query, alpha):
    constant = gammaln(alpha.sum(-1)) - gammaln(alpha).sum(-1)
    return constant[:, None, :] + np.einsum('bnf,bkf->bnk', log_query, alpha - 1)


def _objective(log_query, alpha, assignments, strength):
    likelihood = (assignments * _log_density(log_query, alpha)).sum((1, 2))
    entropy_term = xlogy(assignments, assignments).sum((1, 2))
    proportions = assignments.mean(1)
    partition_penalty = -xlogy(proportions, proportions).sum(-1)
    return -likelihood + entropy_term + strength * assignments.shape[1] * partition_penalty


def _estimate_alpha(alpha, log_mean, iterations):
    """MM-quadratic; nghiệm dương tính ổn định khi b > 0 để tránh triệt tiêu."""
    for step in range(iterations):
        psi = digamma(alpha + 1)
        curvature = np.full_like(alpha, polygamma(1, 1.))
        nonzero = alpha > 1e-11
        numerator = 2 * (psi * alpha - gammaln(alpha + 1))
        np.divide(np.abs(numerator), alpha**2, out=curvature, where=nonzero)
        if not np.isfinite(curvature).all() or np.any(curvature <= 0):
            raise FloatingPointError('Curvature MM không hợp lệ')
        b = psi - digamma(alpha.sum(-1))[..., None] - curvature * alpha - log_mean
        root = np.sqrt(b * b + 4 * curvature)
        updated = np.empty_like(alpha)
        positive = b > 0
        np.divide(2., root + b, out=updated, where=positive)
        np.divide(root - b, 2 * curvature, out=updated, where=~positive)
        if not np.isfinite(updated).all() or np.any(updated <= 0):
            raise FloatingPointError('Tham số Dirichlet không hợp lệ')
        stop = (step > 0 and step % 50 == 0 and
                np.sum((updated - alpha)**2) / np.sum(alpha**2) < 1e-11)
        alpha = updated
        if stop:
            return alpha, step + 1
    return alpha, iterations


def match_clusters(query, assignments):
    """Ghép các cụm đang có với lớp bằng Hungarian, chỉ dùng xác suất CLIP."""
    clusters = assignments.argmax(-1)
    predicted = np.empty_like(clusters)
    mappings = []
    for task in range(len(query)):
        # Giữ thứ tự xuất hiện đầu tiên như code tác giả (liên quan tie matching).
        occupied = list(dict.fromkeys(clusters[task].tolist()))
        prototypes = np.stack([query[task, clusters[task] == k].mean(0) for k in occupied])
        rows, classes = linear_sum_assignment(-prototypes)
        mapping = {occupied[r]: int(k) for r, k in zip(rows, classes)}
        predicted[task] = [mapping[int(k)] for k in clusters[task]]
        mappings.append(mapping)
    return predicted, mappings


def em_dirichlet(query, iterations=20, mm_iterations=1000):
    """Các task cùng kích thước: [tasks, images, classes]; không nhãn, không model.

    Lambda gốc = floor(K/5)*N; strength dưới đây là lambda/N.
    N=1 giữ argmax vì không có thông tin transductive, tránh fit cụm suy biến.
    """
    query = np.asarray(query, dtype=np.float64)
    if (query.ndim != 3 or min(query.shape) < 1 or query.shape[-1] < 2 or
            not np.isfinite(query).all() or np.any(query < 0) or np.any(query > 1) or
            not np.allclose(query.sum(-1), 1., atol=1e-10, rtol=0)):
        raise ValueError('Query phải là các vector xác suất hợp lệ')
    if type(iterations) is not int or iterations < 1 or type(mm_iterations) is not int or mm_iterations < 1:
        raise ValueError('Số vòng lặp phải là số nguyên dương')
    if query.shape[1] == 1:
        return query.argmax(-1), {'singleton_fallback': True, 'objective': [], 'mappings': []}, query.copy()
    tasks, count, classes = query.shape
    strength = classes // 5
    log_query = np.log(query + EPS)
    assignments = query.copy()
    alpha = np.ones((tasks, classes, classes), dtype=np.float64)
    history = [_objective(log_query, alpha, assignments, strength).tolist()]
    inner_steps = []
    for _ in range(iterations):
        sizes = assignments.sum(1)
        active = sizes > EPS
        log_mean = np.einsum('bnk,bnf->bkf', assignments, log_query) / np.maximum(sizes[..., None], EPS)
        log_mean = np.where(active[..., None], log_mean, -10.)
        updated, steps = _estimate_alpha(alpha, log_mean, mm_iterations)
        alpha = np.where(active[..., None], updated, alpha)
        proportions = assignments.mean(1)
        logits = _log_density(log_query, alpha) + strength * (np.log(proportions + EPS)[:, None, :] + 1)
        assignments = softmax(logits, axis=-1)
        if not np.isfinite(assignments).all():
            raise FloatingPointError('Assignment không hữu hạn')
        history.append(_objective(log_query, alpha, assignments, strength).tolist())
        inner_steps.append(steps)
    predicted, mappings = match_clusters(query, assignments)
    history = np.asarray(history)
    if not np.isfinite(history).all():
        raise FloatingPointError('Objective không hữu hạn')
    # MM phải giảm objective; dung sai tương đối chỉ dành cho sai số số học.
    rises = np.diff(history, axis=0)
    tolerance = 1e-8 * (1 + np.abs(history[:-1]))
    if np.any(rises > tolerance):
        raise FloatingPointError('Objective EM-MM tăng ngoài dung sai')
    return predicted, {'singleton_fallback': False, 'objective': history.tolist(),
                       'mappings': mappings, 'mm_steps': inner_steps,
                       'lambda_over_n': strength}, assignments


def predict_two_stage(info_scores, human_scores, info_labels, human_labels,
                      grouping_seed=0, query_size=75, temperature=30.,
                      iterations=20, mm_iterations=1000):
    from experiments.clip_two_stage.run import decide
    baseline, gate = decide(info_scores, human_scores, info_labels, human_labels)
    if type(query_size) is not int or query_size < 2:
        raise ValueError('query_size phải là số nguyên >= 2')
    if type(grouping_seed) is not int or not 0 <= grouping_seed < 2**32:
        raise ValueError('Grouping seed không hợp lệ')
    positive = [i for i, label in enumerate(human_labels) if label != 'not_humanitarian']
    # Dùng nguyên bank tám lớp đã sinh; chỉ giới hạn điểm ở tầng quyết định.
    features = probability_features(np.asarray(human_scores)[:, positive], temperature)
    chosen = np.random.default_rng(grouping_seed).permutation(np.flatnonzero(gate))
    batches = [chosen[start:start + query_size] for start in range(0, len(chosen), query_size)]
    predicted = np.asarray(baseline, dtype=object)
    batch_id = np.full(len(baseline), -1, dtype=int)
    diagnostics = []
    for size in sorted({len(batch) for batch in batches}, reverse=True):
        ids = [i for i, batch in enumerate(batches) if len(batch) == size]
        indices = np.stack([batches[i] for i in ids])
        assigned, log, _ = em_dirichlet(features[indices], iterations, mm_iterations)
        predicted[indices] = np.asarray(human_labels)[np.asarray(positive)[assigned]]
        batch_id[indices] = np.asarray(ids)[:, None]
        diagnostics.append({'batch_ids': ids, 'batch_size': size, **log})
    return predicted.tolist(), batch_id, diagnostics
