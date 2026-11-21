"""EM-Dirichlet: giữ công thức đã kiểm tra; không nhận nhãn thật."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import digamma, gammaln, polygamma, softmax, xlogy
EPS = 1e-15

def probability_features(scores, temperature=30.0):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] < 2 or (not np.isfinite(scores).all()):
        raise ValueError('Điểm phải là ma trận hữu hạn với ít nhất hai lớp')
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('Temperature phải dương')
    if np.any(np.abs(scores) > 1 + 1e-06):
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
        curvature = np.full_like(alpha, polygamma(1, 1.0))
        nonzero = alpha > 1e-11
        numerator = 2 * (psi * alpha - gammaln(alpha + 1))
        np.divide(np.abs(numerator), alpha ** 2, out=curvature, where=nonzero)
        if not np.isfinite(curvature).all() or np.any(curvature <= 0):
            raise FloatingPointError('Curvature MM không hợp lệ')
        b = psi - digamma(alpha.sum(-1))[..., None] - curvature * alpha - log_mean
        root = np.sqrt(b * b + 4 * curvature)
        updated = np.empty_like(alpha)
        positive = b > 0
        np.divide(2.0, root + b, out=updated, where=positive)
        np.divide(root - b, 2 * curvature, out=updated, where=~positive)
        if not np.isfinite(updated).all() or np.any(updated <= 0):
            raise FloatingPointError('Tham số Dirichlet không hợp lệ')
        stop = step > 0 and step % 50 == 0 and (np.sum((updated - alpha) ** 2) / np.sum(alpha ** 2) < 1e-11)
        alpha = updated
        if stop:
            return (alpha, step + 1)
    return (alpha, iterations)

def match_clusters(query, assignments):
    """Ghép các cụm đang có với lớp bằng Hungarian, chỉ dùng xác suất CLIP."""
    clusters = assignments.argmax(-1)
    predicted = np.empty_like(clusters)
    mappings = []
    for task in range(len(query)):
        occupied = list(dict.fromkeys(clusters[task].tolist()))
        prototypes = np.stack([query[task, clusters[task] == k].mean(0) for k in occupied])
        rows, classes = linear_sum_assignment(-prototypes)
        mapping = {occupied[r]: int(k) for r, k in zip(rows, classes)}
        predicted[task] = [mapping[int(k)] for k in clusters[task]]
        mappings.append(mapping)
    return (predicted, mappings)

def em_dirichlet(query, iterations=20, mm_iterations=1000):
    """Các task cùng kích thước: [tasks, images, classes]; không nhãn, không model.

    Lambda gốc = floor(K/5)*N; strength dưới đây là lambda/N.
    N=1 giữ argmax vì không có thông tin transductive, tránh fit cụm suy biến.
    """
    query = np.asarray(query, dtype=np.float64)
    if query.ndim != 3 or min(query.shape) < 1 or query.shape[-1] < 2 or (not np.isfinite(query).all()) or np.any(query < 0) or np.any(query > 1) or (not np.allclose(query.sum(-1), 1.0, atol=1e-10, rtol=0)):
        raise ValueError('Query phải là các vector xác suất hợp lệ')
    if type(iterations) is not int or iterations < 1 or type(mm_iterations) is not int or (mm_iterations < 1):
        raise ValueError('Số vòng lặp phải là số nguyên dương')
    if query.shape[1] == 1:
        return (query.argmax(-1), {'singleton_fallback': True, 'objective': [], 'mappings': []}, query.copy())
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
        log_mean = np.where(active[..., None], log_mean, -10.0)
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
    rises = np.diff(history, axis=0)
    tolerance = 1e-08 * (1 + np.abs(history[:-1]))
    if np.any(rises > tolerance):
        raise FloatingPointError('Objective EM-MM tăng ngoài dung sai')
    return (predicted, {'singleton_fallback': False, 'objective': history.tolist(), 'mappings': mappings, 'mm_steps': inner_steps, 'lambda_over_n': strength}, assignments)
import torch

def stata(features, prototypes, max_iter=10, neighbors=3):
    """features [N,D], prototypes [K,D], đều unit; trả y_hat và z trên CPU.

    Giữ mặc định alpha=1, hard beta, lambda_y_hat=lambda_laplacian=1,
    tau=100, inner iterations=5. Nhóm phải có nhiều hơn neighbors ảnh.
    """
    x = torch.as_tensor(features, dtype=torch.float32, device='cpu')
    t = torch.as_tensor(prototypes, dtype=torch.float32, device='cpu')
    if x.ndim != 2 or t.ndim != 2 or x.shape[1] != t.shape[1] or (t.shape[0] < 2) or (x.shape[1] < 2) or (not torch.isfinite(x).all()) or (not torch.isfinite(t).all()):
        raise ValueError('Sai shape hoặc embedding không hữu hạn')
    if type(neighbors) is not int or neighbors < 1 or len(x) <= neighbors:
        raise ValueError('Nhóm cần nhiều ảnh hơn số láng giềng')
    if type(max_iter) is not int or max_iter < 0:
        raise ValueError('max_iter phải là số nguyên không âm')
    if not torch.allclose(x.norm(dim=1), torch.ones(len(x)), atol=1e-05) or not torch.allclose(t.norm(dim=1), torch.ones(len(t)), atol=1e-05):
        raise ValueError('StatA cần embedding và prototype unit')
    with torch.no_grad():
        prior = (100 * x @ t.T).softmax(1)
        z = prior.clone()
        mu = t.clone()
        init_cov = (z[:, :, None] * (x[:, None, :] - t[None, :, :]).square()).sum((0, 1)) / len(x)
        covariance = init_cov[None].repeat(len(t), 1)
        covariance = covariance.clamp_min(1e-12)
        init_cov = init_cov.clamp_min(1e-12)
        affinity = x @ x.T
        search = affinity.clone()
        search.fill_diagonal_(-float('inf'))
        indices = search.topk(neighbors, dim=1).indices
        weights = torch.zeros_like(affinity)
        weights.scatter_(1, indices, affinity.gather(1, indices))
        for iteration in range(max_iter + 1):
            likelihood = -0.5 * ((x[:, None, :] - mu[None, :, :]).square() / covariance[None, :, :]).sum(-1)
            for _ in range(5):
                logits = likelihood + 50 / (neighbors * 2) * (weights.T @ z + weights @ z)
                logits -= 0.5 * covariance.log().sum(-1)[None, :]
                logits -= logits.max(1, keepdim=True).values
                unnormalized = prior * (logits / 50).exp()
                z = unnormalized / unnormalized.sum(1, keepdim=True).clamp_min(1e-30)
            if iteration == max_iter:
                break
            counts = torch.bincount(z.argmax(1), minlength=len(t))
            beta = counts / (1 + counts + 1e-12)
            mass = z.sum(0).clamp_min(1e-12)
            empirical_mu = z.T @ x / mass[:, None]
            empirical_mu /= empirical_mu.norm(dim=1, keepdim=True).clamp_min(1e-12)
            mu = beta[:, None] * empirical_mu + (1 - beta)[:, None] * t
            mu /= mu.norm(dim=1, keepdim=True).clamp_min(1e-12)
            covariance = (z[:, :, None] * (x[:, None, :] - mu[None, :, :]).square()).sum(0) / mass[:, None]
            covariance = beta[:, None] * covariance + (1 - beta)[:, None] * (init_cov[None, :] + (t - mu).square())
            covariance = covariance.clamp_min(1e-12)
        if not torch.isfinite(z).all() or not torch.allclose(z.sum(1), torch.ones(len(x)), atol=1e-05):
            raise FloatingPointError('StatA assignment không hợp lệ')
        return (prior.numpy(), z.numpy())
