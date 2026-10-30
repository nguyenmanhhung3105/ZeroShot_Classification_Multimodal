"""StatA batch adaptation từ phương trình paper; không nhận ground truth.

Paper: arXiv:2501.03729. Mã đối chiếu: MaxZanella/StatA, commit bên dưới.
Chỉ chạy trong embedding space; backbone CLIP được giữ frozen.
"""
import numpy as np
import torch

REFERENCE_COMMIT = '2635f9355f9389756064becb5d9ed54c2756809c'


def stata(features, prototypes, max_iter=10, neighbors=3):
    """features [N,D], prototypes [K,D], đều unit; trả y_hat và z trên CPU.

    Giữ mặc định alpha=1, hard beta, lambda_y_hat=lambda_laplacian=1,
    tau=100, inner iterations=5. Nhóm phải có nhiều hơn neighbors ảnh.
    """
    x = torch.as_tensor(features, dtype=torch.float32, device='cpu')
    t = torch.as_tensor(prototypes, dtype=torch.float32, device='cpu')
    if (x.ndim != 2 or t.ndim != 2 or x.shape[1] != t.shape[1] or t.shape[0] < 2 or
            x.shape[1] < 2 or not torch.isfinite(x).all() or not torch.isfinite(t).all()):
        raise ValueError('Sai shape hoặc embedding không hữu hạn')
    if type(neighbors) is not int or neighbors < 1 or len(x) <= neighbors:
        raise ValueError('Nhóm cần nhiều ảnh hơn số láng giềng')
    if type(max_iter) is not int or max_iter < 0:
        raise ValueError('max_iter phải là số nguyên không âm')
    if not torch.allclose(x.norm(dim=1), torch.ones(len(x)), atol=1e-5) or not torch.allclose(t.norm(dim=1), torch.ones(len(t)), atol=1e-5):
        raise ValueError('StatA cần embedding và prototype unit')
    with torch.no_grad():
        prior = (100 * x @ t.T).softmax(1)
        z = prior.clone()
        mu = t.clone()
        init_cov = (z[:, :, None] * (x[:, None, :] - t[None, :, :]).square()).sum((0, 1)) / len(x)
        covariance = init_cov[None].repeat(len(t), 1)
        # Chặn giá trị suy biến; ghi nhận trong metadata ở adapter.
        covariance = covariance.clamp_min(1e-12)
        init_cov = init_cov.clamp_min(1e-12)
        affinity = x @ x.T
        search = affinity.clone(); search.fill_diagonal_(-float('inf'))
        indices = search.topk(neighbors, dim=1).indices
        weights = torch.zeros_like(affinity)
        weights.scatter_(1, indices, affinity.gather(1, indices))
        for iteration in range(max_iter + 1):
            likelihood = -.5 * ((x[:, None, :] - mu[None, :, :]).square() / covariance[None, :, :]).sum(-1)
            for _ in range(5):
                logits = likelihood + (50 / (neighbors * 2)) * (weights.T @ z + weights @ z)
                logits -= .5 * covariance.log().sum(-1)[None, :]
                logits -= logits.max(1, keepdim=True).values
                unnormalized = prior * (logits / 50).exp()
                z = unnormalized / unnormalized.sum(1, keepdim=True).clamp_min(1e-30)
            if iteration == max_iter: break
            counts = torch.bincount(z.argmax(1), minlength=len(t))
            beta = counts / (1 + counts + 1e-12)
            mass = z.sum(0).clamp_min(1e-12)
            empirical_mu = (z.T @ x) / mass[:, None]
            empirical_mu /= empirical_mu.norm(dim=1, keepdim=True).clamp_min(1e-12)
            mu = beta[:, None] * empirical_mu + (1 - beta)[:, None] * t
            mu /= mu.norm(dim=1, keepdim=True).clamp_min(1e-12)
            covariance = (z[:, :, None] * (x[:, None, :] - mu[None, :, :]).square()).sum(0) / mass[:, None]
            covariance = beta[:, None] * covariance + (1 - beta)[:, None] * (init_cov[None, :] + (t - mu).square())
            covariance = covariance.clamp_min(1e-12)
        if not torch.isfinite(z).all() or not torch.allclose(z.sum(1), torch.ones(len(x)), atol=1e-5):
            raise FloatingPointError('StatA assignment không hợp lệ')
        return prior.numpy(), z.numpy()


def encode_prototypes(backend, bank, labels, batch_size):
    means = []
    for label in labels:
        blocks = [backend.encode_texts(bank[label][start:start + batch_size]).detach().float().cpu().numpy()
                  for start in range(0, len(bank[label]), batch_size)]
        vectors = np.concatenate(blocks)
        vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
        means.append(vectors.mean(0))
    return np.asarray(means)


def gate_ablation(features, mean_prototypes, info_scores, human_scores, info_labels, human_labels):
    """Ba grouping seeds cố định; control chuẩn hóa để tách tác động aggregation."""
    features = np.asarray(features, dtype=np.float32)
    mean_prototypes = np.asarray(mean_prototypes, dtype=np.float32)
    if not np.allclose(features @ mean_prototypes.T, info_scores, atol=2e-5, rtol=1e-4):
        raise ValueError('Prototype hoặc embedding khác score informativeness gốc')
    norms = np.linalg.norm(mean_prototypes, axis=1, keepdims=True)
    if np.any(norms <= 1e-12): raise ValueError('Prototype bằng zero')
    prototypes = mean_prototypes / norms
    from experiments.clip_two_stage.run import decide
    normalized_scores = features @ prototypes.T
    baseline, _ = decide(normalized_scores, human_scores, info_labels, human_labels)
    positive = [i for i, label in enumerate(human_labels) if label != 'not_humanitarian']
    detail = np.asarray(human_labels)[np.asarray(positive)[np.asarray(human_scores)[:, positive].argmax(1)]]
    predictions = {'normalized_gate_control': baseline}
    gate_labels = {}
    grouping = {}
    probabilities = {}
    old_threads = torch.get_num_threads(); torch.set_num_threads(1)
    try:
        for seed in [0, 1, 2]:
            order = np.random.default_rng(seed).permutation(len(features))
            batches = [order[start:start + 75] for start in range(0, len(order), 75)]
            posterior = np.empty((len(features), len(info_labels)), dtype=np.float32)
            batch_ids = np.full(len(features), -1, dtype=int)
            for batch_id, indices in enumerate(batches):
                if len(indices) <= 3:
                    # Không đủ kNN: giữ prior, ghi rõ fallback.
                    raw = 100 * torch.from_numpy(normalized_scores[indices])
                    posterior[indices] = raw.softmax(1).numpy()
                else:
                    _, posterior[indices] = stata(features[indices], prototypes)
                batch_ids[indices] = batch_id
            binary = np.asarray(info_labels)[posterior.argmax(1)]
            pred = detail.copy(); pred[binary != 'informative'] = 'not_humanitarian'
            key = f'stata_gate_seed_{seed}'
            predictions[key] = pred.tolist(); gate_labels[key] = binary.tolist()
            grouping[key] = batch_ids.tolist(); probabilities[key] = posterior.tolist()
    finally:
        torch.set_num_threads(old_threads)
    return predictions, {'reference_commit': REFERENCE_COMMIT,
                         'grouping_seeds': [0, 1, 2], 'query_size': 75,
                         'alpha': 1, 'soft_beta': False, 'lambda_y_hat': 1, 'lambda_laplacian': 1,
                         'max_iter': 10, 'inner_iterations': 5, 'n_neighbors': 3, 'logit_scale': 100,
                         'normalization': 'unit_image_features_and_unit_mean_text_prototypes',
                         'adaptations': ['Explicit self-neighbor exclusion for duplicate/tied embeddings.',
                                         'Covariance/mass floors for numerical stability.',
                                         'Groups <=3 retain normalized CLIP prior; no such group for 2219 samples.'],
                         'gate_labels': gate_labels, 'batch_ids': grouping, 'posterior': probabilities}
