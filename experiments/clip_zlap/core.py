"""ZLaP transductive, NumPy/SciPy CPU; không nhận nhãn thật.

Theo Eq. (3), (10) của CVPR 2024 và repo chính thức, commit bên dưới.
Giữ k gồm cả self khi tìm hàng xóm ảnh, sau đó xóa đường chéo như tác giả.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg

REFERENCE_COMMIT = '89ab14bef8904a9397f8b6a7e5f3e6af2c288b93'
PARAMETERS = {'k': 5, 'gamma': 5., 'alpha': .3, 'tol': 1e-6, 'maxiter': 50}


def unit_vectors(values):
    x = np.asarray(values, dtype=np.float32)
    if x.ndim != 2 or min(x.shape) < 1 or not np.isfinite(x).all():
        raise ValueError('Cần ma trận embedding hữu hạn, không rỗng')
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norms < 1e-12):
        raise ValueError('Embedding zero')
    return x / norms


def nearest(reference, query, k, block_size=256):
    """Tìm kNN cosine chính xác; tie chọn index nhỏ, không cấp phát N*N."""
    indices = []; similarities = []; ties = 0
    count = min(k, len(reference))
    for start in range(0, len(query), block_size):
        scores = query[start:start + block_size] @ reference.T
        order = np.argsort(-scores, axis=1, kind='stable')
        if count < len(reference):
            ties += int(np.sum(np.take_along_axis(scores, order[:, count-1:count], 1) ==
                               np.take_along_axis(scores, order[:, count:count+1], 1)))
        top = order[:, :count]
        indices.append(top); similarities.append(np.take_along_axis(scores, top, 1))
    return np.concatenate(indices), np.concatenate(similarities), ties


def build_graph(features, prototypes, k=5, gamma=5., alpha=.3):
    if type(k) is not int or k < 1:
        raise ValueError('k phải là số nguyên dương')
    if not np.isfinite(gamma) or gamma <= 0 or not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError('gamma > 0, 0 < alpha < 1')
    x, t = unit_vectors(features), unit_vectors(prototypes)
    if x.shape[1] != t.shape[1] or len(t) < 2:
        raise ValueError('Cần ít nhất hai prototype cùng chiều với ảnh')
    n, c = len(x), len(t)
    image_ids, image_sim, image_ties = nearest(x, x, k)
    text_ids, text_sim, text_ties = nearest(t, x, k)
    # Chỉ lũy thừa cạnh ảnh-text; cạnh ảnh-ảnh giữ cosine không âm.
    columns = np.concatenate([image_ids + c, text_ids], axis=1)
    weights = np.concatenate([np.maximum(image_sim, 0), np.maximum(text_sim, 0) ** gamma], axis=1)
    rows = np.repeat(np.arange(c, c+n), columns.shape[1])
    directed = sparse.csr_matrix((weights.ravel(), (rows, columns.ravel())), shape=(c+n, c+n))
    symmetric = directed + directed.T
    symmetric.setdiag(0); symmetric.eliminate_zeros()
    degree = np.asarray(symmetric.sum(axis=1)).ravel()
    inverse = np.zeros_like(degree)
    np.divide(1., np.sqrt(degree), out=inverse, where=degree > 0)
    scaling = sparse.diags(inverse)
    normalized = scaling @ symmetric @ scaling
    laplacian = sparse.eye(c+n, format='csr', dtype=np.float64) - alpha * normalized
    diagnostics = {'images': n, 'classes': c, 'image_knn_includes_self': True,
                   'image_k': min(k, n), 'text_k': min(k, c), 'directed_edges': int(directed.nnz),
                   'symmetric_edges_without_self': int(symmetric.nnz),
                   'zero_degree_nodes': int(np.sum(degree == 0)),
                   'image_knn_boundary_ties': image_ties, 'text_knn_boundary_ties': text_ties,
                   'tie_policy': 'stable smaller index in current cohort order'}
    return laplacian, diagnostics


def predict(features, prototypes, k=5, gamma=5., alpha=.3, tol=1e-6, maxiter=50):
    if not np.isfinite(tol) or tol <= 0 or type(maxiter) is not int or maxiter < 1:
        raise ValueError('Sai điều kiện dừng CG')
    laplacian, diagnostics = build_graph(features, prototypes, k, gamma, alpha)
    n, c = diagnostics['images'], diagnostics['classes']
    scores = np.empty((n, c)); residuals = []
    for label in range(c):
        seed = np.zeros(n+c); seed[label] = 1.
        solution, info = cg(laplacian, seed, rtol=tol, atol=0., maxiter=maxiter)
        residual = float(np.linalg.norm(laplacian @ solution - seed))
        if info != 0 or not np.isfinite(solution).all() or residual > tol * 1.01:
            raise FloatingPointError(f'CG không hội tụ: class={label}, status={info}, residual={residual}')
        scores[:, label] = solution[c:]; residuals.append(residual)
    diagnostics.update(cg_residuals=residuals, all_cg_converged=True,
                       zero_score_rows=int(np.sum(np.max(scores, axis=1) <= 0)))
    # Giữ quy tắc argmax đầu tiên như tác giả, kể cả hàng không có đường tới text.
    return scores, diagnostics
