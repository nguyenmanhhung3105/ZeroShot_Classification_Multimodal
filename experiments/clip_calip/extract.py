"""Lấy global và projected patch từ cùng forward OpenCLIP ViT, giữ model."""
import torch
import torch.nn.functional as F


@torch.no_grad()
def encode(backend, images, verify_global=False):
    model = backend.model; visual = model.visual
    if (model.training or visual.training or getattr(visual, 'pool_type', None) != 'tok' or
            getattr(visual, 'attn_pool', None) is not None or not hasattr(visual, 'output_tokens') or
            visual.output_tokens or not hasattr(visual, 'ln_post')):
        raise ValueError('CALIP extractor chỉ OpenCLIP ViT eval, CLS pooling, không attention pooling/output_tokens sẵn có')
    batch = torch.stack([backend.preprocess(im) for im in images]).to(backend.device)
    original = visual.output_tokens
    try:
        visual.output_tokens = True
        pooled, tokens = visual(batch)
    finally:
        visual.output_tokens = original
    if getattr(visual, 'final_ln_after_pool', False):
        tokens = visual.ln_post(tokens)
    projection = visual.proj
    if projection is not None:
        if not isinstance(projection, torch.Tensor) or projection.ndim != 2:
            raise ValueError('Chỉ hỗ trợ projection matrix chuẩn của OpenCLIP ViT')
        tokens = tokens @ projection
    g, p = F.normalize(pooled.float(), dim=-1), F.normalize(tokens.float(), dim=-1)
    grid = tuple(int(v) for v in visual.grid_size)
    if p.ndim != 3 or p.shape != (len(images), grid[0]*grid[1], g.shape[1]):
        raise ValueError('Patch không khớp grid/projection của model')
    if not torch.isfinite(g).all() or not torch.isfinite(p).all() or (tokens.norm(dim=-1) == 0).any():
        raise ValueError('Feature rỗng/không hữu hạn')
    error = None
    if verify_global:
        expected = F.normalize(model.encode_image(batch).float(), dim=-1)
        torch.testing.assert_close(g, expected, atol=1e-6, rtol=1e-5)
        error = float(abs(g-expected).max())
    return g, p, {'grid': list(grid), 'patches': p.shape[1], 'embedding_dim': g.shape[1],
                  'global_reference_max_abs_error': error, 'same_encoder_forward': True,
                  'extra_reference_forward': bool(verify_global), 'output_tokens_restored': visual.output_tokens == original}
