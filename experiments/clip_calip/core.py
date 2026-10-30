"""CALIP get_logits: attention *2, mean+max pooling, hệ số ImageNet cố định."""
import torch

REFERENCE_COMMIT = '83e21b6b985934dce6e8414c26300671b4825841'
PARAMETERS = {'attention_scale': 2., 'beta': [1., 1.12, .02], 'logit_scale': 100.}


@torch.no_grad()
def score(global_features, patches, prototypes):
    g, p, t = [torch.as_tensor(v, dtype=torch.float32) for v in [global_features, patches, prototypes]]
    if (g.ndim != 2 or p.ndim != 3 or t.ndim != 2 or len(t) < 2 or
            p.shape[0] != len(g) or p.shape[2] != g.shape[1] or t.shape[1] != g.shape[1] or
            min(p.shape) < 1 or not (g.device == p.device == t.device)):
        raise ValueError('Sai shape/device global/patch/text CALIP')
    for v in [g, p, t]:
        if not torch.isfinite(v).all() or not torch.allclose(v.norm(dim=-1), torch.ones_like(v[..., 0]), atol=1e-5, rtol=0):
            raise ValueError('CALIP cần vector hữu hạn, unit')
    terms, attention = [], []
    for i, patch in enumerate(p):
        # Giữ chiều và phép toán get_logits của tác giả, không normalize lại
        # feature đã tương tác trước khi tính hai term mới.
        spatial = patch.T
        weight = (spatial.T @ t.T) * 2
        wt = weight.softmax(dim=0)
        wv = weight.softmax(dim=1)
        text_a = spatial @ wt
        visual_a = wv @ t
        visual_a = visual_a.mean(0) + visual_a.max(0).values
        terms.append(torch.stack([100.*g[i] @ t.T, 100.*g[i] @ text_a, 100.*visual_a @ t.T]))
        attention.append(wt)
    terms = torch.stack(terms)
    logits = terms[:, 0] + terms[:, 1]*1.12 + terms[:, 2]*.02
    return logits, terms, torch.stack(attention)
