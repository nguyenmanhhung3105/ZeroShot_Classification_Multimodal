"""CALIP theo công thức đã audit ở archive; extractor OpenCLIP 3.3.0 ViT."""
import torch
import torch.nn.functional as F

REFERENCE_COMMIT = '83e21b6b985934dce6e8414c26300671b4825841'
REFERENCE_URL = f'https://github.com/ZiyuGuo99/CALIP/blob/{REFERENCE_COMMIT}/run_imagenet.py'
PARAMETERS = {'attention_scale':2., 'beta':[1.,1.12,.02], 'logit_scale':100.}


@torch.no_grad()
def score(global_features, patches, prototypes):
    g,p,t=[torch.as_tensor(v,dtype=torch.float32) for v in [global_features,patches,prototypes]]
    if (g.ndim!=2 or p.ndim!=3 or t.ndim!=2 or len(t)<2 or p.shape[0]!=len(g)
            or p.shape[2]!=g.shape[1] or t.shape[1]!=g.shape[1] or min(p.shape)<1
            or not (g.device==p.device==t.device)):
        raise ValueError('Sai shape/device global/patch/text CALIP')
    for v in [g,p,t]:
        if not torch.isfinite(v).all() or not torch.allclose(v.norm(dim=-1),torch.ones_like(v[...,0]),atol=1e-5,rtol=0):
            raise ValueError('CALIP cần vector hữu hạn, unit')
    terms=[]
    for i,patch in enumerate(p):
        spatial=patch.T
        attention=(spatial.T@t.T)*2
        text_a=spatial@attention.softmax(dim=0)
        visual_a=attention.softmax(dim=1)@t
        visual_a=visual_a.mean(0)+visual_a.max(0).values
        terms.append(torch.stack([100.*g[i]@t.T,100.*g[i]@text_a,100.*visual_a@t.T]))
    terms=torch.stack(terms)
    return terms[:,0]+1.12*terms[:,1]+.02*terms[:,2],terms


@torch.no_grad()
def encode(backend, images, verify_global=False):
    """Lấy patch đã project; chỉ thêm một forward tham chiếu ở batch đầu."""
    model=backend.model;visual=model.visual
    if (model.training or visual.training or getattr(visual,'pool_type',None)!='tok'
            or getattr(visual,'attn_pool',None) is not None or not hasattr(visual,'output_tokens')
            or visual.output_tokens or not hasattr(visual,'ln_post')):
        raise ValueError('Extractor cần OpenCLIP ViT eval, CLS pooling và output_tokens=False')
    batch=torch.stack([backend.preprocess(im) for im in images]).to(backend.device)
    original=visual.output_tokens
    try:
        visual.output_tokens=True
        pooled,tokens=visual(batch)
    finally:
        visual.output_tokens=original
    if getattr(visual,'final_ln_after_pool',False):tokens=visual.ln_post(tokens)
    if visual.proj is not None:
        if not isinstance(visual.proj,torch.Tensor) or visual.proj.ndim!=2:
            raise ValueError('Chỉ hỗ trợ projection matrix chuẩn của OpenCLIP ViT')
        tokens=tokens@visual.proj
    g,p=F.normalize(pooled.float(),dim=-1),F.normalize(tokens.float(),dim=-1)
    grid=tuple(int(v) for v in visual.grid_size)
    if p.ndim!=3 or p.shape!=(len(images),grid[0]*grid[1],g.shape[1]):
        raise ValueError('Patch không khớp grid/projection')
    if not torch.isfinite(g).all() or not torch.isfinite(p).all() or (tokens.norm(dim=-1)==0).any():
        raise ValueError('Feature rỗng/không hữu hạn')
    error=None
    if verify_global:
        expected=F.normalize(model.encode_image(batch).float(),dim=-1)
        torch.testing.assert_close(g,expected,atol=1e-6,rtol=1e-5)
        error=float(abs(g-expected).max())
    return g,p,{'grid':list(grid),'patches':p.shape[1],'embedding_dim':g.shape[1],
                'global_reference_max_abs_error':error,'same_encoder_forward':True,
                'extra_reference_forward':bool(verify_global),'output_tokens_restored':visual.output_tokens==original}
