"""InMaP: tối ưu class proxy với nhãn giả, không nhận ground truth.

Theo Eq. (2)-(4) và image_opt/sinkhorn của idstcv/InMaP, NeurIPS 2023.
Backbone frozen; có tối ưu vector classifier bằng projected gradient.
"""
import numpy as np
import torch
import torch.nn.functional as F

REFERENCE_COMMIT='5a6f7c94fac040401a5afdf06adc0a31b734534e'

PARAMETERS = {'iterations':2000, 'sinkhorn_iterations':20, 'lr':10.,
              'tau_t':.01, 'tau_i':.04, 'alpha':.6, 'gamma':0.}


def _matrix(value):
    x = torch.as_tensor(value, dtype=torch.float32, device='cpu').clone()
    if x.ndim != 2 or min(x.shape)<1 or not torch.isfinite(x).all():
        raise ValueError('Cần ma trận embedding hữu hạn không rỗng')
    if not torch.allclose(x.norm(dim=1), torch.ones(len(x)), atol=1e-5, rtol=0):
        raise ValueError('Cần embedding/prototype đã chuẩn hóa unit')
    return x


def refine(logits, tau_t=.01, gamma=0., iterations=20):
    if tau_t<=0 or not np.isfinite(tau_t) or not 0<=gamma<=1 or type(iterations) is not int or iterations<1:
        raise ValueError('Sai cấu hình Sinkhorn')
    p = F.softmax(logits/tau_t, dim=1)
    n, c = p.shape
    p = p/n
    target = p.sum(0, keepdim=True).pow(gamma) if gamma>0 else torch.ones((1,c),dtype=p.dtype)/c
    target = target/target.sum()
    for _ in range(iterations):
        mass = p.sum(0, keepdim=True)
        if torch.any(mass<=0): raise FloatingPointError('Sinkhorn có lớp zero mass')
        p = (p/mass)*target
        p = p/p.sum(1,keepdim=True)/n
    return p*n


def optimize(features, prototypes, balanced=False, iterations=2000, sinkhorn_iterations=20,
             lr=10., tau_t=.01, tau_i=.04, alpha=.6, gamma=0.):
    if (type(balanced) is not bool or type(iterations) is not int or iterations<1 or
            not all(np.isfinite(v) and v>0 for v in [lr,tau_t,tau_i]) or not 0<=alpha<=1):
        raise ValueError('Sai cấu hình tối ưu proxy')
    x, t = _matrix(features), _matrix(prototypes)
    if len(t)<2 or x.shape[1]!=t.shape[1]: raise ValueError('Sai chiều hoặc số lớp')
    previous_threads=torch.get_num_threads();torch.set_num_threads(1)
    try:
        with torch.no_grad():
            # Classifier tác giả được stack thành [D,C] contiguous. Giữ cùng
            # layout để tránh đổi accumulation FP32 ở bước tạo nhãn giả.
            initial_classifier=t.T.contiguous()
            logits=x@initial_classifier
            pseudo=refine(logits,tau_t,gamma,sinkhorn_iterations) if balanced else F.softmax(logits/tau_t,dim=1)
            initial_mass=pseudo.mean(0).tolist()
            confidence, labels=pseudo.max(1);hard=confidence>alpha
            pseudo[hard]=F.one_hot(labels[hard],len(t)).to(pseudo.dtype)
            target=x.T@pseudo;classifier=initial_classifier.clone()
            last_norm=float('inf');decays=0;history=[]
            for step in range(iterations):
                probabilities=F.softmax(x@classifier/tau_i,dim=1)
                gradient=x.T@probabilities-target
                norm=gradient.norm()
                if not torch.isfinite(norm):raise FloatingPointError('Gradient proxy không hữu hạn')
                if norm>last_norm:lr/=2.;decays+=1
                last_norm=norm
                classifier-=lr/(len(x)*tau_i)*gradient
                classifier=F.normalize(classifier,dim=0)
                if step in [0,iterations-1] or (step+1)%250==0:
                    loss=-(pseudo*F.log_softmax(x@classifier/tau_i,dim=1)).sum(1).mean()
                    history.append({'iteration':step+1,'pseudo_cross_entropy':float(loss),'lr':lr})
            if not torch.isfinite(classifier).all():raise FloatingPointError('Proxy không hữu hạn')
            scores=(x@classifier).numpy()
            diagnostics={'backbone_frozen':True,'ground_truth_fit':False,'pseudo_label_optimization':True,
                         'balanced_sinkhorn':balanced,'gamma':gamma if balanced else None,
                         'initial_pseudo_mass':initial_mass,'hardened_samples':int(hard.sum()),
                         'hardened_pseudo_mass':pseudo.mean(0).tolist(),'lr_decays':decays,'final_lr':lr,
                         'iterations':iterations,'history':history}
            return scores,classifier.T.numpy().copy(),diagnostics
    finally:torch.set_num_threads(previous_threads)
