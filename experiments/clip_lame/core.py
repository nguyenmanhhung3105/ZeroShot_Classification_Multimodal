"""LAME: hiệu chỉnh posterior trên đồ thị ảnh, giữ model frozen."""
import numpy as np
import torch
import torch.nn.functional as F

REFERENCE_COMMIT='d2e5f63090bc1c8129bf7cbd781029a5955e1a67'
PARAMETERS={'knn':5,'bound_lambda':1.,'max_steps':100,'logit_scale':100.,'force_symmetry':False}


def adapt(features,cosine_scores,affinity='rbf',knn=5,bound_lambda=1.,max_steps=100,logit_scale=100.,force_symmetry=False):
    if affinity not in ['rbf','knn'] or type(knn) is not int or knn<1 or type(max_steps) is not int or max_steps<1:
        raise ValueError('Sai kernel/số bước LAME')
    if not np.isfinite(bound_lambda) or bound_lambda<0 or not np.isfinite(logit_scale) or logit_scale<=0 or type(force_symmetry) is not bool:
        raise ValueError('Sai lambda/scale/symmetry')
    x=torch.as_tensor(features,dtype=torch.float32,device='cpu')
    scores=torch.as_tensor(cosine_scores,dtype=torch.float32,device='cpu')
    if (x.ndim!=2 or scores.ndim!=2 or len(x)!=len(scores) or len(x)==0 or scores.shape[1]<2 or
            not torch.isfinite(x).all() or not torch.isfinite(scores).all() or torch.any(x.norm(dim=1)<1e-12)):
        raise ValueError('Sai feature hoặc score')
    with torch.no_grad():
        x=F.normalize(x,p=2,dim=-1)
        prior=(logit_scale*scores).softmax(-1)
        if len(x)==1:
            return prior.numpy(),{'singleton_prior_fallback':True,'iterations':0,'affinity':affinity}
        # Cùng cách tính Euclidean distance và thứ tự topk như tác giả.
        distance=torch.norm(x.unsqueeze(0)-x.unsqueeze(1),dim=-1,p=2)
        if affinity=='knn':
            nearest=distance.topk(min(knn+1,len(x)),-1,largest=False).indices[:,1:]
            kernel=torch.zeros((len(x),len(x)));kernel.scatter_(-1,nearest,1.)
            sigma=None
        else:
            sigma=distance.topk(min(knn,len(x)),dim=-1,largest=False).values[:,-1].mean()
            if sigma<=0:
                return prior.numpy(),{'degenerate_prior_fallback':True,'iterations':0,'affinity':affinity}
            kernel=torch.exp(-distance.square()/(2*sigma.square()))
        if force_symmetry:kernel=.5*(kernel+kernel.T)
        unary=-torch.log(prior+1e-10);posterior=(-unary).softmax(-1)
        previous=float('inf');energies=[]
        for step in range(max_steps):
            pairwise=bound_lambda*(kernel@posterior)
            posterior=(-unary+pairwise).softmax(-1)
            energy=float((unary*posterior-bound_lambda*pairwise*posterior+posterior*torch.log(posterior.clip(1e-20))).sum())
            if not np.isfinite(energy) or not torch.isfinite(posterior).all():raise FloatingPointError('LAME không hữu hạn')
            energies.append(energy)
            if step>1 and abs(energy-previous)<=1e-8*abs(previous):break
            previous=energy
        return posterior.numpy(),{'affinity':affinity,'iterations':len(energies),'energies':energies,
            'sigma':float(sigma) if sigma is not None else None,'nonzero_diagonal':int((kernel.diagonal()!=0).sum()),
            'posterior_mass':posterior.mean(0).tolist(),'singleton_prior_fallback':False}


def batches(features,scores,seed,affinity='rbf',query_size=75):
    if type(seed) is not int or not 0<=seed<2**32 or type(query_size) is not int or query_size<2:raise ValueError('Sai seed/query size')
    previous=torch.get_num_threads();torch.set_num_threads(1)
    try:
        x=np.asarray(features);s=np.asarray(scores);order=np.random.default_rng(seed).permutation(len(x))
        result=np.empty(s.shape,dtype='float32');ids=np.full(len(x),-1,dtype=int);logs=[]
        for number,start in enumerate(range(0,len(x),query_size)):
            indices=order[start:start+query_size]
            result[indices],log=adapt(x[indices],s[indices],affinity,**PARAMETERS)
            ids[indices]=number;logs.append(log)
        return result,ids,logs
    finally:torch.set_num_threads(previous)
