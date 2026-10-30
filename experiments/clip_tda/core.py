"""TDA (CVPR 2024) single-view stream; CLIP frozen, cache chỉ dùng pseudo-label.

Viết từ công thức paper, đối chiếu kdiAAA/TDA ở commit ghim bên dưới.
Giữ cấu hình ImageNet chính thức; không chọn tham số trên CrisisMMD.
"""
import math
import numpy as np
import torch
import torch.nn.functional as F

REFERENCE_COMMIT='e697fb0c8078cdeff93daa56bcf8860702542069'
PARAMETERS={'positive':{'shot_capacity':3,'alpha':2.,'beta':5.},
            'negative':{'shot_capacity':2,'alpha':.117,'beta':1.,
                        'entropy_lower':.2,'entropy_upper':.5,'mask_lower':.03,'mask_upper':1.}}


def _put(cache,label,item,capacity):
    queue=cache.setdefault(label,[])
    if len(queue)<capacity:queue.append(item)
    elif item['loss']<queue[-1]['loss']:queue[-1]=item
    queue.sort(key=lambda entry:entry['loss'])


def _cache_logits(feature,cache,classes,parameters,negative=False):
    items=[(label,item) for label in sorted(cache) for item in cache[label]]
    if not items:return torch.zeros((1,classes),dtype=torch.float32)
    keys=torch.cat([item['feature'] for _,item in items],0).T
    if negative:
        probs=torch.cat([item['probability'] for _,item in items],0)
        values=((probs>parameters['mask_lower'])&(probs<parameters['mask_upper'])).float()
    else:
        values=F.one_hot(torch.tensor([label for label,_ in items]),num_classes=classes).float()
    affinity=feature@keys
    return parameters['alpha']*torch.exp(-parameters['beta']+parameters['beta']*affinity)@values


def predict_stream(features,prototypes,order=None,negative=True):
    """Không nhận nhãn thật. Mỗi mẫu dùng cache của quá khứ và chính mẫu hiện tại.

    Update-before-predict theo runner gốc. Không gom các ảnh khác nhau thành
    augmented views. Negative=False chỉ dùng kiểm thử tính suy biến binary.
    """
    x=torch.as_tensor(features,dtype=torch.float32,device='cpu')
    t=torch.as_tensor(prototypes,dtype=torch.float32,device='cpu')
    if (x.ndim!=2 or t.ndim!=2 or x.shape[1]!=t.shape[1] or len(x)<1 or len(t)<2 or
            not torch.isfinite(x).all() or not torch.isfinite(t).all()):
        raise ValueError('Sai embedding hoặc prototype')
    if not torch.allclose(x.norm(dim=1),torch.ones(len(x)),atol=1e-5) or not torch.allclose(t.norm(dim=1),torch.ones(len(t)),atol=1e-5):
        raise ValueError('Cần embedding/prototype unit')
    order=np.arange(len(x)) if order is None else np.asarray(order)
    if order.shape!=(len(x),) or not np.issubdtype(order.dtype,np.integer) or sorted(order.tolist())!=list(range(len(x))):
        raise ValueError('Stream order phải là hoán vị đủ các ảnh')
    pos,neg={},{}
    output=np.empty((len(x),len(t)),dtype=np.float32)
    pseudo=np.empty(len(x),dtype=int);entropy=np.empty(len(x));max_negative_spread=0.
    with torch.no_grad():
        for index in order:
            feature=x[index:index+1]
            logits=100.*feature@t.T
            probability=logits.softmax(1)
            loss=float(-(probability*logits.log_softmax(1)).sum(1).item())
            label=int(logits.argmax(1).item());pseudo[index]=label;entropy[index]=loss
            item={'feature':feature.clone(),'loss':loss,'probability':probability.clone(),'index':int(index)}
            _put(pos,label,item,PARAMETERS['positive']['shot_capacity'])
            # Dùng đúng log2(K) của implementation gốc, không tự sửa sang ln(K).
            normalized_entropy=loss/math.log2(len(t))
            if negative and PARAMETERS['negative']['entropy_lower']<normalized_entropy<PARAMETERS['negative']['entropy_upper']:
                _put(neg,label,item,PARAMETERS['negative']['shot_capacity'])
            positive_logits=_cache_logits(feature,pos,len(t),PARAMETERS['positive'])
            negative_logits=_cache_logits(feature,neg,len(t),PARAMETERS['negative'],negative=True)
            max_negative_spread=max(max_negative_spread,float((negative_logits.max()-negative_logits.min()).item()))
            output[index]=(logits+positive_logits-negative_logits).numpy()[0]
    if not np.isfinite(output).all():raise FloatingPointError('TDA logits không hữu hạn')
    return output,{'reference_commit':REFERENCE_COMMIT,'parameters':PARAMETERS,'order':order.tolist(),
                   'positive_cache_indices':{str(k):[item['index'] for item in queue] for k,queue in pos.items()},
                   'negative_cache_indices':{str(k):[item['index'] for item in queue] for k,queue in neg.items()},
                   'pseudo_labels':pseudo.tolist(),'base_entropy':entropy.tolist(),
                   'negative_logit_spread_max':max_negative_spread,
                   'update_cache_before_predict':True,'label_fit':False,'views_per_image':1}
