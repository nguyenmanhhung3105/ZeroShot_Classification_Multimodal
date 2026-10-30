"""DMN-ZS memory read/write trên cached embedding; không nhận nhãn thật.

Theo Eq. (2)-(9), (12) và hai hàm DualMem trong repo CVPR 2024.
Single-view là adaptation; cũng nhận [N,V,D] khi có augmented feature thật.
"""
import numpy as np
import torch
import torch.nn.functional as F

REFERENCE_COMMIT='c6c218c70ebbc701b31cc3d7f7b9597117daad8e'
PARAMETERS={'memory_size':50,'beta':5.5,'logit_scale':100.,'selection_fraction':.1,'text_weight':.5}


def inputs(features,prototypes,order=None):
    values=np.asarray(features)
    if values.ndim==2:values=values[:,None,:]
    x=torch.as_tensor(values,dtype=torch.float32,device='cpu')
    t=torch.as_tensor(prototypes,dtype=torch.float32,device='cpu')
    if (x.ndim!=3 or min(x.shape)<1 or t.ndim!=2 or len(t)<2 or t.shape[1]!=x.shape[2] or
            not torch.isfinite(x).all() or not torch.isfinite(t).all()):raise ValueError('Sai embedding/prototype DMN')
    if not torch.allclose(x.norm(dim=-1),torch.ones(x.shape[:2]),atol=1e-5,rtol=0) or not torch.allclose(t.norm(dim=1),torch.ones(len(t)),atol=1e-5,rtol=0):
        raise ValueError('DMN cần vector ảnh/text unit')
    order=np.arange(len(x)) if order is None else np.asarray(order)
    if order.shape!=(len(x),) or not np.issubdtype(order.dtype,np.integer) or not np.array_equal(np.sort(order),np.arange(len(x))):
        raise ValueError('Order phải là hoán vị đủ mọi ảnh')
    return x,t,order


def predict_stream(features,prototypes,order=None,memory_size=50,beta=5.5,logit_scale=100.,selection_fraction=.1,text_weight=.5):
    if (type(memory_size) is not int or memory_size<1 or not np.isfinite(beta) or beta<=0 or
            not np.isfinite(logit_scale) or logit_scale<=0 or not 0<selection_fraction<=1 or not 0<=text_weight<=1):
        raise ValueError('Sai cấu hình DMN')
    x,t,order=inputs(features,prototypes,order);n,views,dimension=x.shape;c=len(t)
    previous_threads=torch.get_num_threads();torch.set_num_threads(1)
    try:
        with torch.no_grad():
            normalized_text=t/t.norm(dim=1,keepdim=True)
            memory=torch.zeros((c,memory_size,dimension));entropies=torch.zeros((c,memory_size))
            counts=torch.zeros(c,dtype=torch.int64);members=np.full((c,memory_size),-1,dtype=np.int64)
            output=np.empty((n,c),dtype='float32');text=np.empty_like(output);dynamic=np.empty_like(output)
            pseudo=np.empty(n,dtype=np.int64);entropy=np.empty(n,dtype='float32');accepted=np.zeros(n,dtype=bool)
            replaced=np.full(n,-1,dtype=np.int64)
            selected_count=max(1,int(views*selection_fraction))
            for index in order:
                probabilities=(logit_scale*x[index]@normalized_text.T).softmax(-1)
                view_entropy=-(probabilities*torch.log(probabilities+1e-6)).sum(1)
                selected=torch.argsort(view_entropy)[:selected_count]
                p=probabilities[selected].mean(0);label=int(p.argmax());loss=-(p*torch.log(p+1e-8)).sum()
                pseudo[index]=label;entropy[index]=float(loss)
                slot=None
                if counts[label]<memory_size:
                    slot=int(counts[label]);counts[label]+=1
                elif torch.any(loss<entropies[label]):
                    slot=int(torch.sort(entropies[label]).indices[-1])
                if slot is not None:
                    replaced[index]=members[label,slot];memory[label,slot]=x[index,0]
                    entropies[label,slot]=loss;members[label,slot]=index;accepted[index]=True
                # Cập nhật trước khi dự đoán, kể cả ảnh hiện tại, như tác giả.
                bank=torch.cat((memory,t[:,None,:]),dim=1)
                # Giữ marker slot rỗng theo tổng thành phần == 0 của code gốc.
                empty=bank.sum(-1)==0
                keys=bank/bank.norm(dim=-1,keepdim=True);keys[empty]=0
                values=bank/bank.norm(dim=-1,keepdim=True);values[empty]=0
                query=x[index,:1];query=query/query.norm(dim=-1,keepdim=True)
                similarity=(query*keys).sum(-1)
                weight=torch.exp(-beta*(-similarity+1))
                classifier=(values*weight.unsqueeze(-1)).sum(1)
                classifier=classifier/classifier.norm(dim=-1,keepdim=True)
                # get_image_pred chuẩn hóa output lần thứ hai khi mapping=all,
                # bias=0; giữ cùng phép toán FP32 để đối chiếu reference.
                classifier=classifier/classifier.norm(dim=-1,keepdim=True)
                pd=(logit_scale*classifier@x[index,0]).softmax(-1)
                pt=probabilities[0]
                combined=text_weight*pt+(1-text_weight)*pd
                if not torch.isfinite(combined).all():raise FloatingPointError('DMN posterior không hữu hạn')
                output[index]=combined.numpy();text[index]=pt.numpy();dynamic[index]=pd.numpy()
            trace={'text_probability':text,'dynamic_probability':dynamic,'pseudo_label':pseudo,'entropy':entropy,
                   'memory_write':accepted,'replaced_index':replaced}
            diagnostics={'views':views,'selected_views':selected_count,'single_view_adaptation':views==1,
                'order':order.tolist(),'memory_counts':counts.tolist(),'memory_indices':members.tolist(),
                'writes':int(accepted.sum()),'replacements':int((replaced>=0).sum()),
                'support_samples':0,'label_fit':False,'learned_projection':False,
                'text_weight':text_weight,'dynamic_weight':1-text_weight,'weight_search':False,
                'update_before_prediction':True,'empty_slot_rule':'sum(feature components) == 0, as official code',
                'memory_reset':'every call, task, split, stream order'}
            return output,trace,diagnostics
    finally:torch.set_num_threads(previous_threads)
