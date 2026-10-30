"""Đối chứng text hỗ trợ tầng humanitarian; không dùng nhãn thật để fusion."""
import csv
from itertools import islice
import hashlib
import numpy as np

METHODS=('text_only','single_stage_fixed','two_stage_fixed','two_stage_conditional')


def read_texts(config,rows,resolve):
    """Đọc đúng prefix đã chọn, ghép bằng ID; không nhận nhãn annotation text."""
    with resolve(config['data_path']).open(encoding='utf-8',newline='') as stream:
        reader=csv.DictReader(stream,delimiter='\t')
        column=config.get('text_col','clean_text');id_column=config.get('id_col','image_id')
        if not {column,id_column}.issubset(reader.fieldnames or []):raise ValueError('Thiếu ID/text trong TSV')
        selected=list(islice(reader,len(rows)))
    if [r[id_column] for r in selected]!=[r['id'] for r in rows]:raise ValueError('Text khác ID/thứ tự ảnh')
    texts=[r[column] or '' for r in selected]
    identity=[{'id':r['id'],'text_sha256':hashlib.sha256(t.encode()).hexdigest()} for r,t in zip(rows,texts)]
    return texts,identity


def text_scores(backend,texts,banks,batch_size):
    from evaluation_audit import token_lengths
    from text_chunking import token_counter
    lengths,context,status=token_lengths(backend,texts)
    empty=np.array([not t.strip() for t in texts]);indices=np.flatnonzero(~empty)
    # Encode tweet một lần, dùng cùng embedding cho hai task.
    vectors=None
    for start in range(0,len(indices),batch_size):
        current=indices[start:start+batch_size]
        encoded=backend.encode_texts([texts[i] for i in current]).detach().float().cpu().numpy()
        if vectors is None:vectors=np.zeros((len(texts),encoded.shape[1]),dtype=np.float32)
        vectors[current]=encoded
    results=[]
    prompt_lengths={}
    for task,bank in banks.items():
        classes=[]
        for label,prompts in bank.items():
            count,window=token_counter(backend)
            values=[count(prompt) for prompt in prompts]
            if max(values)>window:raise ValueError('Prompt text vượt context; không cắt ngầm')
            prompt_lengths[task+'/'+label]=values
            embeddings=backend.encode_texts(prompts).detach().float().cpu().numpy()
            embeddings/=np.maximum(np.linalg.norm(embeddings,axis=1,keepdims=True),1e-12)
            prototype=embeddings.mean(0);prototype/=max(np.linalg.norm(prototype),1e-12)
            classes.append(prototype)
        classes=np.asarray(classes)
        results.append(np.zeros((len(texts),len(classes))) if vectors is None else vectors@classes.T)
    diagnostics={'context_length':context,'token_length_status':status,'token_lengths':lengths,
                 'truncated_count':sum(length is not None and length>context for length in lengths) if context else None,
                 'empty_count':int(empty.sum()),'prompt_token_lengths':prompt_lengths,
                 'aggregation':'mean_embed','long_text_policy':'model_default_truncation_logged'}
    return results,empty,diagnostics


def distributions(scores,scale):
    scores=np.asarray(scores,float)
    if scores.ndim!=2 or not np.isfinite(scores).all() or not np.isfinite(scale) or scale<=0:
        raise ValueError('Score/scale không hợp lệ')
    logits=scores*scale;exponent=np.exp(logits-logits.max(1,keepdims=True))
    return exponent/exponent.sum(1,keepdims=True)


def predict(info_image,human_image,info_text,human_text,empty,info_labels,labels,scale):
    """Giữ gate ảnh cho cả hai phương án hai tầng để tách tác dụng text ở tầng 2.

    Fixed: 75% ảnh +25% text. Conditional: chỉ dùng text nếu text cũng informative.
    Đồng thuận relevance không bảo đảm ảnh và tweet mô tả đúng cùng sự kiện.
    """
    from experiments.clip_two_stage.run import decide
    baseline,gate=decide(info_image,human_image,info_labels,labels)
    pi=distributions(human_image,scale);pt=distributions(human_text,scale)
    ti=np.asarray(info_text);empty=np.asarray(empty,bool)
    if ti.shape!=np.shape(info_image) or pt.shape!=pi.shape or empty.shape!=(len(pi),) or not np.isfinite(ti).all():
        raise ValueError('Text khác shape/thứ tự lớp')
    text_gate=np.asarray(info_labels)[ti.argmax(1)]=='informative'
    fixed_weight=np.where(empty,0.,.25)
    conditional_weight=np.where(~empty&gate&text_gate,.25,0.)
    fixed=(1-fixed_weight[:,None])*pi+fixed_weight[:,None]*pt
    conditional=(1-conditional_weight[:,None])*pi+conditional_weight[:,None]*pt
    positive=[i for i,l in enumerate(labels) if l!='not_humanitarian']
    def two_stage(p):
        pred=np.asarray(labels)[np.asarray(positive)[p[:,positive].argmax(1)]]
        pred[~gate]='not_humanitarian';return pred.tolist()
    text_only=np.asarray(labels)[pt.argmax(1)].tolist()
    text_only=[p if not missing else '__missing_text__' for p,missing in zip(text_only,empty)]
    predictions={'text_only':text_only,'single_stage_fixed':np.asarray(labels)[fixed.argmax(1)].tolist(),
                 'two_stage_fixed':two_stage(fixed),'two_stage_conditional':two_stage(conditional)}
    return predictions,{'text_gate':text_gate,'text_weight':conditional_weight,
                        'text_used':conditional_weight>0,'empty':empty}
