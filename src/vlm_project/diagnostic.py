"""Chẩn đoán ảnh nhiều lớp: CLIP/StatA/CALIP và gộp xác suất cố định."""
import csv
from datetime import datetime, timezone
from itertools import islice
import time
import uuid

import numpy as np
from scipy.special import softmax
import yaml

from .adaptation import stata
from .backbone import Backbone
from .calip import encode, score, PARAMETERS, REFERENCE_COMMIT, REFERENCE_URL
from .evaluate import metric, publish, table
from .io import ROOT, BENCHMARKS, path, read_json, write_json, sha256, digest, code_hashes
from .paper import verify_sources

SPEC_FILE='configs/fusion_diagnostic.yaml'


def specification():
    spec=yaml.safe_load((ROOT/SPEC_FILE).read_text())
    if spec['version']!=1 or type(spec['samples']) is not int or not 5<=spec['samples']<=512:
        raise ValueError('Diagnostic chỉ nhận 5–512 ảnh')
    for key in ['selection_seed','grouping_seed']:
        if type(spec[key]) is not int or spec[key]<0:raise ValueError('Seed không hợp lệ')
    if (type(spec['adaptation_batch_size']) is not int or type(spec['image_batch_size']) is not int
            or not 4<=spec['adaptation_batch_size']<=75 or not 1<=spec['image_batch_size']<=8):
        raise ValueError('Giữ batch StatA <=75 và batch ảnh <=8 cho diagnostic')
    if (type(spec['cpu_threads']) is not int or type(spec['timeout_seconds']) is not int
            or not 1<=spec['cpu_threads']<=8 or not 60<=spec['timeout_seconds']<=3600):
        raise ValueError('Giới hạn CPU/timeout diagnostic không hợp lệ')
    if spec['clip_logit_scale']!=100. or spec['calibration']!='unfitted_diagnostic_defaults':
        raise ValueError('Diagnostic dùng logit scale 100, chưa fit calibration')
    for values in [spec['temperatures'],spec['fusion_weights']]:
        if set(values)!={'clip','stata','calip'} or not all(np.isfinite(v) for v in values.values()):
            raise ValueError('Cần tham số hữu hạn cho đủ ba nhánh')
    if any(v<=0 for v in spec['temperatures'].values()):raise ValueError('Temperature phải dương')
    weights=spec['fusion_weights']
    if min(weights.values())<0 or not np.isclose(sum(weights.values()),1.,atol=1e-12,rtol=0):
        raise ValueError('Trọng số phải không âm và có tổng 1')
    return spec


def subset(root,name,limit,seed):
    """Chọn index bằng RNG trước khi đọc nhãn; chỉ hash ảnh đã chọn."""
    root=path(root);classes=read_json(root/'classes.json');source=read_json(root/'provenance.json')
    count,size,split=BENCHMARKS[name]
    if source.get('dataset')!=name or source.get('split')!=split:
        raise ValueError('Sai dataset/split diagnostic')
    if len(classes)!=count or len(set(classes))!=count or any(not isinstance(c,str) or not c.strip() for c in classes):
        raise ValueError('Sai taxonomy diagnostic')
    if type(limit) is not int or not 5<=limit<=min(512,size):raise ValueError('Sai số mẫu diagnostic')
    chosen=set(np.random.default_rng(seed).choice(size,limit,replace=False).tolist())
    rows=[];seen=set();scanned=0
    with (root/'manifest.tsv').open(newline='') as stream:
        reader=csv.DictReader(stream,delimiter='\t')
        if not {'id','image','label','bytes','sha256'}<=set(reader.fieldnames or []):raise ValueError('Manifest thiếu schema')
        for index,raw in enumerate(islice(reader,size+1)):
            scanned+=1;label=int(raw['label']);image=path(root/'images'/raw['image'])
            if raw['id'] in seen or not 0<=label<count or not image.is_relative_to(path(root/'images')):
                raise ValueError('Manifest sai ID/nhãn/path')
            seen.add(raw['id'])
            if index in chosen:
                if image.stat().st_size!=int(raw['bytes']) or sha256(image)!=raw['sha256']:
                    raise ValueError('Ảnh diagnostic thiếu hoặc sai SHA256')
                rows.append({'id':raw['id'],'image':str(image),'label':label,'image_sha256':raw['sha256']})
    if scanned!=size or len(rows)!=limit:raise ValueError('Manifest không đủ split hoặc số ảnh chọn')
    identity=[{k:r[k] for k in ['id','label','image_sha256']} for r in rows]
    return classes,rows,{'source':source,'split':split,'selection':'uniform_without_labels',
                        'selection_seed':seed,'cohort_sha256':digest(identity),
                        'manifest_sha256':sha256(root/'manifest.tsv'),'full_split_size':size}


def probability(values,temperature=1.):
    p=np.asarray(values,dtype=np.float64)
    if (p.ndim!=2 or p.shape[1]<2 or not np.isfinite(p).all() or (p<0).any()
            or not np.allclose(p.sum(1),1.,atol=1e-4,rtol=0)):
        raise ValueError('Xác suất nhánh không hợp lệ')
    if not np.isfinite(temperature) or temperature<=0:raise ValueError('Temperature phải dương')
    p=p/p.sum(1,keepdims=True)
    return p if temperature==1. else softmax(np.log(np.maximum(p,1e-300))/temperature,axis=1)


def mix(branches,weights):
    if not weights or any(k not in branches for k in weights):raise ValueError('Thiếu nhánh fusion')
    if (not all(np.isfinite(v) and v>=0 for v in weights.values())
            or not np.isclose(sum(weights.values()),1.,atol=1e-12,rtol=0)):
        raise ValueError('Trọng số fusion không hợp lệ')
    values=[probability(branches[k]) for k in weights]
    if len({v.shape for v in values})!=1:raise ValueError('Fusion phải cùng thứ tự/shape lớp')
    return sum(weight*value for weight,value in zip(weights.values(),values))


def infer_groups(features,prototypes,initial,spec):
    """Không nhận nhãn thật; khôi phục thứ tự ảnh sau chia nhóm StatA."""
    import torch
    order=np.random.default_rng(spec['grouping_seed']).permutation(len(features))
    result=np.empty_like(initial);groups=np.full(len(features),-1,dtype=np.int64)
    old=torch.get_num_threads();torch.set_num_threads(spec['cpu_threads'])
    try:
        for group,start in enumerate(range(0,len(order),spec['adaptation_batch_size'])):
            ids=order[start:start+spec['adaptation_batch_size']];groups[ids]=group
            print(f'StatA: bắt đầu nhóm {group+1}, {len(ids)} ảnh / {len(prototypes)} lớp',flush=True)
            result[ids]=stata(features[ids],prototypes)[1] if len(ids)>3 else initial[ids]
            print(f'StatA: nhóm {group+1}, {len(ids)} ảnh / {len(prototypes)} lớp',flush=True)
    finally:torch.set_num_threads(old)
    return probability(result,spec['temperatures']['stata']),groups


def extract(backend,rows,prototypes,spec,output):
    """Một forward ảnh cho cả hai nhánh; baseline thêm một check ở batch đầu."""
    from PIL import Image
    import torch
    text=torch.as_tensor(prototypes,dtype=torch.float32,device=backend.device)
    features=[];logits=[];diagnostics=[]
    for start in range(0,len(rows),spec['image_batch_size']):
        images=[]
        try:
            for row in rows[start:start+spec['image_batch_size']]:
                with Image.open(row['image']) as image:images.append(image.convert('RGB'))
            g,p,info=encode(backend,images,verify_global=start==0)
            calip,terms=score(g,p,text)
            if start==0:
                np.savez_compressed(output/'patch_verification.npz',global_features=g[:3].cpu().numpy(),
                                    patches=p[:3].cpu().numpy(),prototypes=prototypes,
                                    calip_logits=calip[:3].cpu().numpy(),calip_terms=terms[:3].cpu().numpy())
            features.append(g.cpu().numpy());logits.append(calip.cpu().numpy());diagnostics.append(info)
            print(f'Global/patch/CALIP: {min(start+len(images),len(rows))}/{len(rows)} ảnh',flush=True)
        finally:
            for image in images:image.close()
    return np.concatenate(features),np.concatenate(logits),diagnostics


def run(settings,name,root,limit=None,backend_factory=Backbone):
    spec=specification();limit=spec['samples'] if limit is None else limit
    if limit not in [5,spec['samples']]:raise ValueError('Chỉ chạy smoke 5 hoặc số mẫu diagnostic đã khai báo')
    sources=verify_sources()
    if name == 'fakeddit_2way':
        from .data import fakeddit
        from .fakeddit import profile
        chosen=np.random.default_rng(spec['selection_seed']).choice(3000,limit,replace=False).tolist()
        rows,cohort=fakeddit(root,'dev',selected_indices=chosen)
        rows=[{**row,'label':int(row['2way'])} for row in rows]
        bank=profile(settings)['image_prompts']['2way']['semantic_v1']
        names=list(profile(settings)['names']['2way'].values())
        prompts={names[int(label)]:texts for label,texts in bank.items()}
        cohort.update(selection='uniform_without_labels',selection_seed=spec['selection_seed'],full_split_size=3000)
    else:
        names,rows,cohort=subset(root,name,limit,spec['selection_seed'])
        prompts={n:['A photo of a '+n.replace('_',' ')+'.'] for n in names}
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output=path(settings['output_root'])/'diagnostics'/name/settings['model']/f'diagnostic_n{limit}_{stamp}_{uuid.uuid4().hex[:8]}'
    output.mkdir(parents=True)
    plan={'kind':'fusion_diagnostic','model':settings['model'],'dataset':name,'mode':'smoke' if limit==5 else 'diagnostic',
          'samples':len(rows),'classes':names,'cohort':cohort,'sources':sources,'code_sha256':code_hashes(),
          'specification':spec,'specification_sha256':sha256(ROOT/SPEC_FILE),'prompts':prompts,
          'calip':{'source':REFERENCE_URL,'commit':REFERENCE_COMMIT,'parameters':PARAMETERS},
          'scope':'Image-only diagnostic subset; not full benchmark or selected final method. Fixed weights/temperatures, no label-based search.',
          'calibration_note':'Softmax normalization does not establish calibration; all temperatures are unfitted diagnostic defaults.'}
    write_json(output/'frozen_plan.json',plan);write_json(output/'cohort.json',rows)
    write_json(output/'run.json',{**plan,'status':'running'})
    print(f'[DIAGNOSTIC] {name}: {len(rows)} ảnh / {len(names)} lớp, chưa fit calibration, không tìm trọng số theo test',flush=True)
    started=time.perf_counter();backend=backend_factory(settings['device'])
    model_seconds=time.perf_counter()-started;plan['model_runtime']=backend.metadata
    times={};started=time.perf_counter()
    try:
        text_start=time.perf_counter();prototypes=backend.prototypes(prompts)
        prototypes=prototypes/np.maximum(np.linalg.norm(prototypes,axis=1,keepdims=True),1e-12)
        times['text_seconds']=time.perf_counter()-text_start
        image_start=time.perf_counter();features,calip_logits,extraction=extract(backend,rows,prototypes,spec,output)
        times['image_and_calip_seconds']=time.perf_counter()-image_start
    finally:backend.unload()
    clip_logits=spec['clip_logit_scale']*(features@prototypes.T)
    branches={'clip':softmax(clip_logits/spec['temperatures']['clip'],axis=1),
              'calip':softmax(calip_logits/spec['temperatures']['calip'],axis=1)}
    adaptation_start=time.perf_counter()
    branches['stata'],groups=infer_groups(features,prototypes,branches['clip'],spec)
    times['stata_seconds']=time.perf_counter()-adaptation_start
    branches['fusion_clip_stata']=mix(branches,{'clip':.5,'stata':.5})
    branches['fusion_clip_calip']=mix(branches,{'clip':.5,'calip':.5})
    branches['fusion_three']=mix(branches,spec['fusion_weights'])
    truth=np.asarray([r['label'] for r in rows]);records=[];predictions=[];details={};distribution={};arrays={}
    print(f'Chỉ chẩn đoán subset: có nhãn thật thuộc {len(np.unique(truth))}/{len(names)} lớp; '
          'macro-F1 tính đủ taxonomy, không đối chiếu trực tiếp với full benchmark.',flush=True)
    baseline=branches['clip'].argmax(1)
    for method,p in branches.items():
        p=probability(p);predicted=p.argmax(1);values=metric(truth,predicted,list(range(len(names))),p)
        details[method]=values.pop('per_class')
        records.append({'task':name,'method':method,'seed':None,'samples':len(rows),**values})
        predictions.extend({'task':name,'method':method,'seed':None,'id':row['id'],'truth':row['label'],'predicted':int(pred)}
                           for row,pred in zip(rows,predicted))
        entropy=-(p*np.log(np.maximum(p,1e-300))).sum(1)/np.log(len(names))
        distribution[method]={'mean_confidence':float(p.max(1).mean()),'mean_normalized_entropy':float(entropy.mean()),
                              'nll':float(-np.log(np.maximum(p[np.arange(len(rows)),truth],1e-300)).mean()),
                              'agreement_with_clip':float((predicted==baseline).mean()),
                              'clip_wrong_to_correct':int(((baseline!=truth)&(predicted==truth)).sum()),
                              'clip_correct_to_wrong':int(((baseline==truth)&(predicted!=truth)).sum())}
        arrays['p_'+method]=p
        print(f'{method}: accuracy={values["accuracy"]:.4f}; entropy={entropy.mean():.4f}; '
              f'đổi đúng→sai={distribution[method]["clip_correct_to_wrong"]}',flush=True)
    np.savez_compressed(output/'replay_scores.npz',ids=np.asarray([r['id'] for r in rows]),labels=truth,
                        class_names=np.asarray(names),global_features=features,prototypes=prototypes,
                        clip_logits=clip_logits,calip_logits=calip_logits,groups=groups,**arrays)
    write_json(output/'branch_diagnostics.json',distribution);write_json(output/'extraction.json',extraction)
    # Sàng lọc cố định cùng cohort, không thay sáu đối chứng diagnostic cũ.
    from .single_stage import candidates
    exploration=[]
    for method,p in candidates(branches['clip'],branches['stata'],branches['calip']).items():
        values=metric(truth,p.argmax(1),list(range(len(names))),p)
        values.pop('per_class')
        exploration.append({'method':method,**values})
    table(output/'single_stage_candidates.tsv',exploration)
    plan['single_stage_exploration']='Fixed candidates from ImageNetV2 pilot; development only. Fakeddit: image-only dev 2way, no hierarchy.'
    timings={**times,'model_load_seconds':model_seconds,'encode_and_infer_seconds':time.perf_counter()-started}
    write_json(output/'timings.json',timings)
    plan.update(timings=timings,extraction=extraction,class_coverage=int(len(np.unique(truth))),
                label_usage='labels used only for evaluation after branch probabilities are computed')
    table(output/'sample_top5.tsv',[{'method':m,'id':rows[i]['id'],'rank':rank+1,'label':int(c),
                                     'class_name':names[c],'probability':float(p[i,c])}
                                    for m,p in branches.items() for i in range(len(rows))
                                    for rank,c in enumerate(np.argsort(-p[i],kind='stable')[:5])])
    return publish(output,records,predictions,details,plan)
