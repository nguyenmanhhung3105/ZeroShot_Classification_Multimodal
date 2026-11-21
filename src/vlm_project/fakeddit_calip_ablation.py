"""Fixed CALIP ablations around the original Fakeddit hierarchy; no labels."""
import numpy as np
import time
from scipy.special import softmax

from .calip import encode, score, PARAMETERS, REFERENCE_COMMIT, REFERENCE_URL
from .io import ROOT, read_json
from .methods import hierarchy, fuse
from .adaptation import stata
from .backbone import Backbone
from . import data
from .fakeddit_hierarchy import unit, specification as base_specification
from .io import MODEL, code_hashes, sha256, write_json, new_run
from .evaluate import metric, publish, table, summarize


def specification():
    value=read_json(ROOT/'configs/fakeddit_calip_ablation.json')
    if value!={'version':1,'gate':'stata','calip_stage':2,'temperature':1.,'clip_calip_weight':.5,'image_weight':.75,
               'attention_scale':2.,'beta':[1.,1.12,.02],'logit_scale':100.}:
        raise ValueError('Fixed unfitted CALIP defaults and equal CLIP/CALIP fusion required')
    if {key:value[key] for key in ['attention_scale','beta','logit_scale']}!=PARAMETERS:
        raise ValueError('CALIP implementation parameters differ from configuration')
    return value


def extract(backend,rows,six,batch_size,output):
    """One image forward for global+patch, plus a reference check on first batch."""
    import torch
    from PIL import Image
    prototypes=torch.as_tensor(six,dtype=torch.float32,device=backend.device)
    features,titles,six_logits,checks=[],[],[],[]
    for start in range(0,len(rows),batch_size):
        images=[]
        try:
            batch_rows=rows[start:start+batch_size]
            for row in batch_rows:
                with Image.open(row['image']) as image:images.append(image.convert('RGB'))
            global_features,patches,info=encode(backend,images,verify_global=start==0)
            a6,terms6=score(global_features,patches,prototypes)
            if start==0:
                np.savez_compressed(output/'patch_verification.npz',
                    global_features=global_features[:3].cpu().numpy(),patches=patches[:3].cpu().numpy(),
                    six_prototypes=six,six_terms=terms6[:3].cpu().numpy())
            features.append(global_features.cpu().numpy())
            titles.append(backend.encode_texts([row['text'] for row in batch_rows]))
            six_logits.append(a6.cpu().numpy());checks.append(info)
            done=min(start+batch_size,len(rows))
            if start==0 or done%128==0 or done==len(rows):
                print(f'Global/patch/CALIP six: {done}/{len(rows)}',flush=True)
        finally:
            for image in images:image.close()
    return np.concatenate(features),np.concatenate(titles),np.concatenate(six_logits),checks


def predict(features,titles,binary_raw,six_raw,text_raw,calip_logits,base_spec,calip_spec,stata_solver=stata):
    """StatA binary gate only; CALIP changes fine classification, never the gate."""
    import torch
    x,titles=np.asarray(features),np.asarray(titles)
    binary,six=unit(binary_raw),unit(six_raw)
    logits=np.asarray(calip_logits)
    if (x.ndim!=2 or titles.shape!=x.shape or binary.shape!=(2,x.shape[1])
            or six.shape!=(6,x.shape[1]) or logits.shape!=(len(x),6)
            or np.asarray(text_raw).shape!=six.shape):
        raise ValueError('Invalid aligned feature/prototype/CALIP shapes')
    if not all(np.isfinite(v).all() for v in [x,titles,logits,text_raw]):
        raise ValueError('Nonfinite features/logits')
    if not all(np.allclose(np.linalg.norm(v,axis=1),1,atol=1e-5) for v in [x,titles]):
        raise ValueError('Global/title features must be unit')
    clip_binary=softmax(base_spec['image_scale']*(x@binary.T),axis=1)
    clip=softmax(base_spec['image_scale']*(x@six.T),axis=1)
    calip=softmax(logits/calip_spec['temperature'],axis=1)
    mixed=fuse(clip,calip,calip_spec['clip_calip_weight'])
    text=softmax(base_spec['text_scale']*(titles@np.asarray(text_raw).T),axis=1)
    branches={}
    for name,p in [('clip',clip),('calip',calip),('clip_calip',mixed)]:
        branches[name+'_image']=p
        branches[name+'_mm']=fuse(p,text,calip_spec['image_weight'])
    results=[('flat_'+name,None,p.argmax(1)) for name,p in branches.items()]
    arrays={'p_clip_binary':clip_binary,**{'p_'+name:p for name,p in branches.items()}}
    gates={}
    previous=torch.get_num_threads();torch.set_num_threads(1)
    try:
        for seed in base_spec['grouping_seeds']:
            order=np.random.default_rng(seed).permutation(len(x))
            groups=np.empty(len(x),dtype=np.int32);posterior=clip_binary.copy()
            for group,start in enumerate(range(0,len(x),base_spec['group_size'])):
                ids=order[start:start+base_spec['group_size']];groups[ids]=group
                if len(ids)>base_spec['stata_neighbors']:
                    _,posterior[ids]=stata_solver(x[ids],binary,max_iter=base_spec['stata_iterations'],neighbors=base_spec['stata_neighbors'])
                print(f'StatA binary gate seed={seed}: group {group+1}/{(len(x)+74)//75}',flush=True)
            gate=posterior.argmax(1);gates[seed]=gate
            arrays[f'p_stata_binary_s{seed}']=posterior;arrays[f'groups_s{seed}']=groups
            for name,p in branches.items():
                pred=hierarchy(gate.astype(str),p,list('012345')).astype(int)
                results.append(('hierarchy_stata_'+name,seed,pred))
    finally:torch.set_num_threads(previous)
    return results,arrays,gates


def run(settings,split,root,smoke=False,backend_factory=Backbone,extractor=extract):
    if split!='dev':raise ValueError('StatA/CALIP experiment is dev-only')
    spec=base_specification();calip_spec=specification()
    profile=read_json(settings['fakeddit']['prompt_file'])
    if profile['model']!=MODEL or profile['rules']!={'image_logit_scale':100.,'text_logit_scale':25.,'image_weight':.75}:
        raise ValueError('Unexpected prompt profile/model/rules')
    banks=[profile['image_prompts']['2way']['semantic_v1'],profile['image_prompts']['6way']['legacy'],profile['text_prompts']['6way']]
    if any(list(bank)!=[str(i) for i in range(size)] for bank,size in zip(banks,[2,6,6])):
        raise ValueError('Prompt order must preserve 2way fake/real and 6way True/fake IDs')
    rows,cohort=data.fakeddit(root,'dev',smoke)
    if len(rows)!=(5 if smoke else 3000):raise ValueError('Unexpected cohort size')
    output=new_run(settings,'fakeddit_stata_calip','dev',smoke)
    plan={'kind':'fakeddit_stata_calip','dataset':'fakeddit_6way','model':MODEL,'split':'dev',
          'mode':'smoke' if smoke else 'full','samples':len(rows),'cohort':cohort,
          'specification':spec,'calip_specification':calip_spec,'code_sha256':code_hashes(),
          'profile_sha256':sha256(settings['fakeddit']['prompt_file']),
          'calip_specification_sha256':sha256(ROOT/'configs/fakeddit_calip_ablation.json'),
          'requirements_sha256':sha256(ROOT/'requirements.txt'),
          'calip':{'commit':REFERENCE_COMMIT,'source':REFERENCE_URL,'parameters':PARAMETERS},
          'scope':'Stage1 StatA real/fake; stage2 CLIP vs CALIP vs equal probability fusion; no test',
          'normalization':'Unit image prototypes for all image branches; same raw text mean and 75/25 MM fusion',
          'calibration':'CALIP temperature=1 and fusion weights fixed, uncalibrated',
          'batch_size':settings['batch_size'],'profile':profile}
    write_json(output/'frozen_plan.json',plan);write_json(output/'run.json',{**plan,'status':'running'})
    start=time.perf_counter();backend=backend_factory(settings['device'])
    try:
        binary,six,text=[backend.prototypes(bank) for bank in banks]
        features,titles,logits,checks=extractor(backend,rows,unit(six),settings['batch_size'],output)
        plan['model_runtime']=getattr(backend,'metadata',{'backend':type(backend).__name__})
    finally:backend.unload()
    write_json(output/'extraction.json',checks)
    results,arrays,gates=predict(features,titles,binary,six,text,logits,spec,calip_spec)
    # Labels are introduced only after every real-method prediction exists.
    truth=np.asarray([int(r['6way']) for r in rows]);binary_truth=np.asarray([int(r['2way']) for r in rows])
    if not np.array_equal(binary_truth,(truth==0).astype(int)):raise ValueError('Binary/sixway taxonomy mismatch')
    index={(name,seed):pred for name,seed,pred in results}
    records,predictions,details,comparisons=[],[],{},[]
    for method,seed,pred in results:
        values=metric(truth,pred,list(range(6)));key=method if seed is None else f'{method}_s{seed}'
        details[key]=values.pop('per_class')
        records.append({'task':'6way','method':method,'seed':seed,'samples':len(rows),**values})
        suffix='mm' if method.endswith('_mm') else 'image'
        reference=('hierarchy_stata_clip_'+suffix,seed) if seed is not None else ('flat_clip_'+suffix,None)
        ref=index[reference]
        comparisons.append({'method':method,'seed':seed,'reference':reference[0],
            'accuracy_delta':float((pred==truth).mean()-(ref==truth).mean()),
            'macro_f1_delta':float(values['macro_f1']-metric(truth,ref,list(range(6)))['macro_f1']),
            'corrected':int(((ref!=truth)&(pred==truth)).sum()),'harmed':int(((ref==truth)&(pred!=truth)).sum())})
        predictions.extend({'task':'6way','method':method,'seed':seed,'id':row['id'],'truth':int(y),'predicted':int(p)} for row,y,p in zip(rows,truth,pred))
        arrays['prediction_'+key]=pred
        print(f'{method} seed={seed}: accuracy={values["accuracy"]:.4f}, macro_f1={values["macro_f1"]:.4f}',flush=True)
    table(output/'comparisons.tsv',comparisons)
    binary_records=[]
    for method,seed,pred in [('clip_image',None,arrays['p_clip_binary'].argmax(1)),*[('stata_image',seed,gate) for seed,gate in gates.items()]]:
        values=metric(binary_truth,pred,[0,1]);values.pop('per_class')
        binary_records.append({'task':'2way','method':method,'seed':seed,'samples':len(rows),**values})
    table(output/'binary_metrics_by_seed.tsv',binary_records);table(output/'binary_summary.tsv',summarize(binary_records))
    oracle={}
    for name in ['clip_image','calip_image','clip_calip_image','clip_mm','calip_mm','clip_calip_mm']:
        oracle[name]=metric(truth,hierarchy(binary_truth.astype(str),arrays['p_'+name],list('012345')).astype(int),list(range(6)))
    write_json(output/'oracle_diagnostic.json',{'scope':'Ground-truth binary gate; not a deployed method',**oracle})
    write_json(output/'branch_confidence.json',{key:{'mean_confidence':float(values.max(1).mean()),
        'mean_normalized_entropy':float(-(values*np.log(np.maximum(values,1e-300))).sum(1).mean()/np.log(6))}
        for key,values in arrays.items() if key.startswith('p_') and values.ndim==2 and values.shape[1]==6})
    np.savez_compressed(output/'replay_scores.npz',ids=np.asarray([r['id'] for r in rows]),labels=truth,
        binary_labels=binary_truth,image_features=features,title_features=titles,binary_raw=binary,
        six_raw=six,text_six_raw=text,calip_logits=logits,**arrays)
    plan['elapsed_seconds']=time.perf_counter()-start
    return publish(output,records,predictions,details,plan)


