"""Đối chiếu các method thuần của DualMem, không import model/dataset."""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import numpy as np
import torch
import torch.nn.functional as F
from experiments.clip_dmn.core import predict_stream,inputs,REFERENCE_COMMIT,PARAMETERS


def functions(folder):
    folder=Path(folder);file=folder/'dmn_main.py'
    if subprocess.check_output(['git','-C',str(folder),'rev-parse','HEAD'],text=True).strip()!=REFERENCE_COMMIT:raise ValueError('Sai commit DMN')
    subprocess.run(['git','-C',str(folder),'diff','--exit-code','HEAD','--','dmn_main.py','clip/fix_clip.py','scripts/run.sh'],check=True,capture_output=True)
    tree=ast.parse(file.read_text());parent=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='DualMem')
    names={'update_memory_bank','get_image_pred','get_text_prediction'}
    body=[n for n in parent.body if isinstance(n,ast.FunctionDef) and n.name in names]
    if {n.name for n in body}!=names:raise ValueError('Thiếu method DualMem')
    ns={'torch':torch,'math':math,'F':F};exec(compile(ast.Module(body=body,type_ignores=[]),str(file),'exec'),ns)
    return ns,file


def replay(ns,features,prototypes,order,memory_size=50,logit_scale=100.):
    x,t,order=inputs(features,prototypes,order);n,v,d=x.shape;c=len(t)
    zero=torch.zeros((c,d))
    algorithm=SimpleNamespace(args=SimpleNamespace(position='all'),beta=5.5,
        global_bias=zero,global_bias_key=zero,global_bias_value=zero,global_ffn_bias=zero,text_bias=zero)
    model=SimpleNamespace(text_feat=t,fixed_global_feat_vanilla=t[:,None,:],memory_size=memory_size,
        image_feature_memory=torch.zeros((c,memory_size,d)),image_feature_count=torch.zeros((c,1),dtype=torch.long),
        image_prediction_mem=torch.zeros((c,memory_size,c)),image_entropy_mem=torch.zeros((c,memory_size)),
        logit_scale=SimpleNamespace(exp=lambda:logit_scale))
    expected=np.empty((n,c),dtype='float32');text=np.empty_like(expected);dynamic=np.empty_like(expected)
    with torch.no_grad():
        for i in order:
            model.image_features_global=x[i]
            pt=ns['get_text_prediction'](algorithm,model)
            entropy=-(pt*torch.log(pt+1e-6)).sum(1)
            indices=torch.argsort(entropy)[:max(1,int(v*.1))]
            algorithm.init_pred=pt[indices].mean(0,keepdim=True)
            ns['update_memory_bank'](algorithm,model,None)
            pd=ns['get_image_pred'](algorithm,model)
            text[i]=pt[0].numpy();dynamic[i]=pd[0].numpy();expected[i]=(.5*pt[:1]+.5*pd)[0].numpy()
    return expected,text,dynamic,model


def compare(ns,x,t,order,memory_size=50,logit_scale=100.):
    previous=torch.get_num_threads();torch.set_num_threads(1)
    try:
        expected,text,dynamic,model=replay(ns,x,t,order,memory_size,logit_scale)
        actual,trace,diag=predict_stream(x,t,order,memory_size=memory_size,logit_scale=logit_scale)
        for a,b in [(actual,expected),(trace['text_probability'],text),(trace['dynamic_probability'],dynamic)]:
            np.testing.assert_allclose(a,b,atol=2e-6,rtol=1e-5)
            np.testing.assert_array_equal(a.argmax(1),b.argmax(1))
        np.testing.assert_array_equal(diag['memory_counts'],model.image_feature_count[:,0].numpy())
        values=np.asarray(x);values=values[:,None,:] if values.ndim==2 else values
        for label,ids in enumerate(diag['memory_indices']):
            for slot,index in enumerate(ids):
                if index>=0:np.testing.assert_array_equal(values[index,0],model.image_feature_memory[label,slot].numpy())
        return {'images':len(x),'classes':len(t),'views':values.shape[1],'memory_size':memory_size,'logit_scale':logit_scale,
            'max_combined_probability_abs_error':float(abs(actual-expected).max()),
            'max_dynamic_probability_abs_error':float(abs(trace['dynamic_probability']-dynamic).max()),
            'all_predictions_identical':True,'final_memory_identical':True}
    finally:torch.set_num_threads(previous)


def verify(folder,output,settings=None):
    ns,file=functions(folder);cases=[]
    from experiments.clip_zlap.core import unit_vectors
    rng=np.random.default_rng(31)
    for classes in [2,8]:
        x=unit_vectors(rng.normal(size=(71,16)));t=unit_vectors(rng.normal(size=(classes,16)))
        case=compare(ns,x,t,rng.permutation(len(x)),3);case['scope']='synthetic_single_view';cases.append(case)
        multi=np.stack([x,unit_vectors(rng.normal(size=x.shape)),unit_vectors(rng.normal(size=x.shape))],1)
        case=compare(ns,multi,t,np.arange(len(x)),3);case['scope']='synthetic_multi_view';cases.append(case)
    if settings:
        from experiments.clip_tda.run import load_features
        from experiments.clip_zlap.run import project_path,text_prototypes
        donor=load_features(project_path(settings['text_prototype_run']))
        for scope in ['train','dev']:
            source=load_features(project_path(settings[scope+'_run']))
            for task,t in [('informativeness',source['prototypes']),('humanitarian',text_prototypes(source,donor))]:
                for seed in [0,1,2]:
                    order=np.random.default_rng(seed).permutation(len(source['features']))
                    case=compare(ns,source['features'],t,order);case.update(scope=scope,task=task,shuffle_seed=seed)
                    cases.append(case);print(case,flush=True)
    report={'status':'passed','reference_commit':REFERENCE_COMMIT,'cases':cases,
        'source_sha256':hashlib.sha256(file.read_bytes()).hexdigest(),
        'local_core_sha256':hashlib.sha256((ROOT/'experiments/clip_dmn/core.py').read_bytes()).hexdigest(),
        'adaptations':['Only original DualMem methods extracted; no main/model/dataset or GT weight search executed.',
            'Dummy model holds provided unit embeddings, text anchor, zero mapping biases and checkpoint scale 100.',
            'Single-view always selects its one view; multi-view uses max(1,floor(V*0.1)). Original benchmark uses more views.',
            'Fixed equal text/dynamic probability weights. CPU FP32; no static support or learned projections.']}
    Path(output).write_text(json.dumps(report,indent=2)+'\n');return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference-dir',required=True);p.add_argument('--output',required=True);p.add_argument('--config')
    a=p.parse_args();verify(a.reference_dir,a.output,json.loads(Path(a.config).read_text()) if a.config else None)
