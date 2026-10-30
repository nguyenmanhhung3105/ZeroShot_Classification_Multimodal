"""Nạp hai hàm thuần InMaP tác giả bằng AST, không import dataset/model."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import numpy as np
import torch
import torch.nn.functional as F
from experiments.clip_inmap.core import optimize,refine,PARAMETERS,REFERENCE_COMMIT


def functions(folder):
    folder=Path(folder)
    if subprocess.check_output(['git','-C',str(folder),'rev-parse','HEAD'],text=True).strip()!=REFERENCE_COMMIT:raise ValueError('Sai commit InMaP')
    subprocess.run(['git','-C',str(folder),'diff','--exit-code','HEAD','--','main.py'],check=True,capture_output=True)
    tree=ast.parse((folder/'main.py').read_text());names={'image_opt','sinkhorn'}
    body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    if {n.name for n in body}!=names:raise ValueError('Thiếu hàm InMaP reference')
    ns={'torch':torch,'F':F};exec(compile(ast.Module(body=body,type_ignores=[]),str(folder/'main.py'),'exec'),ns)
    return ns


def compare(ns,x,t,balanced=False,iterations=2000):
    previous=torch.get_num_threads();torch.set_num_threads(1)
    try:
        xt,tt=torch.tensor(x,dtype=torch.float32),torch.tensor(t.T.copy(),dtype=torch.float32)
        logits=xt@tt
        p=ns['sinkhorn'](logits,.01,0,20) if balanced else F.softmax(logits/.01,dim=1)
        expected=ns['image_opt'](xt,tt,p.clone(),10,iterations,.04,.6)
        scores,proxy,diag=optimize(x,t,balanced=balanced,iterations=iterations)
        np.testing.assert_allclose(proxy,expected.T.numpy(),atol=2e-6,rtol=1e-5)
        np.testing.assert_array_equal(scores.argmax(1),(xt@expected).argmax(1).numpy())
        return {'samples':len(x),'classes':len(t),'balanced_sinkhorn':balanced,'iterations':iterations,
                'max_proxy_abs_error':float(abs(proxy-expected.T.numpy()).max()),'predictions_identical':True}
    finally:torch.set_num_threads(previous)


def verify(folder,output,settings=None):
    ns=functions(folder);cases=[]
    from experiments.clip_zlap.core import unit_vectors
    rng=np.random.default_rng(92)
    for classes in [2,8]:
        x=unit_vectors(rng.uniform(.1,1.,(41,16)));t=unit_vectors(rng.uniform(.1,1.,(classes,16)))
        for balanced in [False,True]:
            case=compare(ns,x,t,balanced,100);cases.append(case);print(case,flush=True)
    if settings:
        from experiments.clip_tda.run import load_features
        from experiments.clip_zlap.run import project_path,text_prototypes
        donor=load_features(project_path(settings['text_prototype_run']))
        for scope in ['train','dev']:
            source=load_features(project_path(settings[scope+'_run']))
            for task,t in [('informativeness',source['prototypes']),('humanitarian',text_prototypes(source,donor))]:
                for balanced in [False,True]:
                    case=compare(ns,source['features'],t,balanced);case.update(split=scope,task=task)
                    cases.append(case);print(case,flush=True)
    report={'status':'passed','reference_commit':REFERENCE_COMMIT,'cases':cases,
            'adaptations':['Only original image_opt/sinkhorn functions executed, no main/dataset/model code.',
                'Both use torch CPU FP32 and one thread. Local core clones pseudo labels to avoid mutating caller data.'],
            'reference_sha256':hashlib.sha256((Path(folder)/'main.py').read_bytes()).hexdigest(),
            'local_core_sha256':hashlib.sha256((ROOT/'experiments/clip_inmap/core.py').read_bytes()).hexdigest()}
    Path(output).write_text(json.dumps(report,indent=2)+'\n');return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--reference-dir',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--config');args=parser.parse_args()
    verify(args.reference_dir,args.output,json.loads(Path(args.config).read_text()) if args.config else None)
