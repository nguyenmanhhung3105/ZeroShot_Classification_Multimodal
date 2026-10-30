"""Đối chiếu kernel và optimizer LAME bằng hàm/class thuần từ repo tác giả."""
import argparse
import ast
import hashlib
import json
import logging
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import numpy as np
import torch
from experiments.clip_lame.core import adapt,REFERENCE_COMMIT


def functions(folder):
    folder=Path(folder);file=folder/'src/adaptation/lame.py'
    if subprocess.check_output(['git','-C',str(folder),'rev-parse','HEAD'],text=True).strip()!=REFERENCE_COMMIT:raise ValueError('Sai commit LAME')
    subprocess.run(['git','-C',str(folder),'diff','--exit-code','HEAD','--','src/adaptation/lame.py'],check=True,capture_output=True)
    tree=ast.parse(file.read_text());names={'AffinityMatrix','kNN_affinity','rbf_affinity','laplacian_optimization','entropy_energy'}
    body=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
    if {n.name for n in body}!=names:raise ValueError('Thiếu hàm reference')
    ns={'torch':torch,'logger':logging.getLogger('lame_reference')}
    exec(compile(ast.Module(body=body,type_ignores=[]),str(file),'exec'),ns);return ns,file


def verify(folder,output,settings=None):
    ns,file=functions(folder);previous=torch.get_num_threads();torch.set_num_threads(1);cases=[]
    def compare(x,s,affinity):
        xt=torch.nn.functional.normalize(torch.as_tensor(x,dtype=torch.float32),p=2,dim=-1)
        prior=(100*torch.tensor(s,dtype=torch.float32)).softmax(-1)
        kernel=ns['rbf_affinity' if affinity=='rbf' else 'kNN_affinity'](sigma=1.,knn=5)(xt)
        expected=ns['laplacian_optimization'](-torch.log(prior+1e-10),kernel).numpy()
        actual,_=adapt(x,s,affinity)
        np.testing.assert_allclose(actual,expected,atol=1e-7,rtol=1e-6)
        np.testing.assert_array_equal(actual.argmax(1),expected.argmax(1))
        return float(abs(actual-expected).max())
    try:
        rng=np.random.default_rng(27)
        for classes in [2,8]:
            for affinity in ['rbf','knn']:
                error=compare(rng.normal(size=(19,7)).astype('float32'),rng.uniform(-.01,.1,(19,classes)),affinity)
                cases.append({'scope':'synthetic','classes':classes,'affinity':affinity,'max_posterior_abs_error':error})
        if settings:
            from experiments.clip_tda.run import load_features
            from experiments.clip_zlap.run import project_path
            for scope in ['train','dev']:
                source=load_features(project_path(settings[scope+'_run']));dn=project_path(settings['dn_run'])/scope/'scores.npz'
                with np.load(dn,allow_pickle=False) as pack:leaf=pack['humanitarian'].copy()
                for affinity in ['rbf','knn']:
                    for seed in [0,1,2]:
                        order=np.random.default_rng(seed).permutation(len(source['features']));errors=[]
                        for start in range(0,len(order),75):
                            ix=order[start:start+75]
                            for scores in [source['scores'][0],leaf]:errors.append(compare(source['features'][ix],scores[ix],affinity))
                        case={'scope':scope,'affinity':affinity,'seed':seed,'batches_tasks':len(errors),
                              'all_predictions_identical':True,'max_posterior_abs_error':max(errors)}
                        cases.append(case);print(case,flush=True)
    finally:torch.set_num_threads(previous)
    report={'status':'passed','reference_commit':REFERENCE_COMMIT,'cases':cases,
            'reference_sha256':hashlib.sha256(file.read_bytes()).hexdigest(),
            'local_core_sha256':hashlib.sha256((ROOT/'experiments/clip_lame/core.py').read_bytes()).hexdigest(),
            'adaptations':['Only pure functions/classes; do not import model, dataset or Detectron2.',
                           'CPU FP32 same backend. Singleton/zero-sigma fallback only in local core, tested separately; not needed on real groups.']}
    Path(output).write_text(json.dumps(report,indent=2)+'\n');return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference-dir',required=True);p.add_argument('--output',required=True);p.add_argument('--config')
    a=p.parse_args();verify(a.reference_dir,a.output,json.loads(Path(a.config).read_text()) if a.config else None)
