"""Đối chiếu các hàm cache chính thức; chỉ thay CUDA/half bằng CPU/float32."""
import argparse
import ast
import hashlib
import json
import math
import operator
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import torch.nn.functional as F
from experiments.clip_tda.core import predict_stream,PARAMETERS,REFERENCE_COMMIT


class CpuFloat(ast.NodeTransformer):
    def visit_Call(self,node):
        node=self.generic_visit(node)
        if isinstance(node.func,ast.Attribute):
            if node.func.attr=='cuda':node.func.attr='cpu'
            elif node.func.attr=='half':node.func.attr='float'
        return node


def load_functions(path,names,namespace):
    tree=ast.parse(path.read_text())
    body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    if {n.name for n in body}!=set(names):raise ValueError('Thiếu hàm tác giả')
    tree=CpuFloat().visit(ast.Module(body=body,type_ignores=[]));ast.fix_missing_locations(tree)
    exec(compile(tree,str(path),'exec'),namespace)


def verify(reference,output):
    folder=Path(reference)
    if subprocess.check_output(['git','-C',str(folder),'rev-parse','HEAD'],text=True).strip()!=REFERENCE_COMMIT:
        raise ValueError('Sai commit TDA')
    subprocess.run(['git','-C',str(folder),'diff','--exit-code','HEAD','--','tda_runner.py','utils.py','configs/imagenet.yaml'],check=True,capture_output=True)
    ns=dict(torch=torch,F=F,np=np,math=math,operator=operator)
    load_functions(folder/'tda_runner.py',['update_cache','compute_cache_logits'],ns)
    load_functions(folder/'utils.py',['get_entropy','softmax_entropy'],ns)
    reports=[]
    for classes in [2,8]:
        rng=np.random.default_rng(classes)
        t=np.eye(1,24,dtype='float32')+.03*rng.normal(size=(classes,24)).astype('float32')
        t/=np.linalg.norm(t,axis=1,keepdims=True)
        x=np.eye(1,24,dtype='float32')+.2*rng.normal(size=(60,24)).astype('float32')
        x/=np.linalg.norm(x,axis=1,keepdims=True)
        order=rng.permutation(len(x));prototypes=torch.tensor(t.T);expected=np.empty((len(x),classes),dtype='float32')
        positive,negative={},{}
        with torch.no_grad():
            for index in order:
                feature=torch.tensor(x[index:index+1]);base=100.*feature@prototypes
                loss=ns['softmax_entropy'](base);prob=base.softmax(1);pred=int(base.argmax(1).item())
                ns['update_cache'](positive,pred,[feature,loss],3)
                ent=ns['get_entropy'](loss,prototypes)
                if .2<ent<.5:ns['update_cache'](negative,pred,[feature,loss,prob],2,True)
                logits=base.clone()
                if positive:logits+=ns['compute_cache_logits'](feature,positive,2.,5.,prototypes)
                if negative:logits-=ns['compute_cache_logits'](feature,negative,.117,1.,prototypes,(.03,1.))
                expected[index]=logits.numpy()[0]
        actual,diag=predict_stream(x,t,order)
        np.testing.assert_allclose(actual,expected,atol=1e-5,rtol=1e-6)
        np.testing.assert_array_equal(actual.argmax(1),expected.argmax(1))
        reports.append({'classes':classes,'samples':len(x),'predictions_identical':True,
                        'max_logit_abs_error':float(abs(actual-expected).max()),
                        'negative_logit_spread_max':diag['negative_logit_spread_max']})
    report={'status':'passed','reference_commit':REFERENCE_COMMIT,
            'reference_modifications':'Selected pure functions only; .cuda to .cpu, .half to .float for float32 CPU cache operations.',
            'source_sha256':{name:hashlib.sha256((folder/name).read_bytes()).hexdigest() for name in ['tda_runner.py','utils.py','configs/imagenet.yaml']},
            'local_core_sha256':hashlib.sha256((ROOT/'experiments/clip_tda/core.py').read_bytes()).hexdigest(),'cases':reports}
    Path(output).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();verify(args.reference_dir,args.output)
