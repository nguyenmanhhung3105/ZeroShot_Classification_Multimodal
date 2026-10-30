"""So số với solver StatA gốc trên embedding tổng hợp; chỉ đổi CUDA thành CPU."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from experiments.clip_stata.core import stata,REFERENCE_COMMIT


class CpuOnly(ast.NodeTransformer):
    def visit_Call(self,node):
        node=self.generic_visit(node)
        if isinstance(node.func,ast.Attribute) and node.func.attr=='cuda':node.func.attr='cpu'
        return node


def verify(reference,output):
    folder=Path(reference)
    if subprocess.check_output(['git','-C',str(folder),'rev-parse','HEAD'],text=True).strip()!=REFERENCE_COMMIT:
        raise ValueError('Sai commit StatA')
    subprocess.run(['git','-C',str(folder),'diff','--exit-code','HEAD','--','solvers/StatA.py'],check=True,capture_output=True)
    source=folder/'solvers/StatA.py'
    tree=CpuOnly().visit(ast.parse(source.read_text()));ast.fix_missing_locations(tree)
    namespace={};exec(compile(tree,str(source),'exec'),namespace)
    old_threads=torch.get_num_threads();torch.set_num_threads(1)
    reports=[]
    try:
        for classes in [2,8]:
            rng=np.random.default_rng(classes)
            t=rng.normal(size=(classes,24)).astype('float32');t/=np.linalg.norm(t,axis=1,keepdims=True)
            chosen=rng.choice(2,75,p=[.85,.15])
            x=t[chosen]+rng.normal(scale=.12,size=(75,24)).astype('float32');x/=np.linalg.norm(x,axis=1,keepdims=True)
            with torch.no_grad():
                old_prior,old_z=namespace['StatA_solver'](torch.tensor(x),torch.zeros(75),torch.tensor(t.T[None]))
            prior,z=stata(x,t)
            np.testing.assert_allclose(prior,old_prior.numpy(),atol=1e-6,rtol=1e-5)
            np.testing.assert_allclose(z,old_z.numpy(),atol=2e-5,rtol=2e-4)
            np.testing.assert_array_equal(z.argmax(1),old_z.numpy().argmax(1))
            reports.append({'classes':classes,'represented_synthetic_classes':2,'samples':75,
                            'max_abs_error':float(abs(z-old_z.numpy()).max()),'predictions_identical':True})
    finally:torch.set_num_threads(old_threads)
    report={'status':'passed','reference_commit':REFERENCE_COMMIT,
            'reference_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'local_source_sha256':hashlib.sha256((ROOT/'experiments/clip_stata/core.py').read_bytes()).hexdigest(),
            'reference_modification':'AST replaces .cuda() with .cpu(); formulas unchanged','cases':reports}
    Path(output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();verify(args.reference_dir,args.output)
