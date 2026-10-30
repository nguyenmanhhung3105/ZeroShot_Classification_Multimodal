"""Kiểm tra cùng cơ chế DMN với anchor InMaP và scale 25 của mở rộng dự án."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import numpy as np
from experiments.clip_dmn.verify_reference import functions,compare
from experiments.clip_tda.run import load_features
from experiments.clip_zlap.run import project_path


def verify(folder,output,settings):
    ns,file=functions(folder);cases=[]
    for scope in ['train','dev']:
        source=load_features(project_path(settings[scope+'_run']))
        path=project_path(settings['inmap_run'])/scope/'unbalanced/proxies_scores.npz'
        with np.load(path,allow_pickle=False) as pack:
            for task,key in [('informativeness','info_proxy'),('humanitarian','human_proxy')]:
                proxy=pack[key].copy()
                for seed in [0,1,2]:
                    case=compare(ns,source['features'],proxy,np.random.default_rng(seed).permutation(len(source['features'])),logit_scale=25.)
                    case.update(scope=scope,task=task,shuffle_seed=seed);cases.append(case);print(case,flush=True)
    report={'status':'passed','cases':cases,'reference_sha256':hashlib.sha256(file.read_bytes()).hexdigest(),
        'local_core_sha256':hashlib.sha256((ROOT/'experiments/clip_dmn/core.py').read_bytes()).hexdigest(),
        'scope':'Equation/operator equivalence with visual anchors supplied as class vectors and scale 25; not original text-anchor DMN-ZS benchmark.'}
    Path(output).write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference-dir',required=True);p.add_argument('--output',required=True)
    p.add_argument('--config',default=str(ROOT/'experiments/clip_dmn/config_visual.json'))
    a=p.parse_args();verify(a.reference_dir,a.output,json.loads(Path(a.config).read_text()))
