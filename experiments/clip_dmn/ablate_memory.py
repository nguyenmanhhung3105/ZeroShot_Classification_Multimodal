"""Ablation bỏ memory: InMaP prior + DN leaf, cùng quy tắc parent product."""
import argparse
import csv
import json
from pathlib import Path
import sys
import uuid
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import numpy as np
from experiments.clip_dmn.hierarchy import parent_product
from experiments.clip_tda.run import load_features
from experiments.clip_zlap.run import project_path,validate_majority,evaluate
from experiments.clip_transductive.run import write_json,sha256


def run(visual_run,dn_run,output_dir=None):
    visual_run,dn_run=Path(visual_run),Path(dn_run);vm,dm=[validate_majority(p) for p in [visual_run,dn_run]]
    output=Path(output_dir) if output_dir else ROOT/'results/analysis/clip_inmap_hierarchy'/uuid.uuid4().hex[:12]
    output.mkdir(parents=True,exist_ok=False)
    manifest={'status':'running','rule':'parent product with initial InMaP prior only, remove dynamic memory contribution',
        'label_fit':False,'ablation_only':True,'root_scale':25.,'leaf_scale':100.,
        'input_manifest_sha256':{str(p):sha256((p/'run_manifest.json').read_bytes()) for p in [visual_run,dn_run]},
        'code_sha256':{name:sha256((ROOT/name).read_bytes()) for name in ['experiments/clip_dmn/ablate_memory.py','experiments/clip_dmn/hierarchy.py']}}
    write_json(output/'frozen_plan.json',manifest);write_json(output/'run_manifest.json',manifest)
    try:
        reports={}
        for scope in ['train','dev']:
            s=load_features(project_path(vm['settings'][scope+'_run']));ids=[r['id'] for r in s['rows']]
            if s['feature_sha256']!=vm['source_features'][scope]['sha256'] or s['feature_sha256']!=dm['source_feature_sha256'][scope]:raise ValueError('Source khác cohort')
            with np.load(dn_run/scope/'scores.npz',allow_pickle=False) as z:
                if z['ids'].tolist()!=ids:raise ValueError('Sai DN ID')
                leaf=z['humanitarian'].copy()
            trials=[]
            for seed in [0,1,2]:
                with np.load(visual_run/scope/f'shuffle_{seed}/probabilities.npz',allow_pickle=False) as z:
                    if z['ids'].tolist()!=ids:raise ValueError('Sai InMaP ID')
                    root=z['info_text_probability'].copy()
                predicted,binary,_=parent_product(root,leaf,s['info_labels'],s['labels'])
                trials.append({'seed':seed,**evaluate(s,predicted,binary)})
                with (output/f'{scope}_predictions_seed_{seed}.tsv').open('x',newline='') as f:
                    w=csv.writer(f,delimiter='\t');w.writerow(['id','y_true','binary_truth','y_pred','binary_pred'])
                    w.writerows((r['id'],r['y_true'],s['binary_truth'][i] if s['binary_truth'] is not None else '',predicted[i],binary[i]) for i,r in enumerate(s['rows']))
            reports[scope]={'samples':len(ids),'by_seed':trials,
                'mean':{task:{key:float(np.mean([r[task][key] for r in trials])) for key in ['accuracy','macro_f1']}
                        if trials[0][task] is not None else None for task in ['binary','humanitarian']}}
            print(scope,reports[scope]['mean'],'rescue',trials[0]['humanitarian']['per_class']['rescue_volunteering_or_donation_effort']['recall'],flush=True)
        reports['limitations']=['Component ablation, not a temperature/weight search; remove DMN memory entirely.',
            'Initial InMaP prior is deterministic, three stored shuffle reports are not independent runs.',
            'Retain leaf output unlike original CHiLS; train/dev exploratory only, no test.']
        write_json(output/'analysis.json',reports)
        manifest['status']='complete';manifest['output_sha256']={str(p.relative_to(output)):sha256(p.read_bytes()) for p in output.rglob('*') if p.is_file() and p.name!='run_manifest.json'}
        write_json(output/'run_manifest.json',manifest);print('[SAVED]',output);return output
    except Exception as error:
        manifest.update(status='failed',error=f'{type(error).__name__}: {error}');write_json(output/'run_manifest.json',manifest);raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--visual-run',required=True);p.add_argument('--dn-run',required=True);p.add_argument('--output-dir')
    a=p.parse_args();run(a.visual_run,a.dn_run,a.output_dir)
