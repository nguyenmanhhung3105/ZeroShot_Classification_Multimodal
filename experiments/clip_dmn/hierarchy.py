"""CHiLS-inspired parent reweighting cho root DMN-visual và leaf DN-unit.

Giữ nhãn lá làm đầu ra, khác benchmark CHiLS ánh xạ về nhãn cha. Không phải
P(child|parent) đã hiệu chuẩn; chỉ tích hai nguồn confidence chưa hiệu chuẩn.
"""
import argparse
import csv
import json
from pathlib import Path
import sys
import uuid
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import numpy as np
from scipy.special import softmax
from experiments.clip_tda.run import load_features
from experiments.clip_zlap.run import project_path,validate_majority,evaluate
from experiments.clip_transductive.ensemble import humanitarian
from experiments.clip_transductive.run import sha256,write_json


def parent_product(root_probability,leaf_cosine,info_labels,human_labels):
    root=np.asarray(root_probability,dtype=np.float64);leaf=np.asarray(leaf_cosine,dtype=np.float64)
    if (set(info_labels)!={'informative','not_informative'} or len(info_labels)!=2 or
            len(human_labels)!=8 or len(set(human_labels))!=8 or 'not_humanitarian' not in human_labels or
            root.shape!=(len(leaf),2) or leaf.shape!=(len(root),8) or len(root)==0 or
            not np.isfinite(root).all() or not np.isfinite(leaf).all() or np.any(np.abs(leaf)>1+1e-6) or
            np.any(root<0) or np.any(root>1) or not np.allclose(root.sum(1),1,atol=1e-6,rtol=0)):
        raise ValueError('Sai taxonomy, posterior root hoặc cosine leaf')
    parent=[info_labels.index('not_informative' if label=='not_humanitarian' else 'informative') for label in human_labels]
    weighted=softmax(100.*leaf,axis=1)*root[:,parent]
    if np.any(weighted.sum(1)<=0):raise ValueError('Tích confidence có mass bằng zero')
    relative=weighted/weighted.sum(1,keepdims=True)
    predicted=np.asarray(human_labels)[weighted.argmax(1)].tolist()
    binary=np.where(np.asarray(predicted)!='not_humanitarian','informative','not_informative').tolist()
    return predicted,binary,relative


def run(settings,output_dir=None):
    if settings.get('rule')!='parent_product_coefficient_one_leaf_scale_100' or settings.get('label_fit') is not False:
        raise ValueError('Chỉ dùng tích confidence cố định, không tune weights/scale')
    output=Path(output_dir) if output_dir else ROOT/'results/analysis/clip_dmn_hierarchy'/uuid.uuid4().hex[:12]
    output.mkdir(parents=True,exist_ok=False)
    root_dir,dn_dir=[project_path(settings[k]) for k in ['visual_run','dn_run']]
    rm,dm=[validate_majority(p) for p in [root_dir,dn_dir]]
    manifest={'status':'running','settings':settings,'label_fit':False,'probabilities_calibrated':False,
              'paper_inspiration':'CHiLS ICML 2023 Algorithm 1 parent reweighting; retain leaf output in this project extension',
              'source_manifests':{str(p):sha256((p/'run_manifest.json').read_bytes()) for p in [root_dir,dn_dir]},
              'source_sha256':sha256(Path(__file__).read_bytes())}
    write_json(output/'frozen_plan.json',manifest);write_json(output/'run_manifest.json',manifest)
    try:
        results={}
        for scope in ['train','dev']:
            source=load_features(project_path(rm['settings'][scope+'_run']));ids=[r['id'] for r in source['rows']]
            if source['feature_sha256']!=rm['source_features'][scope]['sha256'] or source['feature_sha256']!=dm['source_feature_sha256'][scope]:
                raise ValueError('Root/leaf khác feature/cohort')
            with np.load(dn_dir/scope/'scores.npz',allow_pickle=False) as z:
                if z['ids'].tolist()!=ids:raise ValueError('DN khác ID')
                leaf=z['humanitarian'].copy()
            reports={name:[] for name in ['hard_gate_control','parent_product']}
            folder=output/scope;folder.mkdir()
            for seed in [0,1,2]:
                with np.load(root_dir/scope/f'shuffle_{seed}/probabilities.npz',allow_pickle=False) as z:
                    if z['ids'].tolist()!=ids:raise ValueError('DMN khác ID')
                    root=z['informativeness'].copy()
                pred,binary,relative=parent_product(root,leaf,source['info_labels'],source['labels'])
                hard_gate=np.asarray(source['info_labels'])[root.argmax(1)]=='informative'
                hard=humanitarian(hard_gate,leaf,source['labels'])
                choices={'hard_gate_control':(hard,np.where(hard_gate,'informative','not_informative').tolist()),'parent_product':(pred,binary)}
                for name,(predicted,binary_pred) in choices.items():reports[name].append({'seed':seed,**evaluate(source,predicted,binary_pred)})
                with (folder/f'predictions_seed_{seed}.tsv').open('x',newline='') as f:
                    w=csv.writer(f,delimiter='\t');w.writerow(['id','y_true','binary_truth','hard_pred','soft_pred','soft_binary_pred'])
                    w.writerows((r['id'],r['y_true'],source['binary_truth'][i] if source['binary_truth'] is not None else '',hard[i],pred[i],binary[i]) for i,r in enumerate(source['rows']))
                np.savez_compressed(folder/f'relative_scores_seed_{seed}.npz',ids=np.asarray(ids),relative_scores=relative)
            mean={method:{task:{key:float(np.mean([r[task][key] for r in rows])) for key in ['accuracy','macro_f1']}
                          if rows[0][task] is not None else None for task in ['binary','humanitarian']} for method,rows in reports.items()}
            results[scope]={'samples':len(ids),'by_method':reports,'mean':mean}
            print(scope,mean,flush=True)
        results['limitations']=['CHiLS-inspired fusion, not original CHiLS benchmark or new label-set generation.',
            'Fine score is over all eight labels, not a conditional distribution over seven; do not call this chain-rule inference.',
            'Binary prediction is model-derived from selected leaf, never ground-truth-derived; no binary dev GT.',
            'One fixed product rule; exploratory on reused train/dev, no test or calibration fitting.']
        write_json(output/'analysis.json',results)
        manifest['status']='complete';manifest['output_sha256']={str(p.relative_to(output)):sha256(p.read_bytes()) for p in output.rglob('*') if p.is_file() and p.name!='run_manifest.json'}
        write_json(output/'run_manifest.json',manifest);print('[SAVED]',output);return output
    except Exception as error:
        manifest.update(status='failed',error=f'{type(error).__name__}: {error}');write_json(output/'run_manifest.json',manifest);raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default=str(ROOT/'experiments/clip_dmn/config_hierarchy.json'));p.add_argument('--output-dir')
    a=p.parse_args();run(json.loads(Path(a.config).read_text()),a.output_dir)
