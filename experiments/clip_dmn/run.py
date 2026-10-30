"""Đo DMN-ZS single-view trên cached feature train/dev, giữ cả 2/8 nhãn."""
import argparse
import csv
import json
from pathlib import Path
import sys
import uuid
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import numpy as np
from experiments.clip_dmn.core import predict_stream,PARAMETERS,REFERENCE_COMMIT
from experiments.clip_tda.run import load_features
from experiments.clip_zlap.run import project_path,text_prototypes,validate_majority,majority_gates,evaluate
from experiments.clip_transductive.ensemble import humanitarian
from experiments.clip_transductive.run import sha256,write_json
from experiments.clip_two_stage.analyze import read_artifact

METHODS=['dmn_gate_raw_leaf','dmn_gate_dn_leaf','majority_dmn_leaf','inmap_dmn_leaf','dmn_both']
ORDERS=['source_order','shuffle_0','shuffle_1','shuffle_2']


def load_companions(source,scope,settings):
    dn_dir,inmap_dir=[project_path(settings[key]) for key in ['dn_run','inmap_run']]
    dm,im=[validate_majority(p) for p in [dn_dir,inmap_dir]]
    if (dm['source_feature_sha256'][scope]!=source['feature_sha256'] or
            im['source_features'][scope]['sha256']!=source['feature_sha256']):raise ValueError('Các head khác source feature')
    with np.load(dn_dir/scope/'scores.npz',allow_pickle=False) as d,np.load(inmap_dir/scope/'unbalanced/proxies_scores.npz',allow_pickle=False) as i:
        ids=[r['id'] for r in source['rows']]
        if d['ids'].tolist()!=ids or i['ids'].tolist()!=ids:raise ValueError('Head DN/InMaP khác ID/thứ tự')
        dn=d['humanitarian'].copy();gate=np.asarray(source['info_labels'])[i['informativeness'].argmax(1)]=='informative'
    return dn,gate


def run(settings,output_dir=None):
    if (settings.get('parameters')!=PARAMETERS or settings.get('methods')!=METHODS or settings.get('stream_orders')!=ORDERS or
            settings.get('label_fit') is not False or settings.get('protocol')!='single_view_dmn_zs_adaptation_fixed_equal_weights_no_search_train_dev'):
        raise ValueError('Chỉ chạy cấu hình DMN single-view, equal weights và bốn order đã khóa')
    output=Path(output_dir) if output_dir else ROOT/'results/analysis/clip_dmn'/uuid.uuid4().hex[:12]
    output.mkdir(parents=True,exist_ok=False)
    manifest={'status':'running','settings':settings,'reference_commit':REFERENCE_COMMIT,'label_fit':False,'support_samples':0,
              'single_view_adaptation':True,'projection_optimization':False,'weight_search':False,
              'code_sha256':{name:sha256((ROOT/name).read_bytes()) for name in ['experiments/clip_dmn/core.py','experiments/clip_dmn/run.py',
                  'experiments/clip_tda/run.py','experiments/clip_zlap/run.py','experiments/clip_transductive/ensemble.py']},
              'input_manifest_sha256':{key:sha256((project_path(settings[key])/'run_manifest.json').read_bytes()) for key in ['majority_run','dn_run','inmap_run']},
              'source_features':{},'source_artifact_sha256':{}}
    write_json(output/'frozen_plan.json',manifest);write_json(output/'run_manifest.json',manifest)
    try:
        donor=load_features(project_path(settings['text_prototype_run']))
        majority_dir=project_path(settings['majority_run']);majority_manifest=validate_majority(majority_dir)
        results={}
        for scope in ['train','dev']:
            source=load_features(project_path(settings[scope+'_run']))
            if source['split']!=scope or source['manifest']['model']!='metaclip2_vith14' or source['manifest'].get('logit_scale')!=100.:
                raise ValueError('Chỉ cùng split/checkpoint MetaCLIP2 đã khóa')
            manifest['source_features'][scope]={'directory':source['directory'],'sha256':source['feature_sha256']}
            manifest['source_artifact_sha256'][scope]={name:sha256(read_artifact(Path(source['directory']),name).encode()) for name in [
                'run_manifest.json','predictions.tsv','informativeness/cohort.json','humanitarian/cohort.json']}
            x,il,hl=source['features'],source['info_labels'],source['labels'];a,b=source['scores']
            human_t=text_prototypes(source,donor);majority=majority_gates(source,scope,majority_dir,majority_manifest)
            dn_leaf,inmap_gate=load_companions(source,scope,settings)
            raw=np.asarray(il)[a.argmax(1)]=='informative'
            normalized=np.asarray(il)[(x@source['prototypes'].T).argmax(1)]=='informative'
            summary={'samples':len(x),'binary_ground_truth_available':source['binary_truth'] is not None,
                'controls':{'baseline':evaluate(source,humanitarian(raw,b,hl),np.where(raw,'informative','not_informative').tolist()),
                    'normalized_both':evaluate(source,humanitarian(normalized,x@human_t.T,hl),np.where(normalized,'informative','not_informative').tolist()),
                    'inmap_dn':evaluate(source,humanitarian(inmap_gate,dn_leaf,hl),np.where(inmap_gate,'informative','not_informative').tolist())},
                'by_method':{name:[] for name in METHODS+['majority_dn_control']}}
            for order_name in ORDERS:
                seed=0 if order_name=='source_order' else int(order_name.rsplit('_',1)[1])
                order=np.arange(len(x)) if order_name=='source_order' else np.random.default_rng(seed).permutation(len(x))
                print(f'[DMN] {scope}; {order_name}; binary memory, then humanitarian memory',flush=True)
                root,root_trace,root_log=predict_stream(x,source['prototypes'],order,**PARAMETERS)
                leaf,leaf_trace,leaf_log=predict_stream(x,human_t,order,**PARAMETERS)
                if order_name=='source_order':
                    root_control=np.asarray(il)[root_trace['text_probability'].argmax(1)]=='informative'
                    if not np.array_equal(root_control,normalized):raise ValueError('DMN text-only root khác normalized control')
                    if not np.array_equal(leaf_trace['text_probability'].argmax(1),(x@human_t.T).argmax(1)):
                        raise ValueError('DMN text-only leaf khác normalized control')
                gate=np.asarray(il)[root.argmax(1)]=='informative'
                choices={'dmn_gate_raw_leaf':humanitarian(gate,b,hl),'dmn_gate_dn_leaf':humanitarian(gate,dn_leaf,hl),
                    'majority_dmn_leaf':humanitarian(majority[seed],leaf,hl),'inmap_dmn_leaf':humanitarian(inmap_gate,leaf,hl),
                    'dmn_both':humanitarian(gate,leaf,hl),'majority_dn_control':humanitarian(majority[seed],dn_leaf,hl)}
                for name,pred in choices.items():
                    g=majority[seed] if name.startswith('majority') else inmap_gate if name.startswith('inmap') else gate
                    report={'order':order_name,'paired_majority_seed':seed,**evaluate(source,pred,np.where(g,'informative','not_informative').tolist())}
                    summary['by_method'][name].append(report)
                    print('[RESULT]',scope,order_name,name,report['humanitarian']['macro_f1'],report['binary']['macro_f1'] if report['binary'] else None,flush=True)
                folder=output/scope/order_name;folder.mkdir(parents=True)
                np.savez_compressed(folder/'probabilities.npz',ids=np.array([r['id'] for r in source['rows']]),
                    informativeness=root,humanitarian=leaf,
                    **{'info_'+k:v for k,v in root_trace.items()},**{'human_'+k:v for k,v in leaf_trace.items()})
                write_json(folder/'diagnostics.json',{'informativeness':root_log,'humanitarian':leaf_log})
                with (folder/'predictions.tsv').open('x',newline='') as f:
                    writer=csv.writer(f,delimiter='\t');writer.writerow(['id','y_true','binary_truth','dmn_informative_gate','majority_gate','inmap_gate','stream_position',*choices])
                    positions=np.argsort(order)
                    writer.writerows((r['id'],r['y_true'],source['binary_truth'][i] if source['binary_truth'] is not None else '',bool(gate[i]),bool(majority[seed][i]),bool(inmap_gate[i]),
                        int(positions[i]),*[p[i] for p in choices.values()]) for i,r in enumerate(source['rows']))
            summary['shuffle_mean']={name:{task:{key:float(np.mean([r[task][key] for r in group if r['order']!='source_order'])) for key in ['accuracy','macro_f1']}
                if group[0][task] is not None else None for task in ['binary','humanitarian']} for name,group in summary['by_method'].items()}
            summary['shuffle_std_population']={name:{task:float(np.std([r[task]['macro_f1'] for r in group if r['order']!='source_order']))
                if group[0][task] is not None else None for task in ['binary','humanitarian']} for name,group in summary['by_method'].items()}
            results[scope]=summary
        results['limitations']=['Single-view adaptation, not full DMN-ZS multi-view benchmark; do not invent augmented views from different images.',
            'Fixed equal probability weights and default memory/beta; author label-based weight search not used.',
            'Cache reset for each task/split/order; first source_order is stress test, excluded from shuffle mean.',
            'Only three paired shuffles; memory stream order and majority grouping share the seed, not independent repetitions.',
            'Binary dev GT absent, metrics null; no label derivation from humanitarian.',
            'Train/dev exploratory and reused; no test or global-best claim.']
        write_json(output/'analysis.json',results)
        with (output/'summary.tsv').open('x',newline='') as f:
            writer=csv.writer(f,delimiter='\t');writer.writerow(['split','method','task','samples','accuracy','macro_f1'])
            for scope in ['train','dev']:
                for name,tasks in results[scope]['shuffle_mean'].items():
                    for task,values in tasks.items():
                        if values is not None:writer.writerow([scope,name,task,results[scope]['samples'],values['accuracy'],values['macro_f1']])
        manifest['status']='complete';manifest['output_sha256']={str(p.relative_to(output)):sha256(p.read_bytes()) for p in output.rglob('*') if p.is_file() and p.name!='run_manifest.json'}
        write_json(output/'run_manifest.json',manifest);print('[SAVED]',output,flush=True);return output
    except Exception as error:
        manifest.update(status='failed',error=f'{type(error).__name__}: {error}');write_json(output/'run_manifest.json',manifest);raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default=str(ROOT/'experiments/clip_dmn/config.json'));p.add_argument('--output-dir')
    a=p.parse_args();run(json.loads(Path(a.config).read_text()),a.output_dir)
