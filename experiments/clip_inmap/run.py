"""Thử InMaP cổng và tầng chi tiết trên cached feature train/dev."""
import argparse
import csv
import json
from pathlib import Path
import sys
import uuid
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import numpy as np
from experiments.clip_inmap.core import optimize,PARAMETERS,REFERENCE_COMMIT
from experiments.clip_tda.run import load_features
from experiments.clip_zlap.run import project_path,text_prototypes,validate_majority,majority_gates,evaluate
from experiments.clip_transductive.ensemble import humanitarian
from experiments.clip_transductive.run import sha256,write_json,metric


def run(settings,output_dir=None):
    if (settings.get('variants')!=['unbalanced','balanced_sinkhorn'] or settings.get('parameters')!=PARAMETERS or
            settings.get('ground_truth_fit') is not False or settings.get('scope')!='independent_train_dev_transductive_proxy_optimization_no_test'):
        raise ValueError('Chỉ chạy hai biến thể có/không Sinkhorn đã khóa')
    output=Path(output_dir) if output_dir else ROOT/'results/analysis/clip_inmap'/uuid.uuid4().hex[:12]
    output.mkdir(parents=True,exist_ok=False)
    manifest={'status':'running','settings':settings,'reference_commit':REFERENCE_COMMIT,'label_fit':False,
              'pseudo_label_optimization':True,'backbone_frozen':True,'source_features':{},
              'fit_scope':'each split independently with no labels; no train/dev image mixing',
              'code_sha256':{name:sha256((ROOT/name).read_bytes()) for name in ['experiments/clip_inmap/core.py','experiments/clip_inmap/run.py',
                  'experiments/clip_tda/run.py','experiments/clip_zlap/run.py','experiments/clip_zlap/core.py']}}
    write_json(output/'frozen_plan.json',manifest);write_json(output/'run_manifest.json',manifest)
    try:
        donor=load_features(project_path(settings['text_prototype_run']))
        directory=project_path(settings['majority_run']);majority_manifest=validate_majority(directory)
        results={}
        for scope in ['train','dev']:
            source=load_features(project_path(settings[scope+'_run']))
            if source['split']!=scope:raise ValueError('Sai split')
            manifest['source_features'][scope]={'directory':source['directory'],'sha256':source['feature_sha256']}
            x,il,labels=source['features'],source['info_labels'],source['labels'];a,b=source['scores']
            human_text=text_prototypes(source,donor);gates=majority_gates(source,scope,directory,majority_manifest)
            raw=np.asarray(il)[a.argmax(1)]=='informative'
            normal=np.asarray(il)[(x@source['prototypes'].T).argmax(1)]=='informative'
            summary={'samples':len(x),'controls':{
                'baseline':evaluate(source,humanitarian(raw,b,labels),np.where(raw,'informative','not_informative').tolist()),
                'normalized_both':evaluate(source,humanitarian(normal,x@human_text.T,labels),np.where(normal,'informative','not_informative').tolist())},'variants':{}}
            for variant in settings['variants']:
                print(f'[INMAP] {scope}; {variant}; root 2 classes; iterations=2000',flush=True)
                root_scores,root_proxy,root_log=optimize(x,source['prototypes'],balanced=variant=='balanced_sinkhorn',**PARAMETERS)
                print(f'[INMAP] {scope}; {variant}; leaf 8 classes; iterations=2000',flush=True)
                leaf_scores,leaf_proxy,leaf_log=optimize(x,human_text,balanced=variant=='balanced_sinkhorn',**PARAMETERS)
                gate=np.asarray(il)[root_scores.argmax(1)]=='informative'
                reports={name:[] for name in ['inmap_gate','majority_inmap_leaf','inmap_two_stage','inmap_flat','majority_control']}
                folder=output/scope/variant;folder.mkdir(parents=True)
                for seed in [0,1,2]:
                    choices={'inmap_gate':humanitarian(gate,b,labels),'majority_inmap_leaf':humanitarian(gates[seed],leaf_scores,labels),
                        'inmap_two_stage':humanitarian(gate,leaf_scores,labels),'inmap_flat':np.asarray(labels)[leaf_scores.argmax(1)].tolist(),
                        'majority_control':humanitarian(gates[seed],b,labels)}
                    for name,pred in choices.items():
                        if name=='inmap_flat':report={'humanitarian':metric([r['y_true'] for r in source['rows']],pred,labels),'binary':None}
                        else:
                            g=gates[seed] if name.startswith('majority') else gate
                            report=evaluate(source,pred,np.where(g,'informative','not_informative').tolist())
                        reports[name].append({'seed':seed,**report})
                    with (folder/f'predictions_seed_{seed}.tsv').open('x',newline='') as f:
                        writer=csv.writer(f,delimiter='\t');writer.writerow(['id','y_true','binary_truth','inmap_informative_gate','majority_gate',*choices])
                        writer.writerows((r['id'],r['y_true'],source['binary_truth'][i] if source['binary_truth'] is not None else '',bool(gate[i]),bool(gates[seed][i]),
                            *[v[i] for v in choices.values()]) for i,r in enumerate(source['rows']))
                means={name:{task:{key:float(np.mean([r[task][key] for r in group])) for key in ['accuracy','macro_f1']}
                             if group[0][task] is not None else None for task in ['binary','humanitarian']} for name,group in reports.items()}
                summary['variants'][variant]={'by_method':reports,'mean':means}
                np.savez_compressed(folder/'proxies_scores.npz',ids=np.array([r['id'] for r in source['rows']]),
                    informativeness=root_scores,humanitarian=leaf_scores,info_proxy=root_proxy,human_proxy=leaf_proxy)
                write_json(folder/'diagnostics.json',{'informativeness':root_log,'humanitarian':leaf_log})
                for name,report in means.items():print('[RESULT]',scope,variant,name,report,flush=True)
            results[scope]=summary
        results['limitations']=['Only train/dev exploratory. No test access; no hyperparameter search.',
            'Pseudo-label proxy fitting is optimization, not encoder fine-tuning or supervised/few-shot learning.',
            'Balanced Sinkhorn assumes uniform class mass before hardening; not appropriate by default for skewed CrisisMMD.',
            'Deterministic full-cohort fitting; repeated seeds only refer to cached majority gates.',
            'Paper backbone/prompts differ; no claim to reproduce original benchmark numbers.']
        write_json(output/'analysis.json',results)
        with (output/'summary.tsv').open('x',newline='') as f:
            writer=csv.writer(f,delimiter='\t');writer.writerow(['split','task','variant','method','samples','accuracy','macro_f1'])
            for scope in ['train','dev']:
                for variant,s in results[scope]['variants'].items():
                    for name,tasks in s['mean'].items():
                        for task,values in tasks.items():
                            if values is not None:writer.writerow([scope,task,variant,name,results[scope]['samples'],values['accuracy'],values['macro_f1']])
        manifest['status']='complete';manifest['output_sha256']={str(p.relative_to(output)):sha256(p.read_bytes()) for p in output.rglob('*') if p.is_file() and p.name!='run_manifest.json'}
        write_json(output/'run_manifest.json',manifest);print('[SAVED]',output,flush=True);return output
    except Exception as error:
        manifest.update(status='failed',error=f'{type(error).__name__}: {error}');write_json(output/'run_manifest.json',manifest);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',default=str(ROOT/'experiments/clip_inmap/config.json'));parser.add_argument('--output-dir')
    args=parser.parse_args();run(json.loads(Path(args.config).read_text()),args.output_dir)
