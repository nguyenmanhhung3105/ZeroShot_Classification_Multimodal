"""LAME ở cổng và/hoặc tầng DN-unit; hai kernel cố định, không đọc test."""
import argparse
import csv
import json
from pathlib import Path
import sys
import uuid
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import numpy as np
from experiments.clip_lame.core import batches,PARAMETERS,REFERENCE_COMMIT
from experiments.clip_tda.run import load_features
from experiments.clip_zlap.run import project_path,validate_majority,majority_gates,evaluate
from experiments.clip_transductive.ensemble import humanitarian
from experiments.clip_transductive.run import sha256,write_json


def run(settings,output_dir=None):
    if (settings.get('affinities')!=['rbf','knn'] or settings.get('grouping_seeds')!=[0,1,2] or
            settings.get('query_size')!=75 or settings.get('parameters')!=PARAMETERS or settings.get('label_fit') is not False):
        raise ValueError('Giữ hai kernel, ba seed và cấu hình đã khóa; không search dev')
    output=Path(output_dir) if output_dir else ROOT/'results/analysis/clip_lame'/uuid.uuid4().hex[:12]
    output.mkdir(parents=True,exist_ok=False)
    manifest={'status':'running','settings':settings,'reference_commit':REFERENCE_COMMIT,'label_fit':False,'source_features':{},
              'method':'LAME on raw CLIP binary scores and cached DN-unit humanitarian scores',
              'code_sha256':{name:sha256((ROOT/name).read_bytes()) for name in ['experiments/clip_lame/core.py','experiments/clip_lame/run.py']}}
    write_json(output/'frozen_plan.json',manifest);write_json(output/'run_manifest.json',manifest)
    try:
        dn_dir=project_path(settings['dn_run']);dn_manifest=validate_majority(dn_dir)
        majority_dir=project_path(settings['majority_run']);majority_manifest=validate_majority(majority_dir)
        manifest['input_manifest_sha256']={str(p):sha256((p/'run_manifest.json').read_bytes()) for p in [dn_dir,majority_dir]}
        results={}
        for scope in ['train','dev']:
            source=load_features(project_path(settings[scope+'_run']))
            if source['split']!=scope or source['feature_sha256']!=dn_manifest['source_feature_sha256'][scope]:raise ValueError('Sai split/source DN')
            manifest['source_features'][scope]={'directory':source['directory'],'sha256':source['feature_sha256']}
            x,il,labels=source['features'],source['info_labels'],source['labels'];a,b=source['scores']
            with np.load(dn_dir/scope/'scores.npz',allow_pickle=False) as pack:
                if pack['ids'].tolist()!=[r['id'] for r in source['rows']]:raise ValueError('DN khác ID/thứ tự')
                dn_leaf=pack['humanitarian'].copy()
            majority=majority_gates(source,scope,majority_dir,majority_manifest)
            raw=np.asarray(il)[a.argmax(1)]=='informative'
            summary={'samples':len(x),'baseline':evaluate(source,humanitarian(raw,b,labels),np.where(raw,'informative','not_informative').tolist()),'affinities':{}}
            for affinity in ['rbf','knn']:
                reports={name:[] for name in ['lame_gate','majority_lame_dn_leaf','lame_both','majority_dn_control']}
                folder=output/scope/affinity;folder.mkdir(parents=True)
                for seed in [0,1,2]:
                    print('[LAME]',scope,affinity,'seed',seed,flush=True)
                    root,groups,root_log=batches(x,a,seed,affinity)
                    leaf,leaf_groups,leaf_log=batches(x,dn_leaf,seed,affinity)
                    if not np.array_equal(groups,leaf_groups):raise ValueError('LAME hai task khác grouping')
                    gate=np.asarray(il)[root.argmax(1)]=='informative'
                    choices={'lame_gate':humanitarian(gate,b,labels),
                             'majority_lame_dn_leaf':humanitarian(majority[seed],leaf,labels),
                             'lame_both':humanitarian(gate,leaf,labels),
                             'majority_dn_control':humanitarian(majority[seed],dn_leaf,labels)}
                    for name,pred in choices.items():
                        g=majority[seed] if name.startswith('majority') else gate
                        reports[name].append({'seed':seed,**evaluate(source,pred,np.where(g,'informative','not_informative').tolist())})
                    with (folder/f'predictions_seed_{seed}.tsv').open('x',newline='') as f:
                        writer=csv.writer(f,delimiter='\t');writer.writerow(['id','y_true','binary_truth','lame_informative_gate','majority_gate','batch_id',*choices])
                        writer.writerows((r['id'],r['y_true'],source['binary_truth'][i] if source['binary_truth'] is not None else '',bool(gate[i]),bool(majority[seed][i]),int(groups[i]),
                            *[v[i] for v in choices.values()]) for i,r in enumerate(source['rows']))
                    np.savez_compressed(folder/f'posteriors_seed_{seed}.npz',ids=np.array([r['id'] for r in source['rows']]),informativeness=root,humanitarian=leaf,batch_ids=groups)
                    write_json(folder/f'diagnostics_seed_{seed}.json',{'informativeness':root_log,'humanitarian':leaf_log})
                means={name:{task:{key:float(np.mean([r[task][key] for r in group])) for key in ['accuracy','macro_f1']}
                             if group[0][task] is not None else None for task in ['binary','humanitarian']} for name,group in reports.items()}
                summary['affinities'][affinity]={'by_method':reports,'mean':means}
                for name,report in means.items():print('[RESULT]',scope,affinity,name,report,flush=True)
            results[scope]=summary
        results['limitations']=['Two fixed author-supported affinities, no tuning of k/lambda/scale on dev.',
            'Query size 75 and same seed grouping as existing EM/StatA; this is a project protocol, not original LAME benchmark.',
            'RBF uses adaptive sigma from fifth neighbor distance as official code; configured sigma field is not an extra tunable coefficient.',
            'No ground-truth input to adaptation, no weight updates or support labels; no test read.',
            'Finite-batch output adaptation; labels/classes/priors need not be uniformly balanced.',
            'Train/dev reused for exploratory research; no independent-test claim.']
        write_json(output/'analysis.json',results)
        with (output/'summary.tsv').open('x',newline='') as f:
            writer=csv.writer(f,delimiter='\t');writer.writerow(['split','affinity','method','task','samples','accuracy','macro_f1'])
            for scope in ['train','dev']:
                for affinity,item in results[scope]['affinities'].items():
                    for name,tasks in item['mean'].items():
                        for task,values in tasks.items():
                            if values is not None:writer.writerow([scope,affinity,name,task,results[scope]['samples'],values['accuracy'],values['macro_f1']])
        manifest['status']='complete';manifest['output_sha256']={str(p.relative_to(output)):sha256(p.read_bytes()) for p in output.rglob('*') if p.is_file() and p.name!='run_manifest.json'}
        write_json(output/'run_manifest.json',manifest);print('[SAVED]',output);return output
    except Exception as error:
        manifest.update(status='failed',error=f'{type(error).__name__}: {error}');write_json(output/'run_manifest.json',manifest);raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default=str(ROOT/'experiments/clip_lame/config.json'));p.add_argument('--output-dir')
    a=p.parse_args();run(json.loads(Path(a.config).read_text()),a.output_dir)
