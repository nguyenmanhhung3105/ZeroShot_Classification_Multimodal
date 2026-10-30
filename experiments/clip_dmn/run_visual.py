"""Mở rộng dự án: DMN single-view với InMaP visual anchor, scale=1/tau_i."""
import argparse
import csv
import json
from pathlib import Path
import sys
import uuid
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import numpy as np
from experiments.clip_dmn.core import predict_stream,PARAMETERS
from experiments.clip_tda.run import load_features
from experiments.clip_zlap.run import project_path,validate_majority,evaluate
from experiments.clip_transductive.ensemble import humanitarian
from experiments.clip_transductive.run import sha256,write_json

VISUAL_PARAMETERS={**PARAMETERS,'logit_scale':25.}


def run(settings,output_dir=None):
    if (settings.get('parameters')!=VISUAL_PARAMETERS or settings.get('anchor')!='unbalanced_inmap_visual_proxy' or settings.get('label_fit') is not False):
        raise ValueError('Chỉ chạy InMaP visual anchor và scale=25 đã khóa')
    output=Path(output_dir) if output_dir else ROOT/'results/analysis/clip_dmn_visual'/uuid.uuid4().hex[:12]
    output.mkdir(parents=True,exist_ok=False)
    manifest={'status':'running','settings':settings,'label_fit':False,'support_samples':0,'single_view_adaptation':True,
        'method':'project extension, not published DMN-ZS benchmark; visual InMaP anchors in place of text',
        'scale':'25 = 1 / InMaP tau_i (0.04), fixed before metrics; no confidence-scale sweep',
        'source_features':{},'code_sha256':{name:sha256((ROOT/name).read_bytes()) for name in ['experiments/clip_dmn/core.py','experiments/clip_dmn/run_visual.py']},
        'input_manifest_sha256':{key:sha256((project_path(settings[key])/'run_manifest.json').read_bytes()) for key in ['inmap_run','dn_run']}}
    write_json(output/'frozen_plan.json',manifest);write_json(output/'run_manifest.json',manifest)
    try:
        im_dir,dn_dir=[project_path(settings[key]) for key in ['inmap_run','dn_run']]
        im,dn=[validate_majority(p) for p in [im_dir,dn_dir]];results={}
        for scope in ['train','dev']:
            source=load_features(project_path(settings[scope+'_run']))
            if source['split']!=scope or source['feature_sha256']!=im['source_features'][scope]['sha256'] or source['feature_sha256']!=dn['source_feature_sha256'][scope]:
                raise ValueError('Các thành phần khác cohort/feature')
            manifest['source_features'][scope]={'directory':source['directory'],'sha256':source['feature_sha256']}
            il,hl=source['info_labels'],source['labels'];x=source['features']
            with np.load(im_dir/scope/'unbalanced/proxies_scores.npz',allow_pickle=False) as pack,np.load(dn_dir/scope/'scores.npz',allow_pickle=False) as cached:
                ids=[r['id'] for r in source['rows']]
                if pack['ids'].tolist()!=ids or cached['ids'].tolist()!=ids:raise ValueError('Sai ID/thứ tự')
                info_proxy=pack['info_proxy'].copy();human_proxy=pack['human_proxy'].copy();dn_leaf=cached['humanitarian'].copy()
                inmap_gate=np.asarray(il)[pack['informativeness'].argmax(1)]=='informative'
                inmap_leaf=pack['humanitarian'].copy()
            summary={'samples':len(x),'control_inmap_dn':evaluate(source,humanitarian(inmap_gate,dn_leaf,hl),np.where(inmap_gate,'informative','not_informative').tolist()),
                'by_method':{name:[] for name in ['visual_dmn_gate_dn_leaf','inmap_visual_dmn_leaf','visual_dmn_both']}}
            for name in ['source_order','shuffle_0','shuffle_1','shuffle_2']:
                order=np.arange(len(x)) if name=='source_order' else np.random.default_rng(int(name[-1])).permutation(len(x))
                print('[DMN VISUAL]',scope,name,flush=True)
                root,rt,rd=predict_stream(x,info_proxy,order,**VISUAL_PARAMETERS)
                leaf,lt,ld=predict_stream(x,human_proxy,order,**VISUAL_PARAMETERS)
                # Giữ nguyên nhãn InMaP trước khi gắn memory; không âm thầm đổi classifier.
                if not np.array_equal(np.asarray(il)[rt['text_probability'].argmax(1)]=='informative',inmap_gate) or not np.array_equal(lt['text_probability'].argmax(1),inmap_leaf.argmax(1)):
                    raise ValueError('Normalizing visual anchor làm đổi nhãn InMaP trước memory')
                gate=np.asarray(il)[root.argmax(1)]=='informative'
                choices={'visual_dmn_gate_dn_leaf':humanitarian(gate,dn_leaf,hl),'inmap_visual_dmn_leaf':humanitarian(inmap_gate,leaf,hl),
                    'visual_dmn_both':humanitarian(gate,leaf,hl)}
                for method,pred in choices.items():
                    used=inmap_gate if method.startswith('inmap') else gate
                    report={'order':name,**evaluate(source,pred,np.where(used,'informative','not_informative').tolist())}
                    summary['by_method'][method].append(report)
                    print('[RESULT]',scope,name,method,report['humanitarian']['macro_f1'],report['binary']['macro_f1'] if report['binary'] else None,flush=True)
                folder=output/scope/name;folder.mkdir(parents=True)
                np.savez_compressed(folder/'probabilities.npz',ids=np.array(ids),informativeness=root,humanitarian=leaf,
                    **{'info_'+k:v for k,v in rt.items()},**{'human_'+k:v for k,v in lt.items()})
                write_json(folder/'diagnostics.json',{'informativeness':rd,'humanitarian':ld})
                with (folder/'predictions.tsv').open('x',newline='') as f:
                    writer=csv.writer(f,delimiter='\t');writer.writerow(['id','y_true','binary_truth','visual_dmn_gate','inmap_gate','stream_position',*choices])
                    positions=np.argsort(order)
                    writer.writerows((r['id'],r['y_true'],source['binary_truth'][i] if source['binary_truth'] is not None else '',bool(gate[i]),bool(inmap_gate[i]),int(positions[i]),
                        *[p[i] for p in choices.values()]) for i,r in enumerate(source['rows']))
            summary['shuffle_mean']={method:{task:{key:float(np.mean([r[task][key] for r in trials if r['order']!='source_order'])) for key in ['accuracy','macro_f1']}
                if trials[0][task] is not None else None for task in ['binary','humanitarian']} for method,trials in summary['by_method'].items()}
            results[scope]=summary
        results['limitations']=['Project extension motivated by pseudo-label errors; not an original DMN-ZS configuration or benchmark reproduction.',
            'Single-view, fixed equal probability weights and visual scale 25; no threshold/temperature/weight search.',
            'Full-cohort InMaP fitting supplies the anchors; this is not a purely causal method overall even though memory is causal.',
            'Only train/dev exploratory, no test; no label fit or labeled support.']
        write_json(output/'analysis.json',results)
        with (output/'summary.tsv').open('x',newline='') as f:
            w=csv.writer(f,delimiter='\t');w.writerow(['split','method','task','samples','accuracy','macro_f1'])
            for scope in ['train','dev']:
                for method,tasks in results[scope]['shuffle_mean'].items():
                    for task,v in tasks.items():
                        if v is not None:w.writerow([scope,method,task,results[scope]['samples'],v['accuracy'],v['macro_f1']])
        manifest['status']='complete';manifest['output_sha256']={str(p.relative_to(output)):sha256(p.read_bytes()) for p in output.rglob('*') if p.is_file() and p.name!='run_manifest.json'}
        write_json(output/'run_manifest.json',manifest);print('[SAVED]',output,flush=True);return output
    except Exception as error:
        manifest.update(status='failed',error=f'{type(error).__name__}: {error}');write_json(output/'run_manifest.json',manifest);raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default=str(ROOT/'experiments/clip_dmn/config_visual.json'));p.add_argument('--output-dir')
    a=p.parse_args();run(json.loads(Path(a.config).read_text()),a.output_dir)
