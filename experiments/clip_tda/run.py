"""Đo TDA binary và humanitarian từ feature artifact; không mở dataset/model."""
import argparse
import csv
import io
import json
from pathlib import Path
import sys
import uuid
import zipfile

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import numpy as np
from experiments.clip_tda.core import predict_stream,REFERENCE_COMMIT,PARAMETERS
from experiments.clip_two_stage.analyze import analyze,read_artifact
from experiments.clip_two_stage.run import decide
from experiments.clip_transductive.run import metric,sha256,write_json
from experiments.clip_transductive.ensemble import humanitarian


def load_features(directory, *, allowed_splits=('train','dev')):
    directory=Path(directory)
    analyze(directory,bootstraps=1)
    manifest=json.loads(read_artifact(directory,'run_manifest.json'))
    if manifest.get('input')!='image_only' or manifest.get('full_frame_view'):
        raise ValueError('Cần feature của ảnh gốc, cùng protocol image-only')
    tasks=['informativeness','humanitarian']
    plans=[json.loads(read_artifact(directory,f'{task}/frozen_plan.json')) for task in tasks]
    cohorts=[json.loads(read_artifact(directory,f'{task}/cohort.json'))['rows'] for task in tasks]
    split=plans[0]['dataset_config'].get('source_split')
    if split not in allowed_splits or any(r['source_split']!=split or r['label_source']!='image' for r in cohorts[1]):
        raise ValueError('Split ngoài protocol được phép; mặc định chỉ train/dev, test cần protocol khóa riêng')
    info_labels=plans[0]['label_order'];labels=plans[1]['label_order']
    if set(info_labels)!={'informative','not_informative'} or len(info_labels)!=2 or len(labels)!=8:
        raise ValueError('Cần đúng task hai nhãn và tám nhãn')
    feature_name='stata_features.npz' if manifest.get('stata_gate') else manifest.get('feature_file','clip_features.npz')
    path=directory/feature_name
    if path.is_file():raw=path.read_bytes()
    else:
        with zipfile.ZipFile(directory/'run_artifacts.zip') as archive:raw=archive.read(feature_name)
    expected=manifest.get('stata_features_sha256') if manifest.get('stata_gate') else manifest.get('feature_sha256')
    if sha256(raw)!=expected:raise ValueError('Feature hash không khớp')
    with np.load(io.BytesIO(raw),allow_pickle=False) as pack:
        if pack['ids'].tolist()!=[r['id'] for r in cohorts[1]]:raise ValueError('Feature khác ID/thứ tự')
        x=pack['image_features'].copy();means=pack['info_mean_prototypes'].copy()
    rows=list(csv.DictReader(io.StringIO(read_artifact(directory,'predictions.tsv')),delimiter='\t'))
    a,b=[np.array([json.loads(r[key]) for r in rows]) for key in ['info_cosine_scores','human_cosine_scores']]
    if not np.allclose(x@means.T,a,atol=2e-5,rtol=1e-4):raise ValueError('Embedding/prototype khác cosine')
    norms=np.linalg.norm(means,axis=1,keepdims=True)
    if np.any(norms<=1e-12):raise ValueError('Prototype zero')
    true_binary=[r['y_true'] for r in cohorts[0]]
    if any(value is not None for value in true_binary):
        if any(value not in info_labels for value in true_binary):raise ValueError('Nhãn binary thiếu/không hợp lệ')
    else:true_binary=None
    return {'directory':str(directory),'manifest':manifest,'split':split,'features':x,'prototypes':means/norms,'mean_prototypes':means,
            'info_labels':info_labels,'labels':labels,'scores':[a,b],'rows':rows,'cohort':cohorts[1],
            'binary_truth':true_binary,'prototype_norms':norms[:,0],'feature_sha256':sha256(raw)}


def compare_reference_ensemble(source):
    """Đo đúng ensemble 2/3 hiện tại trên cả nhãn binary thật và humanitarian."""
    from experiments.clip_stata.core import gate_ablation
    from experiments.clip_transductive.gate import predict_binary
    from experiments.clip_transductive.ensemble import combine
    x=source['features'];il=source['info_labels'];labels=source['labels'];a,b=source['scores']
    normalized=np.asarray(il)[(x@source['prototypes'].T).argmax(1)]=='informative'
    _,stata=gate_ablation(x,source['mean_prototypes'],a,b,il,labels)
    human_truth=[r['y_true'] for r in source['rows']];reports=[];predictions={}
    for seed in [0,1,2]:
        em,_,_=predict_binary(a,il,grouping_seed=seed,query_size=75,temperature=30.,iterations=20,mm_iterations=1000)
        stata_gate=np.asarray(stata['gate_labels'][f'stata_gate_seed_{seed}'])=='informative'
        gate=combine(np.stack([normalized,np.asarray(em)=='informative',stata_gate],1))['majority']
        binary=np.where(gate,'informative','not_informative').tolist()
        human=humanitarian(gate,b,labels)
        reports.append({'seed':seed,'humanitarian':metric(human_truth,human,labels),
                        'binary':metric(source['binary_truth'],binary,il) if source['binary_truth'] is not None else None})
        predictions[str(seed)]={'binary_pred':binary,'humanitarian_pred':human}
    return {'rule':'majority_normalized_clip_raw_em_stata','by_seed':reports,'predictions':predictions,
            'mean':{scope:{key:float(np.mean([r[scope][key] for r in reports])) for key in ['accuracy','macro_f1']}
                    if reports[0][scope] is not None else None for scope in ['binary','humanitarian']}}


def run(directory,output_dir=None,compare_ensemble=False):
    source=load_features(directory)
    output=Path(output_dir) if output_dir else ROOT/'results/analysis/clip_tda'/uuid.uuid4().hex[:12]
    output.mkdir(parents=True,exist_ok=False)
    manifest={'status':'running','source_run':str(directory),'source_split':source['split'],
              'samples':len(source['rows']),'model':source['manifest']['model'],
              'reference_commit':REFERENCE_COMMIT,'parameters':PARAMETERS,
              'stream_orders':['source_order','shuffle_0','shuffle_1','shuffle_2'],
              'label_fit':False,'binary_ground_truth_available':source['binary_truth'] is not None,
              'feature_sha256':source['feature_sha256'],
              'cache_reset':'every_order_and_every_cohort','stage2':'unchanged_independent_argmax',
              'compare_current_ensemble':compare_ensemble,
              'code_sha256':{name:sha256((ROOT/name).read_bytes()) for name in ['experiments/clip_tda/core.py','experiments/clip_tda/run.py']}}
    if compare_ensemble:
        manifest['code_sha256'].update({name:sha256((ROOT/name).read_bytes()) for name in [
            'experiments/clip_stata/core.py','experiments/clip_transductive/core.py',
            'experiments/clip_transductive/gate.py','experiments/clip_transductive/ensemble.py']})
    write_json(output/'run_manifest.json',manifest)
    try:
        x,t=source['features'],source['prototypes'];il=source['info_labels'];labels=source['labels']
        truth=[r['y_true'] for r in source['rows']]
        old,_=decide(*source['scores'],il,labels)
        normalized_gate=np.asarray(il)[(x@t.T).argmax(1)]=='informative'
        normalized=humanitarian(normalized_gate,source['scores'][1],labels)
        binary_base=np.asarray(il)[source['scores'][0].argmax(1)].tolist()
        binary_norm=np.asarray(il)[(x@t.T).argmax(1)].tolist()
        summary={'samples':len(x),'source_split':source['split'],'baseline_humanitarian':metric(truth,old,labels),
                 'normalized_humanitarian':metric(truth,normalized,labels),
                 'binary_baseline':metric(source['binary_truth'],binary_base,il) if source['binary_truth'] is not None else None,
                 'binary_normalized':metric(source['binary_truth'],binary_norm,il) if source['binary_truth'] is not None else None,
                 'by_order':[]}
        orders={'source_order':np.arange(len(x)),**{f'shuffle_{seed}':np.random.default_rng(seed).permutation(len(x)) for seed in [0,1,2]}}
        for name,order in orders.items():
            print(f'[TDA] {source["split"]}; {name}; samples={len(x)}',flush=True)
            logits,diagnostics=predict_stream(x,t,order)
            binary=np.asarray(il)[logits.argmax(1)].tolist()
            predicted=humanitarian(np.asarray(binary)=='informative',source['scores'][1],labels)
            report={'order':name,'humanitarian':metric(truth,predicted,labels),
                    'binary':metric(source['binary_truth'],binary,il) if source['binary_truth'] is not None else None,
                    'negative_logit_spread_max':diagnostics['negative_logit_spread_max']}
            summary['by_order'].append(report)
            folder=output/name;folder.mkdir()
            with (folder/'predictions.tsv').open('x',newline='',encoding='utf-8') as f:
                writer=csv.writer(f,delimiter='\t')
                writer.writerow(['id','humanitarian_truth','binary_truth','binary_pred','y_pred','stream_position','binary_logits'])
                positions=np.argsort(order)
                writer.writerows((r['id'],r['y_true'],source['binary_truth'][i] if source['binary_truth'] is not None else '',
                                  binary[i],predicted[i],int(positions[i]),json.dumps(logits[i].tolist())) for i,r in enumerate(source['rows']))
            write_json(folder/'metrics.json',report);write_json(folder/'diagnostics.json',diagnostics)
            print(f'[RESULT] {name}; humanitarian_f1={report["humanitarian"]["macro_f1"]:.6f}; binary_f1={report["binary"]["macro_f1"] if report["binary"] else None}',flush=True)
        random=[r for r in summary['by_order'] if r['order'].startswith('shuffle_')]
        summary['shuffle_mean']={scope:{key:float(np.mean([r[scope][key] for r in random])) for key in ['accuracy','macro_f1']}
                                 if random[0][scope] is not None else None for scope in ['humanitarian','binary']}
        if compare_ensemble:
            reference=compare_reference_ensemble(source)
            write_json(output/'reference_ensemble.json',reference)
            summary['reference_ensemble_mean']=reference['mean']
            print(f'[REFERENCE ENSEMBLE] {reference["mean"]}',flush=True)
        summary['limitations']=['Binary metric có null khi không có nhãn thật; không suy từ humanitarian.',
                                'Ba shuffle seed chỉ đo thứ tự stream; source_order là stress test riêng, không gộp vào mean.',
                                'Train và dev có scope khác nhau; không diễn giải train thành test.',
                                'Single-view trên MetaCLIP2/prompt dự án, không phải tái lập toàn bộ benchmark TDA.',
                                'Giữ cấu hình ImageNet tác giả; nhánh negative có thể suy biến thành hằng số ở binary.']
        write_json(output/'analysis.json',summary)
        with (output/'tda_summary.tsv').open('x',newline='',encoding='utf-8') as f:
            writer=csv.writer(f,delimiter='\t');writer.writerow(['scope','method','accuracy','macro_f1'])
            for scope,key in [('humanitarian','baseline_humanitarian'),('humanitarian','normalized_humanitarian'),
                              ('binary','binary_baseline'),('binary','binary_normalized')]:
                if summary[key] is not None:writer.writerow([scope,key,summary[key]['accuracy'],summary[key]['macro_f1']])
            for scope,result in summary['shuffle_mean'].items():
                if result is not None:writer.writerow([scope,'tda_shuffle_mean',result['accuracy'],result['macro_f1']])
            if compare_ensemble:
                for scope,result in summary['reference_ensemble_mean'].items():
                    if result is not None:writer.writerow([scope,'reference_ensemble_mean',result['accuracy'],result['macro_f1']])
        manifest['status']='complete'
        manifest['output_sha256']={str(p.relative_to(output)):sha256(p.read_bytes()) for p in output.rglob('*') if p.is_file() and p.name!='run_manifest.json'}
        write_json(output/'run_manifest.json',manifest);print(f'[SAVED] {output}',flush=True);return output
    except Exception as error:
        manifest.update(status='failed',error=f'{type(error).__name__}: {error}')
        write_json(output/'run_manifest.json',manifest);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run',required=True);parser.add_argument('--output-dir')
    parser.add_argument('--compare-ensemble',action='store_true')
    args=parser.parse_args();run(args.source_run,args.output_dir,args.compare_ensemble)
