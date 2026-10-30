"""Kết hợp nhãn cổng đã lưu; giữ nguyên humanitarian argmax và cả tám lớp."""
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
from experiments.clip_transductive.run import load_source,metric,sha256,write_json
from experiments.clip_two_stage.run import decide
from experiments.clip_two_stage.analyze import read_artifact


def combine(votes):
    """Cột: normalized CLIP, EM-Dirichlet, StatA. Không nhận nhãn thật."""
    votes=np.asarray(votes)
    if votes.ndim!=2 or votes.shape[1]!=3 or len(votes)==0 or votes.dtype!=np.bool_:
        raise ValueError('Cần ma trận boolean [N,3] không rỗng')
    return {'majority':votes.sum(1)>=2,'recall_or':votes.any(1)}


def humanitarian(gate,scores,labels):
    gate=np.asarray(gate);scores=np.asarray(scores)
    if (gate.dtype!=np.bool_ or gate.ndim!=1 or scores.shape!=(len(gate),len(labels)) or
            'not_humanitarian' not in labels or len(labels)!=len(set(labels)) or len(labels)<2 or
            not np.isfinite(scores).all()):
        raise ValueError('Gate, score hoặc taxonomy không hợp lệ')
    positive=[i for i,label in enumerate(labels) if label!='not_humanitarian']
    pred=np.asarray(labels)[np.asarray(positive)[scores[:,positive].argmax(1)]]
    pred[~gate]='not_humanitarian'
    return pred.tolist()


def align(reference,rows):
    if ([(r['id'],r['y_true']) for r in reference]!=[(r['id'],r['y_true']) for r in rows] or
            len({r['id'] for r in rows})!=len(rows)):
        raise ValueError('Khác ID, thứ tự, nhãn hoặc có ảnh lặp')


def run(settings,output_dir=None):
    if (settings.get('methods')!=['majority','recall_or'] or settings.get('grouping_seeds')!=[0,1,2] or
            settings.get('label_fit') is not False):
        raise ValueError('Chỉ đánh giá hai quy tắc và ba seed đã khóa')
    def path(key):
        p=Path(settings[key]);return p if p.is_absolute() else ROOT/p
    original=load_source(path('original_run'))
    stata=load_source(path('stata_run'))  # replay kiểm tra từ NPZ/score, không gọi backbone.
    if (original['cohort']!=stata['cohort'] or original['labels']!=stata['labels'] or
            original['info_labels']!=stata['info_labels'] or original['manifest']['model']!=stata['manifest']['model']):
        raise ValueError('Hai nguồn khác cohort, taxonomy hoặc backbone')
    for task in ['informativeness','humanitarian']:
        plans=[json.loads(read_artifact(path(key),f'{task}/frozen_plan.json')) for key in ['original_run','stata_run']]
        if plans[0]['prompt_banks']!=plans[1]['prompt_banks']:raise ValueError('Hai nguồn khác bank prompt')
    for left,right in zip(original['scores'],stata['scores']):
        if not np.allclose(left,right,atol=2e-5,rtol=1e-4):raise ValueError('Hai nguồn khác cosine')
    em_dir=path('em_run');em_manifest=json.loads((em_dir/'run_manifest.json').read_text())
    if (em_manifest.get('status')!='complete' or em_manifest['settings']['grouping_seeds']!=[0,1,2] or
            em_manifest['source_artifact_sha256']['humanitarian_dev']!=original['artifact_sha256']):
        raise ValueError('Lượt EM không hoàn tất hoặc khác nguồn')
    for name,expected in em_manifest['output_sha256'].items():
        if sha256((em_dir/name).read_bytes())!=expected:raise ValueError('Artifact EM đã thay đổi')
    diagnostics=json.loads(read_artifact(path('stata_run'),'stata_diagnostics.json'))
    stata_rows=list(csv.DictReader(io.StringIO(read_artifact(path('stata_run'),'stata_predictions.tsv')),delimiter='\t'))
    align(original['rows'],stata_rows)
    feature_path=path('stata_run')/'stata_features.npz'
    if feature_path.is_file():feature_bytes=feature_path.read_bytes()
    else:
        with zipfile.ZipFile(path('stata_run')/'run_artifacts.zip') as z:feature_bytes=z.read('stata_features.npz')
    if sha256(feature_bytes)!=stata['manifest']['stata_features_sha256']:raise ValueError('Feature hash sai')
    with np.load(io.BytesIO(feature_bytes),allow_pickle=False) as pack:
        proto=pack['info_mean_prototypes'];proto=proto/np.linalg.norm(proto,axis=1,keepdims=True)
        normalized_scores=pack['image_features']@proto.T
    info_labels=original['info_labels'];labels=original['labels'];human_scores=original['scores'][1]
    norm_gate=np.asarray(info_labels)[normalized_scores.argmax(1)]=='informative'
    norm_pred=humanitarian(norm_gate,human_scores,labels)
    if norm_pred!=[r['normalized_gate_control'] for r in stata_rows]:raise ValueError('Normalized control khác score')
    baseline,_=decide(*original['scores'],info_labels,labels)
    prepared=[]
    for seed in settings['grouping_seeds']:
        with (em_dir/'humanitarian_dev'/f'grouping_seed_{seed}'/'predictions.tsv').open() as f:
            em_rows=list(csv.DictReader(f,delimiter='\t'))
        align(original['rows'],em_rows)
        if any(r['info_pred'] not in info_labels for r in em_rows):raise ValueError('Nhãn EM ngoài taxonomy')
        em_gate=np.asarray([r['info_pred'] for r in em_rows])=='informative'
        em_pred=humanitarian(em_gate,human_scores,labels)
        if em_pred!=[r['y_pred'] for r in em_rows] or baseline!=[r['baseline_pred'] for r in em_rows]:
            raise ValueError('Dự đoán EM hoặc baseline khác score')
        key=f'stata_gate_seed_{seed}'
        stata_gate=np.asarray(diagnostics['gate_labels'][key])=='informative'
        stata_pred=humanitarian(stata_gate,human_scores,labels)
        if stata_pred!=[r[key] for r in stata_rows]:raise ValueError('Dự đoán StatA không khớp gate')
        em_batches=[int(r['batch_id']) for r in em_rows]
        if em_batches!=diagnostics['batch_ids'][key]:raise ValueError('Hai phương pháp khác nhóm query')
        votes=np.stack([norm_gate,em_gate,stata_gate],1)
        gates=combine(votes)
        prepared.append({'seed':seed,'votes':votes,'gates':gates,'batch_ids':em_batches,
                         'em_pred':em_pred,'stata_pred':stata_pred,
                         'predictions':{name:humanitarian(gate,human_scores,labels) for name,gate in gates.items()}})
    output=Path(output_dir) if output_dir else ROOT/'results/analysis/clip_gate_ensemble'/uuid.uuid4().hex[:12]
    output.mkdir(parents=True,exist_ok=False)
    manifest={'status':'running','settings':settings,'evaluation_scope':'dev_exploratory_ensemble',
              'samples':len(original['rows']),'label_fit':False,'classes':len(labels),
              'rule_note':'Keep EM unless both other gates agree is exactly majority, not an extra method.',
              'stage2':'unchanged_argmax_over_seven_positive_classes',
              'source_artifact_sha256':{'original':original['artifact_sha256'],'stata':stata['artifact_sha256']},
              'em_manifest_sha256':sha256((em_dir/'run_manifest.json').read_bytes()),
              'source_sha256':sha256(Path(__file__).read_bytes())}
    write_json(output/'frozen_plan.json',manifest)
    write_json(output/'run_manifest.json',manifest)
    try:
        truth=np.asarray([r['y_true'] for r in original['rows']])
        controls={'two_stage':metric(truth,baseline,labels),'normalized_gate':metric(truth,norm_pred,labels),
                  'em_by_seed':[metric(truth,trial['em_pred'],labels) for trial in prepared],
                  'stata_by_seed':[metric(truth,trial['stata_pred'],labels) for trial in prepared]}
        reports={name:[] for name in settings['methods']}
        for trial in prepared:
            seed=trial['seed'];votes=trial['votes']
            for name,pred in trial['predictions'].items():
                pred=np.asarray(pred);em_pred=np.asarray(trial['em_pred']);normal=np.asarray(norm_pred)
                report={'seed':seed,**metric(truth,pred,labels),
                        'fixed_vs_em':int(((em_pred!=truth)&(pred==truth)).sum()),
                        'broken_vs_em':int(((em_pred==truth)&(pred!=truth)).sum()),
                        'fixed_vs_normalized':int(((normal!=truth)&(pred==truth)).sum()),
                        'broken_vs_normalized':int(((normal==truth)&(pred!=truth)).sum()),
                        'informative_predictions':int(trial['gates'][name].sum()),
                        'gate_disagreements':int((votes.any(1)&~votes.all(1)).sum())}
                reports[name].append(report)
                print(f'[ENSEMBLE] {name}; seed={seed}; accuracy={report["accuracy"]:.6f}; macro_f1={report["macro_f1"]:.6f}',flush=True)
            directory=output/f'grouping_seed_{seed}';directory.mkdir()
            with (directory/'predictions.tsv').open('x',newline='',encoding='utf-8') as f:
                writer=csv.writer(f,delimiter='\t')
                writer.writerow(['id','y_true','baseline_pred','normalized_gate','em_gate','stata_gate',
                                 'majority_gate','recall_or_gate','majority_pred','recall_or_pred','batch_id'])
                writer.writerows([row['id'],row['y_true'],baseline[i],*[bool(v) for v in votes[i]],
                                  bool(trial['gates']['majority'][i]),bool(trial['gates']['recall_or'][i]),
                                  trial['predictions']['majority'][i],trial['predictions']['recall_or'][i],trial['batch_ids'][i]]
                                 for i,row in enumerate(original['rows']))
        summary={'samples':len(truth),'label_order':labels,'controls':controls,'by_method':reports,
                 'mean':{name:{key:float(np.mean([r[key] for r in group])) for key in ['accuracy','macro_f1']} for name,group in reports.items()},
                 'std_population':{name:{key:float(np.std([r[key] for r in group])) for key in ['accuracy','macro_f1']} for name,group in reports.items()},
                 'limitations':['Hai quy tắc cố định, không học trọng số từ nhãn dev.',
                                'Cùng CLIP/prompt nên lỗi các cổng có tương quan; không giả định ba model độc lập.',
                                'Ba grouping seed được ghép cùng chỉ số và cùng batch, không chọn seed tốt nhất.',
                                'Dev đã xem nhiều lần; kết quả là thăm dò, chưa phải test độc lập.',
                                'Không có nhãn informativeness thật trên dev, chỉ đánh giá đầu ra tám lớp.']}
        write_json(output/'analysis.json',summary)
        with (output/'summary.tsv').open('x',newline='',encoding='utf-8') as f:
            writer=csv.writer(f,delimiter='\t');writer.writerow(['method','accuracy','macro_f1'])
            for key in ['two_stage','normalized_gate']:writer.writerow([key,controls[key]['accuracy'],controls[key]['macro_f1']])
            for key in ['em','stata']:
                writer.writerow([key+'_mean',*[float(np.mean([r[metric] for r in controls[key+'_by_seed']])) for metric in ['accuracy','macro_f1']]])
            for key,values in summary['mean'].items():writer.writerow([key+'_mean',values['accuracy'],values['macro_f1']])
        manifest['status']='complete'
        manifest['output_sha256']={str(p.relative_to(output)):sha256(p.read_bytes()) for p in output.rglob('*') if p.is_file() and p.name!='run_manifest.json'}
        write_json(output/'run_manifest.json',manifest)
        print(f'[SAVED] {output}',flush=True);return output
    except Exception as error:
        manifest.update(status='failed',error=f'{type(error).__name__}: {error}')
        write_json(output/'run_manifest.json',manifest);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(ROOT/'experiments/clip_transductive/config_ensemble.json'))
    parser.add_argument('--output-dir');args=parser.parse_args()
    run(json.loads(Path(args.config).read_text()),args.output_dir)
