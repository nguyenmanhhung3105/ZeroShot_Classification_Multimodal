"""Xác thực artifact một tầng/hai tầng và phân tích ghép cặp, không chạy model."""
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
from experiments.clip_two_stage.run import decide,decide_margin_pool
from experiments.waffle_lab.run import digest


def read_artifact(directory,name):
    path=Path(directory)/name
    if path.is_file():return path.read_text(encoding='utf-8')
    with zipfile.ZipFile(Path(directory)/'run_artifacts.zip') as archive:
        return archive.read(name).decode('utf-8')


def verify_stata_posterior(replayed,recorded):
    """FP32 giữa Kaggle/local: giữ nhãn chính xác và kiểm tra sai số tuyệt đối.

    Đối chiếu solver tác giả trên đủ 90 nhóm thật xác nhận nhãn không đổi,
    sai số posterior tối đa khoảng 4,7e-4. Không dùng rtol phụ thuộc độ lớn
    xác suất; ghi sai số vào báo cáo thay vì đòi tái lập bitwise giữa nền tảng.
    """
    left,right=np.asarray(replayed),np.asarray(recorded)
    if left.shape!=right.shape or left.ndim!=2 or left.shape[1]!=2 or len(left)==0:
        raise ValueError('StatA posterior sai shape')
    for values in [left,right]:
        if (not np.isfinite(values).all() or np.any(values<0) or np.any(values>1) or
                not np.allclose(values.sum(1),1,atol=1e-5,rtol=0)):
            raise ValueError('StatA posterior không phải xác suất hợp lệ')
    if not np.array_equal(left.argmax(1),right.argmax(1)):
        raise ValueError('StatA posterior đổi nhãn khi replay')
    error=float(np.abs(left-right).max());tolerance=1e-3
    if error>tolerance:raise ValueError('StatA posterior không khớp ngoài dung sai FP32')
    return {'max_abs_error':error,'absolute_tolerance':tolerance,'argmax_identical':True}


def analyze(directory,bootstraps=2000,seed=42):
    if type(bootstraps) is not int or bootstraps<1:raise ValueError('bootstraps phải dương')
    def read(name):return json.loads(read_artifact(directory,name))
    manifest=read('run_manifest.json')
    if manifest.get('status')!='complete':raise ValueError('Chỉ phân tích run hoàn tất')
    plans=[read(task+'/frozen_plan.json') for task in ['informativeness','humanitarian']]
    cohorts=[read(task+'/cohort.json') for task in ['informativeness','humanitarian']]
    for index,(plan,cohort) in enumerate(zip(plans,cohorts)):
        if digest(plan['protocol'])!=manifest['protocol_fingerprints'][index]:raise ValueError('Protocol fingerprint sai')
        if digest(cohort['rows'])!=manifest['cohort_fingerprints'][index]:raise ValueError('Cohort fingerprint sai')
    if [(r['id'],r['image_sha256']) for r in cohorts[0]['rows']]!=[(r['id'],r['image_sha256']) for r in cohorts[1]['rows']]:
        raise ValueError('Hai tầng khác ảnh')
    rows=list(csv.DictReader(io.StringIO(read_artifact(directory,'predictions.tsv')),delimiter='\t'))
    if len(rows)!=manifest['samples'] or len({r['id'] for r in rows})!=len(rows):raise ValueError('Thiếu/lặp dự đoán')
    if [(r['id'],r['y_true']) for r in rows]!=[(r['id'],r['y_true']) for r in cohorts[1]['rows']]:raise ValueError('Dự đoán khác cohort')
    a,b=[np.asarray([json.loads(r[key]) for r in rows]) for key in ['info_cosine_scores','human_cosine_scores']]
    labels=plans[1]['label_order']
    new,gate=decide(a,b,plans[0]['label_order'],labels)
    old=np.asarray(labels)[b.argmax(1)]
    if new!=[r['y_pred'] for r in rows] or old.tolist()!=[r['single_stage_pred'] for r in rows]:raise ValueError('Dự đoán khác score')
    if gate.tolist()!=[r['gate_informative']=='True' for r in rows]:raise ValueError('Gate không khớp score')
    from sklearn.metrics import accuracy_score,f1_score
    from scipy.stats import binomtest
    truth=np.asarray([r['y_true'] for r in rows]);new=np.asarray(new)
    def metric(pred):return {'accuracy':float(accuracy_score(truth,pred)),
                            'macro_f1':float(f1_score(truth,pred,labels=labels,average='macro',zero_division=0))}
    soft,soft_gate,margin=decide_margin_pool(a,b,plans[0]['label_order'],labels)
    ablations={'margin_pool':metric(soft)}
    stata_replay={}
    if manifest.get('stata_gate'):
        from experiments.clip_stata.core import gate_ablation,REFERENCE_COMMIT
        import hashlib
        feature_path=Path(directory)/'stata_features.npz'
        if feature_path.is_file():
            feature_bytes=feature_path.read_bytes()
        else:
            with zipfile.ZipFile(Path(directory)/'run_artifacts.zip') as archive:
                feature_bytes=archive.read('stata_features.npz')
        if hashlib.sha256(feature_bytes).hexdigest()!=manifest['stata_features_sha256']:
            raise ValueError('StatA feature hash sai')
        if manifest['stata_protocol']['reference_commit']!=REFERENCE_COMMIT:
            raise ValueError('Khác implementation StatA tham chiếu')
        with np.load(io.BytesIO(feature_bytes),allow_pickle=False) as pack:
            if pack['ids'].tolist()!=[r['id'] for r in rows]:raise ValueError('StatA features khác ID/thứ tự')
            choices,diagnostics=gate_ablation(pack['image_features'],pack['info_mean_prototypes'],a,b,plans[0]['label_order'],labels)
        stored=list(csv.DictReader(io.StringIO(read_artifact(directory,'stata_predictions.tsv')),delimiter='\t'))
        if [(r['id'],r['y_true']) for r in stored]!=[(r['id'],r['y_true']) for r in rows]:
            raise ValueError('StatA predictions khác cohort')
        recorded=read('stata_diagnostics.json')
        for name,pred in choices.items():
            if pred!=[r[name] for r in stored]:raise ValueError('StatA dự đoán khác embedding')
            ablations[name]=metric(pred)
            if name!='normalized_gate_control':
                if diagnostics['gate_labels'][name]!=recorded['gate_labels'][name]:raise ValueError('StatA gate không khớp')
                if diagnostics['batch_ids'][name]!=recorded['batch_ids'][name]:raise ValueError('StatA groups không khớp')
                stata_replay[name]=verify_stata_posterior(diagnostics['posterior'][name],recorded['posterior'][name])
    if 'margin_pool_pred' in rows[0]:
        if soft!=[r['margin_pool_pred'] for r in rows]:raise ValueError('Margin pool không khớp score')
        if soft_gate.tolist()!=[r['margin_pool_gate']=='True' for r in rows]:raise ValueError('Margin pool gate sai')
        if not np.allclose(margin,[float(r['pooled_margin']) for r in rows],atol=1e-7):raise ValueError('Margin pool giá trị sai')
    text_usage=None
    if manifest.get('full_frame_view'):
        from experiments.clip_two_stage.views import classify_views
        view_rows=list(csv.DictReader(io.StringIO(read_artifact(directory,'view_predictions.tsv')),delimiter='\t'))
        if [(r['id'],r['y_true']) for r in view_rows]!=[(r['id'],r['y_true']) for r in rows]:
            raise ValueError('View khác cohort')
        whole=[np.asarray([json.loads(r[key]) for r in view_rows]) for key in ['full_info_cosine_scores','full_human_cosine_scores']]
        choices=classify_views([a,b],whole,plans[0]['label_order'],labels)
        for method,pred in choices.items():
            if pred!=[r[method] for r in view_rows]:raise ValueError('View prediction khác score')
            ablations[method]=metric(pred)
    if manifest.get('input')=='image_text':
        from experiments.clip_two_stage.multimodal import predict,METHODS
        text_plan=read('text_plan.json');identity=read('text_cohort.json')
        if digest(text_plan)!=manifest['text_plan_fingerprint'] or digest(identity)!=manifest['text_cohort_fingerprint']:
            raise ValueError('Text plan/cohort fingerprint sai')
        if [r['id'] for r in identity]!=[r['id'] for r in rows]:raise ValueError('Text cohort khác ảnh')
        mm=list(csv.DictReader(io.StringIO(read_artifact(directory,'multimodal_predictions.tsv')),delimiter='\t'))
        if [(r['id'],r['y_true']) for r in mm]!=[(r['id'],r['y_true']) for r in rows]:raise ValueError('Dự đoán multimodal khác cohort')
        ti,th=[np.asarray([json.loads(r[key]) for r in mm]) for key in ['text_info_scores','text_human_scores']]
        if any(r['text_empty'] not in ('True','False') for r in mm):raise ValueError('Cờ empty text sai')
        empty=np.asarray([r['text_empty']=='True' for r in mm])
        choices,diagnostics=predict(a,b,ti,th,empty,plans[0]['label_order'],labels,manifest['logit_scale'])
        for method in METHODS:
            if choices[method]!=[r[method] for r in mm]:raise ValueError('Dự đoán multimodal không khớp score')
            ablations[method]=metric(choices[method])
        if not np.allclose(diagnostics['text_weight'],[float(r['conditional_text_weight']) for r in mm]):
            raise ValueError('Trọng số conditional sai')
        text_usage={'conditional_used':int(diagnostics['text_used'].sum()),'empty':int(empty.sum()),
                    'scope':'CI bên dưới chỉ so hard image two-stage với single-stage image; multimodal là đối chứng riêng.'}
    old_ok=old==truth;new_ok=new==truth
    fixed=int((~old_ok&new_ok).sum());broken=int((old_ok&~new_ok).sum())
    order={label:i for i,label in enumerate(labels)}
    y=np.asarray([order[t] for t in truth]);p=np.asarray([order[t] for t in old]);q=np.asarray([order[t] for t in new]);classes=len(labels)
    def f1(y,p):
        matrix=np.bincount(y*classes+p,minlength=classes**2).reshape(classes,classes)
        denom=matrix.sum(0)+matrix.sum(1)
        return np.divide(2*matrix.diagonal(),denom,out=np.zeros(classes),where=denom!=0).mean()
    rng=np.random.default_rng(seed);deltas=[]
    for _ in range(bootstraps):
        indices=rng.integers(len(y),size=len(y));deltas.append(f1(y[indices],q[indices])-f1(y[indices],p[indices]))
    by_event={}
    for event in sorted({r['event'] for r in cohorts[1]['rows']}):
        mask=np.asarray([r['event']==event for r in cohorts[1]['rows']])
        by_event[event]={'samples':int(mask.sum()),'fixed':int((~old_ok&new_ok&mask).sum()),'broken':int((old_ok&~new_ok&mask).sum())}
    return {'source':str(directory),'model':manifest['model'],'seed':manifest['seed'],'samples':len(rows),
            'evaluation_scope':manifest.get('evaluation_scope','exploratory'),
            'single_stage':metric(old),'two_stage':metric(new),'ablations':ablations,'stata_posterior_replay':stata_replay,'text_usage':text_usage,'fixed':fixed,'broken':broken,
            'delta_macro_f1_paired_bootstrap_95pct':np.quantile(deltas,[.025,.975]).tolist(),
            'bootstrap_repetitions':bootstraps,'bootstrap_seed':seed,
            'mcnemar_exact_p_nominal':float(binomtest(broken,fixed+broken,p=.5).pvalue) if fixed+broken else 1.,
            'by_event':by_event,'limitations':'Bootstrap theo mẫu; ảnh cùng tweet/sự kiện có thể phụ thuộc. P-value thăm dò, chưa hiệu chỉnh nhiều phép thử.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',required=True)
    parser.add_argument('--bootstraps',type=int,default=2000)
    args=parser.parse_args();report=analyze(args.run_dir,args.bootstraps)
    output=ROOT/'results/analysis/clip_two_stage'/uuid.uuid4().hex[:12];output.mkdir(parents=True)
    (output/'analysis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2));print(output/'analysis.json')


if __name__=='__main__':main()
