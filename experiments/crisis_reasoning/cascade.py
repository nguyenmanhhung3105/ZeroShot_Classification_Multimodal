"""Đối chứng cascade chọn Qwen khi CLIP và Waffle bất đồng; không fit nhãn."""
import argparse
import csv
import io
import json
from pathlib import Path
import sys
import uuid
import zipfile

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.waffle_lab.analyze import load_run


def artifact(directory,name):
    path=Path(directory)/name
    if path.is_file(): return path.read_text(encoding='utf-8')
    with zipfile.ZipFile(Path(directory)/'run_artifacts.zip') as archive:
        return archive.read(name).decode('utf-8')


def route(primary, contrast, reviewer, labels):
    """Hàm quyết định chỉ nhận nhãn dự đoán; không nhận nhãn thật."""
    if not len(primary)==len(contrast)==len(reviewer):
        raise ValueError('Khác số mẫu')
    if any(label not in labels for label in primary+contrast):
        raise ValueError('Nhãn primary/contrast không hợp lệ')
    flags=[a!=b for a,b in zip(primary,contrast)]
    # Qwen trả lỗi: giữ primary và ghi rõ, không tự chọn bằng nhãn thật.
    return [r if flag and r in labels else p for p,r,flag in zip(primary,reviewer,flags)],flags


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--primary',required=True)
    parser.add_argument('--contrast',required=True)
    parser.add_argument('--reviewer',required=True)
    args=parser.parse_args()
    primary,contrast=[load_run(path) for path in (args.primary,args.contrast)]
    for key in ('task','model','samples','cohort_fingerprint','protocol_fingerprint','device','compute_dtype'):
        if primary['manifest'][key]!=contrast['manifest'][key]:
            raise ValueError(f'Primary/contrast khác {key}')
    pm,qm=[json.loads(artifact(path,'run_manifest.json')) for path in (args.primary,args.reviewer)]
    if (qm.get('status')!='complete' or qm.get('model')!='Qwen/Qwen2.5-VL-3B-Instruct'
            or qm.get('input')!='image_only' or qm.get('inference_backend')!='qwen_real'):
        raise ValueError('Reviewer chưa hoàn tất hoặc khác model/input')
    if pm['task']!=qm['task']:
        raise ValueError('Khác task')
    qc=json.loads(artifact(args.reviewer,'cohort.json'))
    pc=json.loads(artifact(args.primary,'cohort.json'))
    count=qm['samples']
    # Cho phép thử Qwen trên prefix 5/100 ảnh; đánh giá cả ba phương án
    # trên chính prefix đó, không so với metric 3000 ảnh.
    if count>len(pc['rows']): raise ValueError('Reviewer nhiều mẫu hơn primary')
    def identity(rows):
        return [(r['id'],r['image_sha256'],r['y_true']) for r in rows]
    if identity(qc['rows'])!=identity(pc['rows'][:count]):
        raise ValueError('Khác prefix ảnh/nhãn/thứ tự')
    qr=list(csv.DictReader(io.StringIO(artifact(args.reviewer,'predictions.tsv')),delimiter='\t'))
    if len(qr)!=count or [(r['id'],r['y_true']) for r in qr]!=[(r['id'],r['y_true']) for r in qc['rows']]:
        raise ValueError('Reviewer predictions không khớp cohort')
    p=[r['y_pred'] for r in primary['rows'][:count]]
    c=[r['y_pred'] for r in contrast['rows'][:count]]
    q=[r['y_pred'] for r in qr]
    labels=primary['plan']['label_order']
    output,flags=route(p,c,q,labels)
    truth=[r['y_true'] for r in qr]
    from sklearn.metrics import accuracy_score,f1_score
    def metrics(pred):
        return {'accuracy':float(accuracy_score(truth,pred)),
                'macro_f1':float(f1_score(truth,pred,labels=labels,average='macro',zero_division=0))}
    report={'scope':'exploratory_prefix_comparison','samples':count,
            'primary':args.primary,'contrast':args.contrast,'reviewer':args.reviewer,
            'rule':'prompt_disagreement_only_no_label_fit','primary_metrics':metrics(p),
            'qwen_direct_metrics':metrics(q),'cascade_metrics':metrics(output),
            'route_count':sum(flags),'route_fraction':sum(flags)/count,
            'invalid_reviewer_fallback_count':sum(flag and label not in labels for flag,label in zip(flags,q)),
            'fixed':sum(a!=t and b==t for a,b,t in zip(p,output,truth)),
            'broken':sum(a==t and b!=t for a,b,t in zip(p,output,truth)),
            'cost_note':'Qwen toàn bộ đã chạy để đối chứng; route_fraction là tỷ lệ gọi dự kiến, chưa phải tốc độ end-to-end đã đo.'}
    directory=ROOT/'results/analysis/crisis_cascade'/uuid.uuid4().hex[:12]
    directory.mkdir(parents=True)
    (directory/'analysis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    with (directory/'predictions.tsv').open('w',newline='') as stream:
        writer=csv.writer(stream,delimiter='\t');writer.writerow(['id','y_true','primary','contrast','qwen','routed','cascade'])
        writer.writerows(zip([r['id'] for r in qr],truth,p,c,q,flags,output))
    print(json.dumps(report,ensure_ascii=False,indent=2));print(directory)


if __name__=='__main__': main()
