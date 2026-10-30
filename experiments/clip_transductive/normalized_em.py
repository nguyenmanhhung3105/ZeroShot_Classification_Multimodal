"""Đối chứng chuẩn hóa prototype trước EM; đánh giá nhãn binary thật riêng."""
import argparse
import csv
import json
from pathlib import Path
import sys
import uuid
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import numpy as np
from experiments.clip_transductive.core import probability_features
from experiments.clip_transductive.gate import load_binary_source,predict_binary
from experiments.clip_transductive.ensemble import align,humanitarian
from experiments.clip_transductive.run import metric,sha256,write_json
from experiments.clip_two_stage.analyze import read_artifact
from experiments.clip_two_stage.run import decide
from experiments.clip_tda.run import load_features


def normalize_mean_scores(scores,norms):
    scores=np.asarray(scores);norms=np.asarray(norms)
    if (scores.ndim!=2 or norms.shape!=(scores.shape[1],) or not np.isfinite(norms).all() or
            np.any(norms<=1e-12) or np.any(norms>1+1e-6)):
        raise ValueError('Norm của mean prototype không hợp lệ')
    result=scores/norms[None,:]
    probability_features(result)  # score sau chia phải còn là cosine hữu hạn.
    return result


def run(output_dir=None):
    settings=json.loads((ROOT/'experiments/clip_transductive/config_gate.json').read_text())
    binary_path=ROOT/settings['binary_source_run'];train=load_binary_source(binary_path)
    dev_path=ROOT/'results/clip_two_stage_stata_heldout/crisismmd/humanitarian/samples_2219/metaclip2_vith14/humanitarian_ca2d96c17e89'
    dev=load_features(dev_path)
    norms=np.asarray(train['manifest']['prototype_norms']['semantic'])
    if (train['manifest']['model']!=dev['manifest']['model'] or train['labels']!=dev['info_labels'] or
            not np.allclose(norms,dev['prototype_norms'],atol=1e-6,rtol=1e-5)):
        raise ValueError('Hai nguồn khác model, thứ tự lớp hoặc prototype norms')
    dev_plan=json.loads(read_artifact(dev_path,'informativeness/frozen_plan.json'))
    if train['plan']['prompt_banks']['semantic']!=dev_plan['prompt_banks']['semantic']:
        raise ValueError('Khác bank semantic informative')
    em_dir=ROOT/'results/analysis/clip_transductive_gate/c3fa825bc57b'
    previous=json.loads((em_dir/'run_manifest.json').read_text())
    if previous['status']!='complete' or previous['source_artifact_sha256']['binary_train']!=train['artifact_sha256']:
        raise ValueError('Nguồn đối chứng EM khác dữ liệu train hiện tại')
    for name,value in previous['output_sha256'].items():
        if sha256((em_dir/name).read_bytes())!=value:raise ValueError('Artifact EM đã thay đổi')
    output=Path(output_dir) if output_dir else ROOT/'results/analysis/normalized_em_gate'/uuid.uuid4().hex[:12]
    output.mkdir(parents=True,exist_ok=False)
    manifest={'status':'running','normalization':'divide_mean_cosine_by_norm_of_mean_unit_text_embeddings',
              'prototype_norms':norms.tolist(),'label_fit':False,'grouping_seeds':[0,1,2],
              'query_size':75,'temperature':30,'iterations':20,'mm_iterations':1000,
              'binary_scope':'train_3000_true_image_informativeness_labels',
              'humanitarian_scope':'dev_2219_image_humanitarian_labels',
              'binary_source_sha256':train['artifact_sha256'],'dev_feature_sha256':dev['feature_sha256'],
              'source_sha256':sha256(Path(__file__).read_bytes())}
    write_json(output/'frozen_plan.json',manifest);write_json(output/'run_manifest.json',manifest)
    try:
        summary={}
        for scope,source,raw_scores,labels,truth in [
                ('binary_train',train,train['scores'],train['labels'],[r['y_true'] for r in train['rows']]),
                ('humanitarian_dev',dev,dev['scores'][0],dev['labels'],[r['y_true'] for r in dev['rows']])]:
            normalized=normalize_mean_scores(raw_scores,norms)
            info_labels=train['labels']
            def final(binary):
                return binary if scope=='binary_train' else humanitarian(np.asarray(binary)=='informative',dev['scores'][1],labels)
            baseline=final(np.asarray(info_labels)[raw_scores.argmax(1)].tolist())
            normalized_control=final(np.asarray(info_labels)[normalized.argmax(1)].tolist())
            result={'samples':len(truth),'baseline':metric(truth,baseline,labels),
                    'normalized_control':metric(truth,normalized_control,labels),'by_seed':[]}
            for seed in [0,1,2]:
                print(f'[NORMALIZED EM] {scope}; seed={seed}',flush=True)
                binary,batches,diagnostics=predict_binary(normalized,info_labels,grouping_seed=seed,query_size=75,temperature=30.,iterations=20,mm_iterations=1000)
                predicted=final(binary)
                with (em_dir/scope/f'grouping_seed_{seed}/predictions.tsv').open() as f:
                    em_rows=list(csv.DictReader(f,delimiter='\t'))
                align(source['rows'],em_rows)
                raw_em=[r['y_pred'] for r in em_rows]
                if [r['baseline_pred'] for r in em_rows]!=baseline:raise ValueError('Baseline khác đối chứng')
                report={'seed':seed,'raw_em':metric(truth,raw_em,labels),'normalized_em':metric(truth,predicted,labels)}
                result['by_seed'].append(report)
                folder=output/scope/f'grouping_seed_{seed}';folder.mkdir(parents=True)
                with (folder/'predictions.tsv').open('x',newline='',encoding='utf-8') as f:
                    writer=csv.writer(f,delimiter='\t');writer.writerow(['id','y_true','baseline_pred','raw_em_pred','binary_pred','y_pred','batch_id'])
                    writer.writerows((r['id'],r['y_true'],base,old,info,pred,int(batch)) for r,base,old,info,pred,batch in
                                     zip(source['rows'],baseline,raw_em,binary,predicted,batches))
                write_json(folder/'diagnostics.json',diagnostics);write_json(folder/'metrics.json',report)
                print(f'[RESULT] {scope}; seed={seed}; macro_f1={report["normalized_em"]["macro_f1"]:.6f}',flush=True)
            for method in ['raw_em','normalized_em']:
                result[method+'_mean']={key:float(np.mean([r[method][key] for r in result['by_seed']])) for key in ['accuracy','macro_f1']}
            summary[scope]=result
        summary['limitations']=['Thay đổi aggregation trước cùng một thuật toán; không phải model mới.',
                               'Không dùng nhãn chọn norm, nhiệt độ, trọng số hoặc seed.',
                               'Binary đo trên train đã xem; humanitarian đo trên dev đã xem; chưa có test độc lập.',
                               'Dev không có nhãn informativeness thật; không suy nhãn từ humanitarian.']
        write_json(output/'analysis.json',summary)
        manifest['status']='complete'
        manifest['output_sha256']={str(p.relative_to(output)):sha256(p.read_bytes()) for p in output.rglob('*') if p.is_file() and p.name!='run_manifest.json'}
        write_json(output/'run_manifest.json',manifest);print(f'[SAVED] {output}',flush=True);return output
    except Exception as error:
        manifest.update(status='failed',error=f'{type(error).__name__}: {error}')
        write_json(output/'run_manifest.json',manifest);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir')
    args=parser.parse_args();run(args.output_dir)
