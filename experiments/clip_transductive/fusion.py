"""Hai quy tắc kết hợp cố định: soft vote và bảo vệ recall của CLIP gốc."""
import argparse
import csv
import json
from pathlib import Path
import sys
import uuid
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import numpy as np
from experiments.clip_transductive.core import em_dirichlet,probability_features
from experiments.clip_transductive.ensemble import humanitarian,combine
from experiments.clip_transductive.run import metric,sha256,write_json
from experiments.clip_stata.core import gate_ablation
from experiments.clip_tda.run import load_features


def check_probabilities(values):
    p=np.asarray(values,dtype=float)
    if (p.ndim!=2 or p.shape[1]!=2 or len(p)==0 or not np.isfinite(p).all() or
            np.any(p<0) or np.any(p>1) or not np.allclose(p.sum(1),1,atol=1e-6,rtol=0)):
        raise ValueError('Cần xác suất hai lớp hữu hạn, mỗi hàng tổng bằng 1')
    return p


def name_binary_assignments(assignments,mappings):
    """EM u là posterior cụm, không mặc định cột cụm tương ứng tên lớp.

    Giữ Hungarian mapping của cụm đang có. Với binary, cụm không có mẫu
    thắng được ghép với lớp còn lại để hoàn tất hoán vị soft assignments.
    Đây là mở rộng cho soft vote, không gọi posterior là đã hiệu chuẩn.
    """
    values=np.asarray(assignments)
    if values.ndim!=3 or values.shape[2]!=2 or len(mappings)!=len(values):raise ValueError('Sai shape/mapping')
    result=np.empty_like(values)
    for task,mapping in enumerate(mappings):
        check_probabilities(values[task])
        mapping={int(k):int(v) for k,v in mapping.items()}
        if (not mapping or not set(mapping)<=set(range(2)) or not set(mapping.values())<=set(range(2)) or
                len(set(mapping.values()))!=len(mapping)):
            raise ValueError('Mapping cụm/lớp không hợp lệ')
        missing_clusters=sorted(set(range(2))-set(mapping))
        missing_classes=sorted(set(range(2))-set(mapping.values()))
        mapping.update(zip(missing_clusters,missing_classes))
        for cluster,label in mapping.items():result[task,:,label]=values[task,:,cluster]
    return result


def binary_em_probabilities(scores,labels,seed,query_size=75,iterations=20,mm_iterations=1000):
    if len(labels)!=2 or set(labels)!={'informative','not_informative'}:raise ValueError('Sai taxonomy')
    if type(query_size) is not int or query_size<2 or type(seed) is not int or not 0<=seed<2**32:
        raise ValueError('Sai query size/seed')
    query=probability_features(scores,30.)
    order=np.random.default_rng(seed).permutation(len(query))
    batches=[order[start:start+query_size] for start in range(0,len(order),query_size)]
    posterior=np.empty_like(query);hard=np.full(len(query),-1,dtype=int);batch_ids=hard.copy()
    for size in sorted({len(batch) for batch in batches},reverse=True):
        numbers=[i for i,batch in enumerate(batches) if len(batch)==size]
        indices=np.stack([batches[i] for i in numbers])
        predicted,log,assignments=em_dirichlet(query[indices],iterations,mm_iterations)
        named=assignments if log['singleton_fallback'] else name_binary_assignments(assignments,log['mappings'])
        posterior[indices]=named;hard[indices]=predicted;batch_ids[indices]=np.array(numbers)[:,None]
    check_probabilities(posterior)
    difference=posterior.argmax(1)!=hard
    if np.any(difference&(np.abs(posterior[:,0]-posterior[:,1])>1e-12)):
        raise ValueError('Soft mapping thay đổi nhãn EM ngoài tie')
    return posterior,np.asarray(labels)[hard].tolist(),batch_ids,int(difference.sum())


def soft_mean(probabilities,informative_index):
    if len(probabilities)!=3 or informative_index not in [0,1]:raise ValueError('Cần đúng ba nhánh và chỉ số nhãn')
    arrays=[check_probabilities(p) for p in probabilities]
    if len({p.shape for p in arrays})!=1:raise ValueError('Các nhánh khác số ảnh')
    mean=np.mean(arrays,axis=0)
    return mean[:,informative_index]>=.5,mean


def protect(raw_clip_gate,new_gate):
    a,b=np.asarray(raw_clip_gate),np.asarray(new_gate)
    if a.dtype!=np.bool_ or b.dtype!=np.bool_ or a.ndim!=1 or a.shape!=b.shape:
        raise ValueError('Cần hai gate boolean cùng ảnh')
    return a|b


def run(settings,output_dir=None):
    if (settings.get('methods')!=['soft_mean_three','protected_majority'] or settings.get('weights')!='equal_thirds' or
            settings.get('grouping_seeds')!=[0,1,2] or settings.get('label_fit') is not False):
        raise ValueError('Giữ hai quy tắc, equal weights và seed đã khóa')
    output=Path(output_dir) if output_dir else ROOT/'results/analysis/clip_gate_fusion'/uuid.uuid4().hex[:12]
    output.mkdir(parents=True,exist_ok=False)
    manifest={'status':'running','settings':settings,'label_fit':False,'probabilities_calibrated':False,
              'soft_mapping':'Hungarian occupied clusters plus binary complement for remaining cluster',
              'tie_policy':'soft informative probability >= 0.5',
              'protect_rule':'raw_CLIP_gate OR existing_majority_gate',
              'stage2':'unchanged_argmax_seven_positive_classes','source_features':{},
              'code_sha256':{name:sha256((ROOT/name).read_bytes()) for name in [
                  'experiments/clip_transductive/fusion.py','experiments/clip_transductive/core.py',
                  'experiments/clip_stata/core.py','experiments/clip_tda/run.py']}}
    write_json(output/'frozen_plan.json',manifest)
    write_json(output/'run_manifest.json',manifest)
    try:
        results={}
        for scope,key in [('train','train_run'),('dev','dev_run')]:
            directory=Path(settings[key]);directory=directory if directory.is_absolute() else ROOT/directory
            source=load_features(directory)
            if source['split']!=scope:raise ValueError('Không đổi tên split khi đánh giá')
            manifest['source_features'][scope]={'source':str(directory),'sha256':source['feature_sha256'],'samples':len(source['rows'])}
            x,t=source['features'],source['prototypes'];il=source['info_labels'];labels=source['labels'];a,b=source['scores']
            clip_scale=source['manifest'].get('logit_scale')
            if clip_scale!=100.:raise ValueError('Thử nghiệm này khóa checkpoint logit scale 100')
            clip_probability=probability_features(x@t.T,clip_scale)
            norm_gate=np.asarray(il)[(x@t.T).argmax(1)]=='informative'
            raw_gate=np.asarray(il)[a.argmax(1)]=='informative'
            base_human=humanitarian(raw_gate,b,labels)
            _,stata=gate_ablation(x,source['mean_prototypes'],a,b,il,labels)
            prepared=[]
            for seed in [0,1,2]:
                em_prob,em_labels,groups,ties=binary_em_probabilities(a,il,seed)
                stata_key=f'stata_gate_seed_{seed}'
                if groups.tolist()!=stata['batch_ids'][stata_key]:raise ValueError('Hai nhánh khác nhóm')
                stata_gate=np.asarray(stata['gate_labels'][stata_key])=='informative'
                majority=combine(np.stack([norm_gate,np.asarray(em_labels)=='informative',stata_gate],1))['majority']
                soft_gate,mean_probability=soft_mean([clip_probability,em_prob,stata['posterior'][stata_key]],il.index('informative'))
                gates={'majority_control':majority,'soft_mean_three':soft_gate,'protected_majority':protect(raw_gate,majority)}
                choices={name:humanitarian(gate,b,labels) for name,gate in gates.items()}
                if np.asarray(choices['protected_majority'])[raw_gate].tolist()!=np.asarray(base_human)[raw_gate].tolist():
                    raise ValueError('Guard thay đổi lớp của ảnh CLIP gốc đã cho đi tiếp')
                prepared.append({'seed':seed,'gates':gates,'predictions':choices,'mean_probability':mean_probability,
                                 'groups':groups,'em_soft_ties':ties})
            # Các quy tắc phía trên không dùng ground truth.
            truth=[r['y_true'] for r in source['rows']];binary_truth=source['binary_truth']
            base_binary=np.asarray(il)[a.argmax(1)].tolist()
            summary={'samples':len(x),'split':scope,'binary_ground_truth_available':binary_truth is not None,
                     'baseline':{'humanitarian':metric(truth,base_human,labels),
                                 'binary':metric(binary_truth,base_binary,il) if binary_truth is not None else None},
                     'by_method':{name:[] for name in prepared[0]['gates']}}
            for trial in prepared:
                seed=trial['seed']
                for name,gate in trial['gates'].items():
                    binary=np.where(gate,'informative','not_informative').tolist();pred=trial['predictions'][name]
                    report={'seed':seed,'humanitarian':metric(truth,pred,labels),
                            'binary':metric(binary_truth,binary,il) if binary_truth is not None else None,
                            'gate_positive':int(gate.sum()),'em_soft_ties':trial['em_soft_ties']}
                    summary['by_method'][name].append(report)
                    print(f'[FUSION] {scope}; {name}; seed={seed}; human_f1={report["humanitarian"]["macro_f1"]:.6f}; binary_f1={report["binary"]["macro_f1"] if report["binary"] else None}',flush=True)
                folder=output/scope/f'grouping_seed_{seed}';folder.mkdir(parents=True)
                with (folder/'predictions.tsv').open('x',newline='',encoding='utf-8') as f:
                    writer=csv.writer(f,delimiter='\t')
                    writer.writerow(['id','humanitarian_truth','binary_truth','raw_clip_gate','majority_gate','soft_gate','protected_gate',
                                     'majority_pred','soft_pred','protected_pred','soft_informative_probability','batch_id'])
                    writer.writerows((row['id'],row['y_true'],binary_truth[i] if binary_truth is not None else '',bool(raw_gate[i]),
                                      bool(trial['gates']['majority_control'][i]),bool(trial['gates']['soft_mean_three'][i]),bool(trial['gates']['protected_majority'][i]),
                                      trial['predictions']['majority_control'][i],trial['predictions']['soft_mean_three'][i],trial['predictions']['protected_majority'][i],
                                      float(trial['mean_probability'][i,il.index('informative')]),int(trial['groups'][i])) for i,row in enumerate(source['rows']))
            summary['mean']={name:{task:{k:float(np.mean([r[task][k] for r in reports])) for k in ['accuracy','macro_f1']}
                                      if reports[0][task] is not None else None for task in ['binary','humanitarian']}
                             for name,reports in summary['by_method'].items()}
            results[scope]=summary
        results['limitations']=['Hai quy tắc cố định; không search trọng số/ngưỡng theo metric dev.',
                                'EM soft assignments được ghép sang nhãn trước khi trộn; không mặc định cột cụm là cột lớp.',
                                'Soft vote dùng confidence chưa hiệu chuẩn; không diễn giải mean là xác suất đúng thực tế.',
                                'Protected gate giữ recall positive của baseline nhưng có thể tăng false positive.',
                                'Train/dev đã xem nhiều lần, chưa phải test độc lập; không chứng minh tổ hợp tốt nhất toàn cục.']
        write_json(output/'analysis.json',results)
        with (output/'summary.tsv').open('x',newline='',encoding='utf-8') as f:
            writer=csv.writer(f,delimiter='\t');writer.writerow(['split','task','method','samples','accuracy','macro_f1'])
            for scope,summary in results.items():
                if scope=='limitations':continue
                for task,values in summary['baseline'].items():
                    if values is not None:writer.writerow([scope,task,'baseline',summary['samples'],values['accuracy'],values['macro_f1']])
                for name,tasks in summary['mean'].items():
                    for task,values in tasks.items():
                        if values is not None:writer.writerow([scope,task,name,summary['samples'],values['accuracy'],values['macro_f1']])
        manifest['status']='complete'
        manifest['output_sha256']={str(p.relative_to(output)):sha256(p.read_bytes()) for p in output.rglob('*') if p.is_file() and p.name!='run_manifest.json'}
        write_json(output/'run_manifest.json',manifest);print(f'[SAVED] {output}',flush=True);return output
    except Exception as error:
        manifest.update(status='failed',error=f'{type(error).__name__}: {error}')
        write_json(output/'run_manifest.json',manifest);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(ROOT/'experiments/clip_transductive/config_fusion.json'))
    parser.add_argument('--output-dir');args=parser.parse_args();run(json.loads(Path(args.config).read_text()),args.output_dir)
