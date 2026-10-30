"""Một tổ hợp cố định: cổng InMaP không Sinkhorn + tầng chi tiết DN-unit."""
import argparse
import csv
import json
from pathlib import Path
import sys
import uuid
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import numpy as np
from experiments.clip_tda.run import load_features
from experiments.clip_zlap.run import project_path,evaluate
from experiments.clip_transductive.ensemble import humanitarian
from experiments.clip_transductive.run import sha256,write_json


def checked_manifest(directory):
    manifest=json.loads((directory/'run_manifest.json').read_text())
    if manifest.get('status')!='complete' or manifest.get('label_fit') is not False:raise ValueError('Nguồn chưa hoàn tất/đã fit nhãn thật')
    for name,expected in manifest['output_sha256'].items():
        if sha256((directory/name).read_bytes())!=expected:raise ValueError('Hash output nguồn đã đổi')
    return manifest


def run(inmap_run,dn_run,output_dir=None):
    inmap_run,dn_run=Path(inmap_run),Path(dn_run)
    left,right=checked_manifest(inmap_run),checked_manifest(dn_run)
    output=Path(output_dir) if output_dir else ROOT/'results/analysis/clip_inmap_dn'/uuid.uuid4().hex[:12]
    output.mkdir(parents=True,exist_ok=False)
    manifest={'status':'running','rule':'unbalanced_inmap_gate_then_existing_dn_unit_leaf','label_fit':False,
              'input_manifests':{str(path):sha256((path/'run_manifest.json').read_bytes()) for path in [inmap_run,dn_run]},
              'code_sha256':sha256(Path(__file__).read_bytes()),'tuning':False}
    write_json(output/'frozen_plan.json',manifest);write_json(output/'run_manifest.json',manifest)
    try:
        results={}
        for scope in ['train','dev']:
            source=load_features(project_path(left['settings'][scope+'_run']))
            expected=source['feature_sha256']
            if expected!=left['source_features'][scope]['sha256'] or expected!=right['source_feature_sha256'][scope]:raise ValueError('Hai phương pháp khác source feature')
            with np.load(inmap_run/scope/'unbalanced/proxies_scores.npz',allow_pickle=False) as l,np.load(dn_run/scope/'scores.npz',allow_pickle=False) as r:
                ids=[v['id'] for v in source['rows']]
                if l['ids'].tolist()!=ids or r['ids'].tolist()!=ids:raise ValueError('Sai ID/thứ tự')
                gate=np.asarray(source['info_labels'])[l['informativeness'].argmax(1)]=='informative'
                predicted=humanitarian(gate,r['humanitarian'],source['labels'])
            binary=np.where(gate,'informative','not_informative').tolist()
            results[scope]={'samples':len(predicted),**evaluate(source,predicted,binary)}
            with (output/(scope+'_predictions.tsv')).open('x',newline='') as f:
                writer=csv.writer(f,delimiter='\t');writer.writerow(['id','y_true','binary_truth','binary_pred','y_pred'])
                writer.writerows((row['id'],row['y_true'],source['binary_truth'][i] if source['binary_truth'] is not None else '',binary[i],predicted[i]) for i,row in enumerate(source['rows']))
            print(scope,results[scope]['humanitarian']['accuracy'],results[scope]['humanitarian']['macro_f1'],flush=True)
        results['limitations']=['One combination chosen from complementary train/dev findings; still exploratory, no test.',
                               'Deterministic full-cohort InMaP gate, no best seed or weight selection.']
        write_json(output/'analysis.json',results)
        manifest['status']='complete';manifest['output_sha256']={str(p.relative_to(output)):sha256(p.read_bytes()) for p in output.rglob('*') if p.is_file() and p.name!='run_manifest.json'}
        write_json(output/'run_manifest.json',manifest);print('[SAVED]',output);return output
    except Exception as error:
        manifest.update(status='failed',error=f'{type(error).__name__}: {error}');write_json(output/'run_manifest.json',manifest);raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inmap-run',required=True);p.add_argument('--dn-run',required=True);p.add_argument('--output-dir')
    a=p.parse_args();run(a.inmap_run,a.dn_run,a.output_dir)
