"""Test end-to-end min5: dùng nguyên infer/evaluate đã chạy dev, không tune bằng test."""
import time

import numpy as np
import yaml

from . import data, cascade_protocol as protocol, fakeddit_fewshot_cascade as cascade, min5_protocol
from .backbone import Backbone, encode_cohort
from .evaluate import publish, table
from .fewshot_support_thresholds import aggregate
from .io import path, digest, new_run, sha256, write_json
from .min5_test_runner import RUNTIME_KEYS


def inspect(protocol_file,settings_file=None,spec_file=None,*,require_sealed=False):
    lock=protocol.validate(protocol_file,require_sealed=require_sealed,settings_file=settings_file,
                           spec_file=spec_file,check_runtime=require_sealed)
    p=lock['payload'];config_file,recipe_file=protocol.files(p,settings_file,spec_file)
    settings=yaml.safe_load(config_file.read_text())
    if {k:settings.get(k) for k in ['model','batch_size']}!=p['inference_settings']:
        raise ValueError('Model/batch_size khác protocol')
    if settings.get('device','auto') not in ['auto','cpu','cuda','mps']:
        raise ValueError('Device không hợp lệ')
    return lock,settings,config_file,recipe_file


def plan(protocol_file,settings_file=None,spec_file=None):
    lock,settings,_,_=inspect(protocol_file,settings_file,spec_file)
    p=lock['payload'];runtime_matches=p['runtime'] is not None and protocol.runtime_versions()==p['runtime']
    return {'status':'plan_only','kind':p['kind'],'protocol_state':p['state'],'protocol_sha256':lock['sha256'],
            'split':'test','stages':2,'samples':3000,'methods':p['methods'],'seeds':list(range(5)),
            'method_seed_groups':30,'support_per_seed':384,'test_opened':False,
            'runtime_matches':runtime_matches,'executable':p['state']=='sealed_cascade' and runtime_matches,
            'scope':'Six fixed branches; prepare-only does not read train/test roots or load model'}


def run(protocol_file,train_root,test_root,output_root='results/final/cascade_test_min5',settings_file=None,
        *,spec_file=None,backend_factory=Backbone):
    # Protocol/code/config/runtime dừng trước mọi truy cập dataset.
    lock,settings,config_file,recipe_file=inspect(protocol_file,settings_file,spec_file,require_sealed=True)
    p=lock['payload'];recipe=p['recipe']
    train_root,test_root,output_root=path(train_root),path(test_root),path(output_root)
    if (output_root.is_relative_to(train_root) or output_root.is_relative_to(test_root)
            or path(protocol_file).is_relative_to(output_root)):
        raise ValueError('Output phải tách dataset và protocol')
    for root,expected in [(train_root,recipe['manifest_sha256']['train']),(test_root,p['test_manifest_sha256'])]:
        if sha256(root/'manifest.json')!=expected:
            raise ValueError('Sai manifest train/test min5 đã niêm phong')
    train,train_meta=data.fakeddit(train_root,'train')
    if (train_meta.get('split')!='train' or train_meta.get('source_split')!='train'
            or train_meta.get('manifest_sha256')!=recipe['manifest_sha256']['train']
            or min5_protocol.identities(train)!=p['excluded_train']):
        raise ValueError('Train khác source đã khóa')
    min5_protocol.validate_support(recipe['support_snapshot'],train)
    test,test_meta=data.fakeddit(test_root,'test')
    protocol.check_test_cohort(lock,test,test_meta,settings_file=config_file,spec_file=recipe_file)
    selected_ids={i for e in recipe['support_snapshot'] for i in e['ids']}
    selected_train=[r for r in train if r['id'] in selected_ids]
    manifest={'kind':'fakeddit_cascade_test_min5','model':recipe['model'],'dataset':'fakeddit_6way',
              'mode':'full','split':'test','stages':2,'samples':len(test),'test_opened':True,
              'regime':'supervised_fewshot_frozen_backbone','primary_metric':p['primary_metric'],
              'protocol_sha256':lock['sha256'],'protocol_file_sha256':sha256(protocol_file),
              'dev_run_sha256':p['dev_run_sha256'],'specification':recipe,'methods':p['methods'],
              'protocol_fingerprint':p['fingerprint'],'runtime_versions':protocol.runtime_versions(),
              'train_cohort':train_meta,'test_cohort':test_meta,'train_features_samples':len(selected_train),
              'support_budget_per_seed':384,'binary_labels':{'0':'Real','1':'Fake'},
              'scope':'Fixed six branches on sealed test. No dev partition, threshold/seed selection or additional labels. Oracle diagnostic only.',
              'label_usage':'Train labels for all heads. Test labels introduced only after every posterior is computed.',
              'uncertainty':'Mean/sample std on same test IDs and overlapping support seeds; not a confidence interval.'}
    output=new_run({'output_root':output_root},'cascade','test',False)
    write_json(output/'frozen_plan.json',manifest);write_json(output/'support.json',recipe['support_snapshot'])
    write_json(output/'run.json',{**manifest,'status':'running'})
    start=time.perf_counter();backend=backend_factory(settings.get('device','auto'))
    try:
        reference=recipe['model_runtime_reference'];runtime=getattr(backend,'metadata',{})
        keys=RUNTIME_KEYS+(['device'] if 'device' in reference else [])
        if any(digest(runtime.get(k))!=digest(reference[k]) for k in keys):
            raise ValueError('Backbone/device khác runtime reference đã khóa')
        manifest['model_runtime']=runtime
        image,title=encode_cohort(backend,selected_train,settings['batch_size'],with_titles=True)
        query_image,query_title=encode_cohort(backend,test,settings['batch_size'],with_titles=True)
    finally:
        backend.unload()
    manifest['encoding_seconds']=time.perf_counter()-start
    scores,models=cascade.infer(image,title,np.asarray([int(r['6way']) for r in selected_train]),
                                query_image,query_title,[r['id'] for r in selected_train],recipe)
    # Test labels chỉ được dùng đánh giá sau khi đã hoàn tất fit/predict tất cả head.
    truth=np.asarray([int(r['6way']) for r in test]);ids=np.asarray([r['id'] for r in test])
    records,predictions,details,comparisons,routing,gates,subsets,oracle,arrays,subtypes=cascade.evaluate(truth,ids,scores,recipe)
    if subsets or {r['method'] for r in records}!=set(p['methods']) or len(records)!=30:
        raise ValueError('Test không được lẫn partition dev hoặc thay suite đã khóa')
    write_json(output/'cohort.json',[{k:r[k] for k in ['id','6way','image_sha256','text_sha256']} for r in test])
    table(output/'comparisons.tsv',comparisons)
    table(output/'comparisons_summary.tsv',aggregate(comparisons,['policy','decision'],
          ['delta_accuracy_pp','delta_macro_f1_pp','delta_balanced_accuracy_pp']))
    table(output/'routing.tsv',routing);table(output/'gate_metrics.tsv',gates)
    table(output/'subtype_detection.tsv',subtypes)
    table(output/'subtype_summary.tsv',aggregate(subtypes,['policy','decision','class_6way'],
          ['binary_gate_recall','end_to_end_recall']))
    write_json(output/'oracle_diagnostic.json',{'uses_true_binary_labels':True,
               'scope':'Oracle gate diagnostic only, excluded from method ranking','metrics':oracle})
    np.savez_compressed(output/'replay_scores.npz',ids=ids,labels_6way=truth,**arrays)
    np.savez_compressed(output/'linear_models.npz',**models)
    manifest['elapsed_seconds']=time.perf_counter()-start
    return publish(output,records,predictions,details,manifest)
