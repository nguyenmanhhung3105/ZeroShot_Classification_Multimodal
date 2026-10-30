"""CLIP hai tầng; đối chứng ảnh, multimodal hoặc multi-view qua cờ riêng; không fit nhãn."""
import argparse
from copy import deepcopy
import csv
import json
from pathlib import Path
import sys
import uuid

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import numpy as np
import yaml
from experiments.waffle_lab import run as lab
from research.upstream import author_api,verify_sources,choose_device
from results_layout import run_path,model_folder
from experiments.clip_two_stage.validation import check_holdout,lock_holdout


def model_dtype(model):
    """OpenAI CLIP có .dtype; OpenCLIP thường chỉ có dtype trên parameter."""
    dtype = getattr(model, 'dtype', None)
    if dtype is None:
        try:
            dtype = next(model.parameters()).dtype
        except (AttributeError, StopIteration):
            return 'unknown'
    return str(dtype)


def decide(info_scores,human_scores,info_labels,human_labels):
    """Không nhận nhãn thật; giữ argmax/tie order của các bank gốc."""
    a,b=np.asarray(info_scores),np.asarray(human_scores)
    if set(info_labels)!={'informative','not_informative'} or 'not_humanitarian' not in human_labels:
        raise ValueError('Sai taxonomy')
    if a.shape!=(len(b),len(info_labels)) or b.shape!=(len(a),len(human_labels)) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Sai shape hoặc điểm không hữu hạn')
    gate=np.asarray(info_labels)[a.argmax(1)]=='informative'
    active=[i for i,label in enumerate(human_labels) if label!='not_humanitarian']
    predicted=np.asarray(human_labels)[np.asarray(active)[b[:,active].argmax(1)]]
    predicted[~gate]='not_humanitarian'
    return predicted.tolist(),gate


def decide_margin_pool(info_scores,human_scores,info_labels,human_labels):
    """Đối chứng cổng mềm cố định: trung bình hai margin cosine có dấu.

    Không tune trọng số/ngưỡng. Không gọi margin là xác suất đã hiệu chuẩn.
    """
    # Dùng cùng kiểm tra shape/taxonomy với cổng cứng.
    decide(info_scores,human_scores,info_labels,human_labels)
    a,b=np.asarray(info_scores),np.asarray(human_scores)
    active=[i for i,label in enumerate(human_labels) if label!='not_humanitarian']
    first=a[:,info_labels.index('informative')]-a[:,info_labels.index('not_informative')]
    second=b[:,active].max(1)-b[:,human_labels.index('not_humanitarian')]
    combined=(first+second)/2
    gate=combined>=0
    predicted=np.asarray(human_labels)[np.asarray(active)[b[:,active].argmax(1)]]
    predicted[~gate]='not_humanitarian'
    return predicted.tolist(),gate,combined


def run(settings,prepare_only=False):
    lab.validate(settings)
    multimodal=settings.get('multimodal',False)
    if type(multimodal) is not bool:raise ValueError('multimodal phải là boolean')
    full_frame_view=settings.get('full_frame_view',False)
    if type(full_frame_view) is not bool or (full_frame_view and multimodal):
        raise ValueError('full_frame_view là boolean; thử riêng, chưa ghép thêm multimodal')
    stata_gate=settings.get('stata_gate',False)
    if type(stata_gate) is not bool or (stata_gate and (multimodal or full_frame_view)):
        raise ValueError('stata_gate là boolean và thử riêng với multimodal/multi-view')
    export_features=settings.get('export_features',False)
    if type(export_features) is not bool or (export_features and (multimodal or full_frame_view)):
        raise ValueError('export_features là boolean, dùng với ảnh gốc image-only')
    tda_evaluate=settings.get('tda_evaluate',False)
    if type(tda_evaluate) is not bool or (tda_evaluate and not (export_features or stata_gate)):
        raise ValueError('tda_evaluate là boolean và cần export_features hoặc stata_gate')
    if settings['task']!='humanitarian': raise ValueError('Runner cuối xuất tám lớp humanitarian')
    scope=settings.get('evaluation_scope','exploratory')
    if scope not in ('exploratory','heldout'):raise ValueError('evaluation_scope chỉ exploratory/heldout')
    method='clip_two_stage_heldout' if scope=='heldout' else 'clip_two_stage'
    if multimodal:method='clip_two_stage_mm_heldout' if scope=='heldout' else 'clip_two_stage_mm'
    if full_frame_view:method='clip_two_stage_views_heldout' if scope=='heldout' else 'clip_two_stage_views'
    if stata_gate:method='clip_two_stage_stata_heldout' if scope=='heldout' else 'clip_two_stage_stata'
    elif export_features:method='clip_two_stage_features_heldout' if scope=='heldout' else 'clip_two_stage_features'
    output=ROOT/run_path(method,'crisismmd','humanitarian',settings['max_samples'],
                        model_folder(backbones=[settings['backbone']]),uuid.uuid4().hex[:12]) if settings.get('output_dir','auto')=='auto' else lab.project_path(settings['output_dir'])
    output.mkdir(parents=True,exist_ok=True)
    manifest_file=output/'run_manifest.json'
    # Không ghi đè cả lượt đã prepare/lỗi.
    with manifest_file.open('x') as f: json.dump({'status':'preparing'},f)
    backend=None
    manifest={'status':'preparing','model':model_folder(backbones=[settings['backbone']]),'task':'humanitarian',
              'samples':settings['max_samples'],'seed':settings['seed'],'input':'image_text' if multimodal else 'image_only','label_fit':False,
              'rule':'semantic_argmax_gate_then_seven_class_semantic_waffle_argmax',
              'additional_ablation':'equal_mean_signed_cosine_margins_threshold_zero_no_fit',
              'runner_sha256':lab.file_hash(Path(__file__))}
    manifest['evaluation_scope']=scope
    manifest['full_frame_view']=full_frame_view
    manifest['stata_gate']=stata_gate
    manifest['export_features']=export_features or stata_gate
    manifest['tda_evaluate']=tda_evaluate
    if stata_gate:
        from experiments.clip_stata.core import REFERENCE_COMMIT
        manifest['stata_protocol']={'reference_commit':REFERENCE_COMMIT,'scope':'binary_gate_only',
                                   'grouping_seeds':[0,1,2],'query_size':75,'normalized_gate_control':True,
                                   'source_sha256':lab.file_hash(ROOT/'experiments/clip_stata/core.py')}
    if full_frame_view:
        manifest['view_protocol']={'padding':'black_square_preserve_aspect_then_original_preprocess',
                                   'aggregation':'equal_mean_cosine_before_gate_no_renormalization',
                                   'source_sha256':lab.file_hash(ROOT/'experiments/clip_two_stage/views.py')}
    manifest['validation_sha256']=lab.file_hash(ROOT/'experiments/clip_two_stage/validation.py')
    try:
        api,sources=author_api(),verify_sources()
        plans=[]
        for task,variant in [('informativeness','semantic'),('humanitarian','semantic_waffle')]:
            active=deepcopy(settings);active.update(task=task,variant=variant)
            directory=output/task;directory.mkdir()
            plans.append(lab.freeze_plan(active,directory,api,sources))
        final_plan_value=plans[0]['dataset_config'].get('final_dn_unit_plan')
        final_plan_path=None
        if final_plan_value:
            from experiments.clip_final.protocol import load_plan
            final_plan_path=lab.project_path(final_plan_value)
            locked_final=load_plan(final_plan_path)
            if (scope!='heldout' or plans[0]['dataset_config'].get('source_split')!='test' or
                    not export_features or stata_gate or tda_evaluate or multimodal or full_frame_view or
                    settings['seed']!=0 or settings['backbone']!='metaclip2_vith14' or
                    settings['max_samples']>locked_final['protocol']['rules']['max_test_samples']):
                raise ValueError('Final DN-unit cần test image-only, export feature, MetaCLIP2 seed 0, không ablation khác')
            for final_task,final_variant in [('informativeness','semantic'),('humanitarian','semantic_waffle')]:
                final_index=0 if final_task=='informativeness' else 1
                if plans[final_index]['prompt_banks'][final_variant]!=locked_final['protocol']['prompt_banks'][final_task]:
                    raise ValueError('Prompt test khác bank đã khóa trước khi xem test')
            manifest['final_dn_unit_protocol_sha256']=locked_final['protocol_sha256']
        manifest['protocol_fingerprints']=[p['protocol_fingerprint'] for p in plans]
        text_banks=None
        if multimodal:
            from prompts.crisismmd_prompts import get_branch_prompt_sets
            text_banks={task:get_branch_prompt_sets(task)[1] for task in ['informativeness','humanitarian']}
            for plan in plans:
                if list(text_banks[plan['settings']['task']])!=plan['label_order']:
                    raise ValueError('Prompt text khác thứ tự lớp của nhánh ảnh')
            text_plan={'banks':text_banks,'text_aggregation':'mean_embed','text_weight_fixed':.25,
                       'conditional_rule':'image_gate_informative_and_text_gate_informative_and_nonempty',
                       'stage1':'image_only_unchanged','scale':'checkpoint_logit_scale',
                       'source_sha256':lab.file_hash(ROOT/'experiments/clip_two_stage/multimodal.py')}
            lab.write_json(output/'text_plan.json',text_plan)
            manifest['text_plan_fingerprint']=lab.digest(text_plan)
        if scope=='heldout':
            split=plans[0]['dataset_config'].get('source_split')
            if split not in ('dev','val','validate','validation','test'):
                raise ValueError('Chế độ heldout cần processed validation/test; cấu hình hiện đang trỏ train')
        if prepare_only:
            manifest['status']='prepared_no_inference'
            print(f'[PREPARED] {output}; chưa đọc dataset hoặc tải model')
            return output
        cohorts=[]
        # Humanitarian có thể được đánh giá trên split riêng, không cần lấy giao
        # với informative dev/test. Nhãn gate thật không bao giờ là đầu vào model.
        gate_labels_available=plans[0]['dataset_config'].get('gate_labels_available',True)
        human_rows=None
        if not gate_labels_available:
            human_rows=lab.read_cohort(plans[1]['settings'],{'datasets':{'crisismmd':plans[1]['dataset_config']}})
        manifest['gate_ground_truth_available']=gate_labels_available
        for index,plan in enumerate(plans):
            if human_rows is not None:
                rows=deepcopy(human_rows)
                if index==0:
                    for row in rows:row['y_true']=None
            else:
                rows=lab.read_cohort(plan['settings'],{'datasets':{'crisismmd':plan['dataset_config']}})
            if scope=='heldout':
                reference_path=final_plan_path or lab.project_path('configs/prompts/crisismmd_development_exclusion.json')
                reference=locked_final['protocol']['exclusion'] if final_plan_path else json.loads(reference_path.read_text())
                check=check_holdout(rows,reference,plan['dataset_config']['source_split'])
                manifest.setdefault('holdout_checks',[]).append(check)
                manifest['development_reference_sha256']=lab.file_hash(reference_path)
                cohort=lock_holdout(plan['settings'],rows,ROOT)
            else:
                cohort=lab.lock_cohort(plan['settings'],rows)
            lab.write_json(output/plan['settings']['task']/'cohort.json',cohort)
            cohorts.append(cohort)
        def identity(cohort):return [(r['id'],r['image_sha256']) for r in cohort['rows']]
        if identity(cohorts[0])!=identity(cohorts[1]):raise ValueError('Hai task khác ảnh/thứ tự')
        manifest['cohort_fingerprints']=[c['cohort_fingerprint'] for c in cohorts]
        device=choose_device(api.torch,settings.get('device','auto'))
        backend=lab.load_backend(api,settings['backbone'],device)
        manifest.update(status='running',device=device,compute_dtype=model_dtype(backend.model),
                        logit_scale=backend.logit_scale,probabilities_calibrated=False)
        lab.write_json(manifest_file,manifest)
        from PIL import Image
        rows=cohorts[1]['rows'];blocks=[];whole_blocks=[]
        for start in range(0,len(rows),settings['batch_size']):
            images=[]
            try:
                for row in rows[start:start+settings['batch_size']]:
                    with Image.open(lab.project_path(row['image_path'])) as source:images.append(source.convert('RGB'))
                blocks.append(backend.encode_images(images).detach().float().cpu().numpy())
                if full_frame_view:
                    from experiments.clip_two_stage.views import full_frame
                    whole_images=[full_frame(image) for image in images]
                    try:
                        # Encode riêng để không tăng gấp đôi peak batch memory trên T4.
                        whole_blocks.append(backend.encode_images(whole_images).detach().float().cpu().numpy())
                    finally:
                        for image in whole_images:image.close()
            finally:
                for image in images:image.close()
            print(f'[IMAGE] {min(start+settings["batch_size"],len(rows))}/{len(rows)}',flush=True)
        vectors=np.concatenate(blocks);vectors/=np.maximum(np.linalg.norm(vectors,axis=1,keepdims=True),1e-12)
        if full_frame_view:
            whole_vectors=np.concatenate(whole_blocks)
            whole_vectors/=np.maximum(np.linalg.norm(whole_vectors,axis=1,keepdims=True),1e-12)
        scores=[];whole_scores=[]
        for plan in plans:
            bank=plan['prompt_banks'][plan['settings']['variant']]
            scoring_vectors=np.concatenate([vectors,whole_vectors]) if full_frame_view else vectors
            per_prompt,_=lab.prompt_scores(backend,bank,plan['label_order'],scoring_vectors,settings['batch_size'])
            averaged=per_prompt.mean(1)
            scores.append(averaged[:len(rows)])
            if full_frame_view:whole_scores.append(averaged[len(rows):])
        labels=plans[1]['label_order'];info_labels=plans[0]['label_order']
        stata_predictions={}
        if stata_gate or export_features:
            from experiments.clip_stata.core import encode_prototypes
            info_bank=plans[0]['prompt_banks'][plans[0]['settings']['variant']]
            info_prototypes=encode_prototypes(backend,info_bank,info_labels,settings['batch_size'])
            human_bank=plans[1]['prompt_banks'][plans[1]['settings']['variant']]
            human_prototypes=encode_prototypes(backend,human_bank,labels,settings['batch_size'])
            if not np.allclose(vectors@info_prototypes.T,scores[0],atol=2e-5,rtol=1e-4):
                raise ValueError('Feature/prototype export khác score informativeness')
            if not np.allclose(vectors@human_prototypes.T,scores[1],atol=2e-5,rtol=1e-4):
                raise ValueError('Feature/prototype export khác score humanitarian')
            feature_name='stata_features.npz' if stata_gate else 'clip_features.npz'
            np.savez_compressed(output/feature_name,image_features=vectors,
                                info_mean_prototypes=info_prototypes,human_mean_prototypes=human_prototypes,
                                ids=np.asarray([r['id'] for r in rows]))
            manifest.update(feature_file=feature_name,feature_sha256=lab.file_hash(output/feature_name))
        if stata_gate:
            from experiments.clip_stata.core import gate_ablation
            stata_predictions,stata_diagnostics=gate_ablation(vectors,info_prototypes,*scores,info_labels,labels)
            lab.write_json(output/'stata_diagnostics.json',stata_diagnostics)
            manifest['stata_features_sha256']=manifest['feature_sha256']
        predicted,gate=decide(*scores,info_labels,labels)
        soft_pred,soft_gate,soft_margin=decide_margin_pool(*scores,info_labels,labels)
        mm_predictions={};mm_diagnostics=None;text_cosine=None
        if multimodal:
            from experiments.clip_two_stage.multimodal import read_texts,text_scores,predict
            texts,text_identity=read_texts(plans[1]['dataset_config'],rows,lab.project_path)
            lab.write_json(output/'text_cohort.json',text_identity)
            manifest['text_cohort_fingerprint']=lab.digest(text_identity)
            text_cosine,empty,text_diagnostics=text_scores(backend,texts,text_banks,settings['batch_size'])
            mm_predictions,mm_diagnostics=predict(*scores,*text_cosine,empty,info_labels,labels,backend.logit_scale)
            lab.write_json(output/'text_diagnostics.json',text_diagnostics)
            manifest['conditional_text_used_count']=int(mm_diagnostics['text_used'].sum())
        truth=[r['y_true'] for r in rows]
        view_predictions={}
        if full_frame_view:
            from experiments.clip_two_stage.views import classify_views
            view_predictions=classify_views(scores,whole_scores,info_labels,labels)
        # Cùng điểm tầng 2, so trực tiếp một tầng và hai tầng trong một lượt.
        direct=np.asarray(labels)[scores[1].argmax(1)].tolist()
        final_scores=np.full_like(scores[1],-2.)
        positive=[i for i,l in enumerate(labels) if l!='not_humanitarian']
        final_scores[np.ix_(gate,positive)]=scores[1][np.ix_(gate,positive)]
        final_scores[~gate,labels.index('not_humanitarian')]=1.
        metrics={'two_stage':lab.evaluate(truth,predicted,final_scores,labels),
                 'single_stage':lab.evaluate(truth,direct,scores[1],labels),
                 'margin_pool':lab.evaluate(truth,soft_pred,scores[1],labels),
                 'fixed':sum(a!=t and b==t for a,b,t in zip(direct,predicted,truth)),
                 'broken':sum(a==t and b!=t for a,b,t in zip(direct,predicted,truth)),
                 'gate_false_negative':int(sum(not g and t!='not_humanitarian' for g,t in zip(gate,truth))),
                 'gate_false_positive':int(sum(g and t=='not_humanitarian' for g,t in zip(gate,truth)))}
        metrics['margin_pool_vs_hard']={
            'fixed':sum(a!=t and b==t for a,b,t in zip(predicted,soft_pred,truth)),
            'broken':sum(a==t and b!=t for a,b,t in zip(predicted,soft_pred,truth))}
        metrics['multimodal']={}
        if gate_labels_available:
            info_truth=[r['y_true'] for r in cohorts[0]['rows']]
            metrics['informativeness']=lab.evaluate(info_truth,np.asarray(info_labels)[scores[0].argmax(1)].tolist(),scores[0],info_labels)
        metrics['stata']={name:{'metrics':lab.evaluate(truth,choices,scores[1],labels),
                'fixed_vs_original':sum(a!=t and b==t for a,b,t in zip(predicted,choices,truth)),
                'broken_vs_original':sum(a==t and b!=t for a,b,t in zip(predicted,choices,truth))}
                for name,choices in stata_predictions.items()}
        if stata_gate:
            names=[f'stata_gate_seed_{seed}' for seed in [0,1,2]]
            metrics['stata_mean']={key:float(np.mean([metrics['stata'][name]['metrics'][key] for name in names]))
                                   for key in ['accuracy','macro_f1']}
            with (output/'stata_predictions.tsv').open('w',newline='') as stream:
                writer=csv.writer(stream,delimiter='\t');writer.writerow(['id','y_true',*stata_predictions])
                writer.writerows([r['id'],r['y_true'],*[stata_predictions[name][i] for name in stata_predictions]]
                                 for i,r in enumerate(rows))
        metrics['views']={method_name:{'metrics':lab.evaluate(truth,choices,scores[1],labels),
                'fixed_vs_original':sum(a!=t and b==t for a,b,t in zip(predicted,choices,truth)),
                'broken_vs_original':sum(a==t and b!=t for a,b,t in zip(predicted,choices,truth))}
                for method_name,choices in view_predictions.items()}
        for method_name,choices in mm_predictions.items():
            metrics['multimodal'][method_name]={'metrics':lab.evaluate(truth,choices,scores[1],labels),
                'fixed_vs_image_two_stage':sum(a!=t and b==t for a,b,t in zip(predicted,choices,truth)),
                'broken_vs_image_two_stage':sum(a==t and b!=t for a,b,t in zip(predicted,choices,truth))}
        lab.write_json(output/'metrics.json',metrics)
        with (output/'predictions.tsv').open('w',newline='') as f:
            writer=csv.writer(f,delimiter='\t');writer.writerow(['id','y_true','single_stage_pred','y_pred','gate_informative','info_cosine_scores','human_cosine_scores','margin_pool_pred','margin_pool_gate','pooled_margin'])
            writer.writerows((r['id'],r['y_true'],a,b,bool(g),json.dumps(i.tolist()),json.dumps(h.tolist()),s,bool(sg),float(sm)) for r,a,b,g,i,h,s,sg,sm in zip(rows,direct,predicted,gate,*scores,soft_pred,soft_gate,soft_margin))
        if multimodal:
            with (output/'multimodal_predictions.tsv').open('w',newline='') as stream:
                methods=list(mm_predictions)
                writer=csv.writer(stream,delimiter='\t')
                writer.writerow(['id','y_true','text_info_scores','text_human_scores','text_empty','text_gate_informative','conditional_text_weight',*methods])
                writer.writerows([r['id'],r['y_true'],json.dumps(text_cosine[0][i].tolist()),json.dumps(text_cosine[1][i].tolist()),
                                  bool(mm_diagnostics['empty'][i]),bool(mm_diagnostics['text_gate'][i]),float(mm_diagnostics['text_weight'][i]),
                                  *[mm_predictions[m][i] for m in methods]] for i,r in enumerate(rows))
        if full_frame_view:
            with (output/'view_predictions.tsv').open('w',newline='') as stream:
                methods=list(view_predictions);writer=csv.writer(stream,delimiter='\t')
                writer.writerow(['id','y_true','full_info_cosine_scores','full_human_cosine_scores',*methods])
                writer.writerows([r['id'],r['y_true'],json.dumps(whole_scores[0][i].tolist()),json.dumps(whole_scores[1][i].tolist()),
                                  *[view_predictions[m][i] for m in methods]] for i,r in enumerate(rows))
        with (output/'summary.tsv').open('w',newline='') as f:
            writer=csv.writer(f,delimiter='\t');writer.writerow(['method','model','samples','seed','accuracy','macro_f1'])
            for method in ['single_stage','two_stage','margin_pool']:
                writer.writerow([method,manifest['model'],len(rows),settings['seed'],metrics[method]['accuracy'],metrics[method]['macro_f1']])
            for method_name,result in metrics['multimodal'].items():
                writer.writerow([method_name,manifest['model'],len(rows),settings['seed'],result['metrics']['accuracy'],result['metrics']['macro_f1']])
            for method_name,result in metrics['views'].items():
                writer.writerow([method_name,manifest['model'],len(rows),settings['seed'],result['metrics']['accuracy'],result['metrics']['macro_f1']])
            for method_name,result in metrics['stata'].items():
                writer.writerow([method_name,manifest['model'],len(rows),settings['seed'],result['metrics']['accuracy'],result['metrics']['macro_f1']])
            if stata_gate:
                writer.writerow(['stata_gate_mean',manifest['model'],len(rows),settings['seed'],metrics['stata_mean']['accuracy'],metrics['stata_mean']['macro_f1']])
        if 'informativeness' in metrics:
            with (output/'informativeness_summary.tsv').open('w',newline='') as stream:
                writer=csv.writer(stream,delimiter='\t')
                writer.writerow(['task','method','model','samples','accuracy','macro_f1'])
                writer.writerow(['informativeness','semantic_baseline',manifest['model'],len(rows),
                                 metrics['informativeness']['accuracy'],metrics['informativeness']['macro_f1']])
        manifest.update(status='complete',label_order=labels,info_label_order=info_labels,
                        image_encode_passes=2 if full_frame_view else 1,stage2_scores_computed_for_all_for_ablation=True)
        print(f'[RESULT] {output/"summary.tsv"}')
    except Exception as exc:
        manifest.update(status='failed',error=str(exc));raise
    finally:
        lab.write_json(manifest_file,manifest)
        if backend is not None:backend.unload()
    if tda_evaluate:
        # Feature đã complete và model đã unload; TDA chỉ đọc artifact, không
        # mã hóa lại ảnh hoặc cho cache từ train chảy sang cohort khác.
        from experiments.clip_tda.run import run as evaluate_tda
        manifest['tda_analysis_status']='running';lab.write_json(manifest_file,manifest)
        try:
            evaluate_tda(output,output/'tda_analysis',compare_ensemble=True)
            manifest.update(tda_analysis_status='complete',tda_analysis='tda_analysis/analysis.json')
        except Exception as error:
            manifest.update(tda_analysis_status='failed',tda_error=str(error))
            raise
        finally:lab.write_json(manifest_file,manifest)
    if final_plan_path:
        from experiments.clip_final.evaluate import run as evaluate_final
        manifest['final_dn_unit_analysis_status']='running';lab.write_json(manifest_file,manifest)
        try:
            final_dir=evaluate_final(output,final_plan_path,output/'final_dn_unit_analysis')
            final_report=json.loads((final_dir/'analysis.json').read_text())
            with (output/'summary.tsv').open('a',newline='') as stream:
                writer=csv.writer(stream,delimiter='\t')
                for method_name in ['majority','majority_dn_unit']:
                    final_metric=final_report['mean'][method_name]['humanitarian']
                    writer.writerow([method_name+'_mean',manifest['model'],len(rows),settings['seed'],final_metric['accuracy'],final_metric['macro_f1']])
            manifest.update(final_dn_unit_analysis_status='complete',final_dn_unit_analysis='final_dn_unit_analysis/analysis.json')
        except Exception as error:
            manifest.update(final_dn_unit_analysis_status='failed',final_dn_unit_error=str(error));raise
        finally:lab.write_json(manifest_file,manifest)
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='experiments/clip_two_stage/config.yaml')
    parser.add_argument('--max-samples',type=int)
    parser.add_argument('--seed',type=int)
    parser.add_argument('--backbone',choices=lab.BACKBONES)
    parser.add_argument('--evaluation-scope',choices=['exploratory','heldout'])
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--multimodal',action='store_true',default=None)
    parser.add_argument('--full-frame-view',action='store_true',default=None)
    parser.add_argument('--stata-gate',action='store_true',default=None)
    parser.add_argument('--export-features',action='store_true',default=None)
    parser.add_argument('--tda-evaluate',action='store_true',default=None)
    args=parser.parse_args();settings=yaml.safe_load(Path(args.config).read_text())
    for key in ['max_samples','seed','backbone','evaluation_scope','multimodal','full_frame_view','stata_gate','export_features','tda_evaluate']:
        value=getattr(args,key)
        if value is not None:settings[key]=value
    run(settings,args.prepare_only)


if __name__=='__main__':main()
