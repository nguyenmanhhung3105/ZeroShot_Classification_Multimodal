import csv
import json
import io
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import numpy as np
import yaml
from experiments.clip_final.protocol import ROOT, PLAN, RULES, CODE, load_plan
from experiments.waffle_lab.run import digest, file_hash
from experiments.clip_zlap.core import unit_vectors
from experiments.clip_two_stage.run import decide
from prompts.crisismmd_prompts import get_label_names


def synthetic_plan(root):
    il=list(get_label_names('informativeness'));hl=list(get_label_names('humanitarian'))
    rng=np.random.default_rng(81);x=unit_vectors(rng.uniform(.1,1.,(9,11)))
    info=unit_vectors(rng.uniform(.1,1.,(2,11)))*.95
    human=unit_vectors(rng.uniform(.1,1.,(8,11)))*.9
    for name in CODE:
        target=root/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
    p={'model':'metaclip2_vith14','label_fit':False,'rules':RULES,
       'info_labels':il,'human_labels':hl,'info_mean_prototypes':info.tolist(),
       'human_mean_prototypes':human.tolist(),'image_mean':x.mean(0).tolist(),
       'prompt_banks':{task:{l:[l] for l in labels} for task,labels in [('informativeness',il),('humanitarian',hl)]},
       'exclusion':{'image_ids':['100_0'],'tweet_ids':['100'],'image_sha256':[]},
       'code_sha256':{name:file_hash(root/name) for name in CODE},'limits':['synthetic fixture']}
    plan={'status':'frozen_before_test','protocol':p,'protocol_sha256':digest(p)}
    target=root/PLAN;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(plan))
    return plan,x,info,human


def feature_artifact(folder,plan,x,info,human,truth_shift=0):
    il=plan['protocol']['info_labels'];hl=plan['protocol']['human_labels']
    a,b=x@info.T,x@human.T;pred,gate=decide(a,b,il,hl)
    plans=[];cohorts=[]
    for task,labels,variant in [('informativeness',il,'semantic'),('humanitarian',hl,'semantic_waffle')]:
        p={'protocol':{'synthetic':True},'dataset_config':{'source_split':'test'},'label_order':labels,
           'prompt_banks':{variant:plan['protocol']['prompt_banks'][task]}}
        rows=[{'id':f'{200+i}_0','image_sha256':f'hash_{i}','event':'synthetic','source_split':'test','label_source':'image',
               'y_true':labels[(i+truth_shift)%len(labels)]} for i in range(len(x))]
        sub=folder/task;sub.mkdir(parents=True)
        (sub/'frozen_plan.json').write_text(json.dumps(p));(sub/'cohort.json').write_text(json.dumps({'rows':rows}))
        plans.append(p);cohorts.append(rows)
    np.savez_compressed(folder/'clip_features.npz',image_features=x,info_mean_prototypes=info,human_mean_prototypes=human,
                        ids=np.array([r['id'] for r in cohorts[1]]))
    manifest={'status':'complete','model':'metaclip2_vith14','seed':0,'samples':len(x),'input':'image_only',
        'export_features':True,'feature_file':'clip_features.npz','feature_sha256':file_hash(folder/'clip_features.npz'),
        'protocol_fingerprints':[digest(p['protocol']) for p in plans],'cohort_fingerprints':[digest(r) for r in cohorts]}
    (folder/'run_manifest.json').write_text(json.dumps(manifest))
    with (folder/'predictions.tsv').open('w',newline='') as f:
        writer=csv.writer(f,delimiter='\t');writer.writerow(['id','y_true','single_stage_pred','y_pred','gate_informative','info_cosine_scores','human_cosine_scores'])
        writer.writerows((r['id'],r['y_true'],hl[int(b[i].argmax())],pred[i],bool(gate[i]),json.dumps(a[i].tolist()),json.dumps(b[i].tolist())) for i,r in enumerate(cohorts[1]))


class FinalTests(unittest.TestCase):
    def test_frozen_protocol_rejects_changed_code_and_parameters(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();plan,*_=synthetic_plan(root)
            load_plan(root/PLAN,root)
            (root/CODE[-1]).write_text('changed')
            with self.assertRaisesRegex(ValueError,'Code khác'):load_plan(root/PLAN,root)
            plan['protocol']['rules']={**RULES,'coefficient':.5}
            plan['protocol_sha256']=digest(plan['protocol']);(root/PLAN).write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError,'Protocol'):load_plan(root/PLAN,root)

    def test_prepare_reads_only_test_excludes_train_dev_and_preserves_both_labels(self):
        from PIL import Image
        from experiments.clip_final.prepare_test import prepare
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();plan,*_=synthetic_plan(root)
            source=root/'data/raw/CrisisMMD_v2.0';ann=source/'crisismmd_datasplit_all';ann.mkdir(parents=True)
            images=source/'data_image/event';images.mkdir(parents=True)
            for key,color in [('100','red'),('200','blue'),('300','green'),('400','green')]:Image.new('RGB',(8,8),color).save(images/(key+'_0.png'))
            plan['protocol']['exclusion']['image_sha256']=[file_hash(images/'200_0.png')]
            plan['protocol_sha256']=digest(plan['protocol']);(root/PLAN).write_text(json.dumps(plan))
            for task,name in [('informativeness','informative'),('humanitarian','humanitarian')]:
                with (ann/f'task_{name}_text_img_test.tsv').open('w',newline='') as f:
                    writer=csv.DictWriter(f,fieldnames=['tweet_id','image_id','image','event_name','tweet_text','label_image'],delimiter='\t');writer.writeheader()
                    for key in ['100','200','300','400']:
                        writer.writerow({'tweet_id':key,'image_id':key+'_0','image':f'data_image/event/{key}_0.png','event_name':'event','tweet_text':'text',
                                         'label_image':'not_informative' if task=='informativeness' else 'infrastructure_and_utility_damage'})
            shutil.copy2(ROOT/'configs/experiment_config.yaml',root/'configs/experiment_config.yaml')
            (root/'experiments/clip_two_stage').mkdir(exist_ok=True)
            shutil.copy2(ROOT/'experiments/clip_two_stage/launcher.yaml',root/'experiments/clip_two_stage/launcher.yaml')
            output=prepare(root);manifest=json.loads((output/'manifest.json').read_text())
            self.assertEqual(manifest['selected']['samples'],1)
            self.assertEqual(manifest['excluded_counts']['overlap_train_dev_id_or_tweet'],1)
            self.assertEqual(manifest['excluded_counts']['overlap_train_dev_exact_image_hash'],1)
            self.assertEqual(manifest['excluded_counts']['duplicate_image_hash_within_test'],1)
            self.assertEqual(manifest['samples'][0]['y_true'],{'informativeness':'not_informative','humanitarian':'infrastructure_and_utility_damage'})
            self.assertTrue(all(p.endswith('_test.tsv') for p in manifest['raw_files_read']))
            with self.assertRaises(FileExistsError):prepare(root)
            # Kiểm tra gói Kaggle có cả frozen plan và manifest của hai task.
            import sys;sys.path.insert(0,str(ROOT/'scripts'));import launch_runs as launcher
            config=yaml.safe_load((root/'experiments/clip_final/launcher_test.yaml').read_text())
            with patch.object(launcher,'ROOT',root):
                launcher.validate(config,['final_dn_unit_test'])
                shutil.copy2(ROOT/'experiments/clip_two_stage/config.yaml',root/'experiments/clip_two_stage/config.yaml')
                stage=root/'abcdef123456_final';stage.mkdir()
                active,job,_,command,_,_,research=launcher.job_plan(config,'final_dn_unit_test',stage)
                self.assertEqual(active['datasets']['crisismmd']['final_dn_unit_plan'],PLAN)
                self.assertEqual(command[0],'experiments/clip_two_stage/run.py')
                trial=yaml.safe_load(research.read_text())
                self.assertTrue(trial['export_features']);self.assertFalse(trial['tda_evaluate']);self.assertEqual(trial['max_samples'],1)
                experiment=yaml.safe_load((root/'configs/experiment_crisismmd_test_final.yaml').read_text())
                launcher.archive_data(root/'data.zip',experiment,'crisismmd',1)
                with zipfile.ZipFile(root/'data.zip') as z:
                    self.assertIn('data/processed/pro_crisismmd_test_final/manifest.json',z.namelist())
                # archive_code needs a few fixed metadata files in addition to CODE.
                for name in ['configs/research/sources.json','requirements.txt','requirements/research.txt','requirements/vlm_reasoning.txt']:
                    target=root/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
                launcher.archive_code(root/'code.zip')
                with zipfile.ZipFile(root/'code.zip') as z:self.assertIn(PLAN,z.namelist())
            from kaggle_worker import verify_cohort
            spec={'manifest_path':str(output.relative_to(root)/'manifest.json'),'cohort_id':manifest['cohort_id'],
                  'samples_path':str(output.relative_to(root)/'pre_crisismmd.tsv'),'dataset':'crisismmd','task':'humanitarian',
                  'id_col':'image_id','label_col':'label_humanitarian','samples':1,'limit':1,
                  'final_protocol_sha256':plan['protocol_sha256'],'secondary_task':'informativeness','secondary_label_col':'label_informative'}
            verify_cohort(root,{'cohort':spec})
            manifest['samples'][0]['y_true']['informativeness']='informative';(output/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'task thứ hai'):verify_cohort(root,{'cohort':spec})

    def test_test_requires_explicit_loader_protocol_and_truth_only_changes_metrics(self):
        from experiments.clip_tda.run import load_features
        from experiments.clip_final import evaluate
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();plan,x,info,human=synthetic_plan(root);predictions=[]
            for shift in [0,1]:
                source=root/f'source_{shift}';source.mkdir();feature_artifact(source,plan,x,info,human,shift)
                with self.assertRaisesRegex(ValueError,'Split ngoài'):load_features(source)
                with patch.object(evaluate,'load_plan',return_value=plan):output=evaluate.run(source,root/PLAN,root/f'output_{shift}')
                with (output/'predictions_seed_0.tsv').open() as f:
                    rows=list(csv.DictReader(f,delimiter='\t'));predictions.append([[r[k] for k in RULES['methods']] for r in rows])
                analysis=json.loads((output/'analysis.json').read_text())
                self.assertEqual(set(analysis['mean']),set(RULES['methods']))
                self.assertEqual(analysis['mean']['majority']['binary'],analysis['mean']['majority_dn_unit']['binary'])
            self.assertEqual(predictions[0],predictions[1])
            plan['protocol']['exclusion']['image_ids']=['200_0']
            with patch.object(evaluate,'load_plan',return_value=plan),patch.object(evaluate,'predict_locked',side_effect=AssertionError('inference before exclusion')):
                with self.assertRaisesRegex(ValueError,'trùng'):evaluate.run(source,root/PLAN,root/'overlap')

    def test_main_runner_evaluates_final_after_unload_and_appends_summary(self):
        from experiments.clip_two_stage import run as runner
        import torch
        from PIL import Image
        class Backend:
            model=torch.nn.Linear(2,2)
            logit_scale=100.
            closed=False
            def encode_images(self,images):return torch.tensor([[1.,0.]]*len(images))
            def unload(self):self.closed=True
        backend=Backend()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();frozen,*_=synthetic_plan(root)
            Image.new('RGB',(4,4)).save(root/'image.png')
            settings=yaml.safe_load((ROOT/'experiments/clip_two_stage/config.yaml').read_text())
            settings.update(max_samples=2,export_features=True,evaluation_scope='heldout',output_dir=str(root/'run'))
            def plan(active,directory,api,sources):
                task=active['task'];labels=frozen['protocol']['info_labels' if task=='informativeness' else 'human_labels']
                return {'settings':active,'protocol_fingerprint':'synthetic','dataset_config':{'source_split':'test','final_dn_unit_plan':str(root/PLAN)},
                        'label_order':labels,'prompt_banks':{active['variant']:frozen['protocol']['prompt_banks'][task]}}
            def cohort(active,base):
                labels=frozen['protocol']['info_labels' if active['task']=='informativeness' else 'human_labels']
                return [{'id':f'{200+i}_0','y_true':labels[i%len(labels)],'image_path':str(root/'image.png'),
                         'image_sha256':'synthetic','source_split':'test','label_source':'image','event':'synthetic'} for i in range(2)]
            def prototypes(backend,bank,labels,batch):return np.array([[.4+i*.01,.1] for i in range(len(labels))],dtype='float32')
            def scores(backend,bank,labels,vectors,batch):return (vectors@prototypes(backend,bank,labels,batch).T)[:,None,:],None
            def after_unload(source,path,output):
                self.assertTrue(backend.closed)
                current=json.loads((source/'run_manifest.json').read_text())
                self.assertEqual(current['status'],'complete');self.assertEqual(current['final_dn_unit_analysis_status'],'running')
                self.assertTrue((source/'clip_features.npz').exists())
                output.mkdir();(output/'analysis.json').write_text(json.dumps({'mean':{
                    'majority':{'humanitarian':{'accuracy':.5,'macro_f1':.4}},
                    'majority_dn_unit':{'humanitarian':{'accuracy':.6,'macro_f1':.5}}}}))
                return output
            with patch.object(runner,'author_api'),patch.object(runner,'verify_sources',return_value={}), \
                 patch.object(runner,'choose_device',return_value='cpu'),patch.object(runner.lab,'project_path',side_effect=Path), \
                 patch.object(runner.lab,'freeze_plan',side_effect=plan),patch.object(runner.lab,'read_cohort',side_effect=cohort), \
                 patch.object(runner,'check_holdout',return_value={'status':'disjoint'}), \
                 patch.object(runner,'lock_holdout',side_effect=lambda settings,rows,root:{'rows':rows,'cohort_fingerprint':digest(rows)}), \
                 patch.object(runner.lab,'load_backend',return_value=backend),patch.object(runner.lab,'prompt_scores',side_effect=scores), \
                 patch('experiments.clip_stata.core.encode_prototypes',side_effect=prototypes), \
                 patch('experiments.clip_final.protocol.load_plan',return_value=frozen), \
                 patch('experiments.clip_final.evaluate.run',side_effect=after_unload):
                output=runner.run(settings)
            manifest=json.loads((output/'run_manifest.json').read_text())
            self.assertEqual(manifest['final_dn_unit_analysis_status'],'complete')
            with (output/'summary.tsv').open() as f:names=[r['method'] for r in csv.DictReader(f,delimiter='\t')]
            self.assertIn('majority_mean',names);self.assertIn('majority_dn_unit_mean',names)


if __name__=='__main__':unittest.main()
