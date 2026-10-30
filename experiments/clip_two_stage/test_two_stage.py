from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
from experiments.clip_two_stage.run import decide


class TwoStageTests(unittest.TestCase):
    def test_dev_preparation_uses_human_annotation_only_and_excludes_development(self):
        import csv,json
        from PIL import Image
        from experiments.clip_two_stage.prepare_dev import prepare
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            folder=root/'data/raw/CrisisMMD_v2.0/crisismmd_datasplit_all';folder.mkdir(parents=True)
            images=folder.parent/'data_image/event';images.mkdir(parents=True)
            fields=['tweet_id','image_id','image','event_name','tweet_text','label_image']
            with (folder/'task_humanitarian_text_img_dev.tsv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=fields,delimiter='\t');writer.writeheader()
                for key,label,color in [('100','not_humanitarian','red'),('200','infrastructure_and_utility_damage','blue')]:
                    Image.new('RGB',(8,8),color).save(images/f'{key}_0.png')
                    writer.writerow({'tweet_id':key,'image_id':key+'_0','image':f'data_image/event/{key}_0.png',
                                     'event_name':'event','tweet_text':'content','label_image':label})
            (root/'configs/prompts').mkdir(parents=True)
            (root/'configs/prompts/crisismmd_development_exclusion.json').write_text(json.dumps({'image_ids':['100_0'],'tweet_ids':['100'],'image_sha256':[]}))
            (root/'configs/experiment_config.yaml').write_text((ROOT/'configs/experiment_config.yaml').read_text())
            (root/'experiments/clip_two_stage').mkdir(parents=True)
            (root/'experiments/clip_two_stage/launcher.yaml').write_text((ROOT/'experiments/clip_two_stage/launcher.yaml').read_text())
            out=prepare(root)
            manifest=json.loads((out/'manifest.json').read_text())
            self.assertEqual(manifest['selected']['samples'],1)
            self.assertFalse(manifest['test_accessed'])
            self.assertFalse(manifest['gate_labels_available'])
            self.assertEqual(manifest['samples'][0]['y_true']['informativeness'],None)
            with self.assertRaises(FileExistsError):prepare(root)

    def test_margin_pool_can_recover_or_override_gate_without_truth(self):
        from experiments.clip_two_stage.run import decide_margin_pool
        # Tầng 1 hơi nghiêng negative, tầng 2 có bằng chứng positive rõ.
        info=np.array([[.1,.2],[.2,.1]])
        human=np.array([[.1,.8,.2],[.9,.2,.1]])
        labels=['not_humanitarian','a','b']
        hard,_=decide(info,human,['informative','not_informative'],labels)
        soft,gate,margin=decide_margin_pool(info,human,['informative','not_informative'],labels)
        self.assertEqual(hard,['not_humanitarian','a'])
        self.assertEqual(soft,['a','not_humanitarian'])
        np.testing.assert_allclose(margin,[.3,-.3])
        # Thêm offset chung cho mọi lớp không đổi margin/quyết định.
        shifted=decide_margin_pool(info+5,human-3,['informative','not_informative'],labels)
        self.assertEqual(shifted[0],soft)

    def test_analysis_recomputes_metrics_and_rejects_tampered_predictions(self):
        import json
        from experiments.clip_two_stage.analyze import analyze
        from experiments.waffle_lab.run import digest
        records=[{'id':'100_0','image_sha256':'a','y_true':'a','event':'event'},
                 {'id':'200_0','image_sha256':'b','y_true':'not_humanitarian','event':'event'}]
        protocol={'synthetic':True}
        cohort={'rows':records}
        plan={'protocol':protocol,'label_order':['not_humanitarian','a','b']}
        manifest={'status':'complete','model':'synthetic','seed':0,'samples':2,
                  'protocol_fingerprints':[digest(protocol)]*2,'cohort_fingerprints':[digest(records)]*2}
        artifacts={'run_manifest.json':json.dumps(manifest),
                   'humanitarian/frozen_plan.json':json.dumps(plan),
                   'informativeness/frozen_plan.json':json.dumps({**plan,'label_order':['informative','not_informative']}),
                   'humanitarian/cohort.json':json.dumps(cohort),'informativeness/cohort.json':json.dumps(cohort),
                   'predictions.tsv':'id\ty_true\tsingle_stage_pred\ty_pred\tgate_informative\tinfo_cosine_scores\thuman_cosine_scores\n'
                   '100_0\ta\tnot_humanitarian\ta\tTrue\t[0.9,0.1]\t[0.8,0.7,0.2]\n'
                   '200_0\tnot_humanitarian\ta\tnot_humanitarian\tFalse\t[0.1,0.9]\t[0.2,0.8,0.3]\n'}
        with patch('experiments.clip_two_stage.analyze.read_artifact',side_effect=lambda directory,name:artifacts[name]):
            report=analyze('unused',bootstraps=10)
            self.assertEqual(report['fixed'],2)
            self.assertEqual(report['two_stage']['accuracy'],1.)
            artifacts['predictions.tsv']=artifacts['predictions.tsv'].replace('100_0\ta\tnot_humanitarian\ta','100_0\ta\tnot_humanitarian\tb')
            with self.assertRaisesRegex(ValueError,'khác score'):analyze('unused',bootstraps=10)

    def test_holdout_rejects_train_and_development_leakage(self):
        from experiments.clip_two_stage.validation import check_holdout
        reference={'image_ids':['100_0'],'tweet_ids':['100'],'image_sha256':['old']}
        row={'id':'200_0','image_sha256':'new','source_split':'test','label_source':'image'}
        self.assertEqual(check_holdout([row],reference,'test')['status'],'disjoint_from_recorded_development')
        for changes in [{'id':'100_0'},{'id':'100_1'},{'image_sha256':'old'},{'source_split':'train'},{'label_source':'text'}]:
            with self.assertRaises(ValueError):check_holdout([{**row,**changes}],reference,'test')
        with self.assertRaises(ValueError):check_holdout([row],reference,'train')

    def test_heldout_launcher_rejects_train_before_upload(self):
        import launch_runs as launcher
        config=yaml.safe_load((ROOT/'experiments/clip_two_stage/launcher.yaml').read_text())
        config['jobs']['two_stage']['evaluation_scope']='heldout'
        with self.assertRaisesRegex(ValueError,'cohort train'):
            launcher.validate(config,['two_stage'])

    def test_openclip_dtype_without_model_property(self):
        import torch
        from experiments.clip_two_stage.run import model_dtype
        model=torch.nn.Linear(2,2)
        self.assertFalse(hasattr(model,'dtype'))
        self.assertEqual(model_dtype(model),'torch.float32')
        self.assertEqual(model_dtype(model.half()),'torch.float16')
        self.assertEqual(model_dtype(torch.nn.Identity()),'unknown')

    def test_full_run_with_openclip_style_model_saves_manifest_and_summary(self):
        import torch
        from PIL import Image
        from experiments.clip_two_stage import run as runner
        import json
        class Backend:
            model=torch.nn.Linear(2,2)
            logit_scale=100.
            closed=False
            def encode_images(self, images):return torch.tensor([[1.,0.]]*len(images))
            def unload(self):self.closed=True
        backend=Backend()
        settings=yaml.safe_load((ROOT/'experiments/clip_two_stage/config.yaml').read_text())
        settings['max_samples']=2
        def plan(active,directory,api,sources):
            labels=['informative','not_informative'] if active['task']=='informativeness' else ['not_humanitarian','a','b']
            return {'settings':active,'protocol_fingerprint':'synthetic','dataset_config':{},
                    'label_order':labels,'prompt_banks':{active['variant']:{l:['test'] for l in labels}}}
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp);Image.new('RGB',(8,8)).save(path/'image.png')
            settings['output_dir']=str(path/'output')
            def rows(active,base):
                truth=['informative','not_informative'] if active['task']=='informativeness' else ['a','not_humanitarian']
                return [{'id':str(i),'image_sha256':'fake','image_path':str(path/'image.png'),'y_true':y} for i,y in enumerate(truth)]
            def scores(backend,bank,labels,vectors,batch):
                return (np.array([[[.9,.1]],[[.1,.9]]]) if len(labels)==2 else
                        np.array([[[.8,.7,.2]],[[.2,.8,.3]]])),np.ones(len(labels))
            with patch.object(runner,'author_api'),patch.object(runner,'verify_sources',return_value={}), \
                 patch.object(runner,'choose_device',return_value='cpu'), \
                 patch.object(runner.lab,'project_path',side_effect=Path), \
                 patch.object(runner.lab,'freeze_plan',side_effect=plan), \
                 patch.object(runner.lab,'read_cohort',side_effect=rows), \
                 patch.object(runner.lab,'lock_cohort',side_effect=lambda settings,rows:{'rows':rows,'cohort_fingerprint':'fake'}), \
                 patch.object(runner.lab,'load_backend',return_value=backend), \
                 patch.object(runner.lab,'prompt_scores',side_effect=scores):
                output=runner.run(settings)
            manifest=json.loads((output/'run_manifest.json').read_text())
            self.assertEqual(manifest['status'],'complete')
            self.assertEqual(manifest['compute_dtype'],'torch.float32')
            self.assertTrue((output/'summary.tsv').is_file())
            self.assertTrue(backend.closed)

    def test_gate_and_seven_class_decision_override_single_stage(self):
        info=[[.9,.1],[.2,.8],[.8,.2]]
        human=[[.9,.3,.2],[.1,.8,.2],[.9,.2,.4]]
        pred,gate=decide(info,human,['informative','not_informative'],['not_humanitarian','a','b'])
        self.assertEqual(pred,['a','not_humanitarian','b'])
        self.assertEqual(gate.tolist(),[True,False,True])

    def test_label_reordering_and_bad_scores(self):
        pred,_=decide([[.1,.9]],[[.4,.9,.2]],['not_informative','informative'],['b','not_humanitarian','a'])
        self.assertEqual(pred,['b'])
        with self.assertRaises(ValueError):
            decide([[np.nan,.1]],[[.1,.2]],['informative','not_informative'],['not_humanitarian','a'])

    def test_launcher_freezes_one_model_seed_and_correct_command(self):
        import launch_runs as launcher
        config=yaml.safe_load((ROOT/'experiments/clip_two_stage/launcher.yaml').read_text())
        launcher.validate(config,['two_stage']);launcher.validate_single_cloud_trial(config,['two_stage'])
        with tempfile.TemporaryDirectory() as temp:
            _,_,output,command,_,_,research=launcher.job_plan(config,'two_stage',Path(temp))
            self.assertEqual(output.parts[1],'clip_two_stage')
            self.assertEqual(command[0],'experiments/clip_two_stage/run.py')
            settings=yaml.safe_load(research.read_text())
            self.assertEqual(settings['seed'],0)
            self.assertEqual(settings['backbone'],'metaclip2_vith14')

    def test_prepare_does_not_read_cohort_or_load_model(self):
        from experiments.clip_two_stage import run as runner
        import json
        settings=yaml.safe_load((ROOT/'experiments/clip_two_stage/config.yaml').read_text())
        def frozen(active,directory,api,sources):
            plan={'protocol_fingerprint':'synthetic','dataset_config':{}}
            (directory/'frozen_plan.json').write_text(json.dumps(plan))
            return plan
        with tempfile.TemporaryDirectory() as temp:
            # project_path chủ động chỉ cho output trong dự án; dùng root results
            # cho prepare thật đã kiểm tra bằng CLI, test này chỉ kiểm tra thứ tự.
            settings['output_dir']=temp
            with patch.object(runner.lab,'project_path',side_effect=lambda p: Path(p) if Path(p).is_absolute() else ROOT/p), \
                 patch.object(runner,'author_api'),patch.object(runner,'verify_sources',return_value={}), \
                 patch.object(runner.lab,'freeze_plan',side_effect=frozen), \
                 patch.object(runner.lab,'read_cohort',side_effect=AssertionError('dataset accessed')), \
                 patch.object(runner.lab,'load_backend',side_effect=AssertionError('model loaded')):
                output=runner.run(settings,True)
            self.assertTrue((output/'humanitarian/frozen_plan.json').is_file())


if __name__=='__main__':unittest.main()
