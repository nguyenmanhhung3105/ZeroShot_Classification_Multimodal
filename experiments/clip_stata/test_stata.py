from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import yaml
from unittest.mock import patch
from experiments.clip_stata.core import stata, gate_ablation

ROOT=Path(__file__).resolve().parents[2]


class StataTests(unittest.TestCase):
    def test_replay_tolerates_small_float32_drift_but_rejects_invalid_or_changed_predictions(self):
        from experiments.clip_two_stage.analyze import verify_stata_posterior
        base=np.array([[.6,.4],[.3,.7]])
        report=verify_stata_posterior(base,base+np.array([[.0004,-.0004],[0,0]]))
        self.assertTrue(report['argmax_identical'])
        for bad in [base+.01,np.array([[.4,.6],[.3,.7]]),
                    np.array([[.61,.39],[.3,.7]]),np.array([[np.nan,np.nan],[.3,.7]])]:
            with self.assertRaises(ValueError):verify_stata_posterior(base,bad)

    def test_full_runner_writes_features_and_stata_results(self):
        import csv,json,torch
        from PIL import Image
        from experiments.clip_two_stage import run as runner
        class Backend:
            model=torch.nn.Linear(2,2)
            logit_scale=100.
            closed=False
            def encode_images(self,images):
                angle=np.asarray([image.getpixel((0,0))[0]/255 for image in images])*2-1
                return torch.tensor(np.stack([np.cos(angle),np.sin(angle)],1),dtype=torch.float32)
            def encode_texts(self,texts):
                values={'informative':[.8,.6],'not_informative':[.8,-.6],
                        'not_humanitarian':[-1,0],'a':[.6,.8],'b':[.6,-.8]}
                return torch.tensor([values[text] for text in texts],dtype=torch.float32)
            def unload(self):self.closed=True
        backend=Backend()
        settings=yaml.safe_load((ROOT/'experiments/clip_two_stage/config.yaml').read_text())
        settings.update(max_samples=5,stata_gate=True)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);settings['output_dir']=str(root/'output')
            for i in range(5):Image.new('RGB',(4,4),(i*50,0,0)).save(root/f'{i}.png')
            def plan(active,directory,api,sources):
                names=['informative','not_informative'] if active['task']=='informativeness' else ['not_humanitarian','a','b']
                value={'settings':active,'protocol':{'synthetic':True},'protocol_fingerprint':runner.lab.digest({'synthetic':True}),
                       'dataset_config':{},'label_order':names,'prompt_banks':{active['variant']:{name:[name] for name in names}}}
                (directory/'frozen_plan.json').write_text(json.dumps(value))
                return value
            def cohort(active,base):
                return [{'id':str(i),'image_sha256':'fake','image_path':str(root/f'{i}.png'),
                         'y_true':'a' if active['task']=='humanitarian' else 'informative','event':'synthetic'} for i in range(5)]
            with patch.object(runner,'author_api'),patch.object(runner,'verify_sources',return_value={}), \
                 patch.object(runner,'choose_device',return_value='cpu'), \
                 patch.object(runner.lab,'project_path',side_effect=Path), \
                 patch.object(runner.lab,'freeze_plan',side_effect=plan), \
                 patch.object(runner.lab,'read_cohort',side_effect=cohort), \
                 patch.object(runner.lab,'lock_cohort',side_effect=lambda settings,rows:{'rows':rows,'cohort_fingerprint':runner.lab.digest(rows)}), \
                 patch.object(runner.lab,'load_backend',return_value=backend):
                output=runner.run(settings)
                settings.update(stata_gate=False,export_features=True,tda_evaluate=True,output_dir=str(root/'features_only'))
                def evaluate_after_export(source,output_dir,compare_ensemble=False):
                    self.assertTrue(compare_ensemble)
                    self.assertTrue(backend.closed)
                    current=json.loads((source/'run_manifest.json').read_text())
                    self.assertEqual(current['status'],'complete')
                    self.assertEqual(current['tda_analysis_status'],'running')
                    self.assertTrue((source/'clip_features.npz').is_file())
                with patch('experiments.clip_tda.run.run',side_effect=evaluate_after_export) as tda:
                    feature_output=runner.run(settings)
                    tda.assert_called_once()
            manifest=json.loads((output/'run_manifest.json').read_text())
            self.assertEqual(manifest['status'],'complete');self.assertTrue(manifest['stata_gate'])
            metrics=json.loads((output/'metrics.json').read_text())
            self.assertEqual(len(metrics['stata']),4);self.assertIn('stata_mean',metrics)
            with np.load(output/'stata_features.npz',allow_pickle=False) as pack:self.assertEqual(pack['ids'].tolist(),['0','1','2','3','4'])
            with (output/'stata_predictions.tsv').open() as f:self.assertEqual(len(list(csv.DictReader(f,delimiter='\t'))),5)
            self.assertTrue(backend.closed)
            exported=json.loads((feature_output/'run_manifest.json').read_text())
            self.assertTrue(exported['export_features']);self.assertFalse(exported['stata_gate'])
            self.assertEqual(exported['feature_file'],'clip_features.npz')
            self.assertEqual(exported['tda_analysis_status'],'complete')
            exported_metrics=json.loads((feature_output/'metrics.json').read_text())
            self.assertIn('informativeness',exported_metrics)
            self.assertEqual(exported_metrics['stata'],{})
            from experiments.clip_two_stage.analyze import analyze
            report=analyze(output,bootstraps=5)
            self.assertIn('stata_gate_seed_0',report['ablations'])
            path=output/'stata_predictions.tsv'
            with path.open() as f:stored=list(csv.DictReader(f,delimiter='\t'))
            stored[0]['stata_gate_seed_0']='b' if stored[0]['stata_gate_seed_0']=='a' else 'a'
            with path.open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(stored[0]),delimiter='\t');writer.writeheader();writer.writerows(stored)
            with self.assertRaisesRegex(ValueError,'StatA dự đoán khác'):analyze(output,bootstraps=5)

    def test_valid_posteriors_and_unit_validation(self):
        rng=np.random.default_rng(12)
        x=rng.normal(size=(20,12)).astype('float32');x/=np.linalg.norm(x,axis=1,keepdims=True)
        t=rng.normal(size=(2,12)).astype('float32');t/=np.linalg.norm(t,axis=1,keepdims=True)
        prior,z=stata(x,t,max_iter=2)
        np.testing.assert_allclose(prior.sum(1),1,atol=1e-6)
        np.testing.assert_allclose(z.sum(1),1,atol=1e-6)
        self.assertTrue(np.isfinite(z).all())
        with self.assertRaises(ValueError):stata(x*2,t)
        with self.assertRaises(ValueError):stata(x[:3],t)

    def test_ablation_keeps_human_argmax_and_verifies_score_prototypes(self):
        rng=np.random.default_rng(4)
        x=rng.normal(size=(10,8)).astype('float32');x/=np.linalg.norm(x,axis=1,keepdims=True)
        t=rng.normal(size=(2,8)).astype('float32');t/=np.linalg.norm(t,axis=1,keepdims=True);t*=.9
        human=rng.normal(size=(10,3));labels=['not_humanitarian','a','b']
        choices,log=gate_ablation(x,t,x@t.T,human,['informative','not_informative'],labels)
        self.assertIn('normalized_gate_control',choices)
        for name,gates in log['gate_labels'].items():
            expected=np.asarray(labels)[1+human[:,1:].argmax(1)]
            expected[np.asarray(gates)!='informative']='not_humanitarian'
            self.assertEqual(choices[name],expected.tolist())
        with self.assertRaises(ValueError):gate_ablation(x,t,np.zeros((10,2)),human,['informative','not_informative'],labels)

    def test_launcher_has_separate_path_and_forwards_flag(self):
        sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
        import launch_runs as launcher
        config=yaml.safe_load((ROOT/'experiments/clip_two_stage/launcher_dev_stata.yaml').read_text())
        launcher.validate(config,['two_stage_stata_dev'])
        with tempfile.TemporaryDirectory() as tmp:
            _,_,output,_,_,_,research=launcher.job_plan(config,'two_stage_stata_dev',Path(tmp))
            self.assertEqual(output.parts[1],'clip_two_stage_stata_heldout')
            self.assertTrue(yaml.safe_load(research.read_text())['stata_gate'])
        config['jobs']['two_stage_stata_dev']['multimodal']=True
        with self.assertRaises(ValueError):launcher.validate(config,['two_stage_stata_dev'])


if __name__=='__main__':unittest.main()
