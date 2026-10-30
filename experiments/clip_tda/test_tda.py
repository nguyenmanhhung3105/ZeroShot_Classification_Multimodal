import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import yaml
from experiments.clip_tda.core import predict_stream
from experiments.clip_tda.run import load_features,run
from experiments.clip_transductive.test_transductive import fixture,INFO
from experiments.clip_transductive.run import sha256
from experiments.waffle_lab.run import digest

ROOT=Path(__file__).resolve().parents[2]


def binary_features():
    angle=np.linspace(-1,1,40)
    x=np.stack([np.cos(angle),np.sin(angle)],1).astype('float32')
    t=np.array([[np.cos(.02),np.sin(.02)],[np.cos(.02),-np.sin(.02)]],dtype='float32')
    return x,t


def feature_artifact(folder,truth_shift=0,missing=False):
    a,_,_=fixture(folder)
    manifest=json.loads((folder/'run_manifest.json').read_text())
    for index,task in enumerate(['informativeness','humanitarian']):
        plan_file=folder/task/'frozen_plan.json';plan=json.loads(plan_file.read_text())
        plan['dataset_config']['source_split']='train';plan_file.write_text(json.dumps(plan))
        cohort_file=folder/task/'cohort.json';cohort=json.loads(cohort_file.read_text())
        for i,row in enumerate(cohort['rows']):
            row['source_split']='train'
            if index==0:row['y_true']=None if missing else INFO[(i+truth_shift)%2]
        cohort_file.write_text(json.dumps(cohort));manifest['cohort_fingerprints'][index]=digest(cohort['rows'])
    x=np.column_stack([a,np.sqrt(1-(a*a).sum(1))]).astype('float32')
    prototypes=np.array([[1,0,0],[0,1,0]],dtype='float32')
    ids=np.array([f'{i}_0' for i in range(len(a))])
    np.savez_compressed(folder/'clip_features.npz',image_features=x,info_mean_prototypes=prototypes,ids=ids)
    manifest.update(evaluation_scope='exploratory',export_features=True,feature_file='clip_features.npz',
                    feature_sha256=sha256((folder/'clip_features.npz').read_bytes()))
    (folder/'run_manifest.json').write_text(json.dumps(manifest))


class TdaTests(unittest.TestCase):
    def test_binary_ground_truth_affects_metrics_only_and_missing_is_not_inferred(self):
        import csv
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);predictions=[]
            for shift in [0,1]:
                source=root/f'source_{shift}';source.mkdir();feature_artifact(source,truth_shift=shift)
                loaded=load_features(source);self.assertIsNotNone(loaded['binary_truth'])
                output=run(source,root/f'output_{shift}',compare_ensemble=True)
                report=json.loads((output/'analysis.json').read_text())
                self.assertIsNotNone(report['shuffle_mean']['binary'])
                self.assertIsNotNone(report['reference_ensemble_mean']['binary'])
                with (output/'source_order/predictions.tsv').open() as f:
                    predictions.append([(r['id'],r['binary_pred'],r['y_pred']) for r in csv.DictReader(f,delimiter='\t')])
            self.assertEqual(predictions[0],predictions[1])
            with self.assertRaises(FileExistsError):run(source,output)
            missing=root/'missing';missing.mkdir();feature_artifact(missing,missing=True)
            self.assertIsNone(load_features(missing)['binary_truth'])

    def test_binary_negative_cache_is_class_constant_with_official_defaults(self):
        x,t=binary_features()
        logits,diag=predict_stream(x,t)
        positive,_=predict_stream(x,t,negative=False)
        np.testing.assert_array_equal(logits.argmax(1),positive.argmax(1))
        self.assertEqual(diag['negative_logit_spread_max'],0)
        self.assertTrue(diag['negative_cache_indices'])
        self.assertTrue(all(len(v)<=3 for v in diag['positive_cache_indices'].values()))
        self.assertTrue(all(len(v)<=2 for v in diag['negative_cache_indices'].values()))

    def test_causal_prefix_and_cache_reset_and_complete_order(self):
        x,t=binary_features()
        first,_=predict_stream(x,t)
        prefix,_=predict_stream(x[:15],t)
        np.testing.assert_array_equal(first[:15],prefix)
        repeated,_=predict_stream(x,t)
        np.testing.assert_array_equal(first,repeated)
        order=np.random.default_rng(0).permutation(len(x))
        ordered,diag=predict_stream(x,t,order)
        self.assertEqual(ordered.shape,(len(x),2));self.assertEqual(diag['order'],order.tolist())
        with self.assertRaises(ValueError):predict_stream(x,t,np.zeros(len(x),dtype=int))
        with self.assertRaises(ValueError):predict_stream(x*2,t)

    def test_feature_launcher_has_separate_output_and_does_not_enable_stata(self):
        sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
        import launch_runs as launcher
        config=yaml.safe_load((ROOT/'experiments/clip_two_stage/launcher_binary_features.yaml').read_text())
        launcher.validate(config,['binary_features_train'])
        with tempfile.TemporaryDirectory() as temp:
            _,_,output,_,_,_,research=launcher.job_plan(config,'binary_features_train',Path(temp))
            self.assertEqual(output.parts[1],'clip_two_stage_features')
            settings=yaml.safe_load(research.read_text())
            self.assertTrue(settings['export_features']);self.assertFalse(settings['stata_gate'])
            self.assertTrue(settings['tda_evaluate'])
            self.assertEqual(settings['max_samples'],3000)


if __name__=='__main__':unittest.main()
