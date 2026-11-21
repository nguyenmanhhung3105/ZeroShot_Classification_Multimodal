"""Few-shot binary min5: dữ liệu tổng hợp, không dataset/weights/API thật."""
import contextlib
import copy
import csv
import importlib.util
import io as stdio
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.special import expit
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from vlm_project import fakeddit_fewshot_task1 as fs
from vlm_project import io, kaggle, protocol


class Backend:
    def __init__(self, device):
        self.metadata = {'synthetic': True}
        self.unloaded = False

    def unload(self):
        self.unloaded = True


class FewshotTask1Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.settings = io.config()
        self.settings.update(device='cpu', output_root=str(self.root/'runs'))
        self.spec = fs.specification()

    def fixture(self):
        roots = {s:self.root/s for s in ['train','dev']}
        spec = {**self.spec, 'shots_per_6way_class':4, 'train_samples':120, 'dev_samples':24,
                'manifest_sha256':{}}
        rows = {}
        for split, size in [('train',120),('dev',24)]:
            roots[split].mkdir(exist_ok=True)
            (roots[split]/'manifest.json').write_text(split)
            spec['manifest_sha256'][split] = io.sha256(roots[split]/'manifest.json')
            rows[split] = [{'id':f'{split}_{i}', 'image':'unused', 'text':'synthetic',
                            '6way':str(i%6), '2way':str(int(i%6==0)),
                            'image_sha256':f'{split}_image_{i}', 'text_sha256':f'{split}_text_{i}'} for i in range(size)]
        return spec, roots, rows

    @staticmethod
    def encode(backend, rows, batch_size, with_titles=False):
        # Theo ID, độc lập nhãn dev được thay trong test leakage.
        x = np.eye(8)[[int(r['id'].split('_')[-1])%6 for r in rows]]
        return x, np.roll(x, 1, axis=1)

    def test_support_matches_original_sampler_and_binary_budget(self):
        module_spec = importlib.util.spec_from_file_location('old_fewshot', ROOT/'tests/fixtures/legacy_fewshot.py')
        old = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(old)
        y = np.tile(np.arange(6), 80)
        for seed in range(5):
            indices = fs.select_support(y,64,seed)
            np.testing.assert_array_equal(indices,old.select_support(y,64,seed))
            np.testing.assert_array_equal(np.bincount(y[indices]),[64]*6)
            np.testing.assert_array_equal(np.bincount((y[indices]!=0).astype(int)),[64,320])
        with self.assertRaises(ValueError):fs.select_support(np.tile(np.arange(6),63),64,0)

    def test_full_runner_shared_support_replay_and_dev_label_independence(self):
        spec, roots, rows = self.fixture()
        backend = Backend('cpu')
        def loader(root,split,smoke=False):
            self.assertNotEqual(split,'test')
            return rows[split][:5] if smoke else rows[split], {'split':split}
        with patch.object(fs,'specification',return_value=spec), \
             patch.object(fs.data,'fakeddit',side_effect=loader) as load, \
             patch.object(fs,'encode_cohort',side_effect=self.encode) as encoding, \
             contextlib.redirect_stdout(stdio.StringIO()):
            output=fs.run(self.settings,roots['train'],roots['dev'],backend_factory=lambda _:backend)
            for r in rows['dev']:r['6way']='1' if r['6way']=='0' else '0'
            other=fs.run(self.settings,roots['train'],roots['dev'],backend_factory=Backend)
        self.assertTrue(backend.unloaded)
        self.assertEqual([c.args[1] for c in load.call_args_list],['train','dev','train','dev'])
        self.assertEqual(encoding.call_count,4)  # Hai lần/run; không encode riêng cho từng method.
        run=protocol.completed_run(output)
        self.assertIsInstance(io.read_json(output/'cohort.json'),list)
        self.assertIsInstance(io.read_json(output/'train_cohort.json'),list)
        self.assertEqual(run['stages'],1)
        self.assertFalse(run['test_opened'])
        self.assertEqual(run['samples'],24)
        self.assertEqual({r['method']:r['seeds'] for r in run['summary']},{'image':5,'title':5,'image_title':5})
        with np.load(output/'replay_scores.npz',allow_pickle=False) as a, \
             np.load(other/'replay_scores.npz',allow_pickle=False) as b, \
             np.load(output/'features.npz',allow_pickle=False) as f, \
             np.load(output/'linear_models.npz',allow_pickle=False) as models:
            self.assertEqual(len([k for k in a.files if k.startswith('p_')]),15)
            for key in a.files:
                if key.startswith('p_'):np.testing.assert_array_equal(a[key],b[key])
            dev=fs.feature_banks(f['dev_image'],f['dev_title'])
            for summary in run['summary']:
                method=summary['method']
                accuracies=[]
                for seed in spec['seeds']:
                    scores=a[f'p_{method}_s{seed}']
                    accuracies.append((scores.argmax(1)==a['labels']).mean())
                    expected=expit(dev[method]@models[f'{method}_s{seed}_coef'].T+models[f'{method}_s{seed}_intercept'])[:,0]
                    np.testing.assert_allclose(scores[:,1],expected)
                self.assertAlmostEqual(summary['accuracy_mean'],np.mean(accuracies))
                self.assertAlmostEqual(summary['accuracy_std'],np.std(accuracies,ddof=1))
            module_spec=importlib.util.spec_from_file_location('old_probe',ROOT/'tests/fixtures/legacy_fewshot.py')
            old=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(old)
            ix=f['support_s0'];train=fs.feature_banks(f['train_image'],f['train_title'])
            baseline=old.probe(train['image_title'][ix],(f['train_y6'][ix]!=0).astype(int))
            np.testing.assert_allclose(a['p_image_title_s0'],baseline.predict_proba(dev['image_title']))
        with (output/'comparisons.tsv').open() as stream:
            comparison=list(csv.DictReader(stream,delimiter='\t'))
        self.assertEqual(len(comparison),15)
        self.assertTrue(all('fake_as_real' in r and 'recall_real' in r for r in comparison))
        with self.assertRaises(ValueError):protocol.freeze(self.settings,output)

    def test_wrong_cohort_and_leakage_fail_before_backbone(self):
        spec, roots, rows=self.fixture()
        with patch.object(fs,'specification',return_value=spec), \
             patch.object(fs.data,'fakeddit',side_effect=AssertionError('no data on wrong hash')):
            with self.assertRaisesRegex(ValueError,'Sai cohort train'):
                fs.run(self.settings,roots['dev'],roots['train'])
        for key in ['id','image_sha256','text_sha256']:
            modified=copy.deepcopy(rows)
            modified['dev'][0][key]=modified['train'][0][key]
            with patch.object(fs,'specification',return_value=spec), \
                 patch.object(fs.data,'fakeddit',side_effect=lambda p,s,*args:(modified[s],{})), \
                 patch.object(fs,'Backbone',side_effect=AssertionError('no model')):
                with self.assertRaisesRegex(ValueError,'overlap'):
                    fs.run(self.settings,roots['train'],roots['dev'],backend_factory=lambda _:self.fail('loaded model'))

    def test_smoke_preserves_training_support_and_unloads_on_error(self):
        spec,roots,rows=self.fixture()
        backend=Backend('cpu')
        def loader(root,split,smoke=False):return (rows[split][:5] if smoke else rows[split]),{}
        with patch.object(fs,'specification',return_value=spec), \
             patch.object(fs.data,'fakeddit',side_effect=loader), \
             patch.object(fs,'encode_cohort',side_effect=self.encode), \
             contextlib.redirect_stdout(stdio.StringIO()):
            output=fs.run(self.settings,roots['train'],roots['dev'],True,backend_factory=Backend)
        run=protocol.completed_run(output)
        self.assertEqual((run['mode'],run['samples']),('smoke',5))
        support=io.read_json(output/'support.json')
        self.assertEqual([len(s['ids']) for s in support],[24]*5)
        with patch.object(fs,'specification',return_value=spec), \
             patch.object(fs.data,'fakeddit',side_effect=loader), \
             patch.object(fs,'encode_cohort',side_effect=RuntimeError('synthetic failure')):
            with self.assertRaisesRegex(RuntimeError,'synthetic failure'):
                fs.run(self.settings,roots['train'],roots['dev'],True,backend_factory=lambda _:backend)
        self.assertTrue(backend.unloaded)

    def test_launcher_two_min5_datasets_one_pass_no_test_or_legacy_train(self):
        launcher=yaml.safe_load((ROOT/'configs/kaggle_launcher_fewshot_min5.yaml').read_text())
        real_path=kaggle.path
        with patch.object(kaggle,'path',side_effect=lambda p:self.root/'stage' if str(p)=='results/_system/launches/final' else real_path(p)), \
             patch.object(kaggle,'cli',side_effect=AssertionError('no API')):
            for limit in [None,5]:
                plan=kaggle.prepare(launcher,'fakeddit_fewshot_task1_min5',limit)
                doc=io.read_json(Path(plan['root'])/'kernel/run.ipynb')
                code='\n'.join(''.join(c['source']) for c in doc['cells'] if c['cell_type']=='code')
                self.assertEqual(code.count('output = run('),1)
                self.assertIn('fakeddit_fewshot_task1 import run',code)
                self.assertIn('fakeddit_slugs"].update(specification()',code)
                self.assertNotIn('freeze(settings',code)
                self.assertEqual(plan['samples'],5 if limit else 3000)
                self.assertEqual(plan['settings']['kaggle']['fakeddit_slugs']['train'],'vlm-fakeddit-train-min5-v1')
                meta=io.read_json(Path(plan['root'])/'kernel/kernel-metadata.json')
                self.assertEqual(set(meta['dataset_sources']),{plan['code_dataset'],'nguyenhung3105/vlm-fakeddit-train-min5-v1','nguyenhung3105/vlm-fakeddit-dev-min5-v1'})
                self.assertEqual(set(plan['data_datasets']),{'nguyenhung3105/vlm-fakeddit-train-min5-v1','nguyenhung3105/vlm-fakeddit-dev-min5-v1'})
                for c in doc['cells']:
                    if c['cell_type']=='code':compile(''.join(c['source']),'<fewshot task1>','exec')
            for split in ['train','test']:
                invalid=copy.deepcopy(launcher)
                invalid['jobs']['fakeddit_fewshot_task1_min5']['split']=split
                with self.assertRaisesRegex(ValueError,'chỉ đánh giá dev'):
                    kaggle.prepare(invalid,'fakeddit_fewshot_task1_min5')

    def test_submit_checks_train_dataset_before_any_upload_or_push(self):
        calls=[]
        plan={'kernel':'owner/fewshot','data_dataset':'owner/dev','data_datasets':['owner/dev','owner/train']}
        def cli(args,check=True):
            calls.append(args)
            if args==['datasets','status','owner/train']:
                raise RuntimeError('train min5 missing')
            return subprocess.CompletedProcess(args,0,'complete','')
        with patch.object(kaggle,'cli',side_effect=cli),contextlib.redirect_stdout(stdio.StringIO()):
            with self.assertRaisesRegex(RuntimeError,'train min5 missing'):
                kaggle.submit(plan)
        self.assertEqual(calls,[['kernels','status','owner/fewshot'],['quota'],
                                ['datasets','status','owner/dev'],['datasets','status','owner/train']])

    def test_status_403_fallback_requires_all_pages_metadata_and_images(self):
        import json
        plan={'kernel':'owner/fewshot','fewshot_task1_specification':{
            'dataset_slugs':{'train':'train','dev':'dev'},'train_samples':3,'dev_samples':3}}
        calls=[]
        pages=[{'name':'fakeddit_sample.tsv','size':20},{'name':'images/0.jpg','size':10}]
        remaining=[{'name':'images/1.jpg','size':10},{'name':'images/2.jpg','size':10},
                   {'name':'manifest.json','size':30}]
        def cli(args,check=True):
            calls.append(args)
            if args[:2]==['datasets','status']:
                return subprocess.CompletedProcess(args,1,'','403 Forbidden')
            body=json.dumps(remaining) if '--page-token' in args else 'Next Page Token = second\n'+json.dumps(pages)
            return subprocess.CompletedProcess(args,0,body,'')
        with patch.object(kaggle,'cli',side_effect=cli),contextlib.redirect_stdout(stdio.StringIO()):
            kaggle.check_input_dataset(plan,'owner/train')
        self.assertEqual(len(calls),3)
        self.assertEqual(calls[-1][-2:],['--page-token','second'])
        for broken in [[],remaining[:-1]]:
            def incomplete(args,check=True):
                if args[:2]==['datasets','status']:
                    return subprocess.CompletedProcess(args,1,'','403 Forbidden')
                return subprocess.CompletedProcess(args,0,json.dumps(broken),'')
            with patch.object(kaggle,'cli',side_effect=incomplete):
                with self.assertRaisesRegex(RuntimeError,'chưa thấy đủ'):
                    kaggle.check_input_dataset(plan,'owner/train')

    def test_status_403_does_not_ignore_file_api_denial_or_real_missing(self):
        plan={'kernel':'owner/fewshot','fewshot_task1_specification':{
            'dataset_slugs':{'train':'train','dev':'dev'},'train_samples':3,'dev_samples':3}}
        for failure in ['403 Forbidden','401 Unauthorized','404 Not Found']:
            def cli(args,check=True):
                message='403 Forbidden' if args[:2]==['datasets','status'] else failure
                return subprocess.CompletedProcess(args,1,'',message)
            with patch.object(kaggle,'cli',side_effect=cli):
                with self.assertRaisesRegex(RuntimeError,'Không đọc được'):
                    kaggle.check_input_dataset(plan,'owner/train')
        with patch.object(kaggle,'cli',return_value=subprocess.CompletedProcess([],1,'','404 Not Found')) as call:
            with self.assertRaisesRegex(RuntimeError,'404'):
                kaggle.check_input_dataset(plan,'owner/train')
            self.assertEqual(call.call_count,1)


if __name__=='__main__':unittest.main()
