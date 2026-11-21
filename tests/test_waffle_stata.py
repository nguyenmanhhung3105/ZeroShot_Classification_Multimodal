"""Waffle→StatA: chỉ vector tổng hợp, không tải dataset/model/API."""
import contextlib
import hashlib
import io as stdio
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.special import softmax
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from vlm_project import waffle_stata as ws
from vlm_project import io, kaggle, protocol, evaluate, dclip_stata, waffle_stata_full, waffle_semantic_stata


class Backend:
    def __init__(self, device):
        self.metadata = {'synthetic': True}
        self.unloaded = False

    def prototypes(self, bank):
        seed = int(hashlib.sha256(json.dumps(bank).encode()).hexdigest()[:8], 16)
        # Norm khác nhau để kiểm tra đối chứng chuẩn hóa.
        return ws.unit(np.random.default_rng(seed).normal(size=(len(bank), 8))) * np.array([[.8], [.5]])

    def unload(self):
        self.unloaded = True


class WaffleStatATests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        # Isolate runner metadata hashes from an installed author checkout.
        (self.root / 'configs').mkdir()
        for file in [ws.SPEC_FILE, dclip_stata.SPEC_FILE, waffle_stata_full.SPEC_FILE,
                     *waffle_semantic_stata.SPEC_FILES.values()]:
            (self.root / file).write_bytes((ROOT / file).read_bytes())
        for name in ['dtd', 'eurosat']:
            io.write_json(self.root / f'third_party/WaffleCLIP/descriptors/descriptors_{name}.json', {})
        patcher = patch.object(ws, 'ROOT', self.root)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_runner_real_adaptation_label_independence_normalization_and_replay(self):
        settings = io.config()
        settings.update(device='cpu', output_root=str(self.root / 'runs'))
        spec = ws.specification()
        (self.root / 'configs').mkdir(exist_ok=True)
        (self.root / ws.SPEC_FILE).write_text(yaml.safe_dump(spec))
        descriptor = self.root / 'third_party/WaffleCLIP/descriptors/descriptors_dtd.json'
        descriptor.parent.mkdir(parents=True, exist_ok=True)
        descriptor.write_text('{}')
        rows = [{'id': str(i), 'label': i % 2, 'image': 'unused'} for i in range(5)]
        features = ws.unit(np.random.default_rng(8).normal(size=(5, 8)))
        def generate(api, method, names, descriptor, pairs, seed):
            return {n: [f'{method} {seed} {n}'] for n in names}, {}
        backend = Backend('cpu')
        with patch.object(ws, 'ROOT', self.root), \
             patch.object(ws, 'subset', return_value=(['a', 'b'], rows, {'selection_seed': 0})), \
             patch.object(ws, 'verify_sources', return_value={}), \
             patch.object(ws, 'author_api', return_value=None), \
             patch.object(ws, 'generate_prompts', side_effect=generate), \
             patch.object(ws, 'encode_cohort', return_value=(features, None)) as encoding, \
             patch.object(ws, 'adapt', wraps=ws.adapt) as adaptation, \
             contextlib.redirect_stdout(stdio.StringIO()):
            output = ws.run(settings, 'dtd', self.root, 5, backend_factory=lambda _: backend)
            self.assertEqual(adaptation.call_count, 24)
            for call in adaptation.call_args_list:
                np.testing.assert_allclose(np.linalg.norm(call.args[1], axis=1), 1.)
            for row in rows:
                row['label'] = 1 - row['label']
            other = ws.run(settings, 'dtd', self.root, 5, backend_factory=Backend)
        self.assertTrue(backend.unloaded)
        self.assertEqual(encoding.call_count, 2)
        run = protocol.completed_run(output)
        self.assertEqual({v['method']: v['seeds'] for v in run['summary']},
                         {'clip': 1, 'clip_stata': 3, 'waffle': 7, 'waffle_unit': 7, 'waffle_stata': 21})
        with np.load(output / 'replay_scores.npz', allow_pickle=False) as a, \
             np.load(other / 'replay_scores.npz', allow_pickle=False) as b:
            for key in a.files:
                if key.startswith('p_'):
                    np.testing.assert_array_equal(a[key], b[key])
            for seed in range(7):
                mean = a[f'prototypes_waffle_p{seed}']
                np.testing.assert_allclose(a[f'p_waffle_p{seed}'], softmax(100*features@mean.T, axis=1))
                np.testing.assert_allclose(a[f'p_waffle_unit_p{seed}'], softmax(100*features@ws.unit(mean).T, axis=1))
            for row in run['summary']:
                keys = [k for k in a.files if k == 'p_' + row['method'] or k.startswith('p_' + row['method'] + '_')]
                # Tên clip/waffle là prefix của các phương pháp khác: chọn suffix seed.
                if row['method'] == 'clip': keys = ['p_clip']
                elif row['method'] == 'waffle': keys = [f'p_waffle_p{s}' for s in range(7)]
                scores = [evaluate.metric(a['labels'], a[k].argmax(1), [0, 1])['accuracy'] for k in keys]
                self.assertAlmostEqual(row['accuracy_mean'], np.mean(scores))
                self.assertAlmostEqual(row['accuracy_std'], np.std(scores, ddof=1) if len(scores) > 1 else 0.)

    def test_launcher_offline_one_pass_samples_and_existing_dataset(self):
        launcher = yaml.safe_load((ROOT / 'configs/kaggle_launcher.yaml').read_text())
        real_path = kaggle.path
        with patch.object(kaggle, 'path', side_effect=lambda p: self.root / 'stage' if str(p) == 'results/_system/launches/final' else real_path(p)), \
             patch.object(kaggle, 'cli', side_effect=AssertionError('no API')):
            for dataset in ['dtd', 'eurosat']:
                for limit in [None, 5]:
                    plan = kaggle.prepare(launcher, 'waffle_stata_' + dataset, limit)
                    doc = io.read_json(Path(plan['root']) / 'kernel/run.ipynb')
                    code = ''
                    for cell in doc['cells']:
                        if cell['cell_type'] == 'code':
                            source = ''.join(cell['source'])
                            compile(source, '<waffle_stata notebook>', 'exec')
                            code += source
                    self.assertEqual(code.count('output = run('), 1)
                    self.assertIn('from vlm_project.waffle_stata import run', code)
                    self.assertIn("settings = config('configs/project.yaml')", code)
                    self.assertEqual(plan['samples'], 5 if limit else 256)
                    self.assertEqual(plan['mode'], 'smoke' if limit else 'pilot')
                    self.assertEqual(plan['data_dataset'], 'nguyenhung3105/vlm-public-benchmarks-eval-v1')
                    self.assertIn(dataset + '-waffle-stata-pilot', plan['kernel'])
                    self.assertEqual(plan['launcher']['timeout_seconds'], 7200)
            with self.assertRaises(ValueError):
                kaggle.prepare(launcher, 'waffle_stata_dtd', 1880)

    def test_dclip_full_split_uses_dclip_prompts_and_unit_prior(self):
        settings = io.config()
        settings.update(device='cpu', output_root=str(self.root / 'runs'))
        rows = [{'id': str(i), 'label': i % 2, 'image': 'unused'} for i in range(1880)]
        features = ws.unit(np.random.default_rng(11).normal(size=(1880, 8)))
        def generate(api, method, names, descriptor, pairs, seed):
            return {n: [f'{method} {n}'] for n in names}, {}
        def adapt(x, prototypes, initial, spec, seed):
            np.testing.assert_allclose(np.linalg.norm(prototypes, axis=1), 1.)
            return softmax(30*x@prototypes.T, axis=1), np.arange(len(x)) // 75
        backend = Backend('cpu')
        with patch.object(ws.data, 'benchmark', return_value=(['a', 'b'], rows, {'split': 'test-partition-1'})) as load, \
             patch.object(ws, 'read_json', return_value={'b': ['b descriptor'], 'a': ['a descriptor']}), \
             patch.object(ws, 'subset', side_effect=AssertionError('full must not use subset')), \
             patch.object(ws, 'verify_sources', return_value={}), \
             patch.object(ws, 'author_api', return_value=None), \
             patch.object(ws, 'generate_prompts', side_effect=generate) as prompts, \
             patch.object(ws, 'encode_cohort', return_value=(features, None)) as encoding, \
             patch.object(ws, 'adapt', side_effect=adapt), \
             contextlib.redirect_stdout(stdio.StringIO()):
            output = dclip_stata.run(settings, 'dtd', self.root, backend_factory=lambda _: backend)
        self.assertEqual([c.args[1] for c in prompts.call_args_list], ['clip', 'dclip'])
        load.assert_called_once_with(self.root, 'dtd')
        self.assertEqual(encoding.call_count, 1)
        self.assertTrue(backend.unloaded)
        run = protocol.completed_run(output)
        self.assertEqual(run['mode'], 'full')
        self.assertEqual(run['samples'], 1880)
        self.assertEqual(run['kind'], 'dclip_stata')
        self.assertEqual(list(io.read_json(output / 'descriptors.json')), ['a', 'b'])
        self.assertEqual({v['method']: v['seeds'] for v in run['summary']},
                         {'clip': 1, 'clip_stata': 3, 'dclip': 1, 'dclip_unit': 1, 'dclip_stata': 3})
        with np.load(output / 'replay_scores.npz', allow_pickle=False) as a:
            means = a['prototypes_dclip_p0']
            np.testing.assert_allclose(a['p_dclip_p0'], softmax(100*features@means.T, axis=1))
            np.testing.assert_allclose(a['p_dclip_stata_p0_g0'], softmax(30*features@ws.unit(means).T, axis=1))

    def test_dclip_launcher_smoke_and_full(self):
        launcher = yaml.safe_load((ROOT / 'configs/kaggle_launcher.yaml').read_text())
        real_path = kaggle.path
        with patch.object(kaggle, 'path', side_effect=lambda p: self.root / 'stage' if str(p) == 'results/_system/launches/final' else real_path(p)), \
             patch.object(kaggle, 'cli', side_effect=AssertionError('no API')):
            for limit in [None, 5]:
                plan = kaggle.prepare(launcher, 'dclip_stata_dtd', limit)
                doc = io.read_json(Path(plan['root']) / 'kernel/run.ipynb')
                code = '\n'.join(''.join(c['source']) for c in doc['cells'] if c['cell_type'] == 'code')
                self.assertEqual(code.count('output = run('), 1)
                self.assertIn('from vlm_project.dclip_stata import run', code)
                self.assertEqual(plan['samples'], 5 if limit else 1880)
                self.assertEqual(plan['mode'], 'smoke' if limit else 'full')
                self.assertEqual(plan['data_dataset'], 'nguyenhung3105/vlm-public-benchmarks-eval-v1')
                self.assertIn('dtd-dclip-stata-full', plan['kernel'])
                for cell in doc['cells']:
                    if cell['cell_type'] == 'code':
                        compile(''.join(cell['source']), '<dclip notebook>', 'exec')
            with self.assertRaises(ValueError):
                kaggle.prepare(launcher, 'dclip_stata_dtd', 256)

    def test_waffle_full_runner_1880_and_all_prompt_group_pairs(self):
        settings = io.config()
        settings.update(device='cpu', output_root=str(self.root / 'runs'))
        rows = [{'id': str(i), 'label': i % 2, 'image': 'unused'} for i in range(1880)]
        features = ws.unit(np.random.default_rng(9).normal(size=(1880, 8)))
        def generate(api, method, names, descriptor, pairs, seed):
            return {n: [f'{method} {seed} {n}'] for n in names}, {}
        with patch.object(ws.data, 'benchmark', return_value=(['a', 'b'], rows, {'split': 'test-partition-1'})) as load, \
             patch.object(ws, 'subset', side_effect=AssertionError('full must not use subset')), \
             patch.object(ws, 'verify_sources', return_value={}), \
             patch.object(ws, 'author_api', return_value=None), \
             patch.object(ws, 'generate_prompts', side_effect=generate), \
             patch.object(ws, 'encode_cohort', return_value=(features, None)) as encode, \
             patch.object(ws, 'adapt', side_effect=lambda x,p,c,s,g: (softmax(30*x@p.T,axis=1),np.arange(len(x))//75)), \
             contextlib.redirect_stdout(stdio.StringIO()):
            output = waffle_stata_full.run(settings, 'dtd', self.root, backend_factory=Backend)
        load.assert_called_once_with(self.root, 'dtd')
        self.assertEqual(encode.call_count, 1)
        run = protocol.completed_run(output)
        self.assertEqual((run['mode'], run['samples']), ('full', 1880))
        self.assertEqual({v['method']:v['seeds'] for v in run['summary']},
                         {'clip':1,'clip_stata':3,'waffle':7,'waffle_unit':7,'waffle_stata':21})
        with np.load(output/'replay_scores.npz',allow_pickle=False) as a:
            self.assertEqual(a['p_waffle_stata_p6_g2'].shape, (1880,2))

    def test_waffle_full_launcher_separate_from_pilot(self):
        launcher = yaml.safe_load((ROOT / 'configs/kaggle_launcher.yaml').read_text())
        real_path = kaggle.path
        with patch.object(kaggle, 'path', side_effect=lambda p:self.root/'stage' if str(p)=='results/_system/launches/final' else real_path(p)), \
             patch.object(kaggle, 'cli', side_effect=AssertionError('no API')):
            for limit in [None,5]:
                plan = kaggle.prepare(launcher,'waffle_stata_dtd_full',limit)
                doc = io.read_json(Path(plan['root'])/'kernel/run.ipynb')
                code = '\n'.join(''.join(c['source']) for c in doc['cells'] if c['cell_type']=='code')
                self.assertIn('from vlm_project.waffle_stata_full import run',code)
                self.assertEqual(code.count('output = run('),1)
                self.assertEqual(plan['samples'],5 if limit else 1880)
                self.assertEqual(plan['mode'],'smoke' if limit else 'full')
                self.assertIn('dtd-waffle-stata-full',plan['kernel'])
                self.assertEqual(plan['data_dataset'],'nguyenhung3105/vlm-public-benchmarks-eval-v1')
                for cell in doc['cells']:
                    if cell['cell_type']=='code':compile(''.join(cell['source']),'<waffle full>','exec')
            with self.assertRaises(ValueError):kaggle.prepare(launcher,'waffle_stata_dtd_full',256)
        with self.assertRaises(ValueError):waffle_stata_full.run(io.config(),'eurosat',self.root)

    def test_semantic_blend_formula_posteriors_and_label_independence(self):
        import csv
        settings = io.config()
        settings.update(device='cpu', output_root=str(self.root/'runs'))
        rows = [{'id':str(i),'label':i%2,'image':'unused'} for i in range(5)]
        features = ws.unit(np.random.default_rng(24).normal(size=(5,8)))
        def generate(api,method,names,descriptor,pairs,seed):
            return {n:[f'{method} {seed} {n}'] for n in names},{}
        def adapt(x,p,c,s,g):
            np.testing.assert_allclose(np.linalg.norm(p,axis=1),1.)
            return softmax(20*x@p.T,axis=1),np.zeros(len(x),dtype=int)
        with patch.object(ws,'subset',return_value=(['a','b'],rows,{})), \
             patch.object(ws,'verify_sources',return_value={}), \
             patch.object(ws,'author_api',return_value=None), \
             patch.object(ws,'read_json',return_value={'b':['b descriptor'],'a':['a descriptor']}), \
             patch.object(ws,'generate_prompts',side_effect=generate), \
             patch.object(ws,'encode_cohort',return_value=(features,None)) as encode, \
             patch.object(ws,'adapt',side_effect=adapt), \
             contextlib.redirect_stdout(stdio.StringIO()):
            output=waffle_semantic_stata.run(settings,'dtd',self.root,5,backend_factory=Backend)
            for row in rows:row['label']=1-row['label']
            other=waffle_semantic_stata.run(settings,'dtd',self.root,5,backend_factory=Backend)
        self.assertEqual(encode.call_count,2)
        run=protocol.completed_run(output)
        self.assertEqual(run['kind'],'waffle_semantic_stata')
        self.assertEqual({r['method']:r['seeds'] for r in run['summary']},
                         {'clip':1,'clip_stata':3,'semantic_unit':1,'semantic_stata':3,
                          'waffle':7,'waffle_unit':7,'waffle_semantic':7,'waffle_stata':21,'waffle_semantic_stata':21})
        with np.load(output/'replay_scores.npz',allow_pickle=False) as a,np.load(other/'replay_scores.npz',allow_pickle=False) as b:
            self.assertEqual(len([k for k in a.files if k.startswith('p_')]),71)
            for key in a.files:
                if key.startswith('p_'):np.testing.assert_array_equal(a[key],b[key])
            for seed in range(7):
                expected=ws.unit(.5*ws.unit(a[f'prototypes_waffle_p{seed}'])+.5*a['prototypes_semantic'])
                np.testing.assert_allclose(a[f'prototypes_waffle_semantic_p{seed}'],expected)
                np.testing.assert_allclose(a[f'p_waffle_semantic_p{seed}'],softmax(100*features@expected.T,axis=1))
                for g in range(3):
                    np.testing.assert_allclose(a[f'p_waffle_semantic_stata_p{seed}_g{g}'],softmax(20*features@expected.T,axis=1))
            with (output/'comparisons.tsv').open() as f:paired=list(csv.DictReader(f,delimiter='\t'))
            self.assertEqual(len(paired),42)
            for row in paired:
                if row['method']!='waffle_semantic_stata':continue
                s,g=row['prompt_seed'],row['grouping_seed']
                pred=a[f'p_waffle_semantic_stata_p{s}_g{g}'].argmax(1)
                base=a[f'p_waffle_stata_p{s}_g{g}'].argmax(1)
                y=a['labels']
                self.assertEqual(int(row['waffle_stata_wrong_to_correct']),int(((base!=y)&(pred==y)).sum()))
                self.assertEqual(int(row['waffle_stata_correct_to_wrong']),int(((base==y)&(pred!=y)).sum()))

    def test_semantic_launcher_full_and_smoke_separate_kernel(self):
        launcher=yaml.safe_load((ROOT/'configs/kaggle_launcher.yaml').read_text())
        real_path=kaggle.path
        with patch.object(kaggle,'path',side_effect=lambda p:self.root/'stage' if str(p)=='results/_system/launches/final' else real_path(p)), \
             patch.object(kaggle,'cli',side_effect=AssertionError('no API')):
            for limit in [None,5]:
                plan=kaggle.prepare(launcher,'waffle_semantic_stata_dtd',limit)
                doc=io.read_json(Path(plan['root'])/'kernel/run.ipynb')
                code='\n'.join(''.join(c['source']) for c in doc['cells'] if c['cell_type']=='code')
                self.assertIn('from vlm_project.waffle_semantic_stata import run',code)
                self.assertEqual(code.count('output = run('),1)
                self.assertEqual(plan['samples'],5 if limit else 1880)
                self.assertEqual(plan['mode'],'smoke' if limit else 'full')
                self.assertIn('waffle-semantic-stata-full',plan['kernel'])
                self.assertEqual(plan['waffle_stata_specification']['semantic_weight'],.5)
                self.assertEqual(plan['data_dataset'],'nguyenhung3105/vlm-public-benchmarks-eval-v1')
                for c in doc['cells']:
                    if c['cell_type']=='code':compile(''.join(c['source']),'<semantic notebook>','exec')

    def test_eurosat_semantic_config_and_launcher_limits(self):
        spec=waffle_semantic_stata.specification('eurosat')
        dtd=waffle_semantic_stata.specification('dtd')
        self.assertEqual(spec,{**dtd,'samples':27000,'timeout_seconds':14400})
        launcher=yaml.safe_load((ROOT/'configs/kaggle_launcher.yaml').read_text())
        real_path=kaggle.path
        with patch.object(kaggle,'path',side_effect=lambda p:self.root/'stage' if str(p)=='results/_system/launches/final' else real_path(p)), \
             patch.object(kaggle,'cli',side_effect=AssertionError('no API')):
            for limit in [None,5]:
                plan=kaggle.prepare(launcher,'waffle_semantic_stata_eurosat',limit)
                self.assertEqual(plan['samples'],5 if limit else 27000)
                self.assertEqual(plan['mode'],'smoke' if limit else 'full')
                self.assertEqual(plan['launcher']['timeout_seconds'],14400)
                self.assertEqual(plan['waffle_stata_specification']['semantic_weight'],.5)
                self.assertIn('eurosat-waffle-semantic-stata-full',plan['kernel'])
                self.assertEqual(plan['data_dataset'],'nguyenhung3105/vlm-public-benchmarks-eval-v1')
                doc=io.read_json(Path(plan['root'])/'kernel/run.ipynb')
                code='\n'.join(''.join(c['source']) for c in doc['cells'] if c['cell_type']=='code')
                self.assertIn("name = 'eurosat'",code)
                self.assertIn(f'limit={5 if limit else 27000}',code)
                self.assertEqual(code.count('output = run('),1)
                for c in doc['cells']:
                    if c['cell_type']=='code':compile(''.join(c['source']),'<eurosat semantic>','exec')
            with self.assertRaises(ValueError):kaggle.prepare(launcher,'waffle_semantic_stata_eurosat',256)

    def test_eurosat_semantic_full_dispatch_with_tiny_synthetic_split(self):
        # Thu nhỏ schema/split trong fixture để chạy StatA thật, không chạy dataset thật.
        settings=io.config();settings.update(device='cpu',output_root=str(self.root/'runs'))
        spec={**waffle_semantic_stata.specification('eurosat'),'samples':6}
        rows=[{'id':str(i),'label':i%2,'image':'unused'} for i in range(6)]
        features=ws.unit(np.random.default_rng(91).normal(size=(6,8)))
        def generate(api,method,names,descriptor,pairs,seed):
            return {n:[f'{method} {seed} {n}'] for n in names},{}
        with patch.object(waffle_semantic_stata,'specification',return_value=spec), \
             patch.dict(ws.BENCHMARKS,{'eurosat':(2,6,'synthetic-full')}), \
             patch.object(ws.data,'benchmark',return_value=(['a','b'],rows,{'split':'synthetic-full'})) as loader, \
             patch.object(ws,'subset',side_effect=AssertionError('must use full loader')), \
             patch.object(ws,'verify_sources',return_value={}), \
             patch.object(ws,'author_api',return_value=None), \
             patch.object(ws,'read_json',return_value={'b':['b descriptor'],'a':['a descriptor']}), \
             patch.object(ws,'generate_prompts',side_effect=generate), \
             patch.object(ws,'encode_cohort',return_value=(features,None)) as encode, \
             contextlib.redirect_stdout(stdio.StringIO()):
            output=waffle_semantic_stata.run(settings,'eurosat',self.root,backend_factory=Backend)
        loader.assert_called_once_with(self.root,'eurosat')
        self.assertEqual(encode.call_count,1)
        run=protocol.completed_run(output)
        self.assertEqual((run['mode'],run['dataset'],run['samples']),('full','eurosat',6))
        self.assertEqual(len(run['summary']),9)
        self.assertEqual(run['specification_sha256'],io.sha256(ROOT/'configs/waffle_semantic_stata_eurosat.yaml'))
        self.assertIn('eurosat',run['semantic']['source'])
        with np.load(output/'replay_scores.npz',allow_pickle=False) as a:
            self.assertEqual(len([k for k in a.files if k.startswith('p_')]),71)
            self.assertEqual(a['p_waffle_semantic_stata_p6_g2'].shape,(6,2))


if __name__ == '__main__':
    unittest.main()
