"""Support allocation ablation; synthetic metadata/features only."""
import contextlib
import copy
import csv
import io as stdio
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.special import expit
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from vlm_project import fakeddit_fewshot_support as support
from vlm_project import fakeddit_fewshot_task1 as fs
from vlm_project import io,kaggle,protocol
import test_fakeddit_fewshot_task1 as fixtures


class SupportAllocationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.spec=fs.specification()
        self.settings=io.config();self.settings.update(device='cpu',output_root=str(self.root/'runs'))

    def test_budget_nested_overlap_and_rotating_subtype_quotas(self):
        spec=support.specification();y=np.tile(np.arange(6),400)
        extra=[]
        for seed in range(5):
            old=fs.select_support(y,64,seed);new=support.select_balanced_support(y,spec,seed)
            self.assertEqual(len(new),384);self.assertEqual(len(set(new)),384)
            counts=np.bincount(y[new],minlength=6)
            np.testing.assert_array_equal(counts,support.quotas(spec,seed))
            self.assertEqual(counts[0],192);self.assertEqual(counts[1:].sum(),192)
            self.assertTrue(set(counts[1:])<={38,39})
            self.assertEqual(len(set(old)&set(new)),256)
            self.assertTrue(set(old[y[old]==0])<=set(new[y[new]==0]))
            self.assertTrue(set(new[y[new]!=0])<=set(old[y[old]!=0]))
            extra.append(counts[1:])
        np.testing.assert_allclose(np.mean(extra,axis=0),[38.4]*5)
        with self.assertRaises(ValueError):support.select_balanced_support(np.tile(np.arange(6),64),spec,0)

    def test_runner_controls_and_two_policy_cache_replay(self):
        spec,roots,rows=fixtures.FewshotTask1Tests.fixture(self)
        spec.update(support_policies=support.POLICIES,binary_samples_per_class=12,fake_allocation='equal_rotating_remainder')
        def loader(root,split,smoke=False):return (rows[split][:5] if smoke else rows[split]),{'split':split}
        backend=fixtures.Backend('cpu')
        with patch.object(support,'specification',return_value=spec), \
             patch.object(fs.data,'fakeddit',side_effect=loader), \
             patch.object(fs,'encode_cohort',side_effect=fixtures.FewshotTask1Tests.encode) as encode, \
             contextlib.redirect_stdout(stdio.StringIO()):
            output=support.run(self.settings,roots['train'],roots['dev'],backend_factory=lambda _:backend)
            for row in rows['dev']:row['6way']='1' if row['6way']=='0' else '0'
            other=support.run(self.settings,roots['train'],roots['dev'],backend_factory=fixtures.Backend)
        self.assertTrue(backend.unloaded)
        self.assertEqual(encode.call_count,4)
        run=protocol.completed_run(output)
        self.assertEqual(run['kind'],'fakeddit_fewshot_support')
        self.assertEqual(len(run['summary']),6)
        self.assertTrue(all(r['seeds']==5 for r in run['summary']))
        log=io.read_json(output/'support.json')
        self.assertEqual(len(log),10)
        with np.load(output/'features.npz',allow_pickle=False) as cache, \
             np.load(output/'replay_scores.npz',allow_pickle=False) as scores, \
             np.load(other/'replay_scores.npz',allow_pickle=False) as changed, \
             np.load(output/'linear_models.npz',allow_pickle=False) as models:
            self.assertEqual(len([k for k in scores.files if k.startswith('p_')]),30)
            for key in scores.files:
                if key.startswith('p_'):np.testing.assert_array_equal(scores[key],changed[key])
            for entry in log:
                seed,policy=entry['seed'],entry['policy']
                ix=cache[f'support_{policy}_s{seed}']
                self.assertEqual(cache['train_ids'][ix].tolist(),entry['ids'])
                np.testing.assert_array_equal(np.bincount(cache['train_y6'][ix],minlength=6),entry['quotas_6way'])
                if policy=='legacy_6way':
                    y=np.asarray([int(r['6way']) for r in rows['train']])
                    expected=fs.select_support(y,4,seed)
                    self.assertEqual(entry['ids'],[rows['train'][i]['id'] for i in expected])
            banks=fs.feature_banks(cache['dev_image'],cache['dev_title'])
            for policy in support.POLICIES:
                for seed in range(5):
                    for method in spec['methods']:
                        key=f'{policy}_{method}_s{seed}'
                        expected=expit(banks[method]@models[key+'_coef'].T+models[key+'_intercept'])[:,0]
                        np.testing.assert_allclose(expected,scores['p_'+key][:,1])
            with (output/'support_comparisons.tsv').open() as f:paired=list(csv.DictReader(f,delimiter='\t'))
            self.assertEqual(len(paired),15)
            for row in paired:
                m,s=row['modality'],row['seed'];y=scores['labels']
                a=scores[f'p_legacy_6way_{m}_s{s}'].argmax(1);b=scores[f'p_balanced_binary_{m}_s{s}'].argmax(1)
                self.assertEqual(int(row['legacy_wrong_to_correct']),int(((a!=y)&(b==y)).sum()))
                self.assertAlmostEqual(float(row['delta_accuracy_pp']),100*((b==y).mean()-(a==y).mean()))

    def test_launcher_isolated_and_both_datasets_with_403_fallback_schema(self):
        launcher=yaml.safe_load((ROOT/'configs/kaggle_launcher_fewshot_min5.yaml').read_text())
        real_path=kaggle.path
        with patch.object(kaggle,'path',side_effect=lambda p:self.root/'stage' if str(p)=='results/_system/launches/final' else real_path(p)), \
             patch.object(kaggle,'cli',side_effect=AssertionError('no API')):
            for limit in [None,5]:
                plan=kaggle.prepare(launcher,'fakeddit_fewshot_support_min5',limit)
                self.assertEqual(plan['samples'],5 if limit else 3000)
                self.assertEqual(plan['fewshot_task1_specification']['binary_samples_per_class'],192)
                self.assertIn('fewshot-support-min5',plan['kernel'])
                self.assertEqual(set(plan['data_datasets']),{'nguyenhung3105/vlm-fakeddit-train-min5-v1','nguyenhung3105/vlm-fakeddit-dev-min5-v1'})
                doc=io.read_json(Path(plan['root'])/'kernel/run.ipynb')
                code='\n'.join(''.join(c['source']) for c in doc['cells'] if c['cell_type']=='code')
                self.assertIn('from vlm_project.fakeddit_fewshot_support import run',code)
                self.assertEqual(code.count('output = run('),1)
                self.assertNotIn('freeze(settings',code)
                for c in doc['cells']:
                    if c['cell_type']=='code':compile(''.join(c['source']),'<support balance>','exec')
            invalid=copy.deepcopy(launcher);invalid['jobs']['fakeddit_fewshot_support_min5']['split']='test'
            with self.assertRaises(ValueError):kaggle.prepare(invalid,'fakeddit_fewshot_support_min5')


if __name__=='__main__':unittest.main()
