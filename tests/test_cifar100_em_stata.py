import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.special import softmax

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from vlm_project import cifar100_em_stata as adapted, cifar100_two_stage as runner, kaggle


class AdaptedCifarTests(unittest.TestCase):
    def test_multiclass_plurality_and_clip_tie(self):
        votes = np.array([[4, 5, 6], [4, 5, 5], [4, 4, 5], [4, 5, 4]])
        np.testing.assert_array_equal(adapted.majority(votes, 20), [4, 5, 4, 4])
        with self.assertRaises(ValueError):
            adapted.majority(np.array([[0, 0, 20]]), 20)

    def test_shared_query_groups_and_all_controls_no_labels(self):
        spec = runner.specification(ROOT / 'configs/cifar100_em_stata.json')
        x = np.eye(100, dtype=np.float32)[:80]
        fine = np.eye(100, dtype=np.float32)
        coarse = np.asarray([np.repeat(np.eye(20)[i], 5) for i in range(20)], dtype=np.float32)
        coarse /= np.linalg.norm(coarse, axis=1, keepdims=True)
        calls = []
        def em(query, **kwargs):
            calls.append((query.shape, kwargs))
            return (query.argmax(-1)+1) % query.shape[-1], {'objective': [], 'mm_steps': [], 'singleton_fallback':False}, query
        def stata(features, prototypes, **kwargs):
            prior = softmax(100*features@prototypes.T, axis=1)
            shifted = np.eye(len(prototypes))[(prior.argmax(1)+2)%len(prototypes)]
            return prior, shifted
        with contextlib.redirect_stdout(io.StringIO()):
            scores, q, details, arrays = adapted.predict(x, {'fine':fine, 'coarse':coarse},
                100*x@fine.T, 100*x@coarse.T, np.repeat(np.arange(20),5), spec, em, stata)
        self.assertEqual(len(scores), 20)
        self.assertEqual(len(calls), 12)  # Two levels, two groups, three seeds.
        self.assertTrue(all(c[1] == {'iterations':20,'mm_iterations':1000} for c in calls))
        self.assertEqual(q.shape, (80,20))
        for seed in [0,1,2]:
            np.testing.assert_array_equal(scores[f'flat_majority@{seed}'].argmax(1), scores['flat_clip'].argmax(1))
            self.assertEqual(sorted(np.bincount(arrays[f'group_ids_{seed}']).tolist()), [5,75])
            self.assertEqual(arrays[f'fine_votes_{seed}'].shape, (80,3))
            self.assertIn(str(seed),details['seeds'])
        for value in scores.values():
            np.testing.assert_allclose(value.sum(1),1)

    def test_small_group_keeps_clip_stata_and_restores_threads(self):
        import torch
        spec = runner.specification(ROOT / 'configs/cifar100_em_stata.json')
        x = np.eye(100,dtype=np.float32)[:3]
        coarse = np.asarray([np.repeat(np.eye(20)[i],5) for i in range(20)],dtype=np.float32)
        coarse /= np.linalg.norm(coarse,axis=1,keepdims=True)
        def em(q,**kwargs): return q.argmax(-1),{'objective':[]},q
        previous = torch.get_num_threads()
        with contextlib.redirect_stdout(io.StringIO()):
            scores,_,_,_=adapted.predict(x, {'fine':np.eye(100),'coarse':coarse},
                100*x,100*x@coarse.T,np.repeat(np.arange(20),5),spec,em,
                lambda *_a,**_k: self.fail('StatA must not run on <=3 images'))
        self.assertEqual(torch.get_num_threads(), previous)
        np.testing.assert_array_equal(scores['flat_stata@0'].argmax(1),[0,1,2])

    def test_launcher_whitelists_solver_and_separate_job(self):
        import yaml
        launcher=yaml.safe_load((ROOT/'configs/kaggle_launcher.yaml').read_text())
        with tempfile.TemporaryDirectory() as tmp:
            real=kaggle.path
            def local(v): return Path(tmp) if str(v)=='results/_system/launches/final' else real(v)
            with patch.object(kaggle,'path',side_effect=local), patch.object(kaggle,'cli',side_effect=AssertionError('No API')):
                plan=kaggle.prepare(launcher,'cifar100_em_stata_smoke')
            self.assertEqual(plan['samples'],5)
            self.assertEqual(plan['launcher']['poll_seconds'],30)
            self.assertIn('em-stata-smoke',plan['kernel'])
            root=Path(plan['root'])/'kernel'
            manifest=json.loads((root/'source_manifest.json').read_text())
            self.assertIn('src/vlm_project/adaptation.py',manifest['sha256'])
            nb=json.loads((root/'run.ipynb').read_text())
            for cell in nb['cells']:
                if cell['cell_type']=='code':compile(''.join(cell['source']),'adapted-notebook','exec')

    def test_runner_publishes_eight_methods_three_seeds_with_mock_solvers(self):
        sys.path.insert(0,str(ROOT/'tests'))
        from test_cifar100_two_stage import FakeBackend
        spec=runner.specification(ROOT/'configs/cifar100_em_stata.json')
        original_predict=adapted.predict
        def em(q,**kwargs):return q.argmax(-1),{'objective':[]},q
        def stata(x,t,**kwargs):
            p=softmax(100*x@t.T,axis=1)
            return p,p
        def mocked(*args):return original_predict(*args,em_solver=em,stata_solver=stata)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'source';root.mkdir()
            metadata={'dataset':'cifar100','status':'sources_verified','splits':{'train':50000,'test':10000},
                      'class_names':{'fine':[f'fine{i}' for i in range(100)],'coarse':[f'coarse{i}' for i in range(20)]},
                      'fine_to_coarse':{str(i):i//5 for i in range(100)}}
            (root/'download_manifest.json').write_text(json.dumps(metadata))
            with (root/'train.bin').open('wb') as stream:
                stream.truncate(50000*3074)
                for index in runner.split_indices(spec,'dev',True):
                    fine=int(index)%100;stream.seek(int(index)*3074)
                    stream.write(bytes([fine//5,fine])+bytes([fine])*1024+bytes(2048))
            with patch.object(adapted,'predict',side_effect=mocked), contextlib.redirect_stdout(io.StringIO()):
                output=runner.run(root,Path(tmp)/'output',spec_file=ROOT/'configs/cifar100_em_stata.json',backend_factory=FakeBackend)
            result=json.loads((output/'run.json').read_text())
            self.assertEqual(result['kind'],'cifar100_em_stata')
            self.assertEqual(len(result['summary']),8)
            for row in result['summary']:
                self.assertEqual(row['seeds'],1 if row['method'] in ['flat_clip','hierarchy_clip'] else 3)
                self.assertNotIn('top5_mean',row)
            self.assertTrue((output/'adaptation_groups.npz').exists())
            self.assertTrue((output/'completed.json').exists())


if __name__ == '__main__': unittest.main()
