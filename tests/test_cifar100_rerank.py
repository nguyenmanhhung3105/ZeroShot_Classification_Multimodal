import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tests'))
from vlm_project import cifar100_rerank as rerank, cifar100_two_stage as experiment, kaggle
from test_cifar100_two_stage import FakeBackend


class RerankTests(unittest.TestCase):
    def test_routing_uses_group_mass_not_max_fine_or_descriptions(self):
        # The group of the highest single fine score need not have top-2 mass.
        fine = np.log([[.3, .001, .18, .17, .175, .174]])
        mapping = np.array([0, 0, 1, 1, 2, 2])
        desc = np.array([[9., 8., 1., 2., 7., 3.]])
        scores, mass, groups = rerank.predict(fine, desc, mapping)
        np.testing.assert_array_equal(groups, [[1, 2]])
        self.assertEqual(scores['flat_names'].argmax(1).item(), 0)
        self.assertEqual(scores['flat_descriptions'].argmax(1).item(), 0)
        self.assertEqual(scores['top2_descriptions'].argmax(1).item(), 4)
        np.testing.assert_array_equal(scores['top2_names'][:, :2], [[0, 0]])
        for values in scores.values():
            np.testing.assert_allclose(values.sum(1), 1)
        _, _, changed = rerank.predict(fine, -desc, mapping)
        np.testing.assert_array_equal(groups, changed)
        np.testing.assert_allclose(mass.sum(1), 1)

    def test_stable_ties_keep_exact_two_groups_and_ten_children(self):
        mapping = np.repeat(np.arange(20), 5)
        scores, _, groups = rerank.predict(np.zeros((3, 100)), np.zeros((3, 100)), mapping)
        np.testing.assert_array_equal(groups, np.tile([0, 1], (3, 1)))
        self.assertTrue(np.all((scores['top2_names'] > 0).sum(1) == 10))
        with self.assertRaises(ValueError):
            rerank.predict([[0, 0]], [[np.nan, 1]], [0, 1])
        with self.assertRaises(ValueError):
            rerank.predict([[0, 0]], [[0, 1]], [0, 2])

    def test_descriptor_bank_complete_versioned_and_configuration_guarded(self):
        spec = experiment.specification(ROOT / 'configs/cifar100_rerank.json')
        value = rerank.validate_spec(spec)
        names = list(value['descriptors'])
        bank = rerank.description_bank(names, spec)
        self.assertEqual(len(bank), 100)
        self.assertTrue(all(len(texts) == 2 for texts in bank.values()))
        self.assertFalse(value['provenance']['sample_images_used'])
        self.assertFalse(value['provenance']['per_sample_errors_used'])
        self.assertLessEqual(max(len(t.split()) for texts in bank.values() for t in texts), 35)
        with self.assertRaises(ValueError):
            rerank.description_bank(['wrong']*100, spec)
        with self.assertRaises(ValueError):
            rerank.validate_spec({**spec, 'top_groups': 1})

    def test_ablation_reports_harms_and_corrections_against_both_flats(self):
        scores = {name: np.eye(2)[[0, 0]] for name in rerank.METHODS}
        scores['top2_descriptions'] = np.eye(2)[[1, 1]]
        rows = rerank.ablation_comparisons([1, 0], scores)
        row = next(r for r in rows if r['method'] == 'top2_descriptions' and r['reference'] == 'flat_descriptions')
        self.assertEqual(row['corrected'], 1)
        self.assertEqual(row['harmed'], 1)
        self.assertEqual(row['accuracy_delta'], 0)

    def test_end_to_end_four_controls_same_images_and_replay(self):
        spec_file = ROOT / 'configs/cifar100_rerank.json'
        spec = experiment.specification(spec_file)
        names = list(rerank.validate_spec(spec)['descriptors'])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'source'
            root.mkdir()
            source = {'dataset': 'cifar100', 'status': 'sources_verified',
                      'splits': {'train': 50000, 'test': 10000},
                      'class_names': {'fine': names, 'coarse': [f'group{i}' for i in range(20)]},
                      'fine_to_coarse': {str(i): i//5 for i in range(100)}}
            (root / 'download_manifest.json').write_text(json.dumps(source))
            ids = experiment.split_indices(spec, 'dev', True)
            with (root / 'train.bin').open('wb') as stream:
                stream.truncate(50000*3074)
                for index in ids:
                    fine = int(index) % 100
                    stream.seek(int(index)*3074)
                    stream.write(bytes([fine//5, fine])+bytes([fine])*1024+bytes(2048))
            instances = []
            def factory(device):
                instances.append(FakeBackend(device))
                return instances[-1]
            output = experiment.run(root, Path(tmp) / 'output', spec_file=spec_file, backend_factory=factory)
            run = json.loads((output / 'run.json').read_text())
            self.assertEqual(run['kind'], 'cifar100_rerank')
            self.assertEqual(len(run['summary']), 4)
            self.assertEqual(set(run['prompts']), {'fine', 'descriptions'})
            self.assertIn('descriptor_sha256', run['protocol'])
            self.assertTrue((output / 'ablation_comparisons.tsv').exists())
            self.assertTrue(instances[0].unloaded)
            with np.load(output / 'replay_scores.npz', allow_pickle=False) as replay:
                scores, _, groups = rerank.predict(replay['fine_logits'], replay['description_logits'], replay['mapping'])
                for method in rerank.METHODS:
                    np.testing.assert_allclose(scores[method], replay[method])
                np.testing.assert_array_equal(groups, replay['candidate_groups'])
                self.assertEqual(replay['image_features'].shape[0], 5)
            with self.assertRaisesRegex(ValueError, 'smoke'):
                experiment.validate_dev(output, run['protocol'])

    def test_launcher_prepares_separate_rerank_job_without_network(self):
        import yaml
        from unittest.mock import patch
        launcher = yaml.safe_load((ROOT / 'configs/kaggle_launcher.yaml').read_text())
        with tempfile.TemporaryDirectory() as tmp:
            real_path = kaggle.path
            def local_path(value):
                return Path(tmp) if str(value) == 'results/_system/launches/final' else real_path(value)
            with patch.object(kaggle, 'path', side_effect=local_path), patch.object(kaggle, 'cli', side_effect=AssertionError('No API')):
                plan = kaggle.prepare(launcher, 'cifar100_rerank_dev')
            self.assertEqual(plan['task'], 'cifar100_rerank')
            self.assertIn('rerank-dev', plan['kernel'])
            self.assertEqual(plan['samples'], 10000)
            self.assertEqual(plan['launcher']['poll_seconds'], 30)
            folder = Path(plan['root']) / 'kernel'
            manifest = json.loads((folder / 'source_manifest.json').read_text())
            self.assertIn('configs/descriptors/cifar100_visual_v1.json', manifest['sha256'])
            nb = json.loads((folder / 'run.ipynb').read_text())
            code = '\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code')
            compile(code, 'generated-rerank-notebook', 'exec')
            self.assertIn('configs/cifar100_rerank.json', code)
            self.assertIn('metrics_', code)


if __name__ == '__main__':
    unittest.main()
