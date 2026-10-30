import unittest
import numpy as np
from experiments.clip_zlap.core import unit_vectors
from experiments.clip_dpe.core import predict_stream, alignment


class DpeTests(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(17)
        return unit_vectors(rng.normal(size=(31, 12))), unit_vectors(rng.normal(size=(3, 12)))

    def test_prefix_reset_and_permutation_coverage(self):
        x, t = self.fixture()
        a, trace, diag = predict_stream(x, t)
        b, _, _ = predict_stream(x[:9], t)
        again, _, _ = predict_stream(x, t)
        np.testing.assert_array_equal(a[:9], b); np.testing.assert_array_equal(a, again)
        self.assertLessEqual(max(diag['memory_counts']), 3)
        self.assertEqual(diag['text_updates'], int(trace['text_update'].sum()))
        c, _, d = predict_stream(x, t, np.arange(len(x))[::-1])
        self.assertTrue(np.isfinite(c).all()); self.assertEqual(sorted(d['order']), list(range(len(x))))

    def test_memory_lowest_entropy_of_online_pseudo_labels(self):
        x, t = self.fixture(); _, trace, diag = predict_stream(x, t)
        for c, stored in enumerate(diag['memory_indices']):
            candidates = np.flatnonzero(trace['pseudo_label'] == c)
            expected = candidates[np.argsort(trace['cache_entropy'][candidates])[:3]]
            self.assertEqual(set(stored), set(expected))

    def test_rejects_fake_views_bad_parameters_and_bad_order(self):
        x, t = self.fixture()
        with self.assertRaises(ValueError): predict_stream(np.stack([x, x], 1), t)
        with self.assertRaises(ValueError): predict_stream(x, t, alpha=3.)
        with self.assertRaises(ValueError): predict_stream(x, t, np.zeros(len(x), dtype=int))

    def test_alignment_single_class_is_zero(self):
        import torch
        self.assertEqual(float(alignment(torch.tensor([[1., 0.]]), torch.tensor([[0., 1.]]))), 0.)

    def test_confidence_matching_scale_and_identity(self):
        from experiments.clip_dpe.fusion import confidence_match
        rng = np.random.default_rng(45)
        a = rng.normal(size=(37, 8)).astype('float32') * 3
        same, d = confidence_match(a, a)
        self.assertEqual(d['temperature'], 1.)
        np.testing.assert_array_equal(same, 2*a)
        mixed, d = confidence_match(a, 5*a)
        self.assertLess(d['relative_error'], .1)
        self.assertGreater(d['temperature'], 1.)
        np.testing.assert_array_equal(mixed.argmax(1), a.argmax(1))
        with self.assertRaises(ValueError): confidence_match(a, a[:, :2])

    def test_runner_pairs_controls_and_labels_only_change_metrics(self):
        import csv
        import json
        from copy import deepcopy
        from pathlib import Path
        import tempfile
        from unittest.mock import patch
        from experiments.clip_dpe import run as runner
        from prompts.crisismmd_prompts import get_label_names
        rng = np.random.default_rng(13)
        x = unit_vectors(rng.normal(size=(17, 12)))
        info = unit_vectors(rng.normal(size=(2, 12))); human = unit_vectors(rng.normal(size=(8, 12)))
        il = list(get_label_names('informativeness')); hl = list(get_label_names('humanitarian'))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); settings = json.loads((runner.ROOT/'experiments/clip_dpe/config.json').read_text())
            for k in ['majority_run', 'dn_run', 'inmap_run']:
                folder = root/k; folder.mkdir(); (folder/'run_manifest.json').write_text('{}'); settings[k] = str(folder)
            for scope in ['train', 'dev']: settings[scope+'_run'] = str(root/scope)
            settings['text_prototype_run'] = settings['train_run']
            def source(path):
                scope = Path(path).name
                return {'directory': str(path), 'split': scope, 'feature_sha256': 'synthetic',
                        'manifest': {'model': 'metaclip2_vith14', 'logit_scale': 100.},
                        'features': x, 'info_labels': il, 'labels': hl, 'prototypes': info,
                        'scores': [x@info.T*.95, x@human.T*.9],
                        'rows': [{'id': str(i), 'y_true': hl[(i+offset)%8]} for i in range(len(x))],
                        'binary_truth': [il[i%2] for i in range(len(x))] if scope == 'train' else None}
            gates = {s: np.arange(len(x))%2 == s%2 for s in [0, 1, 2]}
            outputs = []
            for offset in [0, 1]:
                with patch.object(runner, 'load_features', side_effect=source), \
                     patch.object(runner, 'text_prototypes', return_value=human), \
                     patch.object(runner, 'validate_majority', return_value={}), \
                     patch.object(runner, 'majority_gates', return_value=gates), \
                     patch.object(runner, 'load_companions', return_value=(x@human.T, np.arange(len(x))%2 == 0)), \
                     patch.object(runner, 'read_artifact', return_value='synthetic'):
                    outputs.append(runner.run(deepcopy(settings), root/f'out_{offset}'))
            for scope in ['train', 'dev']:
                d = json.loads((outputs[0]/'analysis.json').read_text())[scope]
                for name, rows in d['by_method'].items():
                    self.assertAlmostEqual(d['mean'][name]['humanitarian']['macro_f1'],
                                           np.mean([r['humanitarian']['macro_f1'] for r in rows]))
                    self.assertEqual(d['mean'][name]['binary'] is None, scope == 'dev')
                for seed in [0, 1, 2]:
                    with np.load(outputs[0]/scope/f'shuffle_{seed}/logits.npz') as a, \
                         np.load(outputs[1]/scope/f'shuffle_{seed}/logits.npz') as b:
                        np.testing.assert_array_equal(a['informativeness'], b['informativeness'])
                        np.testing.assert_array_equal(a['humanitarian'], b['humanitarian'])
                    with (outputs[0]/scope/f'shuffle_{seed}/predictions.tsv').open() as f:
                        reader = csv.DictReader(f, delimiter='\t')
                        self.assertEqual(len(reader.fieldnames), len(set(reader.fieldnames)))
                        self.assertEqual(len(list(reader)), len(x))


if __name__ == '__main__': unittest.main()
