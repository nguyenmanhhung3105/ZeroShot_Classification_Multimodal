"""Kiểm tra Task 1 bằng vector tổng hợp; không API, ảnh thật hoặc weights."""
import contextlib
import copy
import io as stdio
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
from vlm_project import fakeddit_task1 as task1
from vlm_project import io, kaggle, protocol


class Backend:
    def __init__(self, device):
        self.metadata = {'synthetic': True}
        self.unloaded = False

    def prototypes(self, bank):
        return np.eye(2, 4, dtype=np.float32)

    def unload(self):
        self.unloaded = True


class Task1Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.settings = io.config()
        self.settings.update(device='cpu', output_root=str(self.root / 'runs'))
        self.spec = task1.specification()
        self.features = task1.unit(np.random.default_rng(8).normal(size=(5, 4))).astype(np.float32)
        self.titles = self.features[::-1].copy()

    def test_formula_group_seeds_and_aligned_posteriors_without_labels(self):
        prototypes = np.eye(2, 4)
        seen = []
        def groups(features, prototypes, initial, spec):
            seen.append(spec['grouping_seed'])
            return np.tile([.2, .8], (len(features), 1)), np.zeros(len(features), dtype=int)
        with patch.object(task1, 'infer_groups', side_effect=groups):
            results, ids = task1.predict(self.features, self.titles, prototypes, prototypes, self.spec)
        self.assertEqual(seen, [0, 1, 2])
        values = {(name, seed): p for name, seed, p in results}
        clip = softmax(100 * task1.unit(self.features) @ prototypes.T, axis=1)
        title = softmax(25 * task1.unit(self.titles) @ prototypes.T, axis=1)
        np.testing.assert_allclose(values['clip_image', None], clip, atol=1e-7)
        np.testing.assert_allclose(values['clip_title', None], title, atol=1e-7)
        np.testing.assert_allclose(values['clip_image_title', None], .75*clip+.25*title, atol=1e-7)
        for seed in seen:
            expected = .5*clip+.5*np.tile([.2, .8], (5, 1))
            np.testing.assert_allclose(values['clip_stata_image', seed], expected, atol=1e-7)
            np.testing.assert_allclose(values['clip_stata_image_title', seed], .75*expected+.25*title, atol=1e-7)
        self.assertEqual(len(results), 12)
        self.assertEqual(set(ids), {'groups_s0', 'groups_s1', 'groups_s2'})

    def test_real_stata_runner_replay_label_independence_and_no_freeze(self):
        rows = [{'id': str(i), '2way': str(i % 2), 'text': 'synthetic', 'image': 'unused'} for i in range(5)]
        backend = Backend('cpu')
        with patch.object(task1.data, 'fakeddit', return_value=(rows, {'split': 'dev'})), \
             patch.object(task1, 'encode_cohort', return_value=(self.features, self.titles)) as encode, \
             contextlib.redirect_stdout(stdio.StringIO()):
            output = task1.run(self.settings, 'dev', self.root, smoke=True, backend_factory=lambda _: backend)
            for row in rows:
                row['2way'] = str(1-int(row['2way']))
            other = task1.run(self.settings, 'dev', self.root, smoke=True, backend_factory=Backend)
        self.assertTrue(backend.unloaded)
        self.assertEqual(encode.call_count, 2)  # Mỗi run chỉ encode chung một lần.
        run = protocol.completed_run(output)
        self.assertEqual(run['classes'], ['fake', 'real'])
        self.assertEqual(run['stages'], 1)
        self.assertEqual(len(run['summary']), 6)
        with np.load(output / 'replay_scores.npz', allow_pickle=False) as scores, \
             np.load(other / 'replay_scores.npz', allow_pickle=False) as relabeled:
            for name in scores.files:
                if name.startswith('p_'):
                    np.testing.assert_array_equal(scores[name], relabeled[name])
            for summary in run['summary']:
                seeds = [None] if summary['method'] in ['clip_image', 'clip_title', 'clip_image_title'] else [0, 1, 2]
                accuracies = []
                for seed in seeds:
                    key = 'p_' + summary['method'] + ('' if seed is None else f'_s{seed}')
                    accuracies.append((scores[key].argmax(1) == scores['labels']).mean())
                self.assertAlmostEqual(summary['accuracy_mean'], np.mean(accuracies))
                self.assertAlmostEqual(summary['accuracy_std'], np.std(accuracies, ddof=1) if len(seeds) > 1 else 0.)
                self.assertNotIn('top5_mean', summary)
        with self.assertRaisesRegex(ValueError, 'dev hoàn tất'):
            protocol.freeze(self.settings, output)

    def test_full_runner_3000_samples_without_real_data_or_adaptation(self):
        rows = [{'id': str(i), '2way': str(i % 2)} for i in range(3000)]
        features = np.tile(self.features[:1], (3000, 1))
        with patch.object(task1.data, 'fakeddit', return_value=(rows, {'split': 'dev'})), \
             patch.object(task1, 'encode_cohort', return_value=(features, features)), \
             patch.object(task1, 'infer_groups', side_effect=lambda x, p, c, s: (c, np.arange(len(x)) // 75)), \
             contextlib.redirect_stdout(stdio.StringIO()):
            output = task1.run(self.settings, 'dev', self.root, backend_factory=Backend)
        run = protocol.completed_run(output)
        self.assertEqual(run['samples'], 3000)
        self.assertEqual(run['mode'], 'full')
        self.assertEqual([r['seeds'] for r in run['summary']], [1, 1, 1, 3, 3, 3])

    def test_split_guard_before_data_model_and_unload_on_encoding_error(self):
        with patch.object(task1.data, 'fakeddit', side_effect=AssertionError('no read')):
            for split in ['train', 'test']:
                with self.assertRaisesRegex(ValueError, 'chỉ dùng dev'):
                    task1.run(self.settings, split, self.root)
        backend = Backend('cpu')
        rows = [{'id': str(i), '2way': '0'} for i in range(5)]
        with patch.object(task1.data, 'fakeddit', return_value=(rows, {})), \
             patch.object(task1, 'encode_cohort', side_effect=RuntimeError('encoding failed')):
            with self.assertRaisesRegex(RuntimeError, 'encoding failed'):
                task1.run(self.settings, 'dev', self.root, smoke=True, backend_factory=lambda _: backend)
        self.assertTrue(backend.unloaded)

    def test_launcher_custom_config_reaches_notebook_and_dataset_lookup(self):
        launcher = yaml.safe_load((ROOT / 'configs/kaggle_launcher_min5.yaml').read_text())
        real_path = kaggle.path
        with patch.object(kaggle, 'path', side_effect=lambda p: self.root / 'stage' if str(p) == 'results/_system/launches/final' else real_path(p)), \
             patch.object(kaggle, 'cli', side_effect=AssertionError('no API')):
            plan = kaggle.prepare(launcher, 'fakeddit_task1_dev_min5')
        doc = io.read_json(Path(plan['root']) / 'kernel/run.ipynb')
        setup = ''.join(doc['cells'][2]['source'])
        self.assertIn("settings = config('configs/project_min5.yaml')", setup)
        self.assertNotIn('settings = config()', setup)
        line = next(line for line in setup.splitlines() if line.startswith('settings = config('))
        namespace = {'config': io.config}
        exec(line, namespace)
        settings = namespace['settings']
        self.assertEqual(settings['kaggle']['fakeddit_slugs']['dev'], 'vlm-fakeddit-dev-min5-v1')
        self.assertEqual(plan['data_dataset'], 'nguyenhung3105/' + settings['kaggle']['fakeddit_slugs']['dev'])
        mount = self.root / 'vlm-fakeddit-dev-min5-v1'
        mount.mkdir()
        (mount / 'fakeddit_sample.tsv').touch()
        with patch.object(io, 'ROOT', self.root / 'empty_project'), \
             patch.object(io, 'input_roots', return_value=[mount]):
            self.assertEqual(io.locate_data('fakeddit', 'dev', settings), mount.resolve())

    def test_launcher_smoke_full_offline_notebook_and_invalid_split(self):
        launcher = yaml.safe_load((ROOT / 'configs/kaggle_launcher.yaml').read_text())
        original = copy.deepcopy(launcher)
        real_path = kaggle.path
        with patch.object(kaggle, 'path', side_effect=lambda p: self.root / 'stage' if str(p) == 'results/_system/launches/final' else real_path(p)), \
             patch.object(kaggle, 'cli', side_effect=AssertionError('no API')):
            for limit in [5, None]:
                plan = kaggle.prepare(launcher, 'fakeddit_task1_dev', limit)
                doc = io.read_json(Path(plan['root']) / 'kernel/run.ipynb')
                for cell in doc['cells']:
                    if cell['cell_type'] == 'code':
                        compile(''.join(cell['source']), '<task1 notebook>', 'exec')
                code = '\n'.join(''.join(c['source']) for c in doc['cells'] if c['cell_type'] == 'code')
                self.assertEqual(code.count('output = run('), 1)
                self.assertNotIn('freeze(settings', code)
                self.assertIn('from vlm_project.fakeddit_task1 import run', code)
                self.assertEqual(plan['samples'], 5 if limit else 3000)
                self.assertEqual(plan['data_dataset'], 'nguyenhung3105/vlm-fakeddit-dev-v2')
                self.assertEqual(plan['launcher']['timeout_seconds'], 7200)
                self.assertIn('dev-task1-single-stage', plan['kernel'])
                self.assertEqual(plan['task1_specification']['grouping_seeds'], [0, 1, 2])
            for split in ['test', 'train']:
                invalid = copy.deepcopy(launcher)
                invalid['jobs']['fakeddit_task1_dev']['split'] = split
                with self.assertRaisesRegex(ValueError, 'chỉ hỗ trợ Fakeddit dev'):
                    kaggle.prepare(invalid, 'fakeddit_task1_dev')
        self.assertEqual(launcher, original)


if __name__ == '__main__':
    unittest.main()
