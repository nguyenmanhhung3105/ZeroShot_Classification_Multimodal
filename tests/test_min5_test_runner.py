"""Runner test chỉ dùng model, metadata và ảnh giả lập trong thư mục tạm."""
import contextlib
import copy
import csv
import io as stdio
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from vlm_project import io, protocol, min5_protocol, fewshot_support_thresholds
from vlm_project import min5_test_runner as runner
import test_min5_protocol as fixtures


class Backend:
    def __init__(self, device):
        self.metadata = {'architecture': 'synthetic', 'pretrained': 'synthetic',
                         'tokenizer': 'synthetic', 'context_length': 77,
                         'dtype': 'torch.float32', 'image_size': (224, 224)}
        self.unloaded = False

    def unload(self):
        self.unloaded = True


class Min5RunnerTests(unittest.TestCase):
    def fixture(self, root):
        helper = fixtures.Min5ProtocolTests()
        source, _, settings, train, _ = helper.fixture(root)
        roots = {s: root / s for s in ['train', 'test']}
        for split, directory in roots.items():
            directory.mkdir(); (directory / 'manifest.json').write_text(split)
        run = io.read_json(source / 'run.json')
        run['specification'].update(C=1., class_weight='balanced', solver='lbfgs', max_iter=2000,
                                    tol=1e-6, shots_per_6way_class=64, methods=['image','title','image_title'],
                                    train_samples=3000, dev_samples=3000,
                                    manifest_sha256={'train': io.sha256(roots['train'] / 'manifest.json'), 'dev': 'a'*64})
        run['feature'] = 'unit image; unit title; concatenated unit image/title divided by sqrt(2)'
        run['model_runtime'] = Backend('cpu').metadata
        io.write_json(source / 'run.json', run)
        io.write_json(source / 'completed.json', {'status': 'complete', 'run_sha256': io.sha256(source / 'run.json')})
        with contextlib.redirect_stdout(stdio.StringIO()):
            review = fewshot_support_thresholds.analyze(source, root / 'review')
        min5_protocol.prepare(source, review, root / 'draft.json', settings)
        min5_protocol.seal(root / 'draft.json', io.sha256(roots['test'] / 'manifest.json'), root / 'sealed.json')
        test = [{'id': f'new{i}', '6way': str(i % 6), 'image': 'unused', 'text': 'unused',
                 'image_sha256': io.digest(['new', i]), 'text_sha256': io.digest(['new text', i])} for i in range(3000)]
        rows = {'train': train, 'test': test}
        def load(directory, split):
            return rows[split], {'split': split, 'source_split': 'train' if split == 'train' else 'test_public',
                                 'manifest_sha256': io.sha256(roots[split] / 'manifest.json')}
        return roots, rows, load, settings

    @staticmethod
    def encode(backend, rows, batch_size, with_titles=False):
        indices = [int(r['id'][3:] if r['id'].startswith('new') else r['id'][1:]) % 6 for r in rows]
        x = np.eye(8)[indices]
        return x, np.roll(x, 1, axis=1)

    def test_draft_and_code_drift_stop_before_any_dataset_or_model(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); roots, rows, load, settings = self.fixture(root)
            with patch.object(runner.data, 'fakeddit', side_effect=AssertionError('dataset accessed')):
                with self.assertRaisesRegex(ValueError, 'draft'):
                    runner.run(root / 'draft.json', roots['train'], roots['test'], root / 'output', backend_factory=self.fail)
                with patch.object(min5_protocol, 'fingerprint', return_value={}):
                    with self.assertRaises(ValueError):
                        runner.run(root / 'sealed.json', roots['train'], roots['test'], root / 'output', backend_factory=self.fail)
            self.assertFalse((root / 'output').exists())

    def test_wrong_manifest_stops_before_loader_and_overlap_before_model(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); roots, rows, load, settings = self.fixture(root)
            with patch.object(runner.data, 'fakeddit', side_effect=AssertionError('loader called')):
                with self.assertRaises(ValueError):
                    runner.run(root / 'sealed.json', roots['test'], roots['train'], root / 'output', backend_factory=self.fail)
            rows['test'][0]['id'] = rows['train'][0]['id']
            with patch.object(runner.data, 'fakeddit', side_effect=load):
                with self.assertRaisesRegex(ValueError, 'overlap'):
                    runner.run(root / 'sealed.json', roots['train'], roots['test'], root / 'output', backend_factory=self.fail)
            self.assertFalse((root / 'output').exists())

    def test_end_to_end_reuses_support_fixed_thresholds_and_unloads(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); roots, rows, load, settings = self.fixture(root)
            backend = Backend('cpu')
            with patch.object(runner.data, 'fakeddit', side_effect=load), \
                 patch.object(runner, 'encode_cohort', side_effect=self.encode) as encode, \
                 contextlib.redirect_stdout(stdio.StringIO()):
                output = runner.run(root / 'sealed.json', roots['train'], roots['test'], root / 'output', backend_factory=lambda _: backend)
            self.assertTrue(backend.unloaded)
            self.assertEqual(encode.call_count, 2)
            actual = protocol.completed_run(output)
            self.assertTrue(actual['test_opened']); self.assertEqual(actual['split'], 'test')
            self.assertEqual(len(actual['summary']), 4)
            self.assertTrue(all(r['seeds'] == 5 for r in actual['summary']))
            audit_spec = importlib.util.spec_from_file_location('runner_audit', ROOT / 'scripts/audit_report_results.py')
            audit = importlib.util.module_from_spec(audit_spec); audit_spec.loader.exec_module(audit)
            audit.ROOT = root.resolve()
            checked = audit.audit(output)
            self.assertEqual(checked['status'], 'passed')
            self.assertEqual(checked['prediction_rows'], 60000)
            lock = io.read_json(root / 'sealed.json')['payload']
            self.assertEqual(io.read_json(output / 'support.json'), lock['support'])
            with (output / 'predictions.tsv').open() as stream:
                predictions = list(csv.DictReader(stream, delimiter='\t'))
            self.assertEqual(len(predictions), 60000)
            by_key = {(c['policy'], c['seed']): c['threshold'] for c in lock['candidates'] if c['decision'] != 'default'}
            for row in predictions:
                if '_target_' in row['method']:
                    policy = 'legacy_6way' if row['method'].startswith('legacy') else 'balanced_binary'
                    self.assertEqual(int(row['predicted']), int(float(row['p_fake']) >= by_key[policy, int(row['seed'])]))
            with np.load(output / 'linear_models.npz', allow_pickle=False) as models, \
                 np.load(output / 'replay_scores.npz', allow_pickle=False) as scores:
                image, title = self.encode(backend, rows['test'], 8)
                joint = np.concatenate([image, title], axis=1) / np.sqrt(2)
                expected = expit(joint @ models['legacy_6way_image_title_s0_coef'].T + models['legacy_6way_image_title_s0_intercept'])[:, 0]
                np.testing.assert_allclose(expected, scores['p_legacy_6way_image_title_s0'][:, 1])

    def test_test_labels_never_influence_fit_or_scores(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); roots, rows, load, settings = self.fixture(root)
            with patch.object(runner.data, 'fakeddit', side_effect=load), \
                 patch.object(runner, 'encode_cohort', side_effect=self.encode), \
                 contextlib.redirect_stdout(stdio.StringIO()):
                first = runner.run(root / 'sealed.json', roots['train'], roots['test'], root / 'output', backend_factory=Backend)
                for row in rows['test']: row['6way'] = '0' if row['6way'] != '0' else '1'
                second = runner.run(root / 'sealed.json', roots['train'], roots['test'], root / 'output', backend_factory=Backend)
            with np.load(first / 'replay_scores.npz', allow_pickle=False) as a, \
                 np.load(second / 'replay_scores.npz', allow_pickle=False) as b:
                self.assertFalse(np.array_equal(a['labels'], b['labels']))
                for key in a.files:
                    if key.startswith('p_'): np.testing.assert_array_equal(a[key], b[key])

    def test_unloads_on_encoding_error_and_runtime_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); roots, rows, load, settings = self.fixture(root)
            for mismatch in [False, True]:
                backend = Backend('cpu')
                if mismatch: backend.metadata['tokenizer'] = 'wrong'
                with patch.object(runner.data, 'fakeddit', side_effect=load), \
                     patch.object(runner, 'encode_cohort', side_effect=RuntimeError('encode failure')):
                    with self.assertRaises((ValueError, RuntimeError)):
                        runner.run(root / 'sealed.json', roots['train'], roots['test'], root / 'output', backend_factory=lambda _: backend)
                self.assertTrue(backend.unloaded)

    def test_prepare_only_cli_never_reads_roots_or_loads_model(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); self.fixture(root)
            p = subprocess.run([sys.executable, '-B', str(ROOT / 'scripts/run_min5_test.py'),
                                '--protocol', str(root / 'draft.json'), '--prepare-only',
                                '--train-root', '/does/not/exist', '--test-root', '/does/not/exist'],
                               capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertIn('draft', p.stdout)
            self.assertFalse((root / 'output').exists())

    def test_candidate_suite_and_feature_drift_are_rejected_before_dataset(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); roots, rows, load, settings = self.fixture(root)
            original = io.read_json(root / 'sealed.json')
            for case in ['empty', 'duplicate', 'threshold', 'rule', 'feature', 'classifier', 'metadata']:
                with self.subTest(case=case):
                    lock = copy.deepcopy(original); p = lock['payload']
                    if case == 'empty': p['candidates'] = []
                    if case == 'duplicate': p['candidates'].append(p['candidates'][0])
                    if case == 'threshold': p['candidates'][1]['threshold'] = 2.
                    if case == 'rule': p['candidates'][0]['rule'] = 'other'
                    if case == 'feature': p['source']['feature'] = 'other'
                    if case == 'classifier': p['source']['specification']['C'] = 10.
                    if case == 'metadata': p['source']['model_runtime'] = {}
                    lock['sha256'] = io.digest(p); io.write_json(root / 'bad.json', lock)
                    with patch.object(runner.data, 'fakeddit', side_effect=AssertionError('dataset accessed')):
                        with self.assertRaises(ValueError):
                            runner.run(root / 'bad.json', roots['train'], roots['test'], root / 'output', backend_factory=self.fail)

    def test_train_identity_wrong_test_metadata_and_output_in_dataset_stop_before_model(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); roots, rows, load, settings = self.fixture(root)
            rows['train'][0]['image_sha256'] = 'c'*64
            with patch.object(runner.data, 'fakeddit', side_effect=load):
                with self.assertRaisesRegex(ValueError, 'Train'):
                    runner.run(root / 'sealed.json', roots['train'], roots['test'], root / 'output', backend_factory=self.fail)
            with patch.object(runner.data, 'fakeddit', side_effect=AssertionError('dataset accessed')):
                with self.assertRaisesRegex(ValueError, 'Output'):
                    runner.run(root / 'sealed.json', roots['train'], roots['test'], roots['train'] / 'output', backend_factory=self.fail)


if __name__ == '__main__': unittest.main()
