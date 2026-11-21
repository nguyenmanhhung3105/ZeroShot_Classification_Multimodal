"""Cascade min5 và flat 6way: toàn bộ dữ liệu/encoder tổng hợp."""
import contextlib
import copy
import csv
import importlib.util
import io as stdio
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.special import expit, softmax

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from vlm_project import io, protocol, fewshot_support_thresholds
from vlm_project import fakeddit_fewshot_cascade as cascade
import test_min5_test_runner as fixtures


class CascadeTests(unittest.TestCase):
    def fixture(self, root):
        roots, rows, _, settings_file = fixtures.Min5RunnerTests().fixture(root)
        roots['dev'] = root / 'dev'; roots['dev'].mkdir()
        (roots['dev'] / 'manifest.json').write_text('dev')
        source = root / 'source'
        run = io.read_json(source / 'run.json')
        run['specification']['manifest_sha256']['dev'] = io.sha256(roots['dev'] / 'manifest.json')
        io.write_json(source / 'run.json', run)
        io.write_json(source / 'completed.json', {'status': 'complete', 'run_sha256': io.sha256(source / 'run.json')})
        rows['dev'] = io.read_json(source / 'cohort.json')
        for r in rows['dev']: r.update(image='unused', text='unused')
        with contextlib.redirect_stdout(stdio.StringIO()):
            review = fewshot_support_thresholds.analyze(source, root / 'review')
        snapshot = root / 'cascade.json'
        cascade.export_specification(source, review, snapshot, settings_file)
        settings = {'model': 'synthetic', 'batch_size': 8, 'device': 'cpu', 'output_root': str(root / 'runs')}
        def load(directory, split, smoke=False):
            self.assertIn(split, ['train', 'dev'])
            selected = rows[split][:5] if smoke else rows[split]
            return selected, {'split': split, 'source_split': 'train' if split == 'train' else 'validate',
                              'manifest_sha256': io.sha256(roots[split] / 'manifest.json')}
        return snapshot, settings, roots, rows, load

    @staticmethod
    def encode(backend, rows, batch_size, with_titles=False):
        x = np.eye(8)[[int(''.join(c for c in r['id'] if c.isdigit())) % 6 for r in rows]]
        return x, np.roll(x, 1, axis=1)

    def test_hard_route_is_not_argmax_of_joint_probabilities_and_ties(self):
        gate = np.array([[.4, .6], [.5, .5], [.9, .1]])
        child = np.array([[.21, .2, .2, .2, .19]] * 3)
        np.testing.assert_array_equal(cascade.route(gate, child), [1, 0, 0])
        np.testing.assert_array_equal(cascade.route(gate, child, .5), [1, 1, 0])
        with self.assertRaises(ValueError): cascade.route(gate, child[:, :4])
        with self.assertRaises(ValueError): cascade.route(gate, child, float('nan'))

    def test_error_decomposition_and_oracle_are_separate_from_methods(self):
        gate = np.array([[.1,.9], [.9,.1], [.1,.9]])
        child = np.eye(5)[[1,0,0]]
        flat = np.eye(6)[[0,1,2]]
        spec = {'audit_ids': ['a','b','c'], 'gate_candidates': [
            {'policy': 'legacy_6way', 'seed': 0, 'decision': 'default', 'threshold': None}]}
        result = cascade.evaluate(np.array([0,1,2]), np.array(['a','b','c']),
                                  {('legacy_6way',0): {'gate':gate, 'child':child, 'flat':flat}}, spec)
        records, predictions, details, comparisons, routing, gates, subsets, oracle, arrays, subtypes = result
        self.assertEqual(routing[0]['real_routed_to_fake'], 1)
        self.assertEqual(routing[0]['fake_routed_to_real'], 1)
        self.assertEqual(routing[0]['subtype_errors_after_correct_fake_gate'], 1)
        self.assertEqual(routing[0]['child_accuracy_all_true_fake'], .5)
        self.assertEqual(routing[0]['child_accuracy_given_correct_fake_gate'], 0.)
        self.assertEqual([r['accuracy'] for r in records], [1., 0.])
        self.assertAlmostEqual(oracle['legacy_6way_s0']['accuracy'], 2/3)
        self.assertFalse(any('oracle' in r['method'] for r in records))

    def test_matched_budget_train_only_heads_and_complete_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); snapshot, settings, roots, rows, load = self.fixture(root)
            backend = fixtures.Backend('cpu')
            with patch.object(cascade.data, 'fakeddit', side_effect=load), \
                 patch.object(cascade, 'encode_cohort', side_effect=self.encode) as encode, \
                 patch.object(cascade, 'probe', wraps=cascade.probe) as fit, \
                 contextlib.redirect_stdout(stdio.StringIO()):
                output = cascade.run(settings, roots['train'], roots['dev'], backend_factory=lambda _: backend, spec_file=snapshot)
            self.assertTrue(backend.unloaded); self.assertEqual(encode.call_count, 2)
            self.assertEqual(fit.call_count, 30)  # 2 policy × 5 seed × (gate,child,flat)
            self.assertEqual([len(c.args[0]) for c in fit.call_args_list].count(384), 20)
            self.assertEqual([len(c.args[0]) for c in fit.call_args_list].count(320), 5)
            self.assertEqual([len(c.args[0]) for c in fit.call_args_list].count(192), 5)
            for c in fit.call_args_list:
                self.assertIn(set(np.unique(c.args[1])), [{0,1}, {1,2,3,4,5}, {0,1,2,3,4,5}])
            run = protocol.completed_run(output)
            self.assertFalse(run['test_opened']); self.assertEqual(run['stages'], 2)
            self.assertEqual(len(run['summary']), 6)
            self.assertTrue(all(r['seeds'] == 5 and r['samples'] == 3000 for r in run['summary']))
            self.assertEqual(io.read_json(output / 'support.json'), cascade.specification(snapshot)['support_snapshot'])
            audit_spec = importlib.util.spec_from_file_location('cascade_audit', ROOT / 'scripts/audit_report_results.py')
            audit = importlib.util.module_from_spec(audit_spec); audit_spec.loader.exec_module(audit); audit.ROOT = root.resolve()
            self.assertEqual(audit.audit(output)['prediction_rows'], 90000)
            with (output / 'routing.tsv').open() as f: routing = list(csv.DictReader(f, delimiter='\t'))
            self.assertEqual(len(routing), 20)
            for row in routing:
                errors = sum(int(row[k]) for k in ['real_routed_to_fake','fake_routed_to_real','subtype_errors_after_correct_fake_gate'])
                with np.load(output / 'replay_scores.npz', allow_pickle=False) as replay:
                    key = f"prediction_{row['policy']}_hierarchy_{row['decision']}_s{row['seed']}"
                    self.assertEqual(errors, int((replay[key] != replay['labels_6way']).sum()))
            with (output / 'subtype_detection.tsv').open() as f:
                subtype = list(csv.DictReader(f, delimiter='\t'))
            self.assertEqual(len(subtype), 120)
            for r in subtype:
                self.assertLessEqual(float(r['end_to_end_recall']), float(r['binary_gate_recall']) + 1e-12)
            with (output / 'audit_metrics_by_seed.tsv').open() as f: subset = list(csv.DictReader(f, delimiter='\t'))
            self.assertTrue(all(int(r['samples']) == 1500 for r in subset))
            oracle = io.read_json(output / 'oracle_diagnostic.json')
            self.assertTrue(oracle['uses_true_binary_labels'])
            self.assertFalse(any('oracle' in r['method'] for r in run['summary']))
            with np.load(output / 'replay_scores.npz', allow_pickle=False) as replay:
                gate = replay['p_gate_legacy_6way_s0']; child = replay['p_child_legacy_6way_s0']
                np.testing.assert_array_equal(cascade.route(gate, child), replay['prediction_legacy_6way_hierarchy_default_s0'])
                with np.load(output / 'linear_models.npz', allow_pickle=False) as models:
                    image, title = self.encode(backend, rows['dev'], 8)
                    joint = np.concatenate([image, title], axis=1) / np.sqrt(2)
                    for head in ['gate','child','flat']:
                        key = f'{head}_legacy_6way_s0'
                        logits = joint @ models[key+'_coef'].T + models[key+'_intercept']
                        if head == 'gate':
                            p = expit(logits[:,0]); expected = np.column_stack([1-p,p])
                        else:
                            expected = softmax(logits, axis=1)
                        np.testing.assert_allclose(expected, replay[f'p_{head}_legacy_6way_s0'])

    def test_dev_labels_do_not_change_any_scores_or_real_method_predictions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); snapshot, settings, roots, rows, load = self.fixture(root)
            with patch.object(cascade.data, 'fakeddit', side_effect=load), \
                 patch.object(cascade, 'encode_cohort', side_effect=self.encode), \
                 contextlib.redirect_stdout(stdio.StringIO()):
                a = cascade.run(settings, roots['train'], roots['dev'], backend_factory=fixtures.Backend, spec_file=snapshot)
                # Hash identity không đổi; loader giả cho phép kiểm label usage của inference.
                for r in rows['dev']: r['6way'] = str((int(r['6way']) + 1) % 6)
                b = cascade.run(settings, roots['train'], roots['dev'], backend_factory=fixtures.Backend, spec_file=snapshot)
            with np.load(a / 'replay_scores.npz', allow_pickle=False) as x, np.load(b / 'replay_scores.npz', allow_pickle=False) as y:
                self.assertFalse(np.array_equal(x['labels_6way'], y['labels_6way']))
                for key in x.files:
                    if key.startswith(('p_', 'prediction_')): np.testing.assert_array_equal(x[key], y[key])

    def test_wrong_hash_overlap_and_test_root_fail_before_model(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); snapshot, settings, roots, rows, load = self.fixture(root)
            with patch.object(cascade.data, 'fakeddit', side_effect=AssertionError('data accessed')):
                with self.assertRaises(ValueError):
                    cascade.run(settings, roots['train'], roots['test'], backend_factory=self.fail, spec_file=snapshot)
            rows['dev'][0]['image_sha256'] = rows['train'][0]['image_sha256']
            with patch.object(cascade.data, 'fakeddit', side_effect=load):
                with self.assertRaises(ValueError):
                    cascade.run(settings, roots['train'], roots['dev'], backend_factory=self.fail, spec_file=snapshot)

    def test_smoke_keeps_all_support_and_unloads_on_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); snapshot, settings, roots, rows, load = self.fixture(root)
            with patch.object(cascade.data, 'fakeddit', side_effect=load), \
                 patch.object(cascade, 'encode_cohort', side_effect=self.encode), \
                 contextlib.redirect_stdout(stdio.StringIO()):
                output = cascade.run(settings, roots['train'], roots['dev'], True, fixtures.Backend, spec_file=snapshot)
            run = protocol.completed_run(output)
            self.assertEqual(run['samples'], 5)
            self.assertEqual([len(e['ids']) for e in io.read_json(output / 'support.json')], [384]*10)
            backend = fixtures.Backend('cpu')
            with patch.object(cascade.data, 'fakeddit', side_effect=load), \
                 patch.object(cascade, 'encode_cohort', side_effect=RuntimeError('encoding')):
                with self.assertRaises(RuntimeError):
                    cascade.run(settings, roots['train'], roots['dev'], True, lambda _: backend, spec_file=snapshot)
            self.assertTrue(backend.unloaded)

    def test_snapshot_tampering_invalid_partition_gate_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); snapshot, settings, roots, rows, load = self.fixture(root)
            original = io.read_json(snapshot)
            for case in ['checksum','test','partition','missing_gate','duplicate_gate','threshold','modality']:
                with self.subTest(case=case):
                    value = copy.deepcopy(original); spec = value['payload']
                    if case == 'checksum': value['sha256'] = 'bad'
                    if case == 'test': spec['split'] = 'test'
                    if case == 'partition': spec['audit_ids'][0] = spec['selection_ids'][0]
                    if case == 'missing_gate': spec['gate_candidates'] = []
                    if case == 'duplicate_gate': spec['gate_candidates'].append(spec['gate_candidates'][0])
                    if case == 'threshold': spec['gate_candidates'][1]['threshold'] = float('nan')
                    if case == 'modality': spec['gate_candidates'][0]['modality'] = 'title'
                    if case != 'checksum': value['sha256'] = io.digest(spec)
                    if case == 'threshold':
                        # NaN không được ghi bởi io.write_json; kiểm schema trực tiếp.
                        with self.assertRaises(ValueError): cascade._check_specification(spec)
                    else:
                        io.write_json(root / 'invalid.json', value)
                        with self.assertRaises(ValueError): cascade.specification(root / 'invalid.json')


if __name__ == '__main__': unittest.main()
