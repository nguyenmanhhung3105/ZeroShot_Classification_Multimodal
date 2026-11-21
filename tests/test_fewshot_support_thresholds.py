"""Ngưỡng chọn trên selection, audit chỉ đánh giá; toàn bộ dữ liệu tổng hợp."""
import contextlib
import csv
import io as stdio
from pathlib import Path
import sys
import tempfile
import unittest
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from vlm_project import io, protocol
from vlm_project import fewshot_support_thresholds as analysis


class ThresholdTests(unittest.TestCase):
    def test_highest_feasible_threshold_handles_ties_and_boundary(self):
        y = np.array([0, 0, 1, 1, 1, 1])
        p = np.array([.9, .1, .8, .8, .4, .2])
        self.assertEqual(analysis.choose_threshold(y, p, .5), .8)
        self.assertEqual(analysis.choose_threshold(y, p, .75), .4)
        self.assertEqual(analysis.choose_threshold(y, p, 1.), .2)
        for target in [0, 1.1, float('nan')]:
            with self.assertRaises(ValueError):
                analysis.choose_threshold(y, p, target)
        with self.assertRaises(ValueError):
            analysis.choose_threshold(np.zeros(6), p, .8)

    def test_split_is_disjoint_stratified_and_invariant_to_input_order(self):
        ids = np.array([f'id{i}' for i in range(60)])
        six = np.repeat(np.arange(6), 10)
        a, b = analysis.split_dev(ids, six, .5, 42)
        self.assertFalse(set(a) & set(b))
        self.assertEqual(set(a) | set(b), set(range(60)))
        np.testing.assert_array_equal(np.bincount(six[a]), [5] * 6)
        perm = np.random.default_rng(1).permutation(60)
        c, d = analysis.split_dev(ids[perm], six[perm], .5, 42)
        self.assertEqual(set(ids[a]), set(ids[perm][c]))
        with self.assertRaises(ValueError):
            analysis.split_dev(ids, six, 1., 42)

    def test_pr_curve_matches_confusion_counts_with_tied_scores(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([.8, .1, .8, .2])
        rows = analysis.precision_recall_rows(y, p)
        self.assertEqual(len(rows), 4)
        for row in rows:
            pred = p >= row['threshold']
            self.assertEqual(row['tp'], int((pred & (y == 1)).sum()))
            self.assertEqual(row['fp'], int((pred & (y == 0)).sum()))
            self.assertEqual(row['recall_fake'], float(pred[y == 1].mean()))
            self.assertEqual(row['recall_real'], float((~pred[y == 0]).mean()))

    def source(self, root):
        source = root / 'source'
        source.mkdir()
        six = np.tile(np.arange(6), 500)
        ids = np.array([f'id{i}' for i in range(3000)])
        io.write_json(source / 'cohort.json', [{'id': i, '6way': int(y)} for i, y in zip(ids.tolist(), six)])
        rng = np.random.default_rng(3)
        scores = {}
        for modality in ['image', 'title', 'image_title']:
            for policy in ['legacy_6way', 'balanced_binary']:
                for seed in range(5):
                    p = rng.uniform(.1, .9, 3000)
                    scores[f'p_{policy}_{modality}_s{seed}'] = np.column_stack([1-p, p])
        np.savez_compressed(source / 'replay_scores.npz', ids=ids, labels=(six != 0).astype(int),
                            class_names=np.array(['Real', 'Fake']), **scores)
        self.seal(source)
        return source, ids, six, scores

    def seal(self, source, **changes):
        run = {'kind': 'fakeddit_fewshot_support', 'status': 'complete', 'mode': 'full',
               'split': 'dev', 'samples': 3000, 'test_opened': False, 'model': 'synthetic',
               'binary_labels': {'0': 'Real', '1': 'Fake'},
               'specification': {'seeds': list(range(5)), 'support_policies': ['legacy_6way', 'balanced_binary']},
               'output_sha256': {name: io.sha256(source / name) for name in ['cohort.json', 'replay_scores.npz']}}
        io.write_json(source / 'run.json', {**run, **changes})
        io.write_json(source / 'completed.json', {'status': 'complete', 'run_sha256': io.sha256(source / 'run.json')})

    def test_end_to_end_only_selection_sets_threshold_and_preserves_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, ids, six, scores = self.source(root)
            before = {p.name: io.sha256(p) for p in source.iterdir()}
            with contextlib.redirect_stdout(stdio.StringIO()):
                output = analysis.analyze(source, root / 'output', targets=[.87])
            run = protocol.completed_run(output)
            self.assertFalse(run['test_opened'])
            self.assertEqual(run['samples'], 1500)
            selection, audit = analysis.split_dev(ids, six, .5, 42)
            with (output / 'thresholds.tsv').open() as f:
                rows = list(csv.DictReader(f, delimiter='\t'))
            self.assertEqual(len(rows), 10)
            for r in rows:
                p = scores[f"p_{r['policy']}_image_title_s{r['seed']}"][:, 1]
                expected = analysis.choose_threshold((six[selection] != 0).astype(int), p[selection], .87)
                self.assertEqual(float(r['threshold']), expected)
            with (output / 'comparisons.tsv').open() as f:
                self.assertEqual(len(list(csv.DictReader(f, delimiter='\t'))), 10)
            self.assertEqual(before, {p.name: io.sha256(p) for p in source.iterdir()})
            # Thay chỉ score audit: ngưỡng vẫn hoàn toàn giữ nguyên.
            for p in scores.values():
                p[audit] = [.99, .01]
            np.savez_compressed(source / 'replay_scores.npz', ids=ids, labels=(six != 0).astype(int),
                                class_names=np.array(['Real', 'Fake']), **scores)
            self.seal(source)
            with contextlib.redirect_stdout(stdio.StringIO()):
                other = analysis.analyze(source, root / 'output', targets=[.87])
            self.assertEqual((output / 'thresholds.tsv').read_bytes(), (other / 'thresholds.tsv').read_bytes())

    def test_reject_tampering_wrong_split_scores_labels_and_output_inside_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, ids, six, scores = self.source(root)
            for changes in [{'split': 'test'}, {'test_opened': True}, {'mode': 'smoke'},
                            {'binary_labels': {'0': 'Fake', '1': 'Real'}}]:
                self.seal(source, **changes)
                with self.assertRaises(ValueError):
                    analysis.analyze(source, root / 'out')
            self.seal(source)
            with self.assertRaises(ValueError):
                analysis.analyze(source, source / 'out')
            (source / 'cohort.json').write_text('[]')
            with self.assertRaises(ValueError):
                analysis.analyze(source, root / 'out')
            self.assertFalse((root / 'out').exists())

    def test_invalid_probability_label_order_duplicate_id_and_missing_method(self):
        for case in ['nan', 'range', 'sum', 'shape', 'labels', 'class_names', 'ids', 'missing']:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                source, ids, six, scores = self.source(root)
                key = 'p_balanced_binary_image_title_s0'
                data = {'ids': ids, 'labels': (six != 0).astype(int),
                        'class_names': np.array(['Real', 'Fake']), **scores}
                if case == 'nan': data[key][0] = [np.nan, np.nan]
                if case == 'range': data[key][0] = [-.1, 1.1]
                if case == 'sum': data[key][0] = [.1, .1]
                if case == 'shape': data[key] = data[key][:-1]
                if case == 'labels': data['labels'] = 1-data['labels']
                if case == 'class_names': data['class_names'] = data['class_names'][::-1]
                if case == 'ids': data['ids'] = data['ids'][::-1]
                if case == 'missing': del data[key]
                np.savez_compressed(source / 'replay_scores.npz', **data)
                self.seal(source)
                with self.assertRaises(ValueError):
                    analysis.analyze(source, root / 'out')
                self.assertFalse((root / 'out').exists())

    def test_options_and_tiny_subtypes_are_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, _, _, _ = self.source(root)
            for options in [{'targets': []}, {'targets': [.87, .87]}, {'targets': [87]},
                            {'targets': [float('inf')]}, {'modalities': ['unknown']},
                            {'modalities': []}, {'modalities': ['image', 'image']},
                            {'selection_fraction': float('nan')}]:
                with self.assertRaises(ValueError):
                    analysis.analyze(source, root / 'out', **options)
            with self.assertRaises(ValueError):
                analysis.split_dev(np.array(list('abcdef')), np.arange(6))
            with self.assertRaises(ValueError):
                analysis.split_dev(np.array(['a'] * 12), np.tile(np.arange(6), 2))

    def test_cli_multiple_targets_modalities_and_no_feature_cache_required(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, _, _, _ = self.source(root)
            result = subprocess.run([sys.executable, '-B', str(ROOT / 'scripts/analyze_fakeddit_support_thresholds.py'),
                                     '--run', str(source), '--output-root', str(root / 'out'),
                                     '--target-recall', '.87', '1', '--modalities', 'image', 'title', 'image_title'],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            output = Path(result.stdout.strip().splitlines()[-1])
            run = protocol.completed_run(output)
            self.assertEqual(len(run['summary']), 18)
            self.assertTrue(all(r['seeds'] == 5 for r in run['summary']))
            self.assertIn('no features or weights read', run['verification_scope'])


if __name__ == '__main__':
    unittest.main()
