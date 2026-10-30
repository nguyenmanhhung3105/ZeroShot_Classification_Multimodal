import csv
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from experiments.clip_transductive.drop_class import restrict_task, run, ROOT
from experiments.clip_transductive.run import load_source
from experiments.clip_transductive.test_transductive import fixture, LABELS
from experiments.clip_two_stage.run import decide


class DropClassTests(unittest.TestCase):
    def test_subset_filters_truth_rows_and_exact_score_column_without_mutation(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture(Path(temp))
            source = load_source(temp)
            original = source['scores'][1].copy()
            subset = restrict_task(source, LABELS[-1])
            mask = np.array([r['y_true'] != LABELS[-1] for r in source['rows']])
            self.assertEqual(len(subset['rows']), 17)
            self.assertEqual(len(subset['removed']), 2)
            self.assertEqual(subset['labels'], LABELS[:-1])
            np.testing.assert_array_equal(subset['scores'][0], source['scores'][0][mask])
            np.testing.assert_array_equal(subset['scores'][1], original[mask, :-1])
            np.testing.assert_array_equal(source['scores'][1], original)
            before = decide(*source['scores'], source['info_labels'], source['labels'])[1]
            after = decide(*subset['scores'], subset['info_labels'], subset['labels'])[1]
            np.testing.assert_array_equal(after, before[mask])

    def test_recompute_predictions_replaces_deleted_candidate_not_drop_predicted_rows(self):
        source = {'labels': LABELS, 'info_labels': ['informative', 'not_informative'],
                  'rows': [{'id': 'a', 'y_true': LABELS[1]}, {'id': 'b', 'y_true': LABELS[-1]}],
                  'cohort': [{}, {}],
                  'scores': [np.array([[.8, .1], [.8, .1]]), np.array([[0, .8, 0, 0, 0, 0, 0, .9]] * 2)]}
        self.assertEqual(decide(*source['scores'], source['info_labels'], LABELS)[0], [LABELS[-1]] * 2)
        subset = restrict_task(source, LABELS[-1])
        self.assertEqual(len(subset['rows']), 1)
        self.assertEqual(decide(*subset['scores'], subset['info_labels'], subset['labels'])[0], [LABELS[1]])

    def test_invalid_exclusion_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture(Path(temp)); source = load_source(temp)
            for label in ['not_humanitarian', 'unknown']:
                with self.assertRaises(ValueError): restrict_task(source, label)

    def test_end_to_end_both_methods_same_seven_class_cohort_and_no_overwrite(self):
        config = json.loads((ROOT / 'experiments/clip_transductive/config.json').read_text())
        config.update(grouping_seeds=[0, 1, 2], query_size=6, iterations=2, mm_iterations=50)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source = root / 'source'; source.mkdir(); fixture(source)
            config['source_run'] = str(source)
            output = run(config, LABELS[-1], root / 'output')
            analysis = json.loads((output / 'analysis.json').read_text())
            self.assertEqual(analysis['samples'], 17)
            self.assertEqual(len(analysis['label_order']), 7)
            self.assertEqual(sum(v['support'] for v in analysis['baseline_two_stage']['per_class'].values()), 17)
            ids = []
            for seed in config['grouping_seeds']:
                with (output / f'grouping_seed_{seed}/predictions.tsv').open() as f:
                    rows = list(csv.DictReader(f, delimiter='\t'))
                self.assertEqual(len(rows), 17)
                self.assertTrue(all(r['y_true'] != LABELS[-1] and r['y_pred'] != LABELS[-1] and r['baseline_pred'] != LABELS[-1] for r in rows))
                ids.append([r['id'] for r in rows])
            self.assertEqual(ids[0], ids[1]); self.assertEqual(ids[1], ids[2])
            self.assertEqual(json.loads((output / 'run_manifest.json').read_text())['status'], 'complete')
            with self.assertRaises(FileExistsError): run(config, LABELS[-1], output)


if __name__ == '__main__': unittest.main()
