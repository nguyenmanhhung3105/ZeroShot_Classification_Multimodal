import csv
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from experiments.clip_transductive.core import em_dirichlet, match_clusters, predict_two_stage, probability_features
from experiments.clip_transductive.run import load_source, run, validate_config, ROOT
from experiments.clip_two_stage.run import decide
from experiments.waffle_lab.run import digest


LABELS = ['not_humanitarian'] + [f'class_{i}' for i in range(7)]
INFO = ['informative', 'not_informative']


def fixture(directory, truth_shift=0):
    rng = np.random.default_rng(23)
    a = np.tile([.6, .1], (19, 1)); a[::4] = [.1, .6]
    b = rng.uniform(-.1, .2, (19, 8))
    baseline, gate = decide(a, b, INFO, LABELS)
    rows = [{'id': f'{i}_0', 'y_true': LABELS[(i + truth_shift) % 8], 'image_sha256': f'hash_{i}',
             'source_split': 'dev', 'label_source': 'image', 'event': f'event_{i % 3}'} for i in range(19)]
    manifest = {'status': 'complete', 'evaluation_scope': 'heldout', 'input': 'image_only', 'label_fit': False,
                'model': 'synthetic_test_only', 'samples': 19, 'seed': 0,
                'label_order': LABELS, 'info_label_order': INFO, 'protocol_fingerprints': [], 'cohort_fingerprints': []}
    for index, task in enumerate(['informativeness', 'humanitarian']):
        sub = directory / task; sub.mkdir(parents=True)
        plan = {'protocol': {'task': task}, 'label_order': [INFO, LABELS][index], 'dataset_config': {'source_split': 'dev'}}
        cohort = {'rows': rows}
        (sub / 'frozen_plan.json').write_text(json.dumps(plan))
        (sub / 'cohort.json').write_text(json.dumps(cohort))
        manifest['protocol_fingerprints'].append(digest(plan['protocol']))
        manifest['cohort_fingerprints'].append(digest(rows))
    (directory / 'run_manifest.json').write_text(json.dumps(manifest))
    with (directory / 'predictions.tsv').open('w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['id', 'y_true', 'y_pred', 'single_stage_pred', 'gate_informative', 'info_cosine_scores', 'human_cosine_scores'])
        for i, row in enumerate(rows):
            writer.writerow([row['id'], row['y_true'], baseline[i], LABELS[b[i].argmax()], bool(gate[i]), json.dumps(a[i].tolist()), json.dumps(b[i].tolist())])
    return a, b, gate


class TransductiveTests(unittest.TestCase):
    def test_features_preserve_argmax_and_extremes_are_finite(self):
        scores = np.array([[1., -1., .01], [-.8, .4, .1]])
        p = probability_features(scores)
        np.testing.assert_allclose(p.sum(1), 1.)
        np.testing.assert_array_equal(p.argmax(1), scores.argmax(1))
        self.assertTrue(np.isfinite(p).all())
        for bad in [np.array([[np.nan, 0]]), np.array([[2., 0]])]:
            with self.assertRaises(ValueError): probability_features(bad)

    def test_probability_validation_and_objective_descent(self):
        p = np.random.default_rng(8).dirichlet(np.ones(7), size=(2, 30))
        pred, log, u = em_dirichlet(p, iterations=5, mm_iterations=100)
        self.assertEqual(pred.shape, (2, 30))
        np.testing.assert_allclose(u.sum(-1), 1.)
        self.assertTrue(np.all(np.diff(np.asarray(log['objective']), axis=0) <= 1e-6))
        with self.assertRaises(ValueError): em_dirichlet(np.ones((1, 4, 7)))
        with self.assertRaises(ValueError): em_dirichlet(p, iterations=0)

    def test_hungarian_uses_occupied_clusters_without_forcing_all_classes(self):
        query = np.array([[[.1, .8, .1], [.1, .7, .2], [.8, .1, .1]]])
        u = np.array([[[1., 0, 0], [1., 0, 0], [0, 0, 1.]]])
        pred, mapping = match_clusters(query, u)
        self.assertEqual(pred.tolist(), [[1, 1, 0]])
        self.assertEqual(len(mapping[0]), 2)

    def test_query_permutation_equivariance_with_no_matching_ties(self):
        rng = np.random.default_rng(42)
        p = rng.dirichlet(np.ones(7), size=(1, 35))
        order = rng.permutation(35)
        first, _, _ = em_dirichlet(p, iterations=3, mm_iterations=60)
        second, _, _ = em_dirichlet(p[:, order], iterations=3, mm_iterations=60)
        np.testing.assert_array_equal(first[:, order], second)

    def test_gate_membership_remainder_and_deterministic_seed(self):
        a = np.tile([.8, .2], (12, 1)); a[::4] = [.2, .8]
        b = np.random.default_rng(4).uniform(-.1, .2, (12, 8))
        first = predict_two_stage(a, b, INFO, LABELS, query_size=4, iterations=2, mm_iterations=50)
        second = predict_two_stage(a, b, INFO, LABELS, query_size=4, iterations=2, mm_iterations=50)
        self.assertEqual(first[0], second[0]); np.testing.assert_array_equal(first[1], second[1])
        gate = a[:, 0] > a[:, 1]
        self.assertTrue(np.all(first[1][~gate] == -1))
        self.assertTrue(np.all(first[1][gate] >= 0))
        self.assertEqual([first[0][i] for i in np.flatnonzero(~gate)], ['not_humanitarian'] * 3)
        self.assertEqual(sorted(np.bincount(first[1][gate]).tolist()), [1, 4, 4])
        self.assertTrue(any(log['singleton_fallback'] for log in first[2]))
        a[:] = [.2, .8]
        empty = predict_two_stage(a, b, INFO, LABELS, iterations=1, mm_iterations=1)
        self.assertEqual(empty[0], ['not_humanitarian'] * 12)
        self.assertEqual(empty[2], [])

    def test_artifact_validation_refuses_corruption_and_test_split(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root)
            load_source(root)
            path = root / 'predictions.tsv'
            original = path.read_text()
            path.write_text(original.replace('hash_missing', 'bad') + original.splitlines()[-1] + '\n')
            with self.assertRaises(ValueError): load_source(root)
            path.write_text(original)
            plan_path = root / 'humanitarian/frozen_plan.json'
            plan = json.loads(plan_path.read_text()); plan['dataset_config']['source_split'] = 'test'
            plan_path.write_text(json.dumps(plan))
            with self.assertRaises(ValueError): load_source(root)

    def test_end_to_end_truth_changes_metrics_only_and_refuses_overwrite(self):
        settings = json.loads((ROOT / 'experiments/clip_transductive/config.json').read_text())
        settings.update(grouping_seeds=[0], query_size=6, iterations=2, mm_iterations=50)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for index in range(2):
                source = root / f'source_{index}'; source.mkdir()
                fixture(source, truth_shift=index)
                settings['source_run'] = str(source)
                out = run(settings, root / f'out_{index}')
                self.assertEqual(json.loads((out / 'run_manifest.json').read_text())['status'], 'complete')
            predictions = []
            for index in range(2):
                with (root / f'out_{index}/grouping_seed_0/predictions.tsv').open() as f:
                    predictions.append([(r['id'], r['y_pred'], r['batch_id']) for r in csv.DictReader(f, delimiter='\t')])
            self.assertEqual(predictions[0], predictions[1])
            with self.assertRaises(FileExistsError): run(settings, root / 'out_1')

    def test_invalid_seed_protocol_and_config_are_rejected(self):
        config = json.loads((ROOT / 'experiments/clip_transductive/config.json').read_text())
        for update in [dict(grouping_seeds=[0, 0]), dict(grouping_seeds=[True]), dict(label_fit=True),
                       dict(query_size=1), dict(temperature=float('nan'))]:
            with self.assertRaises(ValueError): validate_config({**config, **update})


if __name__ == '__main__': unittest.main()
