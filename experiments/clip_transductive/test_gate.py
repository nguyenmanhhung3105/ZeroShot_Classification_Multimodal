import csv
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from experiments.clip_transductive.core import em_dirichlet, probability_features
from experiments.clip_transductive.gate import predict_binary, predict_two_stage_gate, run, ROOT
from experiments.clip_transductive.test_transductive import fixture, INFO, LABELS
from experiments.waffle_lab.run import digest


def build_sources(root):
    human = root / 'human'; human.mkdir()
    a, _, _ = fixture(human)
    prompt_bank = {'informative': ['useful photo'], 'not_informative': ['irrelevant photo']}
    path = human / 'informativeness/frozen_plan.json'
    plan = json.loads(path.read_text()); plan['prompt_banks'] = {'semantic': prompt_bank}
    path.write_text(json.dumps(plan))
    binary = root / 'binary'; binary.mkdir()
    rows = [{'id': f'train_{i}_0', 'y_true': INFO[i % 2], 'source_split': 'train',
             'label_source': 'image', 'image_sha256': f'hash{i}', 'event': 'synthetic'} for i in range(len(a))]
    plan = {'protocol': {'test': 'binary'}, 'label_order': INFO,
            'dataset_config': {'source_split': 'train'}, 'prompt_banks': {'semantic': prompt_bank}}
    manifest = {'status': 'complete', 'inference_backend': 'metaclip2_real', 'task': 'informativeness',
                'variant': 'semantic', 'label_fit': False, 'model': 'synthetic_test_only', 'samples': len(a),
                'cohort_fingerprint': digest(rows), 'protocol_fingerprint': digest(plan['protocol'])}
    for name, value in [('frozen_plan.json', plan), ('run_manifest.json', manifest), ('cohort.json', {'rows': rows})]:
        (binary / name).write_text(json.dumps(value))
    with (binary / 'predictions.tsv').open('w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['id', 'y_true', 'y_pred', 'cosine_scores'])
        for r, scores in zip(rows, a):
            writer.writerow([r['id'], r['y_true'], INFO[scores.argmax()], json.dumps(scores.tolist())])
    return binary, human


class GateTests(unittest.TestCase):
    def test_binary_objective_and_author_lambda_for_two_classes(self):
        p = np.random.default_rng(8).dirichlet([2, 3], size=(2, 30))
        _, log, u = em_dirichlet(p, iterations=4, mm_iterations=60)
        self.assertEqual(log['lambda_over_n'], 0)
        self.assertTrue(np.all(np.diff(np.asarray(log['objective']), axis=0) <= 1e-6))
        np.testing.assert_allclose(u.sum(-1), 1.)

    def test_all_samples_are_grouped_and_singleton_is_preserved(self):
        scores = np.random.default_rng(21).uniform(-.2, .2, (11, 2))
        predicted, batches, logs = predict_binary(scores, INFO, query_size=5, iterations=2, mm_iterations=40)
        self.assertEqual(len(predicted), 11)
        self.assertTrue(set(predicted) <= set(INFO))
        self.assertEqual(sorted(np.bincount(batches).tolist()), [1, 5, 5])
        singleton = np.flatnonzero(batches == batches.max())
        self.assertEqual(predicted[singleton[0]], INFO[scores[singleton[0]].argmax()])
        self.assertTrue(any(block['singleton_fallback'] for block in logs))
        second = predict_binary(scores, INFO, query_size=5, iterations=2, mm_iterations=40)
        self.assertEqual(predicted, second[0]); np.testing.assert_array_equal(batches, second[1])

    def test_humanitarian_branch_uses_original_argmax_and_no_transductive(self):
        rng = np.random.default_rng(12)
        info = rng.uniform(-.1, .2, (23, 2)); human = rng.uniform(-.2, .3, (23, 8))
        pred, binary, _, _ = predict_two_stage_gate(info, human, INFO, LABELS, query_size=9, iterations=2, mm_iterations=40)
        expected = np.asarray(LABELS)[1 + human[:, 1:].argmax(1)]
        expected[np.asarray(binary) == 'not_informative'] = 'not_humanitarian'
        self.assertEqual(pred, expected.tolist())
        swapped = predict_two_stage_gate(info[:, ::-1], human, INFO[::-1], LABELS,
                                         query_size=9, iterations=2, mm_iterations=40)
        self.assertEqual(pred, swapped[0])

    def test_bad_input_and_taxonomy_rejected(self):
        for labels in [['a', 'b'], ['informative']]:
            with self.assertRaises(ValueError): predict_binary(np.zeros((2, 2)), labels)
        with self.assertRaises(ValueError): predict_binary(np.zeros((2, 3)), INFO)
        with self.assertRaises(ValueError): predict_binary(np.zeros((2, 2)), INFO, query_size=1)

    def test_end_to_end_keeps_train_and_dev_metrics_separate(self):
        config = json.loads((ROOT / 'experiments/clip_transductive/config_gate.json').read_text())
        config.update(grouping_seeds=[0], query_size=6, iterations=2, mm_iterations=40)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); binary, human = build_sources(root)
            config.update(binary_source_run=str(binary), humanitarian_source_run=str(human))
            out = run(config, root / 'output')
            analysis = json.loads((out / 'analysis.json').read_text())
            self.assertIsNone(analysis['humanitarian_dev']['informativeness_metric'])
            self.assertEqual(len(analysis['humanitarian_dev']['label_order']), 8)
            self.assertEqual(len(analysis['binary_train']['label_order']), 2)
            self.assertEqual(json.loads((out / 'run_manifest.json').read_text())['status'], 'complete')
            with self.assertRaises(FileExistsError): run(config, out)
            plan_file = human / 'informativeness/frozen_plan.json'
            plan = json.loads(plan_file.read_text()); plan['prompt_banks']['semantic'] = {}
            plan_file.write_text(json.dumps(plan))
            with self.assertRaises(ValueError): run(config, root / 'rejected')


if __name__ == '__main__': unittest.main()
