import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from experiments.clip_transductive.confidence_guard import fit_label_free_threshold, apply_guard, run, ROOT
from experiments.clip_transductive.gate import run as run_gate
from experiments.clip_transductive.test_gate import build_sources


class GuardTests(unittest.TestCase):
    def test_threshold_uses_scores_only_and_is_invariant_to_row_and_label_order(self):
        scores = np.array([[.6, .1], [.2, .1], [.4, .1], [.1, .8]])
        labels = ['informative', 'not_informative']
        plan = fit_label_free_threshold(scores, labels)
        self.assertAlmostEqual(plan['threshold'], .3)
        self.assertFalse(plan['uses_ground_truth'])
        self.assertEqual(plan, fit_label_free_threshold(scores[::-1, ::-1], labels[::-1]))
        with self.assertRaises(ValueError): fit_label_free_threshold(np.array([[.1, .8]]), labels)

    def test_guard_restores_only_large_positive_margin_and_does_not_mutate(self):
        scores = np.array([[.6, .1], [.2, .1], [.1, .8], [.3, .1]])
        trans = ['not_informative', 'not_informative', 'informative', 'informative']
        guarded, restored = apply_guard(scores, ['informative', 'not_informative'], trans, .3)
        self.assertEqual(guarded, ['informative', 'not_informative', 'informative', 'informative'])
        self.assertEqual(restored.tolist(), [True, False, False, False])
        self.assertEqual(trans[0], 'not_informative')
        with self.assertRaises(ValueError): apply_guard(scores, ['informative', 'not_informative'], trans, float('nan'))

    def test_end_to_end_freezes_threshold_and_preserves_cohort(self):
        gate_config = json.loads((ROOT/'experiments/clip_transductive/config_gate.json').read_text())
        guard_config = json.loads((ROOT/'experiments/clip_transductive/config_guard.json').read_text())
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); binary, human = build_sources(root)
            gate_config.update(binary_source_run=str(binary), humanitarian_source_run=str(human),
                               grouping_seeds=[0], query_size=6, iterations=2, mm_iterations=40)
            previous = run_gate(gate_config, root/'gate')
            guard_config.update(gate_run=str(previous), grouping_seeds=[0])
            out = run(guard_config, root/'guard')
            plan = json.loads((out/'frozen_guard_plan.json').read_text())
            self.assertFalse(plan['uses_ground_truth'])
            report = json.loads((out/'analysis.json').read_text())
            self.assertEqual(report['binary_train']['samples'], 19)
            self.assertEqual(report['humanitarian_dev']['samples'], 19)
            self.assertEqual(json.loads((out/'run_manifest.json').read_text())['status'], 'complete')
            with self.assertRaises(FileExistsError): run(guard_config, out)


if __name__ == '__main__': unittest.main()
