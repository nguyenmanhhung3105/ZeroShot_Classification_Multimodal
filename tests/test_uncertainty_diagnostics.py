"""Kiểm tra cổng đồng thuận và phép đếm top-2 bằng log giả."""

import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/uncertainty"))
from compare_consensus_gate import MODELS, compare
from compare_prompt_stability import compare as compare_prompt_stability
from diagnose_single_model import diagnose


class UncertaintyDiagnosticsTests(unittest.TestCase):
    def test_consensus_changes_only_low_margin_case(self):
        truth = ("1", "0")
        probabilities = ((.52, .9), (.1, .1), (.1, .1))
        runs = {}
        for model, scores in zip(MODELS, probabilities):
            rows = []
            for i, p0 in enumerate(scores):
                rows.append({"id": str(i), "text": "same", "image_path": f"{i}.jpg",
                             "y_true": truth[i], "y_pred": "0" if p0 > .5 else "1",
                             "score_0": str(p0), "score_1": str(1 - p0)})
            runs[model] = ({"dataset": "fakeddit", "task": "2way", "n_total_samples": 2,
                            "sample_set_id": "same"}, rows)
        report, details = compare(runs)
        self.assertEqual(report["methods"]["aux_consensus_low_margin"]["wrong_to_right"], 1)
        self.assertEqual(report["methods"]["aux_consensus_low_margin"]["right_to_wrong"], 0)
        self.assertEqual(report["methods"]["aux_consensus_all"]["right_to_wrong"], 1)
        self.assertEqual([row["low_margin"] for row in details], [True, False])

    def test_top2_counts_only_true_second_choice(self):
        labels = ("a", "b", "c")
        scores = ((.6, .3, .1), (.6, .1, .3), (.6, .3, .1))
        truth = ("b", "b", "a")
        rows = []
        for i, (prob, actual) in enumerate(zip(scores, truth)):
            row = {"y_true": actual, "y_pred": labels[max(range(3), key=lambda j: prob[j])]}
            for j, label in enumerate(labels):
                row[f"score_{label}"] = str(prob[j])
                row[f"score_image_{label}"] = str(prob[j])
                row[f"score_text_{label}"] = str(prob[j])
            rows.append(row)
        report = diagnose({"dataset": "crisismmd", "task": "humanitarian",
                           "model": "metaclip2_vith14", "n_total_samples": 3}, rows)
        self.assertEqual(report["n_errors"], 2)
        self.assertEqual(report["top2_contains_true"], 2)
        self.assertEqual(report["errors_with_true_as_second_choice"], 1)

    def test_prompt_comparison_checks_rows_even_when_manifest_ids_differ(self):
        def row(identifier, truth, p0):
            return {"id": identifier, "text": "same", "image_path": identifier + ".jpg",
                    "y_true": truth, "y_pred": "0" if p0 > .5 else "1",
                    "score_0": str(p0), "score_1": str(1 - p0),
                    "score_image_0": str(p0), "score_image_1": str(1 - p0)}
        a = [row("a", "1", .6), row("b", "0", .7)]
        b = [row("a", "1", .4), row("b", "0", .7)]
        left = {"dataset": "crisismmd", "task": "informativeness",
                "model": "metaclip2_vith14", "sample_set_id": "old"}
        right = {**left, "sample_set_id": "new"}
        report = compare_prompt_stability(left, a, right, b)
        self.assertFalse(report["sample_set_id_equal"])
        self.assertEqual(report["wrong_to_right"], 1)
        b[1]["image_path"] = "different.jpg"
        with self.assertRaisesRegex(ValueError, "Thứ tự mẫu"):
            compare_prompt_stability(left, a, right, b)


if __name__ == "__main__":
    unittest.main()
