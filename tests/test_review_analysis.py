"""Cờ xem lại chỉ phụ thuộc vào hai nhánh, không nhìn nhãn thật."""

import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from experiments.uncertainty.analyze_review import analyze


class ReviewAnalysisTests(unittest.TestCase):
    def test_flags_branch_disagreement_and_reports_both_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "aaaaaaaaaaaa"
            run.mkdir()
            (run / "run_receipt.json").write_text(json.dumps({"run_id": run.name,
                                                               "job_id": "adapted_crisismmd_info"}))
            fields = ("id", "y_true", "y_pred", "margin", "score_A", "score_B",
                      "score_image_A", "score_image_B", "score_text_A", "score_text_B")
            examples = [
                ("1", "A", "A", .8, .2, .8, .2),  # đồng thuận, đúng
                ("2", "B", "A", .8, .2, .2, .8),  # bất đồng, sai
                ("3", "A", "A", .2, .8, .8, .2),  # bất đồng, đúng
                ("4", "B", "A", .8, .2, .8, .2),  # đồng thuận, sai
            ]
            table = io.StringIO()
            writer = csv.DictWriter(table, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            for sample_id, truth, predicted, ia, ib, ta, tb in examples:
                writer.writerow(dict(id=sample_id, y_true=truth, y_pred=predicted, margin="0.2",
                                     score_A="0.6", score_B="0.4", score_image_A=ia,
                                     score_image_B=ib, score_text_A=ta, score_text_B=tb))
            prefix = "raw_predictions/metaclip2__test"
            with zipfile.ZipFile(run / "run_artifacts.zip", "w") as archive:
                archive.writestr(f"{prefix}/raw_predictions.tsv", table.getvalue())
                archive.writestr(f"{prefix}/manifest.json", json.dumps({
                    "n_total_samples": 4, "sample_set_id": "sample-1",
                    "model": "metaclip2", "dataset": "crisismmd", "task": "informativeness"}))
            target, summary = analyze(run, Path(directory) / "analysis")
            self.assertEqual(summary["n_automatic"], 2)
            self.assertEqual(summary["n_review"], 2)
            self.assertEqual(summary["n_errors_automatic"], 1)
            self.assertEqual(summary["n_errors_review"], 1)
            with (target / "review_flags.tsv").open(newline="") as stream:
                flags = list(csv.DictReader(stream, delimiter="\t"))
            self.assertEqual([row["needs_review"] for row in flags],
                             ["false", "true", "true", "false"])
            self.assertEqual([row["correct"] for row in flags],
                             ["true", "false", "true", "false"])
            self.assertFalse(json.loads((target / "analysis_manifest.json").read_text())[
                "uses_true_labels_for_flag"])


if __name__ == "__main__":
    unittest.main()
