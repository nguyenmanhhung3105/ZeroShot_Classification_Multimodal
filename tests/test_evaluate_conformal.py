"""Kiểm tra lượng tử conformal và việc không dùng nhãn test để tạo tập nhãn."""

import csv
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments/uncertainty"))
from evaluate_conformal import analyze, conformal_threshold


def _write_run(root, run_id, flip_first_truth=False):
    run = root / run_id
    run.mkdir()
    table = io.StringIO()
    fields = ("id", "text", "y_true", "y_pred", "score_0", "score_1")
    writer = csv.DictWriter(table, fieldnames=fields, delimiter="\t")
    writer.writeheader()
    for i in range(20):
        p1 = 0.25 + (i % 10) * 0.055
        truth = i % 2
        if i == 0 and flip_first_truth:
            truth = 1 - truth
        writer.writerow(dict(id=i, text=f"sample {i}", y_true=truth,
                             y_pred=int(p1 > 0.5), score_0=1 - p1, score_1=p1))
    prefix = "raw_predictions/metaclip2_vith14__synthetic"
    with zipfile.ZipFile(run / "run_artifacts.zip", "w") as archive:
        archive.writestr(f"{prefix}/raw_predictions.tsv", table.getvalue())
        archive.writestr(f"{prefix}/manifest.json", json.dumps({
            "model": "metaclip2_vith14", "dataset": "fakeddit", "task": "2way",
            "n_total_samples": 20, "split_verified": False}))
    (run / "run_receipt.json").write_text(json.dumps({"run_id": run_id}))
    return run


class ConformalTests(unittest.TestCase):
    def test_finite_sample_quantile_and_small_sample_case(self):
        self.assertEqual(conformal_threshold([0.1, 0.2, 0.3, 0.4], 0.25), (0.4, 4))
        q, k = conformal_threshold([0.1, 0.2, 0.3, 0.4], 0.10)
        self.assertTrue(math.isinf(q))
        self.assertEqual(k, 5)
        with self.assertRaises(ValueError):
            conformal_threshold([0.1], 0)

    def test_classwise_held_out_set_does_not_look_at_its_truth(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = _write_run(root, "run_a")
            second = _write_run(root, "run_b", flip_first_truth=True)
            output_a, report_a = analyze(first, alpha=0.2, output_dir=root / "out_a",
                                         mode="classwise")
            output_b, report_b = analyze(second, alpha=0.2, output_dir=root / "out_b",
                                         mode="classwise")
            self.assertEqual(report_a["n_samples"], 20)
            self.assertEqual(report_b["n_samples"], 20)
            with (output_a / "prediction_sets.tsv").open(newline="") as stream:
                row_a = next(csv.DictReader(stream, delimiter="\t"))
            with (output_b / "prediction_sets.tsv").open(newline="") as stream:
                row_b = next(csv.DictReader(stream, delimiter="\t"))
            self.assertNotEqual(row_a["y_true"], row_b["y_true"])
            self.assertEqual(row_a["fold"], row_b["fold"])
            self.assertEqual(row_a["prediction_set"], row_b["prediction_set"])
            self.assertEqual(row_a["needs_review"], row_b["needs_review"])
            with (output_a / "folds.tsv").open(newline="") as stream:
                self.assertEqual(sum(int(row["n_held_out"])
                                     for row in csv.DictReader(stream, delimiter="\t")), 20)
            self.assertTrue(0 <= report_a["empirical_label_coverage"] <= 1)


if __name__ == "__main__":
    unittest.main()
