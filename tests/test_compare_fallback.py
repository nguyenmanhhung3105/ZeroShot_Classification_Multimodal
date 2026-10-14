"""Kiểm tra phép ghép log và việc chỉ chuyển các mẫu needs_review."""

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
from experiments.uncertainty.compare_fallback import compare


def _write_run(root, run_id, model, rows):
    run = root / run_id
    run.mkdir()
    table = io.StringIO()
    writer = csv.DictWriter(table, fieldnames=("id", "text", "image_path", "y_true", "y_pred"),
                            delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)
    prefix = f"raw_predictions/{model}__synthetic"
    with zipfile.ZipFile(run / "run_artifacts.zip", "w") as archive:
        archive.writestr(f"{prefix}/raw_predictions.tsv", table.getvalue())
        archive.writestr(f"{prefix}/manifest.json", json.dumps({
            "model": model, "dataset": "crisismmd", "task": "informativeness",
            "n_total_samples": len(rows)}))
    return run


class CompareFallbackTests(unittest.TestCase):
    def test_switches_only_reviewed_and_counts_fixed_and_broken(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            truths = ("A", "B", "A", "B")
            primary_pred = ("A", "A", "A", "B")
            candidate_pred = ("B", "B", "B", "A")
            def rows(predictions):
                return [dict(id=str(i), text=f"text {i}", image_path=f"/images/{i}.jpg",
                             y_true=truth, y_pred=predicted)
                        for i, (truth, predicted) in enumerate(zip(truths, predictions))]
            primary = _write_run(root, "primary", "main", rows(primary_pred))
            candidate = _write_run(root, "candidate", "other", rows(candidate_pred))
            flags = root / "flags"
            flags.mkdir()
            (flags / "analysis_manifest.json").write_text(json.dumps({
                "source_run_id": "primary", "policy": "branch_disagreement_v1",
                "uses_true_labels_for_flag": False}))
            with (flags / "review_flags.tsv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=("id", "y_true", "y_pred", "needs_review"),
                                        delimiter="\t")
                writer.writeheader()
                for i, (truth, prediction, review) in enumerate(
                        zip(truths, primary_pred, (False, True, True, False))):
                    writer.writerow(dict(id=i, y_true=truth, y_pred=prediction,
                                         needs_review=str(review).lower()))
            target, summary = compare(primary, flags, candidate, "other", root / "result")
            self.assertEqual(summary["n_review"], 2)
            self.assertEqual(summary["n_fixed"], 1)
            self.assertEqual(summary["n_broken"], 1)
            self.assertEqual(summary["primary_accuracy"], 0.75)
            self.assertEqual(summary["candidate_accuracy"], 0.25)
            self.assertEqual(summary["gated_accuracy"], 0.75)
            self.assertAlmostEqual(summary["gated_macro_f1"], 0.7333333333333334)
            with (target / "switches.tsv").open(newline="") as stream:
                switched = list(csv.DictReader(stream, delimiter="\t"))
            self.assertEqual([row["id"] for row in switched], ["1", "2"])
            self.assertEqual([row["outcome"] for row in switched], ["fixed", "broken"])
            with self.assertRaisesRegex(ValueError, "khác mẫu/thứ tự"):
                mismatch = rows(candidate_pred)
                mismatch[1]["id"] = "wrong"
                wrong_candidate = _write_run(root, "wrong_candidate", "other", mismatch)
                compare(primary, flags, wrong_candidate, "other", root / "wrong_result")


if __name__ == "__main__":
    unittest.main()
