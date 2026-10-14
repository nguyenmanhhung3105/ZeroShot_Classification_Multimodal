"""Kiểm tra tái tạo fusion và giữ mẫu liên quan cùng fold."""

import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments/uncertainty"))
from sweep_fusion_weight import _group_folds, analyze


class FusionWeightTests(unittest.TestCase):
    def test_grouped_cv_reconstructs_baseline_and_supports_empty_text(self):
        ids = ["shared", "shared", "third", "4", "5", "6", "7", "8", "9", "10"]
        texts = ["first", "bridge", "bridge", "4", "5", "6", "7", "8", "9", ""]
        folds, n_groups = _group_folds(ids, texts)
        self.assertEqual(n_groups, 8)
        self.assertEqual(folds[0], folds[1])
        self.assertEqual(folds[1], folds[2])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "run123"
            run.mkdir()
            table = io.StringIO()
            fields = ("id", "text", "y_true", "y_pred", "_text_was_empty",
                      "score_A", "score_B", "score_image_A", "score_image_B",
                      "score_text_A", "score_text_B")
            writer = csv.DictWriter(table, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            for i in range(10):
                truth = i % 2
                image_pred = truth if i < 7 else 1 - truth
                text_pred = truth if i < 4 else 1 - truth
                image = [0.8 if j == image_pred else 0.2 for j in range(2)]
                text = [0.8 if j == text_pred else 0.2 for j in range(2)]
                empty = i == 9
                fused = image if empty else [(a + b) / 2 for a, b in zip(image, text)]
                writer.writerow(dict(id=ids[i], text=texts[i], y_true=("A", "B")[truth],
                                     y_pred=("A", "B")[max(range(2), key=fused.__getitem__)],
                                     _text_was_empty=str(empty), score_A=fused[0], score_B=fused[1],
                                     score_image_A=image[0], score_image_B=image[1],
                                     score_text_A="" if empty else text[0],
                                     score_text_B="" if empty else text[1]))
            prefix = "raw_predictions/model__synthetic"
            with zipfile.ZipFile(run / "run_artifacts.zip", "w") as archive:
                archive.writestr(f"{prefix}/raw_predictions.tsv", table.getvalue())
                archive.writestr(f"{prefix}/manifest.json", json.dumps({
                    "dataset": "crisismmd", "task": "informativeness", "image_weight": 0.5,
                    "n_total_samples": 10, "relation_enabled": False}))
            target, summary = analyze(run, root / "analysis")
            self.assertEqual(summary["n_samples"], 10)
            self.assertEqual(summary["n_groups"], 8)
            with (target / "folds.tsv").open(newline="") as stream:
                self.assertEqual(sum(int(item["n_held_out"])
                                     for item in csv.DictReader(stream, delimiter="\t")), 10)
            with (target / "full_cohort_sweep.tsv").open(newline="") as stream:
                sweep = {row["image_weight"]: row for row in csv.DictReader(stream, delimiter="\t")}
            self.assertAlmostEqual(float(sweep["1.0"]["accuracy"]), 0.7)
            self.assertAlmostEqual(float(sweep["0.0"]["accuracy"]), 0.4)


if __name__ == "__main__":
    unittest.main()
