"""Đối chiếu hai công thức từ log giả; không dùng dữ liệu CrisisMMD thật."""

import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/uncertainty"))
from compare_crisismmd_agreement import compare


class CompareCrisisMmdAgreementTests(unittest.TestCase):
    def test_reconstructs_paired_baseline_and_rejects_wrong_final_score(self):
        labels = ["informative", "not_informative"]
        image = np.array([[.8, .2], [.4, .6], [.7, .3]])
        text = np.array([[.7, .3], [.9, .1], [0., 0.]])
        baseline = (image + text) / 2
        baseline[2] = image[2]  # text rỗng: dùng nhánh ảnh
        gate = np.array([.25, 0., 0.])
        final = baseline.copy()
        product = image[0] * text[0]
        final[0] = .75 * baseline[0] + .25 * product / product.sum()
        fields = ["id", "y_true", "y_pred", "_text_was_empty", "image_text_fusion_gate",
                  "relation_applied", *[f"score_{label}" for label in labels],
                  *[f"score_image_{label}" for label in labels],
                  *[f"score_text_{label}" for label in labels]]
        rows = []
        for i in range(3):
            row = {"id": str(i), "y_true": labels[i % 2], "y_pred": labels[final[i].argmax()],
                   "_text_was_empty": str(i == 2), "image_text_fusion_gate": str(gate[i]),
                   "relation_applied": str(i == 0)}
            for j, label in enumerate(labels):
                row[f"score_{label}"] = str(final[i, j])
                row[f"score_image_{label}"] = str(image[i, j])
                row[f"score_text_{label}"] = "" if i == 2 else str(text[i, j])
            rows.append(row)
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "run_artifacts.zip"
            def write_archive(items):
                content = io.StringIO()
                writer = csv.DictWriter(content, fieldnames=fields, delimiter="\t")
                writer.writeheader()
                writer.writerows(items)
                with zipfile.ZipFile(archive_path, "w") as archive:
                    folder = "raw_predictions/stub__crisismmd__informativeness__v1"
                    archive.writestr(folder + "/raw_predictions.tsv", content.getvalue())
                    archive.writestr(folder + "/manifest.json", json.dumps({
                        "dataset": "crisismmd", "task": "informativeness", "model": "stub",
                        "relation_enabled": True, "relation_mode": "agreement_fusion",
                        "relation_weight": .25, "image_weight": .5}))
            write_archive(rows)
            report = compare(directory)
            self.assertEqual(report["n_samples"], 3)
            self.assertEqual(report["n_relation_applied"], 1)
            self.assertEqual(report["n_changed"], 0)
            rows[0]["score_informative"] = ".5"
            write_archive(rows)
            with self.assertRaises(ValueError):
                compare(directory)


if __name__ == "__main__":
    unittest.main()
