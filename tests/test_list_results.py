"""Lọc đúng phương pháp/dataset/task/số mẫu/model theo cây thư mục kết quả."""

import csv
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import list_results


class ResultIndexTests(unittest.TestCase):
    def test_find_3000_sample_method_without_confusing_1000_sample_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for run_id, method, samples, score in (
                ("baseline1000", "baseline", "1000", "0.805"),
                ("dclip3000", "dclip", "3000", "0.864"),
            ):
                folder = root / "results" / f"adapted_{method}" / "crisismmd" / "informativeness" / f"samples_{samples}" / "metaclip2_vith14" / f"informativeness_{run_id}"
                folder.mkdir(parents=True)
                with (folder / "summary.tsv").open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, delimiter="\t", fieldnames=(
                        "dataset", "task", "model", "image_method", "n_samples", "macro_f1"))
                    writer.writeheader()
                    writer.writerow(dict(dataset="crisismmd", task="informativeness",
                                         model="metaclip2_vith14", image_method=method,
                                         n_samples=samples, macro_f1=score))
            rows = list_results.result_rows(root)
            self.assertEqual(len(rows), 2)
            filters = SimpleNamespace(dataset="crisismmd", task="informativeness",
                                      variant=None, model="metaclip2", method="adapted_dclip", samples=3000)
            matches = list(list_results.matching_rows(rows, filters))
            self.assertEqual(len(matches), 1)
            self.assertEqual(matches[0]["run_id"], "informativeness_dclip3000")
            self.assertEqual(matches[0]["macro_f1"], "0.864")
            self.assertIn("results/adapted_dclip/crisismmd/informativeness/samples_3000/metaclip2_vith14/informativeness_dclip3000/summary.tsv",
                          matches[0]["summary_path"])

    def test_ensemble_model_and_relation_mode_are_identified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ensemble = root / "results/ensemble/fakeddit/2way/samples_1000/multi_model/2way_abc123/summary.tsv"
            ensemble.parent.mkdir(parents=True)
            ensemble.write_text("dataset\ttask\tmethod\tn_samples\tmacro_f1\n"
                                "fakeddit\t2way\tmetaclip2_vith14\t1000\t0.51\n"
                                "fakeddit\t2way\tequal_mean\t1000\t0.49\n", encoding="utf-8")
            direct = root / "results/direct_pair_fusion/crisismmd/humanitarian/samples_1000/metaclip2_vith14/humanitarian_def456/summary.tsv"
            direct.parent.mkdir(parents=True)
            direct.write_text("dataset\ttask\tmodel\trelation_enabled\trelation_mode\tn_samples\tmacro_f1\n"
                              "crisismmd\thumanitarian\tmetaclip2_vith14\tTrue\tdirect_pair_fusion\t1000\t0.44\n",
                              encoding="utf-8")
            rows = list_results.result_rows(root)
            self.assertEqual([(row["model"], row["variant"]) for row in rows if row["dataset"] == "fakeddit"],
                             [("metaclip2_vith14", "base_model"), ("ensemble", "equal_mean")])
            self.assertEqual(next(row for row in rows if row["dataset"] == "crisismmd")["method"],
                             "direct_pair_fusion")

    def test_header_only_summary_is_listed_separately(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = root / "results/baseline/crisismmd/informativeness/samples_1000/multi_model/informativeness_no_rows/summary.tsv"
            summary.parent.mkdir(parents=True)
            summary.write_text("dataset\ttask\tmodel\n", encoding="utf-8")
            self.assertEqual(list_results.result_rows(root), [])
            self.assertEqual(list(summary.parent.glob("summary.tsv")), [summary])

    def test_legacy_mmimdb_multi_label_appears_under_genres(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = root / "results/baseline/mmimdb/genres/samples_1000/siglip2_so400m/multi_label_old/summary.tsv"
            summary.parent.mkdir(parents=True)
            summary.write_text("dataset\ttask\tn_samples\tmacro_f1\n"
                               "mmimdb\tmulti_label\t1000\t0.5\n", encoding="utf-8")
            rows = list_results.result_rows(root)
            self.assertEqual(rows[0]["task"], "genres")


if __name__ == "__main__":
    unittest.main()
