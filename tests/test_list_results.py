"""Bảng tra cứu giữ được các lượt cùng model nhưng khác số mẫu/phương pháp."""

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
                folder = root / "results/experiments/adapted_crisismmd_info" / run_id
                folder.mkdir(parents=True)
                with (folder / "summary.tsv").open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, delimiter="\t", fieldnames=(
                        "dataset", "task", "model", "image_method", "n_samples", "macro_f1"))
                    writer.writeheader()
                    writer.writerow(dict(dataset="crisismmd", task="informativeness",
                                         model="metaclip2_vith14", image_method=method,
                                         n_samples=samples, macro_f1=score))
            index = list_results.write_index(root)
            with index.open(newline="") as stream:
                rows = list(csv.DictReader(stream, delimiter="\t"))
            self.assertEqual(len(rows), 2)
            filters = SimpleNamespace(dataset="crisismmd", task="informativeness",
                                      job=None, model="metaclip2", method="dclip", samples=3000)
            matches = list(list_results.matching_rows(rows, filters))
            self.assertEqual(len(matches), 1)
            self.assertEqual(matches[0]["run_id"], "dclip3000")
            self.assertEqual(matches[0]["macro_f1"], "0.864")


if __name__ == "__main__":
    unittest.main()
