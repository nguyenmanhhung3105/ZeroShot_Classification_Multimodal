"""Distribution report tests use temporary synthetic data only."""
import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("label_distribution", Path(__file__).resolve().parents[1] / "src/run/test_label_distribution.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


class LabelDistributionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = {"datasets": {}}
        values = {
            "fakeddit": (["2_way_label", "6_way_label"], [[1, 0], [0, 2], [0, 5]]),
            "crisismmd": (["custom_info", "custom_human"], [["informative", "affected_individuals"],
                          ["not_informative", "not_humanitarian"], ["informative", "unknown"]]),
            "mmimdb": (["genres"], [["Action|Drama|Action"], ['["Comedy", "Drama"]'], [""]])}
        for dataset, (columns, rows) in values.items():
            path = self.root / (dataset + ".tsv")
            with path.open("w", newline="") as stream:
                writer = csv.writer(stream, delimiter="\t")
                writer.writerow(columns)
                writer.writerows(rows)
            self.config["datasets"][dataset] = {"data_path": str(path), "max_samples": 1, "enabled": False}
        self.config["datasets"]["crisismmd"].update(label_col_info="custom_info", label_col_human="custom_human")

    def test_all_tasks_counts_and_zero_classes(self):
        counts, overview, issues = report.summarize(self.config)
        self.assertEqual(len(overview), 5)
        self.assertEqual(len(counts), 2 + 6 + 2 + 8 + 23)
        self.assertTrue(all(r["n_rows"] == 3 for r in overview))
        select = lambda dataset, task, label: next(r for r in counts if (r["dataset"], r["task"], r["label_id"]) == (dataset, task, label))
        self.assertEqual(select("fakeddit", "2way", 0)["n_samples"], 2)
        self.assertEqual(select("fakeddit", "6way", 4)["n_samples"], 0)
        self.assertEqual(select("mmimdb", "genres", "Action")["n_samples"], 1)
        self.assertEqual(select("mmimdb", "genres", "Drama")["n_samples"], 2)
        mm = next(r for r in overview if r["dataset"] == "mmimdb")
        self.assertEqual(mm["n_duplicate_label_rows"], 1)
        self.assertEqual(mm["n_missing_labels"], 1)
        self.assertEqual(mm["known_label_assignments"], 4)
        self.assertEqual(issues, [{"dataset": "crisismmd", "task": "humanitarian", "unknown_label": "unknown", "n_rows": 1}])

    def test_bounded_preview_and_no_inference_filter(self):
        counts, overview, issues = report.summarize(self.config, max_rows=2)
        self.assertTrue(all(r["n_rows"] == 2 and r["scope"] == "first_rows_preview" for r in overview))
        self.assertEqual(issues, [])
        self.assertTrue(all(r["status"] == "ok" for r in counts))

    def test_missing_column_is_error_not_zero(self):
        self.config["datasets"]["crisismmd"]["label_col_human"] = "absent"
        counts, overview, _ = report.summarize(self.config)
        human = [r for r in counts if r["task"] == "humanitarian"]
        self.assertTrue(all(r["n_samples"] is None and r["status"] == "error" for r in human))
        self.assertEqual(next(r for r in overview if r["task"] == "informativeness")["status"], "ok")

    def test_raw_guard_and_formats(self):
        with self.assertRaises(ValueError):
            list(report.read_rows(self.root / "dataraw" / "never_read.tsv"))
        with self.assertRaises(ValueError):
            list(report.read_rows(self.root / "data" / "raw" / "never_read.tsv"))
        for text in ('["Action", "Drama"]', "['Action', 'Drama']", "Action|Drama", "Action,Drama"):
            self.assertEqual(report.parse_genres(text), ["Action", "Drama"])

    def test_html_tsv_output_no_overwrite_and_escaping(self):
        counts, overview, issues = report.summarize(self.config, max_rows=2)
        counts[0]["label_name"] = "<script>bad</script>"
        a = report.save_report(counts, overview, issues, self.root / "out")
        b = report.save_report(counts, overview, issues, self.root / "out")
        self.assertNotEqual(a, b)
        html = (a / "report.html").read_text()
        self.assertIn("&lt;script&gt;", html)
        self.assertNotIn("<script>bad", html)
        with (a / "class_counts.tsv").open(encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream, delimiter="\t"))
        self.assertEqual(len(rows), len(counts))
        self.assertTrue((a / "unknown_labels.tsv").exists())


if __name__ == "__main__":
    unittest.main()
