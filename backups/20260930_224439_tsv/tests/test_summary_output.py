"""Summary naming tests; write only synthetic results in temporary directories."""
from concurrent.futures import ThreadPoolExecutor
import importlib
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class SummaryOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        registry = types.ModuleType("models.model_registry")
        registry.load_model = lambda *args, **kwargs: None
        plotting = types.ModuleType("matplotlib")
        plotting.pyplot = types.ModuleType("matplotlib.pyplot")
        with patch.dict(sys.modules, {"models.model_registry": registry,
                                      "matplotlib": plotting, "matplotlib.pyplot": plotting.pyplot}):
            cls.common = importlib.import_module("run.common")
            cls.main = importlib.import_module("run_experiment")
            cls.demo = importlib.import_module("demo")

    def test_dataset_numbering_keeps_existing_files_and_skips_gaps(self):
        with tempfile.TemporaryDirectory() as directory:
            config = {"output": {"results_dir": directory},
                      "datasets": {"fakeddit": {"task": "2way"}}}
            rows = [{"model": "test", "accuracy": 0.5}]
            first = Path(self.common.save_dataset_summary(rows, config, "fakeddit"))
            original = first.read_bytes()
            third = first.with_name("summary_table_fakeddit_2way_3.csv")
            third.write_text("existing\n", encoding="utf-8")
            fourth = Path(self.common.save_dataset_summary(rows, config, "fakeddit"))
            self.assertEqual(first.name, "summary_table_fakeddit_2way_1.csv")
            self.assertEqual(fourth.name, "summary_table_fakeddit_2way_4.csv")
            self.assertEqual(first.read_bytes(), original)
            self.assertEqual(third.read_text(), "existing\n")
            config["datasets"]["fakeddit"]["task"] = "6way"
            self.assertEqual(Path(self.common.save_dataset_summary(rows, config, "fakeddit")).name,
                             "summary_table_fakeddit_6way_1.csv")

    def test_grouped_summary_keeps_columns_and_models(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = [{"dataset": d, "task": t, "model": m, "macro_f1": 0.5}
                    for d, t in (("fakeddit", "2way"), ("crisismmd", "humanitarian"))
                    for m in ("model_a", "model_b")]
            base = str(Path(directory) / "nested" / "summary_table.csv")
            paths = self.common.save_summary_tables(rows, base)
            self.assertEqual([Path(p).name for p in paths], ["summary_table_fakeddit_2way_1.csv",
                                                           "summary_table_crisismmd_humanitarian_1.csv"])
            for path in paths:
                frame = pd.read_csv(path)
                self.assertEqual(frame.columns.tolist(), list(rows[0]))
                self.assertEqual(frame.model.tolist(), ["model_a", "model_b"])
            self.assertEqual(self.common.save_summary_tables([], base), [])

    def test_concurrent_writers_do_not_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            base = str(Path(directory) / "summary_table.csv")
            def save(index):
                return self.common.save_numbered_summary([{"run": index}], base, "fakeddit", "2way")
            with ThreadPoolExecutor(max_workers=4) as pool:
                paths = list(pool.map(save, range(4)))
            self.assertEqual(len(set(paths)), 4)
            self.assertEqual(sorted(pd.read_csv(p).iloc[0]["run"] for p in paths), list(range(4)))

    def test_main_and_demo_use_numbered_summaries(self):
        for module in (self.main, self.demo):
            with self.subTest(entry=module.__name__), tempfile.TemporaryDirectory() as directory:
                config = {"models": ["model_a"],
                          "datasets": {"fakeddit": {"enabled": True, "task": "2way", "task_type": "single_label"}},
                          "output": {"results_dir": directory, "raw_predictions_dir": directory,
                                     "summary_table": str(Path(directory) / "summary_table.csv")},
                          "demo": {"visualization_dir": directory}}
                model = types.SimpleNamespace(unload=lambda: None)
                with patch.object(module, "load_config", return_value=config), \
                     patch.object(module, "load_model", return_value=model):
                    if module is self.main:
                        with patch.dict(module.DATASET_RUNNERS, {"fakeddit": lambda *a: {"accuracy": 0.75}}):
                            module.main()
                            module.main()
                        stem = "summary_table"
                    else:
                        with patch.object(module, "run_dataset", return_value={"accuracy": 0.75}), \
                             patch.multiple(module, N_SAMPLES=20, BATCH_SIZE=4, LOG_DIR=directory,
                                            OUTPUT_DIR=directory, PROMPT_VERSION="v1"):
                            module.main()
                            module.main()
                        stem = "demo_summary"
                self.assertEqual(sorted(p.name for p in Path(directory).glob("*.csv")),
                                 [f"{stem}_fakeddit_2way_1.csv", f"{stem}_fakeddit_2way_2.csv"])


if __name__ == "__main__":
    unittest.main()
