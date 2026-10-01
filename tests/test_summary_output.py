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
            third = first.with_name("summary_table_fakeddit_2way_3.tsv")
            third.write_text("existing\n", encoding="utf-8")
            fourth = Path(self.common.save_dataset_summary(rows, config, "fakeddit"))
            self.assertEqual(first.name, "summary_table_fakeddit_2way_1.tsv")
            self.assertEqual(fourth.name, "summary_table_fakeddit_2way_4.tsv")
            self.assertEqual(first.read_bytes(), original)
            self.assertEqual(third.read_text(), "existing\n")
            config["datasets"]["fakeddit"]["task"] = "6way"
            self.assertEqual(Path(self.common.save_dataset_summary(rows, config, "fakeddit")).name,
                             "summary_table_fakeddit_6way_1.tsv")

    def test_grouped_summary_keeps_columns_and_models(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = [{"dataset": d, "task": t, "model": m, "macro_f1": 0.5}
                    for d, t in (("fakeddit", "2way"), ("crisismmd", "humanitarian"))
                    for m in ("model_a", "model_b")]
            base = str(Path(directory) / "nested" / "summary_table.csv")
            paths = self.common.save_summary_tables(rows, base)
            self.assertEqual([Path(p).name for p in paths], ["summary_table_fakeddit_2way_1.tsv",
                                                           "summary_table_crisismmd_humanitarian_1.tsv"])
            for path in paths:
                frame = pd.read_csv(path, sep="\t")
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
            self.assertEqual(sorted(pd.read_csv(p, sep="\t").iloc[0]["run"] for p in paths), list(range(4)))

    def test_compact_summary_preserves_metrics_errors_and_input(self):
        rows = [{"model": "m", "macro_f1": .4, "fusion_macro_f1": .4,
                 "fusion_samples_f1": .5, "fusion_n_samples": 1000,
                 "image_macro_f1": .3, "text_macro_f1": .2,
                 "config_id": "full-hash", "chunk_long_text": True,
                 "threshold": None, "relative_z_threshold": 1., "hamming_accuracy": .9},
                {"model": "failed", "error": "Sample lock mismatch"}]
        original = pd.DataFrame(rows)
        compact = self.common.compact_summary(original)
        self.assertEqual(compact.iloc[0]['samples_f1'], .5)
        self.assertEqual(compact.iloc[0]['n_samples'], 1000)
        self.assertEqual(compact.iloc[1]['error'], 'Sample lock mismatch')
        self.assertTrue(compact.iloc[0]['chunk_long_text'])
        self.assertNotIn('config_id', compact)
        self.assertNotIn('threshold', compact)
        self.assertNotIn('fusion_macro_f1', compact)
        self.assertEqual(compact.iloc[0]['image_macro_f1'], .3)
        # Missing branch sample counts: no unsupported improvement claim.
        self.assertNotIn('fusion_macro_f1_gain', compact)
        pd.testing.assert_frame_equal(original, pd.DataFrame(rows))
        with tempfile.TemporaryDirectory() as directory:
            path = self.common.save_numbered_summary(rows, str(Path(directory) / 'summary.tsv'), 'mmimdb', 'genres')
            saved = pd.read_csv(path, sep='\t')
            self.assertEqual(saved.columns.tolist(), compact.columns.tolist())

    def test_summary_fusion_gain_requires_same_cohort(self):
        row = {'image_macro_f1': .4, 'text_macro_f1': .3, 'fusion_macro_f1': .45,
               'image_n_samples': 10, 'text_n_samples': 10, 'fusion_n_samples': 10,
               'image_weight': .75, 'text_weight': .25, 'fusion_enabled': True,
               'text_logit_scale_requested': 25, 'text_logit_scale_effective': 25,
               'text_scale_override_applied': True, 'text_logit_scale_status': 'manual'}
        frame = self.common.compact_summary([row, {**row, 'fusion_macro_f1': .35},
                                            {**row, 'text_n_samples': 9}])
        self.assertAlmostEqual(frame.iloc[0]['fusion_macro_f1_gain'], .05)
        self.assertTrue(frame.iloc[0]['fusion_improves_macro_f1'])
        self.assertAlmostEqual(frame.iloc[1]['fusion_macro_f1_gain'], -.05)
        self.assertFalse(frame.iloc[1]['fusion_improves_macro_f1'])
        self.assertTrue(pd.isna(frame.iloc[2]['fusion_improves_macro_f1']))
        self.assertEqual(frame.iloc[0]['image_weight'], .75)
        self.assertTrue(frame.iloc[0]['text_scale_override_applied'])

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
                self.assertEqual(sorted(p.name for p in Path(directory).glob("*.tsv")),
                                 [f"{stem}_fakeddit_2way_1.tsv", f"{stem}_fakeddit_2way_2.tsv"])
                self.assertFalse(list(Path(directory).glob("*.csv")))


if __name__ == "__main__":
    unittest.main()
