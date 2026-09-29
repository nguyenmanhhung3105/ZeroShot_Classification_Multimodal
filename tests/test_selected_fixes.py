"""Regression tests using synthetic rows; no model or dataset is loaded."""

import ast
import importlib
import inspect
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "run"))

from evaluate import evaluate_multi_label
from logging_utils import save_single_label_log, save_multi_label_log
from prompts import fakeddit_prompts, mmimdb_prompts


class SelectedFixTests(unittest.TestCase):
    def test_fakeddit_label_ids_and_invalid_values(self):
        for task, count in (("2way", 2), ("6way", 6)):
            expected = list(range(count))
            for values in (expected, list(map(str, expected)), list(map(float, expected))):
                self.assertEqual(fakeddit_prompts.normalize_labels(values, task), expected)
            for invalid in (count, -1, 0.5, "fake", "", None):
                with self.subTest(task=task, invalid=invalid), self.assertRaises(ValueError):
                    fakeddit_prompts.normalize_labels([invalid], task)

    def test_all_direct_logging_calls_match_existing_signatures(self):
        functions = {
            "save_single_label_log": save_single_label_log,
            "save_multi_label_log": save_multi_label_log,
        }
        tree = ast.parse((ROOT / "src/run_experiment.py").read_text())
        count = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in functions:
                    inspect.signature(functions[node.func.id]).bind(
                        *([None] * len(node.args)),
                        **{kw.arg: None for kw in node.keywords},
                    )
                    count += 1
        self.assertEqual(count, 3)

    def test_logging_preserves_output_schema(self):
        df = pd.DataFrame({"id": ["synthetic"], "text": ["example"], "image_path": ["unused"]})
        with tempfile.TemporaryDirectory() as output:
            common = dict(id_col="id", text_col="text", model_name="stub",
                          dataset_name="synthetic", output_dir=output)
            single = save_single_label_log(df, [0], [1], np.array([[0.2, 0.8]]),
                                           [0, 1], task_name="single", **common)
            multi = save_multi_label_log(df, [["Drama", "Short"]], [["Drama"]],
                                        np.array([[0.8, 0.1]]), ["Drama", "Short"],
                                        task_name="multi", **common)
            expected = [
                (single, ["id", "text", "image_path", "y_true", "y_pred", "correct",
                          "confidence", "margin", "score_0", "score_1"]),
                (multi, ["id", "text", "image_path", "y_true", "y_pred", "missing_genres",
                         "extra_genres", "n_missing", "n_extra", "exact_match",
                         "prob_Drama", "prob_Short"]),
            ]
            for directory, columns in expected:
                self.assertEqual(pd.read_csv(Path(directory) / "raw_predictions.tsv",
                                             sep="\t", nrows=1).columns.tolist(), columns)
                self.assertEqual(len(pd.read_csv(Path(directory) / "error_analysis.tsv",
                                                 sep="\t", nrows=1)), 1)
                manifest = json.loads((Path(directory) / "manifest.json").read_text())
                self.assertEqual(set(manifest), {"model", "dataset", "task", "prompt_version",
                                                 "n_total_samples", "n_errors", "error_rate", "timestamp"})

    def test_mmimdb_labels_match_preprocessing_and_short_counts(self):
        # Parse the constant only: importing this script would access raw data.
        tree = ast.parse((ROOT / "src/preprocessing/data_mmimdb/preprocess_mmimdb.py").read_text())
        valid = next(ast.literal_eval(node.value) for node in tree.body
                     if isinstance(node, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == "VALID_GENRES" for t in node.targets))
        self.assertEqual(set(mmimdb_prompts.get_prompt_set()), valid)
        self.assertEqual(len(valid), 23)
        result = evaluate_multi_label([["Drama", "Short"]], [["Drama"]], ["Drama", "Short"])
        self.assertAlmostEqual(result["macro_f1"], 0.5)
        self.assertAlmostEqual(result["micro_f1"], 2 / 3)
        self.assertAlmostEqual(result["hamming_accuracy"], 0.5)
        for truth, prediction in (([["News"]], [["Drama"]]), ([["Drama"]], [["News"]])):
            with self.assertRaisesRegex(ValueError, "News"):
                evaluate_multi_label(truth, prediction, mmimdb_prompts.get_genre_list())

    def test_fakeddit_all_entry_points_both_tasks(self):
        # Stub only model loading and inference; run real label conversion,
        # evaluation and logging through each entry point on synthetic rows.
        registry = types.ModuleType("models.model_registry")
        registry.load_model = lambda *a, **k: self.fail("Must not load a model")
        matplotlib = types.ModuleType("matplotlib")
        matplotlib.pyplot = types.ModuleType("matplotlib.pyplot")
        with tempfile.TemporaryDirectory() as output, patch.dict(sys.modules, {
            "models.model_registry": registry,
            "matplotlib": matplotlib,
            "matplotlib.pyplot": matplotlib.pyplot,
        }):
            main = importlib.import_module("run_experiment")
            separate = importlib.import_module("run_fakeddit")
            demo = importlib.import_module("demo")
            for module in (main, separate, demo):
                for task, count in (("2way", 2), ("6way", 6)):
                    with self.subTest(entry=module.__name__, task=task):
                        labels = list(range(count))
                        df = pd.DataFrame({
                            "id": list(map(str, labels)),
                            "clean_title": ["Synthetic title with enough characters for filtering"] * count,
                            "image_path": ["never-opened"] * count,
                            "2_way_label": [str(i % 2) for i in labels],
                            "6_way_label": list(map(str, labels)),
                        })
                        cfg = {"task": task, "data_path": "never-read", "task_type": "single_label"}
                        if task == "2way":
                            cfg["positive_label"] = 0
                        config = {"output": {"raw_predictions_dir": output}}
                        scores = np.eye(count)
                        prediction = (labels, scores, scores, scores, np.zeros(count, dtype=bool))
                        with patch.object(module, "load_dataframe", return_value=df), \
                             patch.object(module, "build_class_embeddings", return_value=(None, labels)), \
                             patch.object(module, "resolve_image_paths", return_value=["unused"] * count), \
                             patch.object(module, "load_images_safe", return_value=([object()] * count, labels)), \
                             patch.object(module, "predict_single_label", return_value=prediction):
                            vlm = types.SimpleNamespace(model_name=module.__name__)
                            if module is demo:
                                with patch.object(demo, "LOG_DIR", output), \
                                     patch.object(demo, "OUTPUT_DIR", output), \
                                     patch.object(demo, "visualize"):
                                    result = demo.run_dataset(vlm, vlm.model_name, "fakeddit", cfg, 0.5)
                            else:
                                result = module.run_fakeddit(vlm, cfg, config)
                        self.assertEqual(result["accuracy"], 1.0)
                        self.assertEqual(result["macro_f1"], 1.0)
                        self.assertEqual(result["micro_f1"], 1.0)
                        if task == "2way":
                            self.assertEqual(result["auroc"], 1.0)


if __name__ == "__main__":
    unittest.main()
