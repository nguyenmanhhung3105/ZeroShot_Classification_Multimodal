"""Paper-style mean similarity: synthetic inputs only; no model downloads."""
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image

import test_runtime_fixes
from test_runtime_fixes import FakeVLM, Tensor
from inference import build_class_embeddings, predict_single_label, prompt_aggregation_settings


class PromptAggregationTests(unittest.TestCase):
    def test_mean_sim_equals_explicit_per_prompt_cosine(self):
        embeddings = {"p": [2., 0.], "q": [0., 3.], "r": [1., 1.], "s": [-1., 0.]}
        model = FakeVLM()
        model.encode_texts = lambda texts: Tensor([embeddings[t] for t in texts])
        prompts = {5: ["p", "q"], 2: ["r"], 0: ["q", "s", "r"]}
        vectors, labels = build_class_embeddings(model, prompts, aggregation="mean_sim")
        self.assertEqual(labels, [5, 2, 0])
        samples = np.array([[1., 0.], [0., 1.], [0.6, 0.8]])
        expected = []
        for texts in prompts.values():
            v = np.array([embeddings[t] for t in texts])
            v /= np.linalg.norm(v, axis=1, keepdims=True)
            expected.append((samples @ v.T).mean(axis=1))
        np.testing.assert_allclose(samples @ vectors.T, np.array(expected).T, atol=1e-12)
        self.assertLess(np.linalg.norm(vectors[0]), 1)

    def test_default_preserves_legacy_formula_and_prediction(self):
        model = FakeVLM()
        prompts = {0: ["one", "two"], 1: ["three"]}
        vectors, labels = build_class_embeddings(model, prompts)
        explicit, _ = build_class_embeddings(model, prompts, aggregation="mean_embed")
        old = np.array([[0., 1.], [0., 1.]]) / (1 + 1e-12)
        np.testing.assert_array_equal(vectors, old)
        np.testing.assert_array_equal(vectors, explicit)
        before = predict_single_label(model, [None], ["text"], old, labels, show_progress=False)
        after = predict_single_label(model, [None], ["text"], vectors, labels, show_progress=False)
        for a, b in zip(before, after):
            np.testing.assert_array_equal(a, b)

    def test_two_modes_can_differ_without_changing_prompts(self):
        model = FakeVLM()
        model.encode_texts = lambda texts: Tensor([[1., 0.], [0., 1.]])
        avg, _ = build_class_embeddings(model, {0: ["p", "q"]}, aggregation="mean_sim")
        norm, _ = build_class_embeddings(model, {0: ["p", "q"]})
        self.assertFalse(np.allclose(avg, norm))
        self.assertAlmostEqual(avg[0, 0], .5)
        self.assertAlmostEqual(norm[0, 0], 1 / np.sqrt(2))

    def test_config_scope_and_retired_backend_are_explicit(self):
        for cfg, dataset in (({"task": "6way", "prompt_aggregation": "bad"}, "fakeddit"),
                             ({"task": "genres", "prompt_aggregation": "mean_sim"}, "mmimdb"),
                             ({"task": "6way", "text_backends": ["clip_text", "nli"]}, "fakeddit")):
            with self.assertRaises(ValueError):
                prompt_aggregation_settings(cfg, dataset)
        with self.assertRaises(ValueError):
            build_class_embeddings(FakeVLM(), {0: ["p"]}, aggregation="bad")

    def test_trials_keep_model_and_settings(self):
        test_runtime_fixes.RuntimeFixTests.setUpClass()
        common = importlib.import_module("run.common")
        cfg = {"task": "6way", "prompt_variants": ["current"],
               "prompt_aggregations": ["mean_embed", "mean_sim"], "max_samples": 5}
        rows = common.run_prompt_trials(lambda active: {"macro_f1": .1}, "same_model", "fakeddit", cfg)
        self.assertEqual([r["prompt_aggregation"] for r in rows], ["mean_embed", "mean_sim"])
        self.assertEqual([r["model"] for r in rows], ["same_model"] * 2)
        self.assertNotIn("prompt_aggregation", cfg)
        self.assertIn("prompt_aggregation", common.compact_summary(rows))
        for invalid in ([], "mean_sim", ["mean_sim", "mean_sim"], ["bad"]):
            with self.assertRaises(ValueError):
                common.prompt_trial_configs("fakeddit", {**cfg, "prompt_aggregations": invalid})

    def test_three_runners_log_same_method_and_predictions(self):
        test_runtime_fixes.RuntimeFixTests.setUpClass()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            picture = root / "synthetic.png"
            Image.new("RGB", (3, 3), "blue").save(picture)
            data = root / "synthetic.tsv"
            pd.DataFrame({"id": ["one", "two"], "text": ["synthetic title", "another title"],
                          "6_way_label": [0, 1], "image_path": [str(picture)] * 2}).to_csv(data, sep="\t", index=False)
            task = {"task": "6way", "task_type": "single_label", "data_path": str(data),
                    "text_col": "text", "min_text_length": 0, "max_samples": 2,
                    "prompt_aggregation": "mean_sim"}
            config = {"datasets": {"fakeddit": task}, "inference": {"image_weight": .75},
                      "output": {"raw_predictions_dir": str(root / "logs")}}
            outputs = []
            for entry in ("run_fakeddit", "run_experiment", "demo"):
                module = importlib.import_module(entry)
                model = FakeVLM()
                if entry == "demo":
                    with patch.multiple(module, LOG_DIR=str(root / "demo_logs"), OUTPUT_DIR=str(root / "demo"), N_SAMPLES=2), \
                            patch.object(module, "visualize"):
                        result = module.run_dataset(model, model.model_name, "fakeddit", task, .75, config=config)
                else:
                    result = module.run_fakeddit(model, task, config)
                self.assertEqual(result["prompt_aggregation"], "mean_sim")
                log = Path(result["prediction_log_dir"])
                manifest = json.loads((log / "manifest.json").read_text())
                self.assertEqual(manifest["prompt_aggregation"], "mean_sim")
                outputs.append(pd.read_csv(log / "raw_predictions.tsv", sep="\t"))
            pd.testing.assert_frame_equal(outputs[0], outputs[1])
            pd.testing.assert_frame_equal(outputs[0], outputs[2])


if __name__ == "__main__":
    unittest.main()
