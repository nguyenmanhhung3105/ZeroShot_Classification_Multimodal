"""Small synthetic tests: never read project datasets or load pretrained weights."""
import ast
import importlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/run"))

from evaluate import evaluate_binary
from inference import predict_single_label, predict_multi_label, _softmax, _sigmoid
from logging_utils import save_single_label_log
from prompts import crisismmd_prompts


class Tensor:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class FakeVLM:
    model_name = "synthetic"
    logit_scale = 2.0
    logit_bias = -0.1

    def __init__(self):
        self.image_batches = []
        self.text_batches = []

    def encode_images(self, images):
        self.image_batches.append(len(images))
        return Tensor([[1., 0.]] * len(images))

    def encode_texts(self, texts):
        self.text_batches.append(len(texts))
        return Tensor([[0., 1.]] * len(texts))


class RuntimeFixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        registry = types.ModuleType("models.model_registry")
        registry.load_model = lambda *a, **k: (_ for _ in ()).throw(AssertionError("No model loading"))
        plotting = types.ModuleType("matplotlib")
        plotting.pyplot = types.ModuleType("matplotlib.pyplot")
        with patch.dict(sys.modules, {"models.model_registry": registry,
                                      "matplotlib": plotting, "matplotlib.pyplot": plotting.pyplot}):
            cls.common = importlib.import_module("run.common")
            cls.main = importlib.import_module("run_experiment")
            cls.demo = importlib.import_module("demo")
            cls.crisis = importlib.import_module("run_crisismmd")
            cls.fake = importlib.import_module("run_fakeddit")

    def test_crisis_mapping_keys_display_and_unknown(self):
        for task in ("informativeness", "humanitarian"):
            labels = list(crisismmd_prompts.get_prompt_set(task))
            names = crisismmd_prompts.get_label_names(task)
            self.assertEqual(self.common.normalize_single_labels(labels, names, labels), labels)
            self.assertEqual(self.common.normalize_single_labels(list(names.values()), names, labels), labels)
            with self.assertRaises(ValueError):
                self.common.normalize_single_labels(["unknown"], names, labels)
        self.assertEqual(self.common.crisismmd_label_column(
            {"task": "humanitarian", "label_col": "label_informative"}), "label_humanitarian")

    def test_crisis_task_specific_label_columns(self):
        cfg = {"label_col_info": "custom_info", "label_col_human": "custom_human",
               "label_col": "legacy_column"}
        for task, expected in (("informativeness", "custom_info"), ("humanitarian", "custom_human")):
            self.assertEqual(self.common.crisismmd_label_column({**cfg, "task": task}), expected)
        self.assertEqual(self.common.crisismmd_label_column({"task": "informativeness"}), "label_informative")
        self.assertEqual(self.common.crisismmd_label_column({"task": "humanitarian"}), "label_humanitarian")
        self.assertEqual(self.common.crisismmd_label_column(
            {"task": "humanitarian", "label_col": "legacy_column"}), "legacy_column")
        for value in (None, "", "  ", 12):
            with self.assertRaises(ValueError):
                self.common.crisismmd_label_column({"task": "informativeness", "label_col_info": value})
        with self.assertRaises(ValueError):
            self.common.crisismmd_label_column({"task": "invalid"})

    def test_bounded_reader_and_filter(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "synthetic.tsv")
            pd.DataFrame({"id": ["001", "002", "003", "004"],
                          "text": ["x", "long enough text", "y", "another long text"]}).to_csv(path, sep="\t", index=False)
            with patch.object(self.common.pd, "read_csv", wraps=pd.read_csv) as reader:
                result = self.common.load_dataframe(path, max_samples=1)
                self.assertEqual(reader.call_args.kwargs["nrows"], 1)
                self.assertEqual(result.id.tolist(), ["001"])
            for loader in (self.common.load_dataframe, self.main.load_dataframe, self.demo.load_dataframe):
                result = loader(path, max_samples=2, text_col="text", min_text_length=5)
                self.assertEqual(result.id.tolist(), ["002", "004"])

    def test_lazy_images_skip_bad_and_preserve_extensions(self):
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / "001.jpeg", Path(directory) / "002.jpg"
            for path in (first, second):
                Image.new("RGB", (4, 4), "red").save(path)
            broken = Path(directory) / "broken.jpg"
            broken.write_bytes(b"not an image")
            paths = [str(first), str(Path(directory) / "missing.jpg"), str(broken), str(second)]
            images, indices = self.common.load_images_safe(paths)
            self.assertEqual(indices, [0, 3])
            self.assertEqual(images.paths, [str(first), str(second)])
            with images[0] as image:
                self.assertEqual(image.size, (4, 4))
            df = pd.DataFrame({"id": ["001", "002"], "image_path": ["images/001.jpeg", "images/002.jpg"]})
            self.assertEqual(self.common.resolve_image_paths(df, {"image_dir": directory}), [str(first), str(second)])
            vlm = FakeVLM()
            predict_single_label(vlm, images, ["one", "two"], np.eye(2), [0, 1],
                                 batch_size=1, show_progress=False)
            self.assertEqual(vlm.image_batches, [1, 1])

    def test_single_modality_skips_unused_encoder_and_empty_text_fallback(self):
        for weight, expected_images, expected_texts in ((1., 3, 0), (0., 1, 2), (0.5, 3, 2)):
            vlm = FakeVLM()
            pred, probabilities, _, _, empty = predict_single_label(
                vlm, [object()] * 3, ["text", "", "text"], np.eye(2), [0, 1],
                image_weight=weight, batch_size=2, show_progress=False)
            self.assertEqual(sum(vlm.image_batches), expected_images)
            self.assertEqual(sum(vlm.text_batches), expected_texts)
            self.assertEqual(empty.tolist(), [False, True, False])
            self.assertEqual(pred[1], 0)
            image_p = _softmax(np.array([[2., 0.]]) - 0.1)[0]
            text_p = _softmax(np.array([[0., 2.]]) - 0.1)[0]
            np.testing.assert_allclose(probabilities[0], weight * image_p + (1 - weight) * text_p)
            np.testing.assert_allclose(probabilities[1], image_p)

    def test_multilabel_formula_and_fallback_unchanged(self):
        vlm = FakeVLM()
        pred, probs, _, _, fallback, _ = predict_multi_label(
            vlm, [object()], ["text"], np.eye(2), ["A", "B"],
            threshold=0.99, show_progress=False)
        expected = 0.5 * _sigmoid(np.array([[2., 0.]]) - 0.1) + 0.5 * _sigmoid(np.array([[0., 2.]]) - 0.1)
        np.testing.assert_allclose(probs, expected)
        self.assertEqual(pred, [["A"]])
        self.assertEqual(fallback.tolist(), [True])
        for kwargs in ({"batch_size": 0}, {"image_weight": 2}, {"threshold": -1}):
            with self.assertRaises(ValueError):
                predict_multi_label(vlm, [object()], ["text"], np.eye(2), ["A", "B"],
                                    show_progress=False, **kwargs)

    def test_binary_zero_and_absent_class(self):
        result = evaluate_binary([0, 1], [0, 1], np.array([0.9, 0.1]), 0, label_order=[0, 1])
        self.assertEqual(result["auroc"], 1.)
        self.assertEqual(result["macro_f1"], 1.)
        result = evaluate_binary([1], [1], np.array([0.1]), 0, label_order=[0, 1])
        self.assertIsNone(result["auroc"])
        self.assertEqual(result["macro_f1"], 0.5)

    def test_logging_metadata_and_branch_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            df = pd.DataFrame({"id": ["s"], "text": [""], "image_path": ["old"],
                               "_resolved_image_path": ["resolved"], "_text_was_empty": [True]})
            path = Path(save_single_label_log(
                df, [0], [0], np.array([[0.8, 0.2]]), [0, 1], "id", "text", "stub", "fake", "2way", directory,
                sim_image=np.array([[0.8, 0.2]]), sim_text=np.array([[0.5, 0.5]]),
                extra_manifest={"image_weight": 0.3, "threshold": 0.22, "model": "must-not-override"}))
            row = pd.read_csv(path / "raw_predictions.tsv", sep="\t", nrows=1).iloc[0]
            self.assertEqual(row.image_path, "resolved")
            self.assertTrue(row._text_was_empty)
            self.assertEqual(row.score_image_0, 0.8)
            self.assertTrue(row._image_encoded)
            self.assertFalse(row._text_encoded)
            self.assertTrue(pd.isna(row.score_text_0))
            manifest = json.loads((path / "manifest.json").read_text())
            self.assertEqual(manifest["image_weight"], 0.3)
            self.assertEqual(manifest["model"], "stub")

    def test_crisis_all_entry_points_both_tasks(self):
        for module in (self.main, self.crisis, self.demo):
            for task in ("informativeness", "humanitarian"):
                labels = list(crisismmd_prompts.get_prompt_set(task))
                names = crisismmd_prompts.get_label_names(task)
                df = pd.DataFrame({"id": list(map(str, range(len(labels)))), "text": ["text"] * len(labels),
                                   "label_informative": [names[x] for x in labels] if task == "informativeness" else ["informative"] * len(labels),
                                   "label_humanitarian": [names[x] for x in labels] if task == "humanitarian" else ["not_humanitarian"] * len(labels)})
                cfg = {"task": task, "data_path": "never-read", "text_col": "text", "task_type": "single_label",
                       "label_col": "label_informative"}
                scores = np.eye(len(labels))
                with self.subTest(entry=module.__name__, task=task), tempfile.TemporaryDirectory() as out, \
                     patch.object(module, "load_dataframe", return_value=df), \
                     patch.object(module, "build_class_embeddings", return_value=(None, labels)), \
                     patch.object(module, "resolve_image_paths", return_value=["unused"] * len(labels)), \
                     patch.object(module, "load_images_safe", return_value=([object()] * len(labels), list(range(len(labels))))), \
                     patch.object(module, "predict_single_label", return_value=(labels, scores, scores, scores, np.zeros(len(labels), dtype=bool))):
                    if module is self.demo:
                        with patch.object(module, "LOG_DIR", out), patch.object(module, "OUTPUT_DIR", out), patch.object(module, "visualize"):
                            result = module.run_dataset(FakeVLM(), "stub", "crisismmd", cfg, 0.5)
                    else:
                        result = module.run_crisismmd(FakeVLM(), cfg, {"output": {"raw_predictions_dir": out}})
                    self.assertEqual(result["accuracy"], 1.)
                    self.assertEqual(result["macro_f1"], 1.)

    def test_demo_reads_yaml_settings(self):
        settings = {"n_samples": 3, "batch_size": 2, "log_dir": "unused-logs",
                    "visualization_dir": "unused-demo", "prompt_version": "test"}
        with patch.object(self.demo, "load_config", return_value={"demo": settings, "models": [], "datasets": {}}), \
             patch.multiple(self.demo, N_SAMPLES=20, BATCH_SIZE=4, LOG_DIR="old", OUTPUT_DIR="old", PROMPT_VERSION="v1"):
            self.demo.main()
            self.assertEqual((self.demo.N_SAMPLES, self.demo.BATCH_SIZE, self.demo.LOG_DIR,
                              self.demo.OUTPUT_DIR, self.demo.PROMPT_VERSION), tuple(settings.values()))

    def test_mmimdb_source_extension_selection(self):
        # Execute only filename mapping, never the preprocessing pipeline.
        source = (ROOT / "src/preprocessing/data_mmimdb/preprocess_mmimdb.py").read_text()
        tree = ast.parse(source)
        nodes = []
        collecting = False
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "image_files" for t in node.targets):
                collecting = True
            if collecting:
                nodes.append(node)
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "jpeg_ids" for t in node.targets):
                    break
        import os
        namespace = {"os": os, "all_files": ["one.jpg", "two.JPEG", "three.jpg", "three.jpeg", "one.json"]}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "mapping-only", "exec"), namespace)
        self.assertEqual(namespace["image_files"], {"one": "one.jpg", "two": "two.JPEG", "three": "three.jpeg"})

    def test_fakeddit_ready_uses_processed_schema_and_skips_missing_image(self):
        source = (ROOT / "src/preprocessing/data_fakeddit/check_fakeddit_ready.py").read_text()
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "synthetic.tsv")
            pd.DataFrame({"id": ["one", "two"], "clean_title": ["long text", "longer text"],
                          "image_path": [str(Path(directory) / "missing.jpg")] * 2,
                          "hasImage": [True, True], "2_way_label": [0, 1], "6_way_label": [1, 0]}).to_csv(path, sep="\t", index=False)
            tree = ast.parse(source)
            for node in tree.body:
                if isinstance(node, ast.Assign):
                    names = [t.id for t in node.targets if isinstance(t, ast.Name)]
                    if "CFG" in names:
                        node.value = ast.parse(repr({"task": "2way", "data_path": path}), mode="eval").body
                    elif "MIN_TOTAL_SAMPLES" in names:
                        node.value = ast.Constant(1)
            namespace = {"__name__": "synthetic_check"}
            exec(compile(ast.fix_missing_locations(tree), "synthetic-check", "exec"), namespace)
            self.assertTrue(namespace["ready"])

    def test_mmimdb_runner_logging_and_metrics(self):
        registry = types.ModuleType("models.model_registry")
        registry.load_model = lambda *a, **k: self.fail("No model load")
        with patch.dict(sys.modules, {"models.model_registry": registry}):
            separate = importlib.import_module("run_mmimdb")
        from prompts import mmimdb_prompts
        labels = mmimdb_prompts.get_genre_list()
        df = pd.DataFrame({"id": ["one"], "plot": ["synthetic plot"], "genres": ["Drama|Short"]})
        scores = np.zeros((1, len(labels)))
        scores[0, labels.index("Drama")] = 0.9
        cfg = {"task": "genres", "task_type": "multi_label", "data_path": "never-read"}
        for module in (self.main, separate, self.demo):
            with self.subTest(entry=module.__name__), tempfile.TemporaryDirectory() as out, \
                 patch.object(module, "load_dataframe", return_value=df), \
                 patch.object(module, "build_class_embeddings", return_value=(None, labels)), \
                 patch.object(module, "resolve_image_paths", return_value=["unused"]), \
                 patch.object(module, "load_images_safe", return_value=([object()], [0])), \
                 patch.object(module, "predict_multi_label", return_value=([["Drama"]], scores, scores, scores,
                                                                          np.array([False]), np.array([False]))) as predictor:
                if module is self.demo:
                    with patch.object(module, "LOG_DIR", out), patch.object(module, "OUTPUT_DIR", out), patch.object(module, "visualize"):
                        result = module.run_dataset(FakeVLM(), "stub", "mmimdb", cfg, 0.5)
                else:
                    result = module.run_mmimdb(FakeVLM(), cfg, {"output": {"raw_predictions_dir": out}})
                self.assertAlmostEqual(result["micro_f1"], 2 / 3)
                self.assertAlmostEqual(result["macro_f1"], 1 / 23)
                manifest_path = next(Path(out).glob("*/manifest.json"))
                metadata = json.loads(manifest_path.read_text())
                self.assertIsNone(metadata["threshold"])
                self.assertEqual(metadata["score_type"], "relative_z")
                self.assertEqual(predictor.call_args.kwargs["decision_mode"], "relative_z")
                self.assertEqual(predictor.call_args.kwargs["relative_z_threshold"], 1.0)
                log = pd.read_csv(manifest_path.parent / "raw_predictions.tsv", sep="\t")
                self.assertIn("zscore_Drama", log.columns)
                self.assertNotIn("prob_Drama", log.columns)

    def test_main_relative_predictor_matches_relative_scores(self):
        from inference import relative_multilabel_scores, multilabel_settings
        image = np.array([[0.1, 0.2, 0.4], [0.3, 0.3, 0.3]])
        text = np.array([[0.6, 0.7, 0.9], [0., 0., 0.]])
        empty = np.array([False, True])
        expected = relative_multilabel_scores(image, text, 0.5, empty)
        with patch("inference.encode_multimodal_embeddings", return_value=(image, text, empty)):
            output = predict_multi_label(FakeVLM(), [0, 1], ["text", ""], np.eye(3), ["a", "b", "c"],
                                         decision_mode="relative_z", threshold=None,
                                         relative_z_threshold=1.0, show_progress=False)
        for result, scores in zip(output[1:4], expected):
            np.testing.assert_array_equal(result, scores)
        self.assertEqual(output[0], [["c"], ["a"]])
        self.assertEqual(output[4].tolist(), [False, True])
        options, meta = multilabel_settings({"decision_mode": "fixed_threshold"}, 0.22)
        self.assertEqual(options["decision_mode"], "fixed_threshold")
        self.assertEqual(meta["threshold"], 0.22)
        self.assertEqual(meta["score_type"], "probability")

    def test_all_source_files_parse(self):
        for path in (ROOT / "src").rglob("*.py"):
            ast.parse(path.read_text(), filename=str(path))

    def test_relative_scores_are_per_sample_and_constant_safe(self):
        from inference import relative_multilabel_scores
        image = np.array([[0.1, 0.2, 0.4], [0.3, 0.3, 0.3]])
        text = np.array([[0.6, 0.7, 0.9], [0.7, 0.7, 0.7]])
        together = relative_multilabel_scores(image, text)
        alone = relative_multilabel_scores(image[:1], text[:1])
        shifted = relative_multilabel_scores(image * 2 + 3, text * 4 + 1)
        for whole, one, other in zip(together, alone, shifted):
            np.testing.assert_array_equal(whole[:1], one)
            np.testing.assert_array_equal(whole[1], np.zeros(3))
            np.testing.assert_allclose(whole, other, atol=1e-12)

    def test_relative_mode_avoids_all_labels_when_sigmoid_saturates(self):
        cosine = np.linspace(0.1, 0.3, 23)[None, :]
        model = FakeVLM()
        model.logit_scale, model.logit_bias = 100., 0.
        with patch("inference.encode_multimodal_embeddings", return_value=(cosine, cosine, np.array([False]))):
            old = predict_multi_label(model, [0], ["text"], np.eye(23), list(range(23)), show_progress=False)
            new = predict_multi_label(model, [0], ["text"], np.eye(23), list(range(23)),
                                      decision_mode="relative_z", show_progress=False)
        self.assertEqual(len(old[0][0]), 23)
        self.assertGreater(len(new[0][0]), 0)
        self.assertLess(len(new[0][0]), 23)


if __name__ == "__main__":
    unittest.main()
