"""Synthetic-only regression tests; never read data or load a pretrained model."""
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
from inference import (build_class_embeddings, encode_multimodal_embeddings,
                       predict_single_label, predict_multi_label)
from branch_ablation import compare, route_prompts, metrics, run
from prompts import fakeddit_prompts, crisismmd_prompts, mmimdb_prompts


class Tensor:
    def __init__(self, data):
        self.data = np.asarray(data, dtype=float)
    def detach(self):
        return self
    def cpu(self):
        return self
    def numpy(self):
        return self.data


class Model:
    model_name = "synthetic"
    logit_scale = 2.0
    logit_bias = -0.7
    def __init__(self):
        self.image_calls = 0
    def encode_images(self, images):
        self.image_calls += 1
        return Tensor([[1, 0] if i == 0 else [0, 1] for i in images])
    def encode_texts(self, texts):
        return Tensor([[1, 0] if t.startswith("a") else [0, 1] for t in texts])
    def unload(self):
        pass


class BranchTests(unittest.TestCase):
    def test_default_matches_independent_legacy_formula(self):
        classes = np.eye(2)
        for multi in (False, True):
            for weight in (0, 0.5, 1):
                model = Model()
                texts = ["a", "b", ""]
                cache = encode_multimodal_embeddings(model, [0, 1, 0], texts,
                                                     show_progress=False, image_weight=weight)
                image, text, empty = cache
                def transform(embeds):
                    logits = embeds @ classes.T * model.logit_scale + model.logit_bias
                    if multi:
                        return 1 / (1 + np.exp(-logits))
                    values = np.exp(logits - logits.max(axis=1, keepdims=True))
                    return values / values.sum(axis=1, keepdims=True)
                pi, pt = transform(image), transform(text)
                weights = np.where(empty, 1, weight)[:, None]
                expected = weights * pi + (1 - weights) * pt
                predictor = predict_multi_label if multi else predict_single_label
                options = {"threshold": 0.9} if multi else {}
                actual = predictor(model, [0, 1, 0], texts, classes, [0, 1],
                                   image_weight=weight, show_progress=False, **options)
                np.testing.assert_array_equal(actual[1], expected)
                np.testing.assert_array_equal(actual[2], pi)
                np.testing.assert_array_equal(actual[3], pt)
                if multi:
                    self.assertEqual(actual[0], [[0], [1], [0]])
                    self.assertTrue(actual[4].all())
                else:
                    np.testing.assert_allclose(actual[1].sum(axis=1), 1)

    def test_explicit_same_prototypes_and_cache_preserve_output(self):
        for predictor in (predict_single_label, predict_multi_label):
            model = Model()
            texts = ["a", "b", ""]
            baseline = predictor(model, [0, 1, 0], texts, np.eye(2), [0, 1], show_progress=False)
            cache = encode_multimodal_embeddings(model, [0, 1, 0], texts, show_progress=False)
            cached = predictor(model, [0, 1, 0], texts, np.eye(2), [0, 1],
                               img_class_embeds=np.eye(2), txt_class_embeds=np.eye(2),
                               encoded_embeddings=cache, show_progress=False)
            for a, b in zip(baseline, cached):
                self.assertEqual(a, b) if isinstance(a, list) else np.testing.assert_array_equal(a, b)

    def test_routing_all_37_labels_uses_original_words(self):
        routes = json.loads((ROOT / "configs/branch_prompt_routes.json").read_text())
        for dataset, task, prompts in (
            ("fakeddit", "6way", fakeddit_prompts.get_prompt_set("6way")),
            ("crisismmd", "humanitarian", crisismmd_prompts.get_prompt_set("humanitarian")),
            ("mmimdb", "genres", mmimdb_prompts.get_prompt_set())):
            branches = route_prompts(prompts, routes, dataset, task)
            for branch in branches.values():
                self.assertEqual(list(branch), list(prompts))
                for label, sentences in branch.items():
                    self.assertTrue(sentences)
                    self.assertTrue(set(sentences) <= set(prompts[label]))
                embeds, _ = build_class_embeddings(Model(), branch)
                np.testing.assert_allclose(np.linalg.norm(embeds, axis=1), 1)
        with self.assertRaises(ValueError):
            route_prompts({0: ["a"]}, routes, "fakeddit", "2way")

    def test_compare_reuses_inputs_and_reports_branch_metrics(self):
        model = Model()
        prompts = {0: ["a"], 1: ["b"]}
        branches = {"image": prompts, "text": {0: ["b"], 1: ["a"]}}
        rows, predictions = compare(model, [0, 1], ["a", "b"], [0, 1], prompts, branches,
                                    show_progress=False)
        self.assertEqual(model.image_calls, 1)
        self.assertEqual(rows[0]["fusion_macro_f1"], 1)
        self.assertEqual(rows[1]["text_macro_f1"], 0)
        self.assertEqual(rows[1]["image_macro_f1"], 1)
        self.assertEqual(len(predictions), 12)
        for row in rows:
            for branch in ("image", "text", "fusion"):
                self.assertIn(f"{branch}_balanced_accuracy", row)

    def test_empty_text_is_not_reported_as_text_only_image_fallback(self):
        rows, _ = compare(Model(), [0, 1], ["", "b"], [0, 1], {0: ["a"], 1: ["b"]},
                          show_progress=False)
        self.assertEqual(rows[0]["text_n_samples"], 1)
        self.assertEqual(rows[0]["fusion_n_samples"], 2)
        self.assertEqual(rows[0]["n_empty_text"], 1)

    def test_multilabel_metrics_and_fallback(self):
        rows, _ = compare(Model(), [0, 1], ["a", "b"], [[0], [1]], {0: ["a"], 1: ["b"]},
                          multilabel=True, threshold=0.99, show_progress=False)
        self.assertEqual(rows[0]["fusion_samples_f1"], 1)
        self.assertEqual(rows[0]["fusion_fallback_rate"], 1)
        self.assertTrue(np.isfinite(rows[0]["text_entropy"]))
        with self.assertRaises(ValueError):
            metrics([[3]], [[0]], [0, 1], True)

    def test_reject_misaligned_cache_and_invalid_prototypes(self):
        with self.assertRaises(ValueError):
            predict_single_label(Model(), [0], ["a"], np.eye(2), [0, 1],
                                 img_class_embeds=np.zeros((2, 2)), show_progress=False)
        with self.assertRaises(ValueError):
            predict_single_label(Model(), [0], ["a"], np.eye(2), [0, 1],
                                 encoded_embeddings=(np.eye(2), np.eye(2), np.zeros(2, dtype=bool)))

    def test_runner_end_to_end_with_two_synthetic_rows(self):
        registry = types.ModuleType("models.model_registry")
        registry.load_model = lambda *args, **kwargs: Model()
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {"models.model_registry": registry}):
            directory = Path(tmp)
            image_path = directory / "synthetic.png"
            Image.new("RGB", (2, 2)).save(image_path)
            data_path = directory / "synthetic.tsv"
            pd.DataFrame({"id": ["a", "b"], "clean_title": ["a" * 30, "b" * 30],
                          "6_way_label": [0, 1], "image_path": [str(image_path)] * 2}).to_csv(data_path, sep="\t", index=False)
            base = {"models": ["synthetic"], "datasets": {"fakeddit": {
                "data_path": str(data_path), "id_col": "id", "text_col": "clean_title"}}}
            base_path = directory / "base.json"
            base_path.write_text(json.dumps(base))
            config = {"base_config": str(base_path), "dataset": "fakeddit", "task": "6way",
                      "max_samples": 2, "split_prompts": True,
                      "prompt_routes": str(ROOT / "configs/branch_prompt_routes.json"),
                      "output_dir": str(directory / "results")}
            path = directory / "config.json"
            path.write_text(json.dumps(config))
            output = run(path)
            result = pd.read_csv(output / "results.tsv", sep="\t")
            self.assertFalse(list(output.glob("*.csv")))
            self.assertEqual(result.variant.tolist(), ["baseline", "split_prompts"])
            self.assertEqual(result.n_samples.tolist(), [2, 2])
            self.assertEqual(result.split.tolist(), ["exploratory", "exploratory"])
            snapshot = json.loads((output / "config_and_prompts.json").read_text())
            self.assertEqual(len(snapshot["branch_prompts"]["image"]), 6)
            self.assertEqual(len(json.loads((output / "samples.json").read_text())), 2)
            config["split"] = "test"
            path.write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                run(path)


if __name__ == "__main__":
    unittest.main()
