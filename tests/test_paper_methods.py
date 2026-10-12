"""Kiểm tra độc lập hai nhánh bằng dữ liệu giả, không dùng trọng số thật."""
import hashlib
import importlib
import io
import json
from pathlib import Path
import pickle
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image

import test_runtime_fixes as runtime
from test_runtime_fixes import FakeVLM, Tensor
from inference import predict_single_label, predict_multi_label
from research.adapted_methods import waffle_prompts, research_embeddings, load_descriptors, _StringsOnlyUnpickler
from research.run_adapted_methods import prepare_trials, aggregate_results, paper_summary


class DiverseVLM(FakeVLM):
    def encode_texts(self, texts):
        return Tensor([[1 + hashlib.sha256(t.encode()).digest()[0],
                        1 + hashlib.sha256(t.encode()).digest()[1]] for t in texts])


class PaperMethodTests(unittest.TestCase):
    def test_shared_suffixes_seed_and_no_global_rng_mutation(self):
        names = {5: "vehicle damage", 0: "affected individuals"}
        words = ["orange", "table", "purple", "cloud", "stone"]
        before = np.random.get_state()
        prompts = waffle_prompts(names, words, 3, 2)
        after = np.random.get_state()
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])
        self.assertEqual(prompts, waffle_prompts(names, words, 3, 2))
        self.assertNotEqual(prompts, waffle_prompts(names, words, 3, 3))
        self.assertEqual(list(prompts), [5, 0])
        self.assertEqual(len(prompts[5]), 6)
        self.assertEqual([p.replace(names[5], "CLASS") for p in prompts[5]],
                         [p.replace(names[0], "CLASS") for p in prompts[0]])
        for count in (0, True, 257):
            with self.assertRaises(ValueError):
                waffle_prompts(names, words, count)

    def test_pickle_forbids_executable_objects(self):
        with self.assertRaises(pickle.UnpicklingError):
            _StringsOnlyUnpickler(io.BytesIO(pickle.dumps(eval))).load()

    def test_noop_and_branch_isolation(self):
        labels = [0, 1]
        classes = np.eye(2)
        alternative = classes[::-1].copy()
        for fn, kwargs in ((predict_single_label, {}),
                           (predict_multi_label, {"decision_mode": "relative_z"}),
                           (predict_multi_label, {"decision_mode": "fixed_threshold"})):
            base = fn(FakeVLM(), [None, None], ["hello", ""], classes, labels, show_progress=False, **kwargs)
            noop = fn(FakeVLM(), [None, None], ["hello", ""], classes, labels,
                      show_progress=False, image_class_embeds=None, **kwargs)
            changed = fn(FakeVLM(), [None, None], ["hello", ""], classes, labels,
                         show_progress=False, image_class_embeds=alternative, **kwargs)
            for left, right in zip(base, noop):
                self.assertEqual(str(left), str(right))
            np.testing.assert_array_equal(base[3], changed[3])  # giữ nguyên nhánh text
            self.assertFalse(np.allclose(base[2], changed[2]))
            np.testing.assert_array_equal(changed[1][1], changed[2][1])  # text rỗng dùng ảnh dự phòng
            if fn is predict_single_label:
                np.testing.assert_allclose(changed[1].sum(axis=1), 1.)

    def test_invalid_matrix_and_label_order(self):
        with self.assertRaises(ValueError):
            predict_single_label(FakeVLM(), [None], ["x"], np.eye(2), [0, 1],
                                 image_class_embeds=np.ones((3, 2)), show_progress=False)
        with self.assertRaises(ValueError):
            research_embeddings(FakeVLM(), {"_research_image_prompts": {1: ["x"], 0: ["y"]}},
                                {0: ["x"], 1: ["y"]}, [0, 1])

    def test_descriptors_require_complete_labels_and_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "descriptors.json"
            path.write_text(json.dumps({"provenance": "synthetic", "descriptors": {"0": ["red roof"]}}))
            prompts, meta = load_descriptors(path, {0: "building"})
            self.assertIn("which has red roof", prompts[0][0])
            self.assertEqual(len(meta["descriptor_sha256"]), 64)
            with self.assertRaises(ValueError):
                load_descriptors(path, {0: "building", 1: "road"})

    def test_trials_preserve_config_and_average_all_seeds(self):
        base = {"datasets": {"crisismmd": {"task": "humanitarian", "prompt_aggregation": "mean_embed"}}}
        settings = {"dataset": "crisismmd", "methods": ["baseline", "waffle"], "seeds": [0, 1]}
        with patch("research.run_adapted_methods.official_words", return_value=(["apple", "tree"], "hash")):
            trials = prepare_trials(settings, base)
        self.assertNotIn("max_samples", base["datasets"]["crisismmd"])
        self.assertEqual(len(trials), 3)
        self.assertEqual(trials[0]["prompt_aggregation"], "mean_embed")
        self.assertEqual(trials[1]["prompt_aggregation"], "mean_embed")  # giữ cách xử lý text nền
        result = aggregate_results([{"model": "x", "image_method": "waffle", "macro_f1": .2},
                                    {"model": "x", "image_method": "waffle", "macro_f1": .4}])
        self.assertAlmostEqual(result.iloc[0]["macro_f1_mean"], .3)
        self.assertAlmostEqual(result.iloc[0]["macro_f1_std"], .1)

    def test_summary_keeps_method_identity(self):
        runtime.RuntimeFixTests.setUpClass()
        row = {"image_method": "waffle", "image_prompt_aggregation": "mean_sim", "research_seed": 2}
        frame = paper_summary([row])
        self.assertEqual(frame.iloc[0]["image_method"], "waffle")
        self.assertEqual(frame.iloc[0]["image_prompt_aggregation"], "mean_sim")

    def test_chunk_max_text_scores_unchanged(self):
        from test_text_chunking import ChunkVLM
        classes = np.array([[.8, .1], [-.2, .7]])
        options = dict(show_progress=False, chunk_long_text=True, chunk_aggregation="max_sim",
                       decision_mode="relative_z")
        before = predict_multi_label(ChunkVLM(), [None], ["First sentence. Second longer sentence."],
                                     classes, [0, 1], **options)
        after = predict_multi_label(ChunkVLM(), [None], ["First sentence. Second longer sentence."],
                                    classes, [0, 1], image_class_embeds=classes[::-1], **options)
        np.testing.assert_array_equal(before[3], after[3])

    def test_fixed_crisis_descriptor_file(self):
        from research.adapted_methods import ROOT
        from prompts.crisismmd_prompts import get_label_names
        names = get_label_names("humanitarian")
        prompts, _ = load_descriptors(ROOT / "configs/research/descriptors/crisismmd_humanitarian.json", names)
        self.assertEqual(list(prompts), list(names))
        self.assertTrue(all(len(values) == 3 for values in prompts.values()))

    def test_research_cli_synthetic_end_to_end(self):
        from research.run_adapted_methods import main
        runtime.RuntimeFixTests.setUpClass()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            picture = root / "image.png"
            Image.new("RGB", (2, 2)).save(picture)
            data = root / "rows.tsv"
            pd.DataFrame({"id": ["a", "b"], "text": ["synthetic one", "synthetic two"],
                          "label_humanitarian": ["vehicle_damage", "not_humanitarian"],
                          "image_path": [str(picture)] * 2}).to_csv(data, sep="\t", index=False)
            base = root / "base.json"
            base.write_text(json.dumps({"models": ["synthetic"], "datasets": {"crisismmd": {
                "task": "humanitarian", "id_col": "id", "text_col": "text", "data_path": str(data)}},
                "inference": {"image_weight": .5}, "evaluation": {"measure_truncation": False}}))
            config = root / "paper.json"
            config.write_text(json.dumps({"base_config": str(base), "dataset": "crisismmd", "max_samples": 2,
                "methods": ["baseline", "class_name", "waffle"], "seeds": [0, 1], "waffle_pairs": 2,
                "output_dir": str(root / "results")}))
            model = DiverseVLM()
            unloaded = []
            model.unload = lambda: unloaded.append(True)
            registry = types.ModuleType("models.model_registry")
            registry.load_model = lambda name: model
            with patch.dict("sys.modules", {"models.model_registry": registry}), \
                    patch("research.run_adapted_methods.official_words", return_value=(["apple", "tree"], "synthetic")):
                main(["--config", str(config)])
            summary = pd.read_csv(next((root / "results").glob("*/summary.tsv")), sep="\t")
            self.assertEqual(summary["image_method"].tolist(), ["baseline", "class_name", "waffle", "waffle"])
            seeds = pd.read_csv(next((root / "results").glob("*/seed_summary.tsv")), sep="\t")
            self.assertEqual(seeds.loc[seeds.image_method == "waffle", "n_runs"].tolist(), [2])
            self.assertEqual(unloaded, [True])
            frozen = json.loads(next((root / "results").glob("*/frozen_plan.json")).read_text())
            self.assertEqual(len(frozen["baseline_text_prompts"]), 8)

    def test_all_dataset_runners_record_prompts_and_preserve_text(self):
        runtime.RuntimeFixTests.setUpClass()
        cases = [("fakeddit", "2way", "2_way_label", [0, 1]),
                 ("fakeddit", "6way", "6_way_label", [0, 1]),
                 ("crisismmd", "informativeness", "label_informative", ["informative", "not_informative"]),
                 ("crisismmd", "humanitarian", "label_humanitarian", ["vehicle_damage", "not_humanitarian"]),
                 ("mmimdb", "genres", "genres", ["Action|Drama", "Short"])]
        for dataset, task_name, column, truth in cases:
            with self.subTest(dataset=dataset, task=task_name), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                picture = root / "synthetic.png"
                Image.new("RGB", (3, 3), "blue").save(picture)
                data = root / "synthetic.tsv"
                pd.DataFrame({"id": ["one", "two"], "text": ["synthetic title", "another synthetic title"],
                              column: truth, "image_path": [str(picture)] * 2}).to_csv(data, sep="\t", index=False)
                task = {"task": task_name, "data_path": str(data), "id_col": "id", "text_col": "text",
                        "label_col": column, "min_text_length": 0, "max_samples": 2}
                config = {"datasets": {dataset: task}, "inference": {"image_weight": .5},
                          "evaluation": {"enabled": True, "measure_truncation": False},
                          "output": {"raw_predictions_dir": str(root / "logs")}}
                settings = {"dataset": dataset, "methods": ["baseline", "class_name"], "max_samples": 2}
                runner = getattr(importlib.import_module(f"run.run_{dataset}"), f"run_{dataset}")
                outputs = []
                for active in prepare_trials(settings, config):
                    result = runner(DiverseVLM(), active, config)
                    directory = Path(result["prediction_log_dir"])
                    manifest = json.loads((directory / "manifest.json").read_text())
                    self.assertEqual(manifest["image_method"], active["_research_metadata"]["image_method"])
                    outputs.append((result, pd.read_csv(directory / "raw_predictions.tsv", sep="\t")))
                self.assertEqual(outputs[0][0]["sample_set_id"], outputs[1][0]["sample_set_id"])
                cols = [c for c in outputs[0][1] if c.startswith("score_text_")]
                self.assertTrue(cols)
                pd.testing.assert_frame_equal(outputs[0][1][cols], outputs[1][1][cols])


if __name__ == "__main__":
    unittest.main()
