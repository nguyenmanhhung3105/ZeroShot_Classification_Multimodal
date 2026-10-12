"""Kiểm tra hàm tác giả và adapter bằng dữ liệu giả; không tải checkpoint."""
from copy import deepcopy
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
import yaml

from research import run_upstream as runner
from research.upstream import (ROOT, author_api, generate_prompts, description_encodings,
                               image_scores, verify_sources)


class SyntheticModel:
    context_length = 77

    def encode_text(self, tokens):
        tokens = tokens.float()
        return torch.stack([tokens.sum(1) % 97 + 1, tokens[:, 1] % 31 + 1,
                            tokens[:, 2] % 17 + 1, tokens[:, 3] % 23 + 1,
                            tokens.count_nonzero(1).float() + 1, tokens[:, 4] % 41 + 1], dim=1)


class SyntheticBackend:
    """Tokenizer thật, embedding giả; không khởi tạo model pretrained."""
    instances = []

    def __init__(self, api, backbone, device):
        self.api, self.backbone, self.device = api, backbone, device
        self.model_name = "synthetic_openai_clip"
        self.model = SyntheticModel()
        self.tokenizer = api.clip.simple_tokenizer.SimpleTokenizer()
        self.logit_scale, self.logit_bias = 2.0, 0.0
        self.image_batches, self.text_batches, self.unloaded = [], [], False
        self.instances.append(self)

    def encode_images(self, images):
        self.image_batches.append(len(images))
        return F.normalize(torch.tensor([[*(c / 255 for c in image.getpixel((0, 0))), 1., .3, .1]
                                         for image in images]))

    def encode_texts(self, texts):
        self.text_batches.append(list(texts))
        return F.normalize(self.model.encode_text(self.api.clip.tokenize(texts, truncate=True)))

    def unload(self):
        self.unloaded = True


class UpstreamProjectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if any(not (ROOT / "third_party" / name).exists() for name in ("OpenAI_CLIP", "WaffleCLIP", "DCLIP")):
            raise unittest.SkipTest("Cần clone repo theo docs/research.md để kiểm tra hàm tác giả")
        cls.api = author_api()
        cls.sources = verify_sources()
        cls.settings = yaml.safe_load((ROOT / "configs/research/upstream.yaml").read_text())

    def test_original_prompt_generator_seed_and_rng_restore(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "descriptors.json"
            path.write_text(json.dumps({"blue car": ["a blue vehicle"], "red boat": ["a red ship"]}))
            before_np, before_py = np.random.get_state(), random.getstate()
            prompts, _ = generate_prompts(self.api, "waffle", ["blue car", "red boat"], path, 2, 3)
            np.testing.assert_array_equal(before_np[1], np.random.get_state()[1])
            self.assertEqual(before_np[2:], np.random.get_state()[2:])
            self.assertEqual(before_py, random.getstate())
            same, _ = generate_prompts(self.api, "waffle", ["blue car", "red boat"], path, 2, 3)
            other, _ = generate_prompts(self.api, "waffle", ["blue car", "red boat"], path, 2, 4)
            self.assertEqual(prompts, same)
            self.assertNotEqual(prompts, other)
            self.assertEqual(len(prompts["blue car"]), 4)
            self.assertEqual([p.replace("blue car", "CLASS") for p in prompts["blue car"]],
                             [p.replace("red boat", "CLASS") for p in prompts["red boat"]])
            dclip, _ = generate_prompts(self.api, "dclip", ["blue car", "red boat"], path, 2, 0)
            self.assertEqual(dclip["blue car"], ["blue car, which is a blue vehicle"])
            labels, options = generate_prompts(self.api, "dclip_labels", ["blue car", "red boat"], path, 2, 0)
            encoded = description_encodings(self.api, SyntheticModel(), "dclip_labels", labels, options, "cpu")
            expected = F.normalize(SyntheticModel().encode_text(self.api.clip.tokenize(["blue car", "red boat"])))
            torch.testing.assert_close(torch.cat(list(encoded.values())), expected)

    def test_original_encoding_and_mean_similarity(self):
        prompts = {"car": ["a blue car", "a yellow car"], "boat": ["a boat"]}
        model = SyntheticModel()
        samples = F.normalize(torch.tensor([[1., 2., 3., 4., 5., 6.], [6., 1., 4., 3., 2., 5.]]))
        for method in ("clip", "waffle", "dclip"):
            options = {"device": "cpu"} if method == "dclip" else SimpleNamespace(classes_to_load=None)
            encodings = description_encodings(self.api, model, method, prompts, options, "cpu")
            expected = []
            for texts, vectors in zip(prompts.values(), encodings.values()):
                direct = F.normalize(model.encode_text(self.api.clip.tokenize(texts)))
                torch.testing.assert_close(vectors, direct)
                torch.testing.assert_close(vectors.norm(dim=1), torch.ones(len(texts)))
                expected.append((samples @ direct.T).mean(dim=1))
            actual = image_scores(self.api, samples, encodings, method)
            torch.testing.assert_close(actual, torch.stack(expected, dim=1))
            self.assertEqual(actual.argmax(1).tolist(), torch.stack(expected, dim=1).argmax(1).tolist())

    def fixture(self, root, dataset, task):
        picture = root / "synthetic.png"
        Image.new("RGB", (3, 3), "blue").save(picture)
        data = root / "synthetic.tsv"
        names = runner.prompt_spec(dataset, task)
        label_keys = list(names)
        if dataset == "fakeddit":
            column, truth = ("2_way_label" if task == "2way" else "6_way_label"), label_keys[:2]
        elif dataset == "crisismmd":
            column, truth = ("label_informative" if task == "informativeness" else "label_humanitarian"), label_keys[:2]
        else:
            column, truth = "genres", ["Action|Drama", "Short"]
        text = "A fictional plot. " * 90 if dataset == "mmimdb" else "synthetic text"
        pd.DataFrame({"id": ["one", "two", "missing"], "text": [text, "", "synthetic"], column: [*truth, truth[0]],
                      "image_path": [str(picture), str(picture), str(root / "missing.png")]}).to_csv(data, sep="\t", index=False)
        active = {"task": task, "data_path": str(data), "text_col": "text", "max_samples": 3,
                  "min_text_length": 0, "chunk_long_text": dataset == "mmimdb",
                  "decision_mode": "relative_z", "relative_z_threshold": 1.0}
        base = {"models": ["synthetic"], "datasets": {dataset: active}, "inference": {"image_weight": .5},
                "evaluation": {"measure_truncation": True, "split": "synthetic"}}
        base_path = root / "base.json"
        base_path.write_text(json.dumps(base))
        settings = deepcopy(self.settings)
        settings.update(base_config=str(base_path), seeds=[0, 1], waffle_pairs=2, output_dir=str(root / "results"))
        settings["datasets"] = {dataset: {"enabled": True, "task": task, "max_samples": 3}}
        config = root / "config.json"
        config.write_text(json.dumps(settings))
        return config, base, settings

    def test_prepare_only_never_reads_dataset_or_loads_model(self):
        with tempfile.TemporaryDirectory() as directory:
            config, _, _ = self.fixture(Path(directory), "crisismmd", "humanitarian")
            with patch.object(runner, "OpenAIClipBackend", side_effect=AssertionError("Không tải model")), \
                    patch.object(runner, "load_cohort", side_effect=AssertionError("Không đọc dữ liệu")):
                root = runner.main(["--config", str(config), "--prepare-only"])
            frozen = json.loads((root / "frozen_plan.json").read_text())
            self.assertEqual(len(frozen["trials"]), len(self.settings["methods"]) + 1)
            self.assertEqual(set(frozen["cohorts"]["crisismmd"]["text_prompts"]), set(runner.prompt_spec("crisismmd", "humanitarian")))
            self.assertFalse((root / "summary.tsv").exists())

    def test_synthetic_all_five_tasks_logs_and_cached_inputs(self):
        cases = [("fakeddit", "2way"), ("fakeddit", "6way"), ("crisismmd", "informativeness"),
                 ("crisismmd", "humanitarian"), ("mmimdb", "genres")]
        for dataset, task in cases:
            with self.subTest(dataset=dataset, task=task), tempfile.TemporaryDirectory() as directory:
                config, _, _ = self.fixture(Path(directory), dataset, task)
                with (patch.object(runner, "OpenAIClipBackend", SyntheticBackend),
                      redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO())):
                    root = runner.main(["--config", str(config)])
                backend = SyntheticBackend.instances[-1]
                self.assertEqual(sum(backend.image_batches), 2)  # không mã hóa lại giữa trial/variant
                self.assertTrue(backend.unloaded)
                summary = pd.read_csv(root / "summary.tsv", sep="\t")
                self.assertEqual(len(summary), 2 * (len(self.settings["methods"]) + 1))
                self.assertNotIn("descriptor_provenance", summary.columns)
                native_rows = summary[summary.output_variant == "original_image"]
                self.assertTrue(native_rows["fusion_macro_f1_gain"].isna().all())
                self.assertTrue(native_rows["text_macro_f1"].isna().all())
                self.assertEqual(set(summary["output_variant"]), {"original_image", "multimodal"})
                self.assertTrue(summary["n_samples"].eq(2).all())
                self.assertTrue(summary["n_skipped_images"].eq(1).all())
                self.assertTrue(summary["device"].eq("cpu").all())
                self.assertEqual(summary.loc[summary.output_variant == "multimodal", "text_macro_f1"].nunique(), 1)
                seed_table = pd.read_csv(root / "seed_summary.tsv", sep="\t")
                self.assertEqual(seed_table.loc[seed_table.method == "waffle", "n_runs"].tolist(), [2, 2])
                metric = "samples_f1" if dataset == "mmimdb" else "balanced_accuracy"
                expected = summary.groupby(["method", "output_variant"])[metric].mean()
                actual = seed_table.set_index(["method", "output_variant"])[metric + "_mean"]
                pd.testing.assert_series_equal(actual.sort_index(), expected.sort_index(), check_names=False)
                self.assertFalse((root / "failure.json").exists())
                snapshots = []
                for path in summary.prediction_log_dir:
                    manifest = json.loads((Path(path) / "manifest.json").read_text())
                    self.assertEqual(manifest["device"], "cpu")
                    snapshots.append(json.loads((Path(path) / "evaluation_snapshot.json").read_text()))
                    rows = pd.read_csv(Path(path) / "raw_predictions.tsv", sep="\t")
                    if dataset == "mmimdb":
                        self.assertEqual(manifest["classification_scope"], "multi_label_extension")
                        self.assertTrue(any(c.startswith("zscore_") for c in rows))
                        self.assertGreater(snapshots[-1]["truncation"]["n_truncated"], 0)
                    else:
                        scores = rows[[f"score_{k}" for k in runner.prompt_spec(dataset, task)]]
                        np.testing.assert_allclose(scores.sum(axis=1), 1.)
                self.assertEqual(len({s["sample_set_id"] for s in snapshots}), 1)

    def test_native_argmax_and_relative_z_ignore_calibration(self):
        backend = SimpleNamespace(logit_scale=100.)
        image, text = np.array([[.5, .2, .1], [.1, .7, .2]]), np.array([[.1, .9, .2], [.8, .1, .2]])
        empty = np.array([False, True])
        pred, scores, p_image, _, _ = runner.score_predictions(backend, image, text, empty, [0, 1, 2], 1., 25.,
            multilabel=False, decision={}, native=True)
        self.assertEqual(pred, image.argmax(1).tolist())
        np.testing.assert_allclose(scores, p_image)
        decision = {"score_type": "relative_z", "relative_z_threshold": 1., "fallback_top1": True}
        before = runner.score_predictions(backend, image, text, empty, [0, 1, 2], .5, 25.,
            multilabel=True, decision=decision, native=False)
        backend.logit_scale = 2.
        after = runner.score_predictions(backend, image, text, empty, [0, 1, 2], .5, 50.,
            multilabel=True, decision=decision, native=False)
        self.assertEqual(before[0], after[0])
        for left, right in zip(before[1:], after[1:]):
            np.testing.assert_array_equal(left, right)
        np.testing.assert_allclose(before[1][1], before[2][1])  # text rỗng dùng ảnh

    def test_max_sim_cache_uses_same_chunking_as_main(self):
        from inference import build_class_embeddings
        from text_chunking import encode_chunked_max_sim
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, base, settings = self.fixture(root, "mmimdb", "genres")
            base["datasets"]["mmimdb"]["chunk_aggregation"] = "max_sim"
            specs, _ = runner.prepare_plan(settings, base, self.api, root, self.sources)
            spec = specs["mmimdb"]
            cohort = runner.load_cohort("mmimdb", spec)
            backend = SyntheticBackend(self.api, "ViT-B/32", "cpu")
            cache = runner.encode_inputs(backend, "mmimdb", spec, cohort, base, settings)
            classes, _ = build_class_embeddings(backend, spec["text_prompts"], aggregation="mean_embed")
            expected, counts = encode_chunked_max_sim(backend, [cohort["texts"][0]], settings["batch_size"], classes)
            np.testing.assert_allclose(cache["text_scores"][0], expected[0])
            self.assertEqual(cache["chunking"]["counts"], [counts[0], 0])
            self.assertGreater(counts[0], 1)

    def test_fakeddit_loader_keeps_numeric_id_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, base, settings = self.fixture(root, "fakeddit", "2way")
            specs, _ = runner.prepare_plan(settings, base, self.api, root, self.sources)
            path = Path(base["datasets"]["fakeddit"]["data_path"])
            frame = pd.read_csv(path, sep="\t", dtype=str)
            frame.loc[0, "2_way_label"] = "fake"
            frame.to_csv(path, sep="\t", index=False)
            with self.assertRaisesRegex(ValueError, "nhãn không hợp lệ"):
                runner.load_cohort("fakeddit", specs["fakeddit"])

    def test_failure_keeps_plan_and_does_not_publish_completed_sweep(self):
        class BrokenBackend(SyntheticBackend):
            def encode_images(self, images):
                raise RuntimeError("Lỗi giả lập")
        with tempfile.TemporaryDirectory() as directory:
            config, _, _ = self.fixture(Path(directory), "fakeddit", "2way")
            with patch.object(runner, "OpenAIClipBackend", BrokenBackend), self.assertRaisesRegex(RuntimeError, "Lỗi giả lập"):
                runner.main(["--config", str(config)])
            root = next((Path(directory) / "results").iterdir())
            self.assertTrue((root / "frozen_plan.json").exists())
            self.assertTrue((root / "failure.json").exists())
            self.assertFalse((root / "seed_summary.tsv").exists())
            self.assertTrue(BrokenBackend.instances[-1].unloaded)

    def test_bad_config_and_incomplete_descriptor_fail_before_inference(self):
        for override in ({"waffle_pairs": 0}, {"seeds": [0, 0]}, {"methods": ["unknown"]}, {"outputs": []}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                runner.validate_settings({**self.settings, **override})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, base, settings = self.fixture(root, "fakeddit", "2way")
            descriptor = root / "bad.json"
            descriptor.write_text(json.dumps({"provenance": "Giả lập", "descriptors": {"0": ["an object"]}}))
            settings["descriptor_files"]["fakeddit"]["2way"] = str(descriptor)
            with self.assertRaisesRegex(ValueError, "thiếu/thừa nhãn"):
                runner.prepare_plan(settings, base, self.api, root, self.sources)


if __name__ == "__main__":
    unittest.main()
