"""NLI backend tests: synthetic texts/images, no pretrained downloads or data."""
from contextlib import ExitStack
import importlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/run"))
from inference import predict_single_label, _softmax
from nli_text import NLITextScorer, make_text_scorer, backend_summary, validate_backend
from text_scale import text_scale_settings
from test_runtime_fixes import FakeVLM


class FakeTokenizer:
    model_max_length = 1024

    def __init__(self, scorer):
        self.values = {h: float(label) for label, h in zip(scorer.labels, scorer.hypotheses)}
        self.calls = []

    def encode(self, text, **kwargs):
        return list(range(len(text.split())))

    def num_special_tokens_to_add(self, pair=False):
        return 4 if pair else 2

    def __call__(self, texts, hypotheses, **kwargs):
        self.calls.append((texts, hypotheses, kwargs))
        return {"input_ids": torch.tensor([[self.values[h]] for h in hypotheses])}


class FakeNLI(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Deliberately NOT the BART order; code must read label2id.
        self.config = types.SimpleNamespace(label2id={"entailment": 0, "neutral": 1, "contradiction": 2},
                                            num_labels=3, max_position_embeddings=1024, _commit_hash="synthetic-sha")
        self.grad_states = []

    def forward(self, input_ids):
        self.grad_states.append((torch.is_grad_enabled(), self.training))
        entail = input_ids[:, 0]
        return types.SimpleNamespace(logits=torch.stack([entail, entail * 0 + 100, entail * 0 - 100], dim=1))


def fake_load(scorer):
    scorer.runtime.update(device="cpu", revision_resolved="synthetic-sha")
    return FakeTokenizer(scorer), FakeNLI().eval(), "cpu"


class NLITextTests(unittest.TestCase):
    def scorer(self, **settings):
        return NLITextScorer(settings, [5, 0, 4, 1, 2, 3])

    def test_defaults_and_config_validation(self):
        with patch.object(NLITextScorer, "_load", side_effect=AssertionError("must not load")):
            self.assertIsNone(make_text_scorer({}, "fakeddit", [0, 1]))
        for dataset, task, backend in (("mmimdb", "genres", "nli"), ("fakeddit", "2way", "nli"),
                                        ("crisismmd", "humanitarian", "nli"), ("fakeddit", "6way", "bad")):
            with self.assertRaises(ValueError):
                validate_backend({"task": task, "text_backend": backend}, dataset)
        for settings in ({"batch_size": 0}, {"max_length": True}, {"batch_size": 1.5},
                         {"model": "other-model"}, {"unknown": 1}):
            with self.assertRaises(ValueError):
                self.scorer(**settings)
        with self.assertRaises(ValueError):
            NLITextScorer({}, [0, 1])

    def test_entailment_softmax_order_batches_and_truncation(self):
        scorer = self.scorer(batch_size=4, max_length=24)
        tokenizer, model, device = fake_load(scorer)
        with patch.object(scorer, "_load", return_value=(tokenizer, model, device)) as load, \
                patch.object(scorer, "_release") as release:
            scores = scorer.predict(["short text", " ", "long " * 30], show_progress=False)
        load.assert_called_once()
        release.assert_called_once_with("cpu")
        expected = _softmax(np.array(scorer.labels, dtype=float))
        np.testing.assert_allclose(scores[0], expected, rtol=1e-6)
        np.testing.assert_allclose(scores[2], expected, rtol=1e-6)
        np.testing.assert_allclose(scores[1], 1 / 6)
        np.testing.assert_allclose(scores.sum(axis=1), 1)
        self.assertEqual(len(tokenizer.calls), 3)  # 2 nonempty texts * 6 classes / 4
        self.assertTrue(all(len(c[0]) <= 4 and c[2]["truncation"] == "only_first" for c in tokenizer.calls))
        self.assertEqual(scorer.runtime["n_truncated"], 1)
        self.assertIsNone(scorer.length_info["lengths"][1])
        self.assertTrue(all(state == (False, False) for state in model.grad_states))
        self.assertEqual(backend_summary(scorer)["nli_status"], "applied")

    def test_loader_pins_revision_and_uses_eval(self):
        scorer = self.scorer()
        tokenizer, model, _ = fake_load(scorer)
        model.train()
        transformers = types.ModuleType("transformers")
        transformers.AutoTokenizer = types.SimpleNamespace(from_pretrained=Mock(return_value=tokenizer))
        transformers.AutoModelForSequenceClassification = types.SimpleNamespace(from_pretrained=Mock(return_value=model))
        with patch.dict(sys.modules, {"transformers": transformers}), patch.object(scorer, "_release"):
            scorer.predict(["synthetic text"], show_progress=False)
        options = transformers.AutoModelForSequenceClassification.from_pretrained.call_args.kwargs
        self.assertEqual(options["revision"], scorer.settings["revision"])
        self.assertTrue(options["use_safetensors"])
        self.assertFalse(options["trust_remote_code"])
        self.assertFalse(model.training)

    def test_failure_releases_and_does_not_fallback_silently(self):
        for failure in ("mapping", "context", "forward"):
            scorer = self.scorer(max_length=2048 if failure == "context" else 1024)
            tokenizer, model, device = fake_load(scorer)
            if failure == "mapping":
                model.config.label2id = {"LABEL_0": 0, "LABEL_1": 1, "LABEL_2": 2}
            if failure == "forward":
                model.forward = Mock(side_effect=RuntimeError("synthetic failure"))
            with patch.object(scorer, "_load", return_value=(tokenizer, model, device)), \
                    patch.object(scorer, "_release") as release, self.assertRaises((ValueError, RuntimeError)):
                scorer.predict(["hello"], show_progress=False)
            release.assert_called_once()

    def test_image_branch_unchanged_and_vlm_text_scale_ignored(self):
        texts = ["first", "", "last"]
        classes = np.array([[1., 0.], [0., 1.], [1., 1.], [-1., 0.], [0., -1.], [-1., -1.]])
        labels = self.scorer().labels
        args = ([None] * 3, texts, classes, labels)
        baseline = predict_single_label(FakeVLM(), *args, show_progress=False)
        outputs = []
        for scale in (None, 25, 100):
            scorer, vlm, diagnostics = self.scorer(), FakeVLM(), {}
            with patch.object(scorer, "_load", side_effect=lambda: fake_load(scorer)), patch.object(scorer, "_release"):
                outputs.append(predict_single_label(vlm, *args, text_scorer=scorer, text_logit_scale=scale,
                                                     diagnostics=diagnostics, show_progress=False))
            self.assertEqual(vlm.text_batches, [])
            np.testing.assert_array_equal(outputs[-1][2], baseline[2])
            np.testing.assert_array_equal(outputs[-1][1][1], baseline[2][1])
            self.assertNotIn("text", diagnostics["cosine"])
        for result in outputs[1:]:
            np.testing.assert_array_equal(result[1], outputs[0][1])
        np.testing.assert_allclose(outputs[0][1][0], .5 * outputs[0][2][0] + .5 * outputs[0][3][0])
        _, meta = text_scale_settings({"inference": {"text_logit_scale": 25}}, "fakeddit", FakeVLM(),
                                     {"task": "6way", "text_backend": "nli"}, .75)
        self.assertEqual(meta["text_logit_scale_status"], "ignored_nli")
        self.assertIsNone(meta["text_logit_scale_effective"])
        self.assertFalse(meta["text_scale_override_applied"])

    def test_image_only_empty_and_text_only_paths(self):
        for weight, texts in ((1., ["hello", "world"]), (.5, ["", " "]), (0., ["hello", "", "last"])):
            scorer, vlm = self.scorer(), FakeVLM()
            with patch.object(scorer, "_load", side_effect=lambda: fake_load(scorer)) as load, patch.object(scorer, "_release"):
                result = predict_single_label(vlm, [None] * len(texts), texts, np.ones((6, 2)), scorer.labels,
                                              batch_size=1, image_weight=weight, text_scorer=scorer, show_progress=False)
            np.testing.assert_allclose(result[1].sum(axis=1), 1)
            self.assertEqual(vlm.text_batches, [])
            if weight == 0:
                self.assertEqual(vlm.image_batches, [1])  # empty-text image fallback only
                np.testing.assert_array_equal(result[1][0], result[3][0])
            else:
                load.assert_not_called()


class NLIRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        registry = types.ModuleType("models.model_registry")
        registry.load_model = Mock(side_effect=AssertionError("no weights"))
        plotting = types.ModuleType("matplotlib")
        plotting.pyplot = types.ModuleType("matplotlib.pyplot")
        with patch.dict(sys.modules, {"models.model_registry": registry, "matplotlib": plotting,
                                      "matplotlib.pyplot": plotting.pyplot}):
            cls.common = importlib.import_module("run.common")
            cls.main = importlib.import_module("run_experiment")
            cls.demo = importlib.import_module("demo")
            cls.separate = importlib.import_module("run_fakeddit")

    def test_trial_expansion_and_summary_backend(self):
        task = {"task": "6way", "prompt_variants": ["current"], "text_backends": ["clip_text", "nli"]}
        rows = self.common.run_prompt_trials(lambda active: {"macro_f1": .1}, "stub", "fakeddit", task)
        self.assertEqual([r["text_backend"] for r in rows], ["clip_text", "nli"])
        self.assertNotIn("text_backend", task)
        self.assertIn("text_backend", self.common.compact_summary(rows).columns)
        for values in ([], "nli", ["nli", "nli"], ["unknown"]):
            with self.assertRaises(ValueError):
                self.common.prompt_trial_configs("fakeddit", {**task, "text_backends": values})

    def test_all_fakeddit_entrypoints_write_nli_metadata_and_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            picture = root / "synthetic.png"
            Image.new("RGB", (3, 3), "blue").save(picture)
            data = root / "synthetic.tsv"
            pd.DataFrame({"id": ["one", "two"], "text": ["synthetic title", "another synthetic title"],
                          "6_way_label": [0, 5], "image_path": [str(picture)] * 2}).to_csv(data, sep="\t", index=False)
            task = {"task": "6way", "task_type": "single_label", "data_path": str(data), "text_col": "text",
                    "max_samples": 2, "min_text_length": 0, "text_backend": "nli", "nli": {"max_length": 24}}
            config = {"datasets": {"fakeddit": task}, "inference": {"image_weight": .75, "text_logit_scale": 25},
                      "evaluation": {"enabled": True, "split": "val", "measure_truncation": True},
                      "output": {"raw_predictions_dir": str(root / "logs")}}
            outputs = []
            for entry, module in (("main", self.main), ("separate", self.separate), ("demo", self.demo)):
                with ExitStack() as stack:
                    stack.enter_context(patch.object(NLITextScorer, "_load", new=fake_load))
                    stack.enter_context(patch.object(NLITextScorer, "_release"))
                    model = FakeVLM()
                    if entry == "demo":
                        stack.enter_context(patch.multiple(module, LOG_DIR=str(root / "demo_logs"),
                                                          OUTPUT_DIR=str(root / "demo"), N_SAMPLES=2))
                        stack.enter_context(patch.object(module, "visualize"))
                        result = module.run_dataset(model, model.model_name, "fakeddit", task, .75, config=config)
                    else:
                        result = module.run_fakeddit(model, task, config)
                self.assertEqual(result["nli_status"], "applied")
                self.assertEqual(result["text_logit_scale_status"], "ignored_nli")
                log = Path(result["prediction_log_dir"])
                manifest = json.loads((log / "manifest.json").read_text())
                self.assertEqual(manifest["nli"]["runtime"]["revision_resolved"], "synthetic-sha")
                self.assertEqual(len(manifest["nli"]["hypothesis_spec"]["hypotheses"]), 6)
                self.assertEqual(manifest["truncation"]["context_length"], 24)
                snapshot = json.loads((log / "evaluation_snapshot.json").read_text())
                self.assertIn("text_backend", snapshot["effective_inference"])
                report = pd.read_csv(log / "diagnostics.tsv", sep="\t")
                text = report[report.branch == "text"].iloc[0]
                self.assertTrue(pd.isna(text["cosine_mean"]))
                self.assertEqual(text["input_token_count_unit"], "max_premise_hypothesis_pair_tokens")
                predictions = pd.read_csv(log / "raw_predictions.tsv", sep="\t")
                outputs.append(predictions)
                # VLM encodes only the six sets of class prompts, not sample texts.
                self.assertEqual(model.text_batches, [4] * 6)
            pd.testing.assert_frame_equal(outputs[0], outputs[1])
            pd.testing.assert_frame_equal(outputs[0], outputs[2])


if __name__ == "__main__":
    unittest.main()
