"""Audit regressions using synthetic rows/images and model stubs only."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import copy
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

from evaluation_audit import prepare_audit, finish_audit, save_audit, token_lengths
from inference import predict_single_label, predict_multi_label
from test_runtime_fixes import FakeVLM


class SimpleTokenizer:
    def encode(self, text):
        return list(range(len(text.split())))


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.frame = pd.DataFrame({"id": ["one", "two", "three"],
                                   "text": ["alpha beta", "", "gamma delta epsilon zeta"],
                                   "label": [0, 1, 0], "image_path": ["synthetic"] * 3})
        self.task = {"data_path": "synthetic.tsv", "max_samples": 3}
        self.config = {"evaluation": {"enabled": True, "split": "val",
                       "sample_lock_mode": "create_or_verify", "sample_lock_dir": str(self.root / "locks")}}
        self.model = FakeVLM()
        self.model.tokenizer = SimpleTokenizer()
        self.model.model = types.SimpleNamespace(context_length=5)

    def prepare(self, frame=None, config=None, task=None, scope="main"):
        return prepare_audit(self.frame if frame is None else frame, task or self.task,
                             self.config if config is None else config, "synthetic", "classes",
                             "id", "text", "label", {0: ["class zero"], 1: ["class one"]}, scope=scope)

    def test_lock_detects_changed_samples_and_preserves_original(self):
        first = self.prepare()
        lock = Path(first["sample_lock_path"])
        original = lock.read_bytes()
        self.assertEqual(first["sample_set_id"], self.prepare()["sample_set_id"])
        modified = []
        for column, value in (("id", "other"), ("text", "changed"), ("label", 1), ("image_path", "other.jpg")):
            frame = self.frame.copy()
            frame.loc[0, column] = value
            modified.append(frame)
        modified.extend([self.frame.iloc[::-1].copy(), self.frame.iloc[:2].copy()])
        for frame in modified:
            with self.assertRaisesRegex(ValueError, "Sample lock mismatch"):
                self.prepare(frame=frame)
        with self.assertRaisesRegex(ValueError, "Sample lock mismatch"):
            self.prepare(task={**self.task, "max_samples": 4})
        self.assertEqual(original, lock.read_bytes())
        self.assertNotEqual(first["sample_lock_path"], self.prepare(scope="demo")["sample_lock_path"])
        self.assertNotIn("evaluation_audit", self.frame.attrs)

    def test_lock_creation_is_atomic_across_concurrent_runs(self):
        with ThreadPoolExecutor(max_workers=6) as pool:
            audits = list(pool.map(lambda _: self.prepare(frame=self.frame.copy()), range(12)))
        self.assertEqual(len({a["sample_set_id"] for a in audits}), 1)
        self.assertEqual(len(json.loads(Path(audits[0]["sample_lock_path"]).read_text())["samples"]), 3)
        self.assertFalse(list((self.root / "locks").glob(".cohort-*.tmp")))

    def test_lock_modes_and_disabled_audit(self):
        verify = copy.deepcopy(self.config)
        verify["evaluation"]["sample_lock_mode"] = "verify"
        with self.assertRaisesRegex(ValueError, "Sample lock missing"):
            self.prepare(config=verify)
        self.assertFalse((self.root / "locks").exists())
        self.assertIsNone(self.prepare(config={}))
        self.assertEqual(finish_audit(None, None, None, None, None, None, None, None, None, None, None), {})
        off = copy.deepcopy(self.config)
        off["evaluation"]["sample_lock_mode"] = "off"
        self.assertIsNone(self.prepare(config=off)["sample_lock_path"])
        self.assertFalse((self.root / "locks").exists())

    def test_single_label_metrics_truncation_and_output_files(self):
        audit = self.prepare()
        image = np.array([[0.9, 0.1], [0.1, 0.9], [0.1, 0.9]])
        text = np.array([[0.1, 0.9], [0.5, 0.5], [0.8, 0.2]])
        fused = (image + text) / 2
        fused[1] = image[1]
        summary = finish_audit(audit, self.model, self.frame, self.frame.text.tolist(), [0, 1, 0],
                               [0, 1], fused, image, text, [False, True, False], 0.5)
        self.assertAlmostEqual(summary["image_macro_f1"], 2 / 3)
        self.assertAlmostEqual(summary["text_macro_f1"], 1 / 3)
        self.assertAlmostEqual(summary["fusion_macro_f1"], 2 / 3)
        self.assertEqual(summary["text_n_samples"], 2)
        self.assertEqual(audit["truncation"]["n_truncated"], 1)
        self.assertEqual(audit["truncation"]["truncated_rate"], 0.5)
        self.assertEqual(audit["text_lengths"][0]["token_count"], 4)
        self.assertIsNone(audit["text_lengths"][1]["truncated"])
        with patch("evaluation_audit.subprocess.run", side_effect=FileNotFoundError):
            save_audit(audit, self.root)
        report = pd.read_csv(self.root / "diagnostics.tsv", sep="\t")
        self.assertEqual(report.branch.tolist(), ["image", "text", "fusion"])
        self.assertTrue(np.isfinite(report.entropy).all())
        snapshot = json.loads((self.root / "evaluation_snapshot.json").read_text())
        self.assertEqual(snapshot["config_id"], summary["config_id"])
        self.assertIsNone(snapshot["git_hash"])
        self.assertIn("inference.py", snapshot["source_sha256"])
        self.assertNotIn("evaluation_audit", self.frame.attrs)

    def test_multilabel_relative_metrics_and_fallback(self):
        frame = self.frame.iloc[:2].copy()
        audit = self.prepare(frame=frame)
        scores = np.array([[1.4, -0.7, -0.7], [0., 0., 0.]])
        summary = finish_audit(audit, self.model, frame, ["a", "b"], [["a"], ["b"]],
                               ["a", "b", "c"], scores, scores, scores, [False, False], 0.5,
                               multilabel=True, decision_meta={"score_type": "relative_z", "relative_z_threshold": 1.0})
        self.assertAlmostEqual(summary["fusion_micro_f1"], 0.5)
        self.assertAlmostEqual(summary["fusion_samples_f1"], 0.5)
        self.assertAlmostEqual(summary["fusion_macro_f1"], 2 / 9)
        for row in audit["diagnostics"]:
            self.assertEqual(row["mean_predicted_labels"], 1)
            self.assertEqual(row["fallback_rate"], 0.5)
            self.assertNotIn("entropy", row)

    def test_audit_does_not_change_predictions_and_omits_unused_branch(self):
        for predictor, options in ((predict_single_label, {}), (predict_multi_label, {}),
                                   (predict_multi_label, {"decision_mode": "relative_z"})):
            for weight in (0., 0.5, 1.):
                args = ([object()] * 3, self.frame.text.tolist(), np.eye(2), [0, 1])
                before = predictor(FakeVLM(), *args, image_weight=weight, show_progress=False, **options)
                diagnostics = {}
                after = predictor(FakeVLM(), *args, image_weight=weight, show_progress=False,
                                  diagnostics=diagnostics, **options)
                for a, b in zip(before, after):
                    if isinstance(a, list):
                        self.assertEqual(a, b)
                    else:
                        np.testing.assert_array_equal(a, b)
                if weight == 1 and predictor is predict_single_label:
                    audit = self.prepare()
                    result = finish_audit(audit, self.model, self.frame, self.frame.text.tolist(), [0, 1, 0],
                                          [0, 1], after[1], after[2], after[3], after[-1], 1.)
                    self.assertEqual(result["text_n_samples"], 0)
                    self.assertNotIn("text_macro_f1", result)
                    self.assertNotIn("text", diagnostics["cosine"])

    def test_hf_token_lengths_cleaning_and_unknown_backend(self):
        backend = types.SimpleNamespace(encode=lambda text, **kw: list(range(len(text.split()) + 1)))
        tokenizer = type("HFTokenizer", (), {})()
        tokenizer.clean_fn = lambda text: text.replace("_", " ")
        tokenizer.tokenizer = backend
        self.model.tokenizer = tokenizer
        lengths, context, status = token_lengths(self.model, [" one_two ", "", "three"])
        self.assertEqual((lengths, context, status), ([3, None, 2], 5, "measured"))
        self.model.tokenizer = object()
        with self.assertWarns(UserWarning):
            self.assertEqual(token_lengths(self.model, ["text"]), ([None], 5, "unknown"))


class AuditRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        registry = types.ModuleType("models.model_registry")
        registry.load_model = lambda *a, **k: (_ for _ in ()).throw(AssertionError("No pretrained loading"))
        plotting = types.ModuleType("matplotlib")
        plotting.pyplot = types.ModuleType("matplotlib.pyplot")
        with patch.dict(sys.modules, {"models.model_registry": registry, "matplotlib": plotting,
                                      "matplotlib.pyplot": plotting.pyplot}):
            cls.main = importlib.import_module("run_experiment")
            cls.demo = importlib.import_module("demo")
            cls.separate = {name: importlib.import_module(f"run_{name}") for name in ("fakeddit", "crisismmd", "mmimdb")}

    def test_all_runners_with_real_synthetic_reader_inference_and_logging(self):
        cases = [("fakeddit", "2way", "2_way_label", [0, 1]),
                 ("fakeddit", "6way", "6_way_label", [0, 1]),
                 ("crisismmd", "informativeness", "label_informative", ["informative", "not_informative"]),
                 ("crisismmd", "humanitarian", "label_humanitarian", ["not_humanitarian", "other_relevant_information"]),
                 ("mmimdb", "genres", "genres", ["Action|Drama", "Short"])]
        for dataset, task, column, truth in cases:
            with self.subTest(dataset=dataset, task=task), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                picture = root / "synthetic.png"
                Image.new("RGB", (3, 3), "red").save(picture)
                data_path = root / "synthetic.tsv"
                # Short first row tests consistent Fakeddit filtering in all entry points.
                pd.DataFrame({"id": ["short", "one", "two"], "text": ["tiny", "long title " * 4, "another title " * 4],
                              column: [truth[0], *truth], "image_path": [str(picture)] * 3}).to_csv(data_path, sep="\t", index=False)
                task_cfg = {"task": task, "task_type": "multi_label" if dataset == "mmimdb" else "single_label",
                            "data_path": str(data_path), "id_col": "id", "text_col": "text", "max_samples": 2,
                            "label_col": column, "decision_mode": "relative_z"}
                config = {"datasets": {dataset: task_cfg}, "inference": {"batch_size": 2, "image_weight": 0.5},
                          "output": {"raw_predictions_dir": str(root / "logs")},
                          "evaluation": {"enabled": True, "sample_lock_mode": "create_or_verify",
                                         "sample_lock_dir": str(root / "locks"), "split": "val"}}
                outputs = []
                for entry, module in (("main", self.main), ("separate", self.separate[dataset]), ("demo", self.demo)):
                    model = FakeVLM()
                    model.tokenizer, model.model = SimpleTokenizer(), types.SimpleNamespace(context_length=5)
                    with ExitStack() as stack:
                        if entry == "demo":
                            stack.enter_context(patch.multiple(module, LOG_DIR=str(root / "demo_logs"),
                                                              OUTPUT_DIR=str(root / "demo"), N_SAMPLES=2, BATCH_SIZE=2))
                            stack.enter_context(patch.object(module, "visualize"))
                            result = module.run_dataset(model, model.model_name, dataset, task_cfg, 0.5, config=config)
                        else:
                            result = getattr(module, f"run_{dataset}")(model, task_cfg, config)
                    outputs.append(result)
                    self.assertAlmostEqual(result["fusion_macro_f1"], result["macro_f1"])
                    self.assertEqual(result["fusion_n_samples"], 2)
                    self.assertIn("image_macro_f1", result)
                    self.assertIn("text_macro_f1", result)
                    # Prompts plus one input batch; diagnostics never re-encode samples.
                    self.assertEqual(model.image_batches, [2])
                self.assertEqual(outputs[0]["sample_set_id"], outputs[1]["sample_set_id"])
                snapshots = list(root.glob("**/evaluation_snapshot.json"))
                self.assertEqual(len(snapshots), 3)
                for path in snapshots:
                    snapshot = json.loads(path.read_text())
                    self.assertEqual(snapshot["truncation"]["status"], "measured")
                    manifest = json.loads((path.parent / "manifest.json").read_text())
                    self.assertEqual(manifest["sample_set_id"], snapshot["sample_set_id"])
                    self.assertEqual(manifest["config_id"], snapshot["config_id"])
                    self.assertFalse(manifest["split_verified"])
                    if dataset == "fakeddit":
                        self.assertEqual([r["id"] for r in snapshot["cohort"]["samples"]], ["one", "two"])
                    report = pd.read_csv(path.parent / "diagnostics.tsv", sep="\t")
                    self.assertEqual(report.branch.tolist(), ["image", "text", "fusion"])
                    self.assertEqual(report.loc[report.branch == "text", "input_status"].item(), "measured")
                    self.assertEqual(len(pd.read_csv(path.parent / "text_lengths.tsv", sep="\t")), 2)
                # Reordering locked samples must stop before encoding input images.
                changed = pd.read_csv(data_path, sep="\t", dtype=str).iloc[::-1]
                changed.to_csv(data_path, sep="\t", index=False)
                model = FakeVLM()
                with self.assertRaisesRegex(ValueError, "Sample lock mismatch"):
                    getattr(self.main, f"run_{dataset}")(model, task_cfg, config)
                self.assertEqual(model.image_batches, [])


if __name__ == "__main__":
    unittest.main()
