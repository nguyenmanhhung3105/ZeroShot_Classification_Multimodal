"""Prompt comparison tests without real data or model weights."""
import importlib
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from prompts import fakeddit_prompts as prompts


class PromptVariantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        registry = types.ModuleType("models.model_registry")
        registry.load_model = Mock()
        with patch.dict(sys.modules, {"models.model_registry": registry}):
            cls.common = importlib.import_module("run.common")

    def test_baseline_unchanged(self):
        self.assertIs(prompts.get_prompt_set("2way"), prompts.PROMPTS_2WAY)
        self.assertIs(prompts.get_prompt_set("6way"), prompts.PROMPTS_6WAY)

    def test_variants_preserve_ids_and_counts(self):
        for variant, count in (("current", 4), ("class_name", 1), ("descriptors", 4)):
            rendered = prompts.get_prompt_set("6way", variant)
            self.assertEqual(list(rendered), list(range(6)))
            self.assertTrue(all(len(v) == count and all(s.strip() for s in v)
                                for v in rendered.values()))
            self.assertEqual(prompts.prompt_metadata({"prompt_variant": variant}, rendered)
                             ["prompts_per_label"], count)

    def test_invalid_variants_rejected(self):
        for task, variant in (("2way", "descriptors"), ("6way", "unknown")):
            with self.assertRaises(ValueError):
                prompts.get_prompt_set(task, variant)
        for variants in ([], "current", ["current", "current"], ["unknown"]):
            with self.assertRaises(ValueError):
                self.common.prompt_trial_configs("fakeddit", {"task": "6way", "prompt_variants": variants})

    def test_trials_keep_settings_and_report_individual_errors(self):
        config = {"task": "6way", "max_samples": 5,
                  "prompt_variants": ["current", "class_name", "descriptors"]}
        seen = []
        def run(active):
            seen.append(active)
            if active["prompt_variant"] == "class_name":
                raise ValueError("synthetic failure")
            return {"macro_f1": .3}
        rows = self.common.run_prompt_trials(run, "fake", "fakeddit", config)
        self.assertEqual([r["prompt_variant"] for r in rows], config["prompt_variants"])
        self.assertIn("error", rows[1])
        self.assertEqual(rows[2]["macro_f1"], .3)
        self.assertTrue(all(c["max_samples"] == 5 for c in seen))
        self.assertNotIn("prompt_variant", config)

    def test_model_loaded_once_for_three_trials(self):
        config = {"models": ["fake"], "datasets": {"fakeddit": {
            "enabled": True, "task": "6way",
            "prompt_variants": ["current", "class_name", "descriptors"]}},
            "output": {"results_dir": "unused", "raw_predictions_dir": "unused"}}
        model = Mock()
        with patch.object(self.common, "load_model", return_value=model) as load, \
                patch.object(self.common.os, "makedirs"), \
                patch.object(self.common, "save_dataset_summary", return_value="unused.tsv"):
            runner = Mock(return_value={"macro_f1": .3})
            result = self.common.run_all_models_for_dataset("fakeddit", runner, config)
        load.assert_called_once()
        model.unload.assert_called_once()
        self.assertEqual(runner.call_count, 3)
        self.assertEqual(len(result), 3)


if __name__ == "__main__":
    unittest.main()
