"""Kiểm tra YAML mới, cấu hình cũ và các entry point không lệch nghiệp vụ."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import load_config
from run.common import get_image_weight
from text_scale import text_scale_settings


class ConfigTests(unittest.TestCase):
    def test_main_yaml_is_valid_and_exposes_dataset_options(self):
        config = load_config(Path(__file__).resolve().parents[1] / "configs/experiment_config.yaml")
        for dataset, options in config["datasets"].items():
            self.assertIsInstance(options["data_path"], str)
            self.assertGreaterEqual(get_image_weight(config, dataset), 0.)
            self.assertLessEqual(get_image_weight(config, dataset), 1.)
            self.assertIn("text_logit_scale", options["inference"])
        self.assertIn(config["evaluation"]["sample_lock_mode"], ("off", "create_or_verify", "verify"))

    def test_new_nested_settings_equal_old_overrides(self):
        class Model:
            logit_scale = 100.
        old = {"inference": {"image_weight": .5, "image_weight_overrides": {"fakeddit": .75},
                              "text_logit_scale": None, "text_logit_scale_overrides": {"fakeddit": 25}}}
        new = {"inference": {"image_weight": .5}, "datasets": {"fakeddit": {"inference": {
               "image_weight": .75, "text_logit_scale": 25}}}}
        for cfg in (old, new):
            self.assertEqual(get_image_weight(cfg, "fakeddit"), .75)
            self.assertEqual(text_scale_settings(cfg, "fakeddit", Model(), {}, .75)[0], 25)
        # null trong lựa chọn riêng phải bỏ override cũ, không rơi về giá trị 25.
        old["datasets"] = {"fakeddit": {"inference": {"text_logit_scale": None}}}
        self.assertIsNone(text_scale_settings(old, "fakeddit", Model(), {}, .75)[0])

    def test_invalid_task_limit_weight_rejected(self):
        for options in ({"task": "8way"}, {"task": "6way", "max_samples": True},
                        {"task": "6way", "inference": {"image_weight": 1.5}},
                        {"task": "6way", "task_type": "multi_label"}):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "config.json"
                path.write_text(json.dumps({"models": ["synthetic"], "datasets": {"fakeddit": options}}))
                with self.assertRaises(ValueError):
                    load_config(path)

    def test_main_and_demo_delegate_to_same_runners(self):
        import demo
        import run_experiment
        from run.run_fakeddit import run_fakeddit
        from run.run_crisismmd import run_crisismmd
        from run.run_mmimdb import run_mmimdb
        for name, fn in (("fakeddit", run_fakeddit), ("crisismmd", run_crisismmd), ("mmimdb", run_mmimdb)):
            self.assertIs(demo.RUNNERS[name], fn)
            self.assertIs(run_experiment.RUNNERS[name], fn)


if __name__ == "__main__":
    unittest.main()
