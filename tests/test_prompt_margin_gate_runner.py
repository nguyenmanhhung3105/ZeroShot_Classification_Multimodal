"""Kiểm tra job chuyển prompt tạo summary từ hai lượt cùng model/cohort giả."""

from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/uncertainty"))
import run_crisismmd_prompt_gate as runner


class PromptMarginGateRunnerTests(unittest.TestCase):
    def test_runner_uses_adapted_prompt_only_for_low_margin(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            job = {"id": "test_prompt_gate", "dataset": "crisismmd", "task": "informativeness",
                   "models": [runner.MODEL], "max_samples": 2, "output_dir": str(output),
                   "prompt_gate": {"method": "dclip_adapted", "margin_threshold": 0.10}}
            job_path = output / "job.yaml"
            job_path.write_text(yaml.safe_dump(job), encoding="utf-8")
            config = {"models": [runner.MODEL], "datasets": {"crisismmd": {
                "task": "informativeness", "prompt_aggregation": "mean_embed",
                "image_text_relation": {"enabled": False}, "inference": {"image_weight": 0.5}}},
                "evaluation": {}, "output": {"raw_predictions_dir": "unused"}}

            class Model:
                def unload(self):
                    pass

            def fake_run(model, task, active_config):
                method = "dclip_adapted" if "_research_image_prompts" in task else "baseline"
                path = Path(active_config["output"]["raw_predictions_dir"]) / method
                path.mkdir(parents=True)
                probs = [0.2, 0.1] if method == "dclip_adapted" else [0.52, 0.9]
                fields = ["id", "text", "image_path", "y_true", "y_pred",
                          "score_informative", "score_not_informative",
                          "score_image_informative", "score_image_not_informative"]
                with (path / "raw_predictions.tsv").open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
                    writer.writeheader()
                    for index, p in enumerate(probs):
                        writer.writerow({"id": str(index), "text": "same", "image_path": f"{index}.jpg",
                                         "y_true": "not_informative" if index == 0 else "informative",
                                         "y_pred": "informative" if p > 0.5 else "not_informative",
                                         "score_informative": p, "score_not_informative": 1 - p,
                                         "score_image_informative": p,
                                         "score_image_not_informative": 1 - p})
                (path / "manifest.json").write_text(json.dumps({"dataset": "crisismmd",
                    "task": "informativeness", "model": runner.MODEL, "n_total_samples": 2,
                    "sample_set_id": "same"}), encoding="utf-8")
                return {"prediction_log_dir": str(path)}

            with patch.object(runner, "load_config", return_value=config), \
                    patch.object(runner, "load_model", return_value=Model()), \
                    patch.object(runner, "run_crisismmd", side_effect=fake_run), \
                    redirect_stdout(io.StringIO()):
                runner.main(["--config", "unused.yaml", "--job-config", str(job_path)])

            summary = next(output.rglob("summary.tsv"))
            with summary.open(encoding="utf-8", newline="") as stream:
                by_method = {row["method"]: row for row in csv.DictReader(stream, delimiter="\t")}
            self.assertEqual(by_method["prompt_margin_gate"]["accuracy"], "1.0")
            self.assertEqual(by_method["prompt_margin_gate"]["wrong_to_right"], "1")
            self.assertEqual(by_method["prompt_margin_gate"]["right_to_wrong"], "0")


if __name__ == "__main__":
    unittest.main()
