"""Kiểm tra phép ghép score và ngăn so sánh nhầm cohort."""

from pathlib import Path
from contextlib import redirect_stdout
from copy import deepcopy
import csv
import io
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/ensemble"))

import run_ensemble as ensemble
from run_ensemble import combine_runs, combine_multilabel_runs


class EnsembleTests(unittest.TestCase):
    @staticmethod
    def rows(scores, truths):
        result = []
        for i, ((a, b), truth) in enumerate(zip(scores, truths)):
            result.append({"id": str(i), "text": f"text {i}", "image_path": f"image {i}",
                           "y_true": truth, "y_pred": "a" if a >= b else "b",
                           "score_a": str(a), "score_b": str(b)})
        return result

    def test_gate_only_changes_low_margin_and_counts_corrections(self):
        truths = ["b", "a", "b"]
        runs = {
            "metaclip2_vith14": self.rows([(0.52, 0.48), (0.9, 0.1), (0.2, 0.8)], truths),
            "dfn5b_vith14": self.rows([(0.1, 0.9), (0.1, 0.9), (0.4, 0.6)], truths),
            "siglip2_so400m": self.rows([(0.2, 0.8), (0.2, 0.8), (0.7, 0.3)], truths),
        }
        summary, comparison, detail = combine_runs(runs, ["a", "b"])
        by_method = {row["method"]: row for row in summary}
        self.assertEqual(detail["uncertain_count"], 1)
        self.assertEqual(by_method["equal_mean"]["wrong_to_right"], 1)
        self.assertEqual(by_method["equal_mean"]["right_to_wrong"], 1)
        self.assertEqual(by_method["margin_gate"]["wrong_to_right"], 1)
        self.assertEqual(by_method["margin_gate"]["right_to_wrong"], 0)
        self.assertEqual([row["gated_pred"] for row in comparison], truths)

    def test_rejects_mismatched_cohort(self):
        runs = {"metaclip2_vith14": self.rows([(0.52, 0.48)], ["b"]),
                "dfn5b_vith14": self.rows([(0.1, 0.9)], ["b"])}
        runs["dfn5b_vith14"][0]["text"] = "different"
        with self.assertRaisesRegex(ValueError, "Cohort không khớp"):
            combine_runs(runs, ["a", "b"])

    def test_fakeddit_numeric_labels_are_compared_as_strings(self):
        runs = {"metaclip2_vith14": self.rows([(0.6, 0.4), (0.4, 0.6)], ["0", "1"]),
                "dfn5b_vith14": self.rows([(0.7, 0.3), (0.2, 0.8)], ["0", "1"])}
        for rows in runs.values():
            for row in rows:
                row["score_0"], row["score_1"] = row.pop("score_a"), row.pop("score_b")
                row["y_pred"] = "0" if float(row["score_0"]) >= float(row["score_1"]) else "1"
        summary, _, _ = combine_runs(runs, [0, 1])
        self.assertEqual(next(row for row in summary if row["method"] == "equal_mean")["accuracy"], 1.0)

    def test_multilabel_mean_uses_independent_scores(self):
        def row(index, truth, pred, a, b):
            return {"id": str(index), "text": "text", "image_path": "image",
                    "y_true": truth, "y_pred": pred, "prob_A": str(a), "prob_B": str(b)}
        runs = {"metaclip2_vith14": [row(1, "A", "A", 0.8, 0.2), row(2, "B", "A", 0.6, 0.4)],
                "dfn5b_vith14": [row(1, "A", "A", 0.9, 0.1), row(2, "B", "B", 0.1, 0.9)]}
        summary, comparison, _ = combine_multilabel_runs(
            runs, ["A", "B"], {"decision_mode": "fixed_threshold", "threshold": 0.5},
            "metaclip2_vith14")
        self.assertEqual(next(row for row in summary if row["method"] == "equal_mean")["exact_match"], 1.0)
        self.assertEqual(comparison[1]["ensemble_pred"], "B")

    def test_runner_writes_isolated_summary_with_mocked_models(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "experiment"
            config = {"models": [], "datasets": {"crisismmd": {"task": "informativeness",
                "prompt_aggregation": "mean_embed"}}, "evaluation": {},
                "output": {"raw_predictions_dir": "unused"}}
            job = {"id": "crisis_test", "dataset": "crisismmd", "task": "informativeness",
                   "max_samples": 2, "models": ["metaclip2_vith14", "dfn5b_vith14"],
                   "output_dir": str(destination),
                   "ensemble": {"methods": ["equal_mean"], "margin_threshold": 0.1}}
            job_file = Path(directory) / "job.yaml"
            job_file.write_text(yaml.safe_dump(job))

            class Model:
                def __init__(self, name):
                    self.model_name = name

                def unload(self):
                    pass

            def fake_runner(model, task, cfg):
                path = destination / model.model_name
                path.mkdir(parents=True)
                scores = [(0.6, 0.4), (0.2, 0.8)]
                with (path / "raw_predictions.tsv").open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=["id", "text", "image_path", "y_true",
                        "y_pred", "score_a", "score_b"], delimiter="\t")
                    writer.writeheader()
                    for i, (a, b) in enumerate(scores):
                        writer.writerow({"id": i, "text": f"text {i}", "image_path": f"image {i}",
                                         "y_true": "a" if i == 0 else "b", "y_pred": "a" if a > b else "b",
                                         "score_a": a, "score_b": b})
                (path / "manifest.json").write_text(json.dumps({"sample_set_id": "same",
                                                                  "prompt_id": "same"}))
                return {"prediction_log_dir": str(path)}

            with patch.object(ensemble, "load_config", return_value=deepcopy(config)), \
                    patch.object(ensemble, "load_model", side_effect=lambda name, device=None: Model(name)), \
                    patch.dict(ensemble.RUNNERS, {"crisismmd": fake_runner}), \
                    patch.object(ensemble, "labels_for", return_value=["a", "b"]), \
                    redirect_stdout(io.StringIO()):
                ensemble.main(["--config", "unused.yaml", "--job-config", str(job_file)])
            self.assertEqual(len(list(destination.rglob("summary.tsv"))), 1)


if __name__ == "__main__":
    unittest.main()
