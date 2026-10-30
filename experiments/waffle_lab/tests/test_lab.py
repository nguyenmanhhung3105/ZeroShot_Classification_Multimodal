"""Kiểm tra bằng dữ liệu giả nhỏ; không dùng checkpoint hoặc dataset thực."""
from contextlib import redirect_stdout
from copy import deepcopy
import csv
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "tests"))

import numpy as np
from PIL import Image
import yaml

from experiments.waffle_lab import run as runner
from experiments.waffle_lab import analyze
from experiments.waffle_lab.core import attach_semantics, combine, VARIANTS
from research.upstream import author_api, verify_sources, generate_prompts
from test_upstream_projects import SyntheticBackend
import launch_runs as launcher


class CombiningTests(unittest.TestCase):
    def test_metaclip_backend_uses_registry_and_freezes_weights(self):
        from unittest.mock import Mock
        backend = Mock()
        with patch("models.model_registry.load_model", return_value=backend) as load:
            self.assertIs(runner.load_backend(None, "metaclip2_vith14", "cpu"), backend)
        load.assert_called_once_with("metaclip2_vith14", device="cpu")
        backend.model.requires_grad_.assert_called_once_with(False)

    def test_stability_can_change_prediction_and_does_not_just_mark_review(self):
        anchor = np.array([[[.2, .1]], [[.2, .1]]])
        # Hai phiếu A yếu và hai phiếu B mạnh: mean nghiêng B nhưng không đồng thuận.
        uncertain = [[.11, .10]] * 4 + [[0., 1.]] * 4
        # Cùng dự đoán B ở mọi cặp: được phép đóng góp đến noise_cap.
        certain = [[.1, .9]] * 8
        banks = {"semantic": anchor, "semantic_waffle": np.array([uncertain, certain])}
        stable = combine("stable_anchor", banks)
        fixed = combine("anchor_mix", banks)
        self.assertEqual(stable["scores"].argmax(1).tolist(), [0, 1])
        self.assertEqual(fixed["scores"].argmax(1).tolist(), [1, 1])
        np.testing.assert_allclose(stable["noise_weight"], [0., .5])

    def test_tied_votes_are_not_confident_agreement(self):
        result = combine("stable_anchor", {"semantic": [[[.2, .1]]],
                                            "semantic_waffle": np.zeros((1, 6, 2))})
        self.assertEqual(result["noise_weight"].tolist(), [0.])
        self.assertEqual(result["prompt_agreement"].tolist(), [0.])

    def test_common_prompt_offset_and_class_permutation_preserve_decisions(self):
        rng = np.random.default_rng(7)
        banks = {"semantic": rng.uniform(-.5, .5, (5, 3, 8)),
                 "semantic_waffle": rng.uniform(-.5, .5, (5, 30, 8))}
        before = combine("stable_anchor", banks)
        changed = {key: values + rng.normal(size=(5, values.shape[1], 1)) for key, values in banks.items()}
        np.testing.assert_array_equal(before["scores"].argmax(1), combine("stable_anchor", changed)["scores"].argmax(1))
        order = rng.permutation(8)
        permuted = combine("stable_anchor", {k: v[:, :, order] for k, v in banks.items()})
        np.testing.assert_allclose(before["scores"][:, order], permuted["scores"])
        np.testing.assert_allclose(before["noise_weight"], permuted["noise_weight"])

    def test_explicit_unit_control_detects_norm_confounded_ranking(self):
        banks = {"waffle": np.array([[[.3, .2], [.3, .2]]])}
        self.assertEqual(combine("waffle", banks)["scores"].argmax(1).tolist(), [0])
        self.assertEqual(combine("waffle_unit", banks, prototype_norm=[1., .25])["scores"].argmax(1).tolist(), [1])

    def test_invalid_scores_are_rejected(self):
        for value in (np.zeros((0, 3, 2)), np.array([[[np.nan, .1]]]), np.ones((3, 2))):
            with self.assertRaises(ValueError):
                combine("waffle", {"waffle": value})
        with self.assertRaises(ValueError):
            combine("stable_anchor", {"semantic": np.ones((1, 3, 2)),
                                       "semantic_waffle": np.ones((1, 3, 2))})


class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.api = author_api()
        cls.sources = verify_sources()

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.settings = yaml.safe_load((REPO / "experiments/waffle_lab/config.yaml").read_text())
        self.settings.update(max_samples=2, output_dir="results/trial")
        self.data = self.root / "data/processed/pro_crisismmd"
        (self.data / "images").mkdir(parents=True)
        self.fields = ["image_id", "image_path", "label_informative", "label_humanitarian", "source_split",
                       "label_source", "event", "clean_text"]
        self.rows = []
        for index, color in enumerate(("red", "green", "blue")):
            Image.new("RGB", (3, 3), color).save(self.data / f"images/{index}.png")
            self.rows.append({"image_id": str(index), "image_path": f"data/processed/pro_crisismmd/images/{index}.png",
                "label_informative": "informative" if index % 2 else "not_informative",
                "label_humanitarian": "not_humanitarian", "source_split": "train", "label_source": "image",
                "event": "synthetic_event", "clean_text": "DO NOT ENCODE THIS CAPTION"})
        self.write_rows()
        self.base = {"datasets": {"crisismmd": {"data_path": "data/processed/pro_crisismmd/samples.tsv",
            "image_dir": "data/processed/pro_crisismmd/images", "id_col": "image_id",
            "label_col_info": "label_informative", "label_col_human": "label_humanitarian", "source_split": "train"}}}
        (self.root / "configs/research/descriptors").mkdir(parents=True)
        for filename in ("crisismmd_info.json", "crisismmd_humanitarian.json"):
            shutil.copy2(REPO / "configs/research/descriptors" / filename,
                         self.root / "configs/research/descriptors" / filename)
        (self.root / "configs/experiment_config.yaml").write_text(yaml.safe_dump(self.base))
        self.root_patch = patch.object(runner, "ROOT", self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)

    def write_rows(self):
        with (self.data / "samples.tsv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.fields, delimiter="\t")
            writer.writeheader()
            writer.writerows(self.rows)

    def test_prompt_freezing_exact_author_reference_and_balanced_anchors(self):
        directory = self.root / "prepared"
        directory.mkdir()
        for task in ("informativeness", "humanitarian"):
            settings = {**self.settings, "task": task}
            plan = runner.freeze_plan(settings, directory, self.api, self.sources)
            names = plan["protocol"]["labels"]
            original, _ = generate_prompts(self.api, "waffle", names.values(), directory / "author_descriptors.json", 15, 0)
            self.assertEqual(plan["prompt_banks"]["waffle"], {key: original[name] for key, name in names.items()})
            for key in names:
                clean = plan["prompt_banks"]["semantic"][key]
                noisy = plan["prompt_banks"]["semantic_waffle"][key]
                self.assertEqual(len(noisy), 30)
                for anchor in clean:
                    self.assertEqual(sum(s.startswith(anchor.rstrip('.') + ", ") for s in noisy), 10)

    def test_prepare_only_never_reads_dataset_or_loads_weights(self):
        with patch.object(runner, "read_cohort", side_effect=AssertionError("Không đọc dữ liệu")), \
                patch.object(runner, "OpenAIClipBackend", side_effect=AssertionError("Không tải weights")), \
                redirect_stdout(io.StringIO()):
            directory = runner.run(self.settings, prepare_only=True)
        self.assertTrue((directory / "frozen_plan.json").is_file())
        self.assertFalse((directory / "summary.tsv").exists())

    def test_reader_is_bounded_and_never_uses_caption_for_admission(self):
        self.rows[0]["clean_text"] = ""
        self.rows[2]["label_informative"] = "INVALID_ROW_BEYOND_LIMIT"
        self.write_rows()
        selected = runner.read_cohort(self.settings, self.base)
        self.assertEqual([r["id"] for r in selected], ["0", "1"])
        self.assertTrue(all("clean_text" not in row for row in selected))

    def test_cohort_lock_rejects_image_label_or_order_change(self):
        selected = runner.read_cohort(self.settings, self.base)
        runner.lock_cohort(self.settings, selected)
        runner.lock_cohort(self.settings, selected)
        for changed in (list(reversed(selected)), [{**selected[0], "y_true": "informative"}, selected[1]],
                        [{**selected[0], "image_sha256": "changed"}, selected[1]]):
            with self.assertRaisesRegex(ValueError, "Cohort khác"):
                runner.lock_cohort(self.settings, changed)

    def test_bad_sources_and_missing_images_fail_without_replacement(self):
        for field, value in (("label_source", "text"), ("source_split", "test"), ("image_id", ""),
                             ("image_path", "data/processed/pro_crisismmd/images/missing.png")):
            with self.subTest(field=field):
                before = deepcopy(self.rows)
                self.rows[0][field] = value
                self.write_rows()
                with self.assertRaises(ValueError):
                    runner.read_cohort(self.settings, self.base)
                self.rows = before
        self.write_rows()

    def test_all_variants_end_to_end_with_synthetic_backend_and_one_image_pass(self):
        for variant in VARIANTS:
            settings = {**self.settings, "variant": variant, "output_dir": f"results/{variant}"}
            with patch.object(runner, "OpenAIClipBackend", SyntheticBackend), redirect_stdout(io.StringIO()):
                directory = runner.run(settings)
            backend = SyntheticBackend.instances[-1]
            self.assertEqual(backend.image_batches, [2])
            self.assertTrue(backend.unloaded)
            self.assertTrue(all("DO NOT ENCODE THIS CAPTION" not in text for batch in backend.text_batches for text in batch))
            manifest = json.loads((directory / "run_manifest.json").read_text())
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["inference_backend"], "synthetic_or_custom")
            self.assertFalse(manifest["label_fit"])
            self.assertTrue((directory / "summary.tsv").is_file())

    def test_decode_failure_releases_model_and_marks_failed(self):
        (self.data / "images/0.png").write_bytes(b"invalid-image")
        with patch.object(runner, "OpenAIClipBackend", SyntheticBackend), redirect_stdout(io.StringIO()):
            with self.assertRaises(OSError):
                runner.run(self.settings)
        self.assertTrue(SyntheticBackend.instances[-1].unloaded)
        manifest = json.loads((self.root / "results/trial/run_manifest.json").read_text())
        self.assertEqual(manifest["status"], "failed")

    def test_binary_auc_uses_informative_column_even_when_index_zero(self):
        labels = ["informative", "not_informative"]
        scores = np.array([[.4, .1], [.1, .5]])
        metrics = runner.evaluate(labels, labels, scores, labels)
        self.assertEqual(metrics["auroc"], 1.)

    def test_analysis_validates_real_vs_fake_and_pairs_same_cohort(self):
        with patch.object(runner, "OpenAIClipBackend", SyntheticBackend), redirect_stdout(io.StringIO()):
            directory = runner.run(self.settings)
        with self.assertRaisesRegex(ValueError, "giả/custom"):
            analyze.load_run(directory)
        loaded = analyze.load_run(directory, allow_synthetic=True)
        result = analyze.compare(loaded, loaded, bootstraps=20)
        self.assertEqual(result["delta_macro_f1_paired_bootstrap_95pct"], [0., 0.])
        self.assertEqual(result["mcnemar_exact_p_nominal"], 1.)
        self.assertEqual(result["flips"]["unchanged"], 2)
        downloaded = self.root / "downloaded"
        downloaded.mkdir()
        shutil.make_archive(str(downloaded / "run_artifacts"), "zip", root_dir=directory)
        zipped = analyze.load_run(downloaded, allow_synthetic=True)
        self.assertEqual(zipped["rows"], loaded["rows"])
        changed = deepcopy(loaded)
        changed["manifest"]["cohort_fingerprint"] = "different_cohort"
        with self.assertRaisesRegex(ValueError, "cohort_fingerprint"):
            analyze.compare(loaded, changed, bootstraps=20)

    def test_seed_summary_reports_accuracy_and_rejects_duplicate_seed(self):
        runs = [{"manifest": {"variant": "stable_anchor", "seed": seed,
                 "cohort_fingerprint": "same", "protocol_fingerprint": "same", "inference_backend": "synthetic"},
                 "metrics": {"accuracy": accuracy, "macro_f1": accuracy, "auroc": None}}
                for seed, accuracy in ((0, .4), (1, .6))]
        result = analyze.summarize(runs)[0]
        self.assertAlmostEqual(result["accuracy_mean"], .5)
        self.assertAlmostEqual(result["accuracy_std"], .1)
        with self.assertRaisesRegex(ValueError, "Lặp seed"):
            analyze.summarize([runs[0], runs[0]])


class LauncherIntegrationTests(unittest.TestCase):
    def test_command_generator_produces_separate_jobs_and_skips_deterministic_repeats(self):
        from experiments.waffle_lab.commands import commands
        self.assertEqual(len(list(commands(seeds=(0,)))), 14)
        self.assertEqual(len(list(commands(seeds=tuple(range(7))))), 74)
        with patch.object(launcher, "run_one") as launch:
            launcher.main(["--launcher-config", str(REPO / "experiments/waffle_lab/launcher.yaml"),
                           "--job", "waffle_info", "--method", "semantic_waffle", "--seed", "3",
                           "--max-samples", "5", "--use-kaggle", "true", "--dry-run"])
        config, name, dry_run = launch.call_args.args
        self.assertEqual(name, "waffle_info")
        self.assertTrue(dry_run)
        self.assertEqual(config["jobs"][name]["variant"], "semantic_waffle")
        self.assertEqual(config["jobs"][name]["seeds"], [3])

    def test_every_variant_freezes_correct_command_and_result_path(self):
        config = yaml.safe_load((REPO / "experiments/waffle_lab/launcher.yaml").read_text())
        config["use_kaggle"] = True
        for variant in VARIANTS:
            for job_name in ("waffle_info", "waffle_humanitarian"):
                with tempfile.TemporaryDirectory() as temporary:
                    active = deepcopy(config)
                    active["jobs"][job_name]["variant"] = variant
                    launcher.validate(active, [job_name])
                    launcher.validate_single_cloud_trial(active, [job_name])
                    _, job, output, command, _, _, research_file = launcher.job_plan(active, job_name, Path(temporary))
                    self.assertEqual(command[0], "experiments/waffle_lab/run.py")
                    self.assertEqual(output.parts[1], "waffle_lab_" + variant)
                    self.assertEqual(output.parts[-2], "openai_clip_vit_b_32")
                    research = yaml.safe_load(research_file.read_text())
                    self.assertEqual(research["variant"], variant)
                    self.assertEqual(research["base_config"], "configs/active.yaml")
                    self.assertEqual(research["max_samples"], 5)

    def test_multi_seed_or_model_is_rejected(self):
        config = yaml.safe_load((REPO / "experiments/waffle_lab/launcher.yaml").read_text())
        for key, value in (("seeds", [0, 1]), ("backbones", ["RN50", "ViT-B/32"])):
            active = deepcopy(config)
            active["jobs"]["waffle_info"][key] = value
            with self.assertRaises(ValueError):
                launcher.validate(active, ["waffle_info"])

    def test_metaclip_launcher_preserves_model_and_method(self):
        config = yaml.safe_load((REPO / "experiments/waffle_lab/launcher.yaml").read_text())
        config["jobs"]["waffle_info"]["backbones"] = ["metaclip2_vith14"]
        config["jobs"]["waffle_info"]["variant"] = "waffle"
        launcher.validate(config, ["waffle_info"])
        with tempfile.TemporaryDirectory() as temporary:
            _, _, output, _, _, _, research_file = launcher.job_plan(config, "waffle_info", Path(temporary))
            self.assertEqual(output.parts[-2], "metaclip2_vith14")
            research = yaml.safe_load(research_file.read_text())
            self.assertEqual(research["backbone"], "metaclip2_vith14")
            self.assertEqual(research["variant"], "waffle")


if __name__ == "__main__":
    unittest.main()
