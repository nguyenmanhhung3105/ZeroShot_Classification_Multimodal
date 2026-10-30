"""Kiểm tra launcher đa dataset bằng dữ liệu giả rất nhỏ, không gọi mạng/model."""

import ast
from contextlib import redirect_stderr, redirect_stdout
import csv
import io
import json
from pathlib import Path
import shlex
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import launch_runs as launcher
import list_kaggle_commands as command_list
import audit_kaggle_cohorts as cpu_audit
from kaggle_worker import find_dataset, tree_sha256, unpack_archive, verify_cohort


class LauncherTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        (self.root / "configs/research/descriptors").mkdir(parents=True)
        (self.root / "configs/prompts").mkdir(parents=True)
        shutil.copy2(REPO / "configs/prompts/fakeddit_6way_variants.json",
                     self.root / "configs/prompts/fakeddit_6way_variants.json")
        (self.root / "src/research").mkdir(parents=True)
        (self.root / "src/research/run_adapted_methods.py").write_text("print('test')\n")
        (self.root / "experiments/ensemble").mkdir(parents=True)
        (self.root / "experiments/ensemble/run_ensemble.py").write_text("print('test')\n")
        (self.root / "scripts").mkdir()
        shutil.copy2(REPO / "scripts/kaggle_worker.py", self.root / "scripts/kaggle_worker.py")
        shutil.copy2(REPO / "configs/experiment_config.yaml", self.root / "configs/experiment_config.yaml")
        fixture_config = yaml.safe_load((self.root / "configs/experiment_config.yaml").read_text())
        for task in fixture_config["datasets"].values():
            task.pop("cohort_manifest", None)
        (self.root / "configs/experiment_config.yaml").write_text(yaml.safe_dump(fixture_config))
        shutil.copy2(REPO / "configs/research/adapted.yaml", self.root / "configs/research/adapted.yaml")
        shutil.copy2(REPO / "configs/research/upstream.yaml", self.root / "configs/research/upstream.yaml")
        shutil.copy2(REPO / "configs/research/sources.json", self.root / "configs/research/sources.json")
        (self.root / "requirements").mkdir()
        (self.root / "requirements.txt").write_text("pyyaml\n")
        (self.root / "requirements/research.txt").write_text("regex\n")
        (self.root / "requirements/vlm_reasoning.txt").write_text("accelerate\n")
        for dataset, stem in (("fakeddit", "fakeddit_sample"),
                              ("crisismmd", "pre_crisismmd"), ("mmimdb", "mmimdb_sample")):
            folder = self.root / f"data/processed/pro_{dataset}"
            (folder / "images").mkdir(parents=True)
            for name in ("a.jpg", "b.jpg", "c.jpg"):
                (folder / "images" / name).write_bytes(b"fake-image")
            with (folder / f"{stem}.tsv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["id", "tweet_id", "image_path",
                    "clean_title", "clean_text", "plot", "2_way_label", "6_way_label",
                    "label_informative", "label_humanitarian", "genres"], delimiter="\t")
                writer.writeheader()
                for name in ("a.jpg", "b.jpg", "c.jpg"):
                    writer.writerow({"id": name, "tweet_id": name, "image_path": name,
                        "clean_title": "valid text longer than twenty characters",
                        "clean_text": name, "plot": name, "2_way_label": "0", "6_way_label": "0",
                        "label_informative": "informative", "label_humanitarian": "not_humanitarian",
                        "genres": "Comedy"})
        self.config = yaml.safe_load((REPO / "configs/launcher.yaml").read_text())
        self.config["kaggle"]["username"] = "testuser"
        for job in self.config["jobs"].values():
            job["max_samples"] = 2 if job["kind"] != "relation_ablation" else 2
        self.root_patch = patch.object(launcher, "ROOT", self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)

    def test_every_standard_dataset_builds_bounded_cloud_bundle(self):
        self.config["use_kaggle"] = True
        self.config["kaggle"]["data_mode"] = "bounded"
        for name in ("fakeddit_2way", "fakeddit_6way", "crisismmd_info",
                     "crisismmd_humanitarian", "mmimdb_genres"):
            with self.subTest(name=name), patch.object(launcher, "run_cli") as cli, redirect_stdout(io.StringIO()):
                launcher.validate(self.config, [name])
                launcher.run_one(self.config, name, dry_run=True)
                cli.assert_not_called()
        stages = list((self.root / "results/_system/launches/staging").iterdir())
        self.assertEqual(len(stages), 5)
        for stage in stages:
            with zipfile.ZipFile(stage / "data/data_bundle.zip") as archive:
                self.assertEqual(sum(item.startswith("data/processed/") for item in archive.namelist()), 3)

    def test_fakeddit_filter_scans_only_until_enough_valid_rows(self):
        source = self.root / "data/processed/pro_fakeddit/fakeddit_sample.tsv"
        with source.open("r+", newline="") as stream:
            rows = list(csv.DictReader(stream, delimiter="\t"))
            rows[0]["clean_title"] = "short"
            stream.seek(0)
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
            stream.truncate()
        experiment = yaml.safe_load((self.root / "configs/experiment_config.yaml").read_text())
        path = self.root / "data_bundle.zip"
        _, selected, _, scanned = launcher.archive_data(path, experiment, "fakeddit", 2)
        self.assertEqual((selected, scanned), (2, 3))
        with zipfile.ZipFile(path) as archive:
            text = archive.read("data/processed/pro_fakeddit/fakeddit_sample.tsv").decode()
            self.assertEqual(len(list(csv.DictReader(io.StringIO(text), delimiter="\t"))), 3)

    def test_research_jobs_use_yaml_methods_and_separate_output(self):
        for name in ("demo_mmimdb_genres", "adapted_mmimdb_genres",
                     "upstream_fakeddit_2way", "relation_fakeddit_6way"):
            with self.subTest(name=name):
                launcher.validate(self.config, [name])
                stage = self.root / f"{name}_stage"
                stage.mkdir()
                _, job, output, command, _, _, research = launcher.job_plan(self.config, name, stage)
                self.assertTrue(str(output).startswith("results/"))
                self.assertEqual(output.parts[-3], "samples_2")
                self.assertIn(output.parts[-2], {"multi_model", "metaclip2_vith14", "openai_clip_vit_b_32"})
                self.assertEqual(job["output_dir"], str(output))
                if research:
                    self.assertEqual(yaml.safe_load(research.read_text())["methods"], job["methods"])
                    self.assertIn("--config", command)

    def test_crisismmd_agreement_jobs_enable_relation_only_for_selected_job(self):
        for name in ("crisismmd_info_agreement", "crisismmd_humanitarian_agreement",
                     "adapted_crisismmd_info_agreement", "adapted_crisismmd_humanitarian_agreement"):
            with self.subTest(name=name):
                launcher.validate(self.config, [name])
                stage = self.root / (name + "_stage")
                stage.mkdir()
                experiment, _, _, _, active, _, research = launcher.job_plan(self.config, name, stage)
                relation = experiment["datasets"]["crisismmd"]["image_text_relation"]
                self.assertEqual(relation["mode"], "agreement_fusion")
                self.assertTrue(relation["enabled"])
                self.assertFalse(experiment["datasets"]["fakeddit"]["image_text_relation"].get("enabled", False))
                self.assertEqual(launcher.load_config(active)["datasets"]["crisismmd"]["task"],
                                 self.config["jobs"][name]["task"])
                if research is not None:
                    self.assertTrue(yaml.safe_load(research.read_text())["image_text_relation"]["enabled"])

    def test_crisismmd_direct_pair_jobs_are_opt_in(self):
        for name in ("crisismmd_info_direct_pair", "crisismmd_humanitarian_direct_pair"):
            with self.subTest(name=name):
                launcher.validate(self.config, [name])
                stage = self.root / (name + "_stage")
                stage.mkdir()
                experiment, _, _, _, active, _, _ = launcher.job_plan(self.config, name, stage)
                relation = experiment["datasets"]["crisismmd"]["image_text_relation"]
                self.assertEqual(relation, {"enabled": True, "mode": "direct_pair_fusion", "weight": .25})
                self.assertEqual(launcher.load_config(active)["datasets"]["crisismmd"]["task"],
                                 self.config["jobs"][name]["task"])

    def test_crisismmd_entropy_jobs_freeze_fusion_and_separate_results(self):
        for name in ("crisismmd_info_entropy", "crisismmd_humanitarian_entropy",
                     "adapted_crisismmd_info_entropy", "adapted_crisismmd_humanitarian_entropy"):
            with self.subTest(name=name):
                self.config["use_kaggle"] = True
                launcher.validate(self.config, [name])
                launcher.validate_single_cloud_trial(self.config, [name])
                stage = self.root / (name + "_stage")
                stage.mkdir()
                experiment, job, output, _, active, _, research = launcher.job_plan(self.config, name, stage)
                task = experiment["datasets"]["crisismmd"]
                self.assertEqual(task["inference"]["fusion_mode"], "entropy_weighted")
                self.assertFalse(task["image_text_relation"]["enabled"])
                self.assertEqual(output.parts[1], job["result_method"])
                self.assertEqual(launcher.load_config(active)["datasets"]["crisismmd"]["task"], job["task"])
                if job["kind"] == "adapted":
                    self.assertEqual(yaml.safe_load(research.read_text())["methods"], ["dclip"])

    def test_crisismmd_uncertainty_jobs_have_runnable_cloud_plans(self):
        self.config["use_kaggle"] = True
        self.config["kaggle"]["data_mode"] = "bounded"
        cases = {
            "crisismmd_info_margin_majority": ("margin_majority", "experiments/ensemble/run_ensemble.py"),
            "crisismmd_humanitarian_margin_majority": ("margin_majority", "experiments/ensemble/run_ensemble.py"),
            "crisismmd_info_prompt_margin_gate": ("prompt_margin_gate", "experiments/uncertainty/run_crisismmd_prompt_gate.py"),
            "crisismmd_humanitarian_prompt_margin_gate": ("prompt_margin_gate", "experiments/uncertainty/run_crisismmd_prompt_gate.py"),
        }
        for name, (method, entry) in cases.items():
            with self.subTest(name=name), patch.object(launcher, "run_cli") as cli, redirect_stdout(io.StringIO()):
                launcher.validate(self.config, [name])
                launcher.validate_single_cloud_trial(self.config, [name])
                stage = self.root / (name + "_stage")
                stage.mkdir()
                _, _, output, command, _, _, _ = launcher.job_plan(self.config, name, stage)
                self.assertEqual((output.parts[1], command[0]), (method, entry))
                launcher.run_one(self.config, name, dry_run=True)
                cli.assert_not_called()

    def test_crisismmd_image_only_jobs_keep_results_separate(self):
        cases = (("crisismmd_info_image_only", "standard", None),
                 ("adapted_crisismmd_info_image_only", "adapted", "dclip"),
                 ("upstream_crisismmd_info_image_only", "upstream", "dclip"))
        for name, kind, method in cases:
            with self.subTest(name=name):
                job = self.config["jobs"][name]
                if method:
                    job["methods"] = [method]
                launcher.validate(self.config, [name])
                stage = self.root / (name + "_stage")
                stage.mkdir()
                active_config, saved_job, output, _, _, _, research_file = launcher.job_plan(
                    self.config, name, stage)
                self.assertEqual(active_config["datasets"]["crisismmd"]["image_text_relation"]["enabled"], False)
                self.assertTrue(output.parts[1].startswith("image_only_"))
                self.assertEqual(saved_job["result_protocol"], "image_only")
                if kind == "upstream":
                    self.assertEqual(yaml.safe_load(research_file.read_text())["outputs"], ["original_image"])
                    self.assertEqual(output.parts[1], "image_only_upstream_dclip")
                else:
                    self.assertEqual(active_config["datasets"]["crisismmd"]["inference"]["image_weight"], 1)

    def test_image_only_job_rejects_multimodal_configuration(self):
        name = "crisismmd_info_image_only"
        self.config["jobs"][name]["image_weight"] = .5
        with self.assertRaisesRegex(ValueError, "image_weight: 1.0"):
            launcher.validate(self.config, [name])

    def test_inherited_relation_gets_its_own_method_folder(self):
        path = self.root / "configs/experiment_config.yaml"
        experiment = yaml.safe_load(path.read_text(encoding="utf-8"))
        experiment["inference"]["image_text_relation"] = {"enabled": True, "mode": "auto", "weight": .25}
        path.write_text(yaml.safe_dump(experiment), encoding="utf-8")
        job = self.config["jobs"]["fakeddit_6way"]
        job.pop("image_text_relation", None)
        stage = self.root / "relation_stage"
        stage.mkdir()
        _, saved_job, output, _, _, _, _ = launcher.job_plan(self.config, "fakeddit_6way", stage)
        self.assertEqual(output.parts[1], "class_aware")
        self.assertEqual(saved_job["result_method"], "class_aware")

    def test_upstream_job_can_select_original_image_only(self):
        name = "upstream_crisismmd_humanitarian"
        self.config["jobs"][name]["outputs"] = ["original_image"]
        launcher.validate(self.config, [name])
        stage = self.root / "upstream_stage"
        stage.mkdir()
        _, _, _, _, active, _, research = launcher.job_plan(self.config, name, stage)
        self.assertEqual(launcher.load_config(active)["models"],
                         yaml.safe_load((self.root / "configs/experiment_config.yaml").read_text())["models"])
        self.assertEqual(yaml.safe_load(research.read_text())["outputs"], ["original_image"])

    def test_upstream_outputs_reject_unknown_variant(self):
        name = "upstream_crisismmd_humanitarian"
        self.config["jobs"][name]["outputs"] = ["unknown"]
        with self.assertRaisesRegex(ValueError, "outputs"):
            launcher.validate(self.config, [name])

    def test_research_job_can_override_waffle_seeds(self):
        name = "adapted_crisismmd_info"
        self.config["jobs"][name]["seeds"] = [2, 3]
        launcher.validate(self.config, [name])
        stage = self.root / "adapted_seed_stage"
        stage.mkdir()
        research = launcher.job_plan(self.config, name, stage)[-1]
        self.assertEqual(yaml.safe_load(research.read_text())["seeds"], [2, 3])
        self.config["jobs"][name]["seeds"] = [2, 2]
        with self.assertRaisesRegex(ValueError, "seeds"):
            launcher.validate(self.config, [name])

    def test_cli_selects_one_standard_model_without_running(self):
        with patch.object(launcher, "run_one") as run:
            launcher.main(["--job", "crisismmd_info", "--model", "dfn5b_vith14",
                           "--use-kaggle", "true", "--dry-run"])
        config, name, dry_run = run.call_args.args
        self.assertEqual(name, "crisismmd_info")
        self.assertTrue(dry_run)
        self.assertEqual(config["jobs"][name]["models"], ["dfn5b_vith14"])

    def test_openai_clip_uses_same_crisismmd_fusion_for_baseline_and_dclip(self):
        model = "openai_clip_vit_b_32"
        self.assertEqual(launcher.MODEL_CONFIGS[model]["pretrained"], "openai")
        for task, suffix in (("informativeness", "info"), ("humanitarian", "humanitarian")):
            for kind, name in (("standard", f"crisismmd_{suffix}"),
                               ("adapted", f"adapted_crisismmd_{suffix}")):
                with self.subTest(task=task, kind=kind):
                    job = self.config["jobs"][name]
                    job["models"] = [model]
                    if kind == "adapted":
                        job["methods"] = ["dclip"]
                    launcher.validate(self.config, [name])
                    stage = self.root / f"openai_{name}"
                    stage.mkdir()
                    experiment, _, output, _, _, _, research_file = launcher.job_plan(
                        self.config, name, stage)
                    self.assertEqual(experiment["models"], [model])
                    self.assertEqual(experiment["datasets"]["crisismmd"]["inference"]["image_weight"], 0.5)
                    self.assertEqual(output.parts[-2], model)
                    self.assertEqual(output.parts[1], "baseline" if kind == "standard" else "adapted_dclip")
                    if kind == "adapted":
                        self.assertEqual(Path(yaml.safe_load(research_file.read_text())["descriptor_file"]).name,
                                         f"crisismmd_{suffix}.json")

    def test_cli_selects_one_upstream_method_backbone_output_and_seed(self):
        with patch.object(launcher, "run_one") as run:
            launcher.main(["--job", "upstream_crisismmd_info", "--method", "dclip",
                           "--backbone", "ViT-L/14", "--output", "original_image", "--seed", "3",
                           "--use-kaggle", "true", "--dry-run"])
        config, name, _ = run.call_args.args
        job = config["jobs"][name]
        self.assertEqual(job["methods"], ["dclip"])
        self.assertEqual(job["backbones"], ["ViT-L/14"])
        self.assertEqual(job["outputs"], ["original_image"])
        self.assertEqual(job["seeds"], [3])
        stage = self.root / "selected_upstream_stage"
        stage.mkdir()
        research = launcher.job_plan(config, name, stage)[-1]
        self.assertEqual(yaml.safe_load(research.read_text())["backbones"], ["ViT-L/14"])

    def test_cloud_upstream_rejects_multiple_methods_before_staging(self):
        error = io.StringIO()
        with patch.object(launcher, "run_one") as run, redirect_stderr(error), \
                self.assertRaises(SystemExit):
            launcher.main(["--job", "upstream_crisismmd_info", "--backbone", "ViT-B/32",
                           "--output", "original_image", "--use-kaggle", "true"])
        self.assertIn("--method", error.getvalue())
        run.assert_not_called()

    def test_cli_rejects_model_for_upstream(self):
        with self.assertRaises(SystemExit), patch.object(launcher, "run_one") as run:
            launcher.main(["--job", "upstream_crisismmd_info", "--model", "metaclip2_vith14",
                           "--dry-run"])
        run.assert_not_called()

    def test_listed_kaggle_commands_are_accepted_by_launcher(self):
        config = yaml.safe_load((REPO / "configs/launcher.yaml").read_text())
        count = 0
        with patch.object(launcher, "run_one") as run:
            for _, _, command in command_list.commands(config):
                launcher.main(shlex.split(command)[3:] + ["--dry-run"])
                self.assertTrue(run.call_args.args[0]["use_kaggle"])
                self.assertEqual(run.call_args.args[2], True)
                count += 1
        self.assertGreater(count, 250)
        self.assertEqual(run.call_count, count)

    def test_research_cloud_bundle_keeps_per_run_yaml_in_worker(self):
        self.config["use_kaggle"] = True
        for name in ("demo_crisismmd_info", "adapted_crisismmd_info",
                     "upstream_crisismmd_info", "relation_fakeddit_6way"):
            with self.subTest(name=name), patch.object(launcher, "run_cli") as cli, redirect_stdout(io.StringIO()):
                launcher.validate(self.config, [name])
                launcher.run_one(self.config, name, dry_run=True)
                cli.assert_not_called()
        for stage in (self.root / "results/_system/launches/staging").iterdir():
            with zipfile.ZipFile(stage / "code/code_bundle.zip") as archive:
                self.assertNotIn("configs/active.yaml", archive.namelist())
                self.assertIn("requirements/research.txt", archive.namelist())
            self.assertIn("active_yaml", (stage / "kernel/worker.py").read_text())

    def test_archive_hash_matches_kaggle_mount_layout(self):
        experiment = yaml.safe_load((self.root / "configs/experiment_config.yaml").read_text())
        archive = self.root / "data_bundle.zip"
        launcher.archive_data(archive, experiment, "mmimdb", 2)
        mount = self.root / "input/datasets/testuser/vlm-runs-data"
        unpack_archive(archive, mount)
        expected = launcher.archive_tree_sha256(archive)
        self.assertEqual(expected, tree_sha256(mount))
        self.assertEqual(find_dataset(self.root / "input", "vlm-runs-data", expected), mount)

    def test_unchanged_data_bundle_has_stable_sha(self):
        experiment = yaml.safe_load((self.root / "configs/experiment_config.yaml").read_text())
        first = launcher.archive_data(self.root / "first.zip", experiment, "crisismmd", 2)[0]
        second = launcher.archive_data(self.root / "second.zip", experiment, "crisismmd", 2)[0]
        self.assertEqual(first, second)

    def test_full_dataset_bundle_reused_between_tasks_and_limits(self):
        self.config["use_kaggle"] = True
        self.config["kaggle"]["data_mode"] = "full_dataset"
        (self.root / "data/processed/pro_crisismmd/images/.DS_Store").write_bytes(b"macOS metadata")
        jobs = ("crisismmd_info", "crisismmd_humanitarian")
        with redirect_stdout(io.StringIO()):
            for name in jobs:
                launcher.validate(self.config, [name])
                launcher.run_one(self.config, name, dry_run=True)
        stages = list((self.root / "results/_system/launches/staging").iterdir())
        self.assertEqual(len(stages), 2)
        settings = [launcher.staged_run(stage)[0] for stage in stages]
        self.assertEqual(settings[0]["data_sha256"], settings[1]["data_sha256"])
        self.assertTrue(all(item["data_dataset_slug"] == "vlm-runs-data-crisismmd" for item in settings))
        with zipfile.ZipFile(stages[0] / "data/data_bundle.zip") as archive:
            self.assertEqual(sum(name.startswith("data/processed/pro_crisismmd/images/")
                                 for name in archive.namelist()), 3)
            self.assertFalse(any(".DS_Store" in name for name in archive.namelist()))
        self.assertTrue((self.root / "results/_system/launches/full_data_cache/crisismmd/data_bundle.zip").is_file())

    def test_full_dataset_cache_detects_changed_bytes_with_same_metadata(self):
        experiment = yaml.safe_load((self.root / "configs/experiment_config.yaml").read_text())
        image = self.root / "data/processed/pro_crisismmd/images/a.jpg"
        old_stat = image.stat()
        first = launcher.archive_full_data(self.root / "first.zip", experiment, "crisismmd", 2)[0]
        image.write_bytes(b"other-data")  # Cùng số byte với b"fake-image".
        import os
        os.utime(image, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
        second = launcher.archive_full_data(self.root / "second.zip", experiment, "crisismmd", 2)[0]
        self.assertNotEqual(first, second)

    def test_sync_full_data_does_not_start_kernel(self):
        self.config["kaggle"]["data_mode"] = "full_dataset"
        with patch.object(launcher, "find_kaggle_cli", return_value="kaggle"), \
                patch.object(launcher, "sync_dataset", return_value=None) as sync, redirect_stdout(io.StringIO()):
            launcher.sync_full_data(self.config, ("crisismmd", "mmimdb"))
        self.assertEqual([call.args[1] for call in sync.call_args_list],
                         ["testuser/vlm-runs-data-crisismmd", "testuser/vlm-runs-data-mmimdb"])
        self.assertTrue(all(call.kwargs["verify_remote"] for call in sync.call_args_list))

    def attach_cohort(self, dataset):
        config_path = self.root / "configs/experiment_config.yaml"
        experiment = yaml.safe_load(config_path.read_text())
        task = experiment["datasets"][dataset]
        source = self.root / task["data_path"]
        with source.open(newline="") as stream:
            rows = list(csv.DictReader(stream, delimiter="\t"))
        entries = []
        for row in rows:
            row["image_id"] = row["id"]
            row["source_split"] = "test"
            row["image_path"] = f"data/processed/pro_{dataset}/images/{row['id']}"
            truth = ({"informativeness": row["label_informative"], "humanitarian": row["label_humanitarian"]}
                     if dataset == "crisismmd" else [row["genres"]] if dataset == "mmimdb" else row["6_way_label"])
            if dataset == "fakeddit":
                row["2_way_label"] = "1" if row["6_way_label"] == "0" else "0"
            entries.append({"id": row[task["id_col"]], "y_true": truth})
        with source.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
        relative = f"data/processed/pro_{dataset}/manifest.json"
        manifest = {"status": "ready", "dataset": dataset, "task": "both" if dataset == "crisismmd" else "genres" if dataset == "mmimdb" else "6way",
                    "cohort_id": f"test-{dataset}", "selected": {"samples": 3}, "samples": entries,
                    "samples_sha256": launcher.sha256(source), "source_split": "test", "label_source": "image"}
        (self.root / relative).write_text(json.dumps(manifest))
        task.update(cohort_manifest=relative, source_split="test")
        config_path.write_text(yaml.safe_dump(experiment))
        return experiment

    def test_full_bundle_contains_manifest_and_worker_verifies_each_task(self):
        self.config["use_kaggle"] = True
        self.config["kaggle"]["data_mode"] = "full_dataset"
        for dataset in ("crisismmd", "fakeddit", "mmimdb"):
            self.attach_cohort(dataset)
        for job in ("crisismmd_info", "crisismmd_humanitarian", "fakeddit_2way", "fakeddit_6way", "mmimdb_genres"):
            with self.subTest(job=job), redirect_stdout(io.StringIO()):
                launcher.run_one(self.config, job, dry_run=True)
                stages = sorted((self.root / "results/_system/launches/staging").iterdir(), key=lambda p: p.stat().st_mtime_ns)
                stage = stages[-1]
                settings, _, _ = launcher.staged_run(stage)
                mount = self.root / f"mount_{job}"
                unpack_archive(stage / "data/data_bundle.zip", mount)
                receipt = verify_cohort(mount, settings)
                self.assertEqual(receipt["cohort_samples"], 3)
                self.assertEqual(receipt["requested_samples"], 2)
                self.assertEqual(receipt["source_split"], "test")
                source = mount / settings["cohort"]["samples_path"]
                source.write_text(source.read_text() + "tampered")
                with self.assertRaisesRegex(ValueError, "checksum"):
                    verify_cohort(mount, settings)

    def test_explicit_sync_checks_remote_even_if_state_matches(self):
        staged = self.root / "remote_check"
        staged.mkdir()
        source = staged / "data_bundle.zip"
        with zipfile.ZipFile(source, "w") as archive:
            archive.writestr("fresh.txt", "current bytes")
        name = "testuser/vlm-runs-data"
        state = {name: "same-local-sha"}
        def cli(args, check=True):
            if args[1:3] == ["datasets", "list"]:
                return SimpleNamespace(stdout=json.dumps([{"ref": name}]), stderr="", returncode=0)
            self.assertEqual(args[1:3], ["datasets", "download"])
            shutil.copy2(source, Path(args[args.index("-p") + 1]) / "vlm-runs-data.zip")
            return SimpleNamespace(stdout="", stderr="", returncode=0)
        with patch.object(launcher, "run_cli", side_effect=cli) as remote, redirect_stdout(io.StringIO()):
            result = launcher.sync_dataset(staged, name, "same-local-sha", state, "kaggle", verify_remote=True)
        self.assertEqual(result["verification"], "sha256")
        self.assertEqual(remote.call_count, 2)

    def test_existing_empty_dataset_gets_a_new_version_without_state(self):
        staged = self.root / "empty_remote"
        staged.mkdir()
        source = staged / "data_bundle.zip"
        with zipfile.ZipFile(source, "w") as archive:
            archive.writestr("fresh.txt", "current bytes")
        name, updated = "testuser/vlm-runs-data", False
        def cli(args, check=True):
            nonlocal updated
            action = args[1:3]
            if action == ["datasets", "list"]:
                return SimpleNamespace(stdout=json.dumps([{"ref": name}]), stderr="", returncode=0)
            if action == ["datasets", "files"]:
                return SimpleNamespace(stdout="[]", stderr="", returncode=0)
            if action == ["datasets", "version"]:
                updated = True
                return SimpleNamespace(stdout="created", stderr="", returncode=0)
            self.assertEqual(action, ["datasets", "download"])
            if not updated:
                return SimpleNamespace(stdout="", stderr="404 Not Found", returncode=1)
            shutil.copy2(source, Path(args[args.index("-p") + 1]) / "vlm-runs-data.zip")
            return SimpleNamespace(stdout="", stderr="", returncode=0)
        with patch.object(launcher, "run_cli", side_effect=cli), redirect_stdout(io.StringIO()):
            result = launcher.sync_dataset(staged, name, "fresh", {}, "kaggle", verify_remote=True)
        self.assertTrue(updated)
        self.assertTrue(result["updated"])
        self.assertEqual(result["verification"], "sha256")

    def test_bulk_file_check_does_not_download_stale_zip(self):
        stage = self.root / "file_first"
        stage.mkdir()
        with zipfile.ZipFile(stage / "data_bundle.zip", "w") as archive:
            archive.writestr("manifest.json", "{}")
        name = "testuser/vlm-runs-data"
        def cli(args, check=True):
            if args[1:3] == ["datasets", "list"]:
                return SimpleNamespace(stdout=json.dumps([{"ref": name}]), stderr="", returncode=0)
            self.assertEqual(args[1:3], ["datasets", "files"])
            return SimpleNamespace(stdout='[{"name":"manifest.json","size":2}]', stderr="", returncode=0)
        with patch.object(launcher, "run_cli", side_effect=cli), redirect_stdout(io.StringIO()):
            result = launcher.sync_dataset(stage, name, "hash", {}, "kaggle", verify_remote=True, file_manifest_first=True)
        self.assertEqual(result["verification"], "file_manifest")

    def test_configured_command_list_preserves_launcher_and_dclip_only(self):
        config = yaml.safe_load((REPO / "configs/launcher_3000.yaml").read_text())
        generated = list(command_list.commands(config, launcher_path="configs/launcher_3000.yaml", configured_only=True))
        self.assertEqual(len(generated), 28)
        for kind, name, command in generated:
            self.assertIn("--launcher-config configs/launcher_3000.yaml", command)
            self.assertEqual(config["jobs"][name]["max_samples"], 3000)
            if kind == "adapted":
                self.assertIn("--method dclip", command)

    def test_result_receipt_must_match_verified_cohort(self):
        folder = self.root / "receipt"
        folder.mkdir()
        (folder / "summary.tsv").write_text("accuracy\n1.0\n")
        settings = {"run_id": "run", "job_id": "job", "code_sha256": "code", "data_sha256": "data",
                    "cohort": {"cohort_id": "fresh", "samples": 3000, "limit": 3000, "task": "humanitarian"}}
        receipt = {key: value for key, value in settings.items() if key != "cohort"}
        path = folder / "run_receipt.json"
        path.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(RuntimeError, "cohort"):
            launcher.verify_result(folder, settings)
        receipt["cohort"] = {"cohort_id": "fresh", "cohort_samples": 3000, "requested_samples": 3000, "task": "humanitarian"}
        path.write_text(json.dumps(receipt))
        launcher.verify_result(folder, settings)

    def test_cpu_audit_mounts_four_datasets_without_gpu_or_models(self):
        for dataset in ("fakeddit", "crisismmd", "mmimdb"):
            self.attach_cohort(dataset)
        expected, kernel_id = cpu_audit.prepare(self.config, self.root / "cpu_audit")
        metadata = json.loads((self.root / "cpu_audit/kernel/kernel-metadata.json").read_text())
        self.assertEqual(metadata["enable_gpu"], "false")
        self.assertEqual(metadata["enable_internet"], "false")
        self.assertEqual(metadata["is_private"], "true")
        self.assertEqual(len(metadata["dataset_sources"]), 4)
        self.assertEqual(len(expected["cohorts"]), 5)
        worker = (self.root / "cpu_audit/kernel/worker.py").read_text()
        ast.parse(worker)
        self.assertIn("audit_only(AUDIT_SETTINGS)", worker)
        result = {"audit_id": expected["audit_id"], "gpu_used": False, "models_loaded": False,
                  "sources": [{"slug": slug, "tree_sha256": sha} for slug, sha in expected["trees"].items()],
                  "cohorts": [{"task": spec["cohort"]["task"], "cohort_id": spec["cohort"]["cohort_id"],
                               "cohort_samples": 3, "requested_samples": 3} for spec in expected["cohorts"]]}
        cpu_audit.validate_result(result, expected)
        result["sources"][0]["tree_sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "không khớp"):
            cpu_audit.validate_result(result, expected)

    def test_cpu_audit_resume_collects_same_version_without_submitting(self):
        stage = self.root / "cpu_resume"
        (stage / "kernel").mkdir(parents=True)
        (stage / "expected.json").write_text(json.dumps({"audit_id": "same-run"}))
        (stage / "kernel/kernel-metadata.json").write_text(json.dumps({"id": "testuser/cpu"}))
        (stage / "kernel_version.txt").write_text("7")
        (self.root / "docs/experiments").mkdir(parents=True)
        config = self.root / "launcher.yaml"
        config.write_text(yaml.safe_dump(self.config))
        def collect(args):
            self.assertEqual(args[1:4], ["kernels", "output", "testuser/cpu/7"])
            folder = Path(args[args.index("-p") + 1])
            (folder / "kaggle_data_check.json").write_text(json.dumps({"audit_id": "same-run"}))
        with patch.object(cpu_audit, "prepare") as prepare, patch.object(launcher, "find_kaggle_cli", return_value="kaggle"), \
                patch.object(launcher, "await_kernel") as wait, patch.object(launcher, "run_cli", side_effect=collect) as cli, \
                patch.object(cpu_audit, "validate_result") as validate, redirect_stdout(io.StringIO()):
            cpu_audit.main(["--launcher-config", str(config), "--resume-stage", str(stage)])
        prepare.assert_not_called()
        wait.assert_called_once_with("kaggle", "testuser/cpu/7", 20, 1200)
        self.assertEqual(cli.call_count, 1)
        validate.assert_called_once()
        self.assertEqual(validate.call_args.args[0]["audit_id"], "same-run")
        self.assertEqual(validate.call_args.args[1], {"audit_id": "same-run"})
        result = json.loads((self.root / "docs/experiments/kaggle_3000_remote_check.json").read_text())
        self.assertEqual(result["kernel_version"], 7)

    def test_existing_dataset_can_be_verified_from_paginated_files_when_download_is_404(self):
        stage = self.root / "dataset_stage"
        stage.mkdir()
        with zipfile.ZipFile(stage / "data_bundle.zip", "w") as archive:
            archive.writestr("a.txt", "alpha")
            archive.writestr("b.txt", "beta")
        name, digest, state = "testuser/vlm-runs-data-mmimdb", "abc123", {}

        def fake_cli(args, check=True):
            if args[1:3] == ["datasets", "list"]:
                return SimpleNamespace(stdout=json.dumps([{"ref": name}]), stderr="", returncode=0)
            if args[1:3] == ["datasets", "download"]:
                return SimpleNamespace(stdout="", stderr="404 Not Found", returncode=1)
            self.assertEqual(args[1:3], ["datasets", "files"])
            if "--page-token" not in args:
                return SimpleNamespace(stdout='Next Page Token = page2\n[{"name":"a.txt","size":5}]',
                                       stderr="", returncode=0)
            self.assertEqual(args[-1], "page2")
            return SimpleNamespace(stdout='[{"name":"b.txt","size":4}]', stderr="", returncode=0)

        with patch.object(launcher, "run_cli", side_effect=fake_cli) as cli, redirect_stdout(io.StringIO()):
            launcher.sync_dataset(stage, name, digest, state, "kaggle")
        self.assertEqual(state[name], digest)
        self.assertFalse(any(call.args[0][1:3] in (["datasets", "create"], ["datasets", "version"])
                             for call in cli.call_args_list))

    def test_code_bundle_is_independent_of_job_and_run_id(self):
        first = launcher.archive_code(self.root / "code_first.zip")
        second = launcher.archive_code(self.root / "code_second.zip")
        self.assertEqual(first, second)
        with zipfile.ZipFile(self.root / "code_first.zip") as archive:
            self.assertNotIn("configs/job.yaml", archive.namelist())
            self.assertIn("configs/prompts/fakeddit_6way_variants.json", archive.namelist())
        mount = self.root / "input/datasets/testuser/vlm-runs-code"
        unpack_archive(self.root / "code_first.zip", mount)
        self.assertEqual(launcher.archive_tree_sha256(self.root / "code_first.zip"),
                         tree_sha256(mount))

    def test_missing_local_state_reuses_private_dataset(self):
        experiment = yaml.safe_load((self.root / "configs/experiment_config.yaml").read_text())
        staged = self.root / "staged"
        staged.mkdir()
        source = staged / "data_bundle.zip"
        digest, _, _, _ = launcher.archive_data(source, experiment, "crisismmd", 2)
        state = {}
        name = "testuser/vlm-runs-data"

        def fake_cli(args, check=True):
            if args[1:3] == ["datasets", "list"]:
                return SimpleNamespace(stdout=json.dumps([{"ref": name}]), stderr="", returncode=0)
            self.assertEqual(args[1:3], ["datasets", "download"])
            shutil.copy2(source, Path(args[args.index("-p") + 1]) / "vlm-runs-data.zip")
            return SimpleNamespace(stdout="", stderr="", returncode=0)

        with patch.object(launcher, "run_cli", side_effect=fake_cli) as cli, redirect_stdout(io.StringIO()):
            launcher.sync_dataset(staged, name, digest, state, "kaggle")
        self.assertEqual(state[name], digest)
        self.assertFalse(any(call.args[0][1:3] == ["datasets", "create"] for call in cli.call_args_list))

    def test_local_dry_run_does_not_call_kaggle(self):
        with patch.object(launcher, "run_cli") as cli, redirect_stdout(io.StringIO()):
            launcher.run_one(self.config, "crisismmd_info", dry_run=True)
        cli.assert_not_called()

    def test_remote_mocked_flow_recovers_and_verifies_receipt(self):
        self.config["use_kaggle"] = True
        remote, calls, settings = {}, [], {}

        def fake_cli(args, check=True):
            calls.append(args[1:3])
            if args[1:3] == ["datasets", "download"]:
                name = args[3]
                if name not in remote:
                    return SimpleNamespace(stdout="", stderr="404 Not Found", returncode=1)
                shutil.copy2(remote[name], Path(args[args.index("-p") + 1]) / f"{name.split('/')[1]}.zip")
            if args[1:3] == ["datasets", "create"]:
                staged = Path(args[args.index("-p") + 1])
                name = json.loads((staged / "dataset-metadata.json").read_text())["id"]
                remote[name] = next(staged.glob("*.zip"))
            if args[1:3] == ["kernels", "push"]:
                worker = Path(args[args.index("-p") + 1]) / "worker.py"
                line = worker.read_text().splitlines()[-3]
                settings.update(ast.literal_eval(line.split(" = ", 1)[1]))
                return SimpleNamespace(stdout="Kernel version 7 successfully pushed.",
                                       stderr="", returncode=0)
            if args[1:3] == ["kernels", "output"]:
                self.assertEqual(args[3], "testuser/vlm-runs/7")
                output = Path(args[args.index("-p") + 1])
                (output / "run_receipt.json").write_text(json.dumps(settings))
                (output / "summary.tsv").write_text("task\tmacro_f1\n")
            if args[1:3] == ["kernels", "status"]:
                self.assertEqual(args[3], "testuser/vlm-runs/7")
            return SimpleNamespace(stdout="complete", stderr="", returncode=0)

        with patch.object(launcher, "find_kaggle_cli", return_value="kaggle"), \
                patch.object(launcher, "run_cli", side_effect=fake_cli), redirect_stdout(io.StringIO()):
            output = launcher.run_one(self.config, "crisismmd_info")
        self.assertTrue((output / "summary.tsv").is_file())
        self.assertEqual(calls.count(["datasets", "create"]), 2)
        self.assertNotIn(["datasets", "status"], calls)

    def test_resume_stage_downloads_without_pushing_a_new_kernel(self):
        self.config["use_kaggle"] = True
        with redirect_stdout(io.StringIO()):
            launcher.run_one(self.config, "crisismmd_info", dry_run=True)
        stage = next((self.root / "results/_system/launches/staging").iterdir())
        settings, kernel_id, output = launcher.staged_run(stage)

        def fake_cli(args, check=True):
            if args[1:3] == ["kernels", "status"]:
                self.assertEqual(args[3], kernel_id)
                return SimpleNamespace(stdout="complete", stderr="", returncode=0)
            self.assertEqual(args[1:3], ["kernels", "output"])
            directory = Path(args[args.index("-p") + 1])
            (directory / "run_receipt.json").write_text(json.dumps(settings))
            (directory / "summary.tsv").write_text("task\tmacro_f1\n")
            return SimpleNamespace(stdout="", stderr="", returncode=0)

        with patch.object(launcher, "find_kaggle_cli", return_value="kaggle"), \
                patch.object(launcher, "run_cli", side_effect=fake_cli) as cli, \
                redirect_stdout(io.StringIO()):
            self.assertEqual(launcher.resume_run(stage, self.config), output)
        self.assertTrue((output / "summary.tsv").is_file())
        self.assertEqual([call.args[0][1:3] for call in cli.call_args_list],
                         [["kernels", "status"], ["kernels", "output"]])

    def test_resume_accepts_stages_created_before_model_folder(self):
        self.config["use_kaggle"] = True
        with redirect_stdout(io.StringIO()):
            launcher.run_one(self.config, "crisismmd_info", dry_run=True)
        stage = next((self.root / "results/_system/launches/staging").iterdir())
        settings, _, expected = launcher.staged_run(stage)
        old = Path(settings["output_dir"])
        worker = stage / "kernel/worker.py"
        for previous in (old.parent.parent / old.name,
                         old.parent.parent.parent.parent / old.parent.parent.name / old.name):
            settings["output_dir"] = str(previous)
            worker.write_text("SETTINGS = " + repr(settings) + "\n", encoding="utf-8")
            self.assertEqual(launcher.staged_run(stage)[2], expected)

    def test_resume_uses_saved_version_for_status_and_output(self):
        self.config["use_kaggle"] = True
        with redirect_stdout(io.StringIO()):
            launcher.run_one(self.config, "crisismmd_info", dry_run=True)
        stage = next((self.root / "results/_system/launches/staging").iterdir())
        (stage / "kernel_version.txt").write_text("54\n")
        settings, kernel_id, output = launcher.staged_run(stage)
        with patch.object(launcher, "find_kaggle_cli", return_value="kaggle"), \
                patch.object(launcher, "await_kernel") as wait, \
                patch.object(launcher, "collect_result") as collect:
            launcher.resume_run(stage, self.config)
        wait.assert_called_once_with("kaggle", f"{kernel_id}/54", self.config["kaggle"]["poll_seconds"],
                                     self.config["kaggle"]["timeout_seconds"])
        collect.assert_called_once_with("kaggle", f"{kernel_id}/54", settings, output)

    def test_resume_rejects_invalid_saved_version_without_calling_kaggle(self):
        self.config["use_kaggle"] = True
        with redirect_stdout(io.StringIO()):
            launcher.run_one(self.config, "crisismmd_info", dry_run=True)
        stage = next((self.root / "results/_system/launches/staging").iterdir())
        (stage / "kernel_version.txt").write_text("54/other")
        with patch.object(launcher, "run_cli") as cli, self.assertRaisesRegex(ValueError, "kernel_version"):
            launcher.resume_run(stage, self.config)
        cli.assert_not_called()

    def test_collect_result_uses_migrated_image_only_run_without_download(self):
        old = self.root / ("results/upstream_dclip/crisismmd/informativeness/samples_2/"
                           "openai_clip_vit_b_32/informativeness_abc123")
        new = self.root / ("results/image_only_upstream_dclip/crisismmd/informativeness/samples_2/"
                           "openai_clip_vit_b_32/informativeness_abc123")
        new.mkdir(parents=True)
        settings = {key: "same" for key in ("run_id", "job_id", "code_sha256", "data_sha256")}
        (new / "run_receipt.json").write_text(json.dumps(settings), encoding="utf-8")
        (new / "summary.tsv").write_text("method\n", encoding="utf-8")
        mapping = self.root / "results/_system/migration_map.json"
        mapping.parent.mkdir(parents=True)
        mapping.write_text(json.dumps([{"from": old.relative_to(self.root / "results").as_posix(),
            "to": new.relative_to(self.root / "results").as_posix()}]), encoding="utf-8")
        with patch.object(launcher, "run_cli") as cli, redirect_stdout(io.StringIO()):
            self.assertEqual(launcher.collect_result("kaggle", "owner/kernel", settings, old), new)
            cli.assert_not_called()

    def test_resume_rejects_receipt_from_other_run(self):
        self.config["use_kaggle"] = True
        with redirect_stdout(io.StringIO()):
            launcher.run_one(self.config, "crisismmd_info", dry_run=True)
        stage = next((self.root / "results/_system/launches/staging").iterdir())
        _, _, output = launcher.staged_run(stage)

        def fake_cli(args, check=True):
            if args[1:3] == ["kernels", "status"]:
                return SimpleNamespace(stdout="complete", stderr="", returncode=0)
            directory = Path(args[args.index("-p") + 1])
            (directory / "run_receipt.json").write_text(json.dumps({"run_id": "different"}))
            (directory / "summary.tsv").write_text("task\tmacro_f1\n")
            return SimpleNamespace(stdout="", stderr="", returncode=0)

        with patch.object(launcher, "find_kaggle_cli", return_value="kaggle"), \
                patch.object(launcher, "run_cli", side_effect=fake_cli), \
                self.assertRaisesRegex(RuntimeError, "không khớp"):
            launcher.resume_run(stage, self.config)
        self.assertFalse(output.exists())

    def test_transient_private_kernel_access_error_is_retried(self):
        denied = SimpleNamespace(stdout="", stderr="Permission 'kernels.get' was denied", returncode=1)
        ready = SimpleNamespace(stdout="complete", stderr="", returncode=0)
        with patch.object(launcher, "run_cli", side_effect=[denied, ready]) as cli, \
                patch.object(launcher.time, "sleep") as sleep, redirect_stdout(io.StringIO()):
            launcher.await_kernel("kaggle", "testuser/vlm-runs", 5, 300)
        self.assertEqual(cli.call_count, 2)
        sleep.assert_called_once_with(5)

    def test_persistent_private_kernel_access_error_suggests_relogin(self):
        denied = SimpleNamespace(stdout="", stderr="Authentication required", returncode=1)
        with patch.object(launcher, "run_cli", return_value=denied), \
                patch.object(launcher.time, "sleep"), redirect_stdout(io.StringIO()), \
                self.assertRaisesRegex(RuntimeError, "kaggle auth login --force"):
            launcher.await_kernel("kaggle", "testuser/vlm-runs", 5, 300)

    def test_failed_kernel_is_terminal_and_does_not_suggest_resume(self):
        failed = SimpleNamespace(stdout="failed", stderr="", returncode=0)
        logs = SimpleNamespace(stdout='[{"data":"ValueError: models phải là danh sách model không rỗng"}]',
                               stderr="", returncode=0)
        with patch.object(launcher, "run_cli", side_effect=[failed, logs]), \
                redirect_stdout(io.StringIO()), \
                self.assertRaisesRegex(launcher.KernelExecutionError, "models phải là danh sách"):
            launcher.await_kernel("kaggle", "testuser/vlm-runs", 5, 300)

    def test_invalid_ensemble_method_for_mmimdb_rejected(self):
        self.config["jobs"]["mmimdb_genres"]["ensemble"].update(enabled=True, methods=["margin_gate"])
        with self.assertRaisesRegex(ValueError, "MM-IMDb"):
            launcher.validate(self.config, ["mmimdb_genres"])

    def test_cancel_acknowledged_stops_polling(self):
        cancelled = SimpleNamespace(stdout='kernelworkerstatus.cancel_acknowledged', stderr="", returncode=0)
        logs = SimpleNamespace(stdout="[]", stderr="", returncode=0)
        with patch.object(launcher, "run_cli", side_effect=[cancelled, logs]), \
                patch.object(launcher.time, "sleep") as sleep, redirect_stdout(io.StringIO()), \
                self.assertRaises(launcher.KernelExecutionError):
            launcher.await_kernel("kaggle", "testuser/cpu", 5, 300)
        sleep.assert_not_called()

    def test_cancelled_version_without_logs_explains_resume_limit(self):
        cancelled = SimpleNamespace(stdout='kernelworkerstatus.cancel_acknowledged', stderr="", returncode=0)
        for text in ("", "[]", '[{"data":""}]'):
            with self.subTest(logs=text), patch.object(launcher, "run_cli", side_effect=[
                    cancelled, SimpleNamespace(stdout=text, stderr="", returncode=0)]) as cli, \
                    redirect_stdout(io.StringIO()), self.assertRaises(launcher.KernelExecutionError) as error:
                launcher.await_kernel("kaggle", "testuser/vlm-runs/54", 5, 300)
            message = str(error.exception)
            self.assertIn("testuser/vlm-runs/54 bị hủy", message)
            self.assertIn("Kaggle không trả log", message)
            self.assertIn("không khởi động lại", message)
            self.assertEqual(cli.call_args.args[0], ["kaggle", "kernels", "logs", "testuser/vlm-runs/54"])

    def test_failed_kernel_without_logs_reports_failure_not_cancellation(self):
        failure = SimpleNamespace(stdout='kernelworkerstatus.error', stderr="", returncode=0)
        with patch.object(launcher, "run_cli", side_effect=[
                failure, SimpleNamespace(stdout="[]", stderr="", returncode=0)]), \
                redirect_stdout(io.StringIO()), self.assertRaises(launcher.KernelExecutionError) as error:
            launcher.await_kernel("kaggle", "testuser/vlm-runs/54", 5, 300)
        self.assertIn("thất bại", str(error.exception))
        self.assertIn("Kaggle không trả log", str(error.exception))


if __name__ == "__main__":
    unittest.main()
