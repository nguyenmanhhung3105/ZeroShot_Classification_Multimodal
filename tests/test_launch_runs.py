"""Kiểm tra launcher đa dataset bằng dữ liệu giả rất nhỏ, không gọi mạng/model."""

import ast
from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
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
from kaggle_worker import find_dataset, tree_sha256, unpack_archive


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
        shutil.copy2(REPO / "configs/research/adapted.yaml", self.root / "configs/research/adapted.yaml")
        shutil.copy2(REPO / "configs/research/upstream.yaml", self.root / "configs/research/upstream.yaml")
        shutil.copy2(REPO / "configs/research/sources.json", self.root / "configs/research/sources.json")
        (self.root / "requirements").mkdir()
        (self.root / "requirements.txt").write_text("pyyaml\n")
        (self.root / "requirements/research.txt").write_text("regex\n")
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
        for name in ("fakeddit_2way", "fakeddit_6way", "crisismmd_info",
                     "crisismmd_humanitarian", "mmimdb_genres"):
            with self.subTest(name=name), patch.object(launcher, "run_cli") as cli, redirect_stdout(io.StringIO()):
                launcher.validate(self.config, [name])
                launcher.run_one(self.config, name, dry_run=True)
                cli.assert_not_called()
        stages = list((self.root / "results/launches/staging").iterdir())
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
                self.assertTrue(str(output).startswith("results/experiments/"))
                self.assertEqual(job["output_dir"], str(output))
                if research:
                    self.assertEqual(yaml.safe_load(research.read_text())["methods"], job["methods"])
                    self.assertIn("--config", command)

    def test_research_cloud_bundle_keeps_per_run_yaml_in_worker(self):
        self.config["use_kaggle"] = True
        for name in ("demo_crisismmd_info", "adapted_crisismmd_info",
                     "upstream_crisismmd_info", "relation_fakeddit_6way"):
            with self.subTest(name=name), patch.object(launcher, "run_cli") as cli, redirect_stdout(io.StringIO()):
                launcher.validate(self.config, [name])
                launcher.run_one(self.config, name, dry_run=True)
                cli.assert_not_called()
        for stage in (self.root / "results/launches/staging").iterdir():
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
                output = Path(args[args.index("-p") + 1])
                (output / "run_receipt.json").write_text(json.dumps(settings))
                (output / "summary.tsv").write_text("task\tmacro_f1\n")
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
        stage = next((self.root / "results/launches/staging").iterdir())
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

    def test_resume_rejects_receipt_from_other_run(self):
        self.config["use_kaggle"] = True
        with redirect_stdout(io.StringIO()):
            launcher.run_one(self.config, "crisismmd_info", dry_run=True)
        stage = next((self.root / "results/launches/staging").iterdir())
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

    def test_invalid_ensemble_method_for_mmimdb_rejected(self):
        self.config["jobs"]["mmimdb_genres"]["ensemble"].update(enabled=True, methods=["margin_gate"])
        with self.assertRaisesRegex(ValueError, "MM-IMDb"):
            launcher.validate(self.config, ["mmimdb_genres"])


if __name__ == "__main__":
    unittest.main()
