"""Cây kết quả giữ phương pháp, dataset, task, số mẫu và lượt chạy tách biệt."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from results_layout import analysis_path, job_path, model_folder, standalone_config
import migrate_results_layout


class ResultsLayoutTests(unittest.TestCase):
    def test_new_jobs_put_task_in_its_own_folder(self):
        base = dict(kind="standard", dataset="crisismmd", task="humanitarian", max_samples=1000,
                    models=["metaclip2_vith14"])
        self.assertEqual(str(job_path(base, "abc123")),
                         "results/baseline/crisismmd/humanitarian/samples_1000/metaclip2_vith14/humanitarian_abc123")
        direct = {**base, "image_text_relation": {"enabled": True, "mode": "direct_pair_fusion"}}
        self.assertEqual(str(job_path(direct, "abc123")),
                         "results/direct_pair_fusion/crisismmd/humanitarian/samples_1000/metaclip2_vith14/humanitarian_abc123")
        adapted = {**base, "kind": "adapted", "methods": ["dclip"]}
        self.assertEqual(str(job_path(adapted, "abc123")),
                         "results/adapted_dclip/crisismmd/humanitarian/samples_1000/metaclip2_vith14/humanitarian_abc123")
        self.assertEqual(str(job_path({**adapted, "result_protocol": "image_only"}, "abc123")),
                         "results/image_only_adapted_dclip/crisismmd/humanitarian/samples_1000/metaclip2_vith14/humanitarian_abc123")
        self.assertEqual(model_folder(models=["metaclip2_vith14", "siglip2_so400m"]), "multi_model")
        upstream = dict(kind="upstream", dataset="crisismmd", task="humanitarian",
                        max_samples=1000, methods=["dclip"], backbones=["ViT-B/32"])
        self.assertIn("/openai_clip_vit_b_32/", str(job_path(upstream, "abc123")))

    def test_standalone_config_routes_default_only(self):
        config = {"models": ["metaclip2_vith14"],
                  "inference": {"image_text_relation": {"enabled": False}},
                  "datasets": {"crisismmd": {"task": "informativeness", "max_samples": 3000}},
                  "output": {"results_dir": "auto", "raw_predictions_dir": "auto",
                             "summary_table": "auto"}}
        active = standalone_config(config, "crisismmd", "run01")
        self.assertEqual(active["output"]["results_dir"],
                         "results/baseline/crisismmd/informativeness/samples_3000/metaclip2_vith14/informativeness_run01")
        self.assertEqual(config["output"]["results_dir"], "auto")
        self.assertIs(standalone_config(active, "crisismmd"), active)

    def test_analysis_inherits_task_from_source_run(self):
        source = Path("results/baseline/fakeddit/6way/samples_1000/metaclip2_vith14/6way_abc123")
        self.assertEqual(str(analysis_path("fallback", source, "6way_review")),
                         "results/analysis/fallback/fakeddit/6way/samples_1000/metaclip2_vith14/6way_review")
        self.assertEqual(str(analysis_path("fallback", source, "6way_review", model="multi_model")),
                         "results/analysis/fallback/fakeddit/6way/samples_1000/multi_model/6way_review")

    def test_group_analysis_preserves_reports_and_migration_map(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "results"
            source = (results / "analysis_fallback/fakeddit/6way/samples_1000"
                      / "metaclip2_vith14/6way_review")
            source.mkdir(parents=True)
            (source / "summary.json").write_text('{"n_fixed": 2}', encoding="utf-8")
            manifest = results / "_system/migration_map.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text(json.dumps([{"from": "analysis/fallback/6way_review",
                "to": "analysis_fallback/fakeddit/6way/samples_1000/metaclip2_vith14/6way_review"}]),
                encoding="utf-8")
            moves = migrate_results_layout.plan_analysis_group(root)
            self.assertEqual(len(moves), 1)
            migrate_results_layout.apply_analysis_group(moves, root)
            target = results / "analysis/fallback/fakeddit/6way/samples_1000/metaclip2_vith14/6way_review"
            self.assertEqual(json.loads((target / "summary.json").read_text())["n_fixed"], 2)
            self.assertFalse(source.exists())
            self.assertEqual(json.loads(manifest.read_text())[0]["to"],
                             target.relative_to(results).as_posix())
            self.assertEqual(migrate_results_layout.plan_analysis_group(root), [])

    def test_group_image_only_moves_only_complete_original_image_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "results"
            source = (results / "upstream_dclip/crisismmd/informativeness/samples_1000"
                      / "openai_clip_vit_b_32/informativeness_abc123")
            mixed = (results / "upstream_suite/crisismmd/humanitarian/samples_5"
                     / "openai_clip_vit_b_32/humanitarian_def456")
            for folder, variants in ((source, ["original_image"]),
                                     (mixed, ["original_image", "multimodal"])):
                folder.mkdir(parents=True)
                (folder / "summary.tsv").write_text(
                    "output_variant\n" + "".join(variant + "\n" for variant in variants),
                    encoding="utf-8")
            manifest = results / "_system/migration_map.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text(json.dumps([{"from": "experiments/abc123",
                "to": source.relative_to(results).as_posix()}]), encoding="utf-8")
            moves = migrate_results_layout.plan_image_only_runs(root)
            self.assertEqual(len(moves), 1)
            migrate_results_layout.apply_image_only_runs(moves, root)
            target = (results / "image_only_upstream_dclip/crisismmd/informativeness/samples_1000"
                      / "openai_clip_vit_b_32/informativeness_abc123")
            self.assertTrue((target / "summary.tsv").is_file())
            self.assertTrue((mixed / "summary.tsv").is_file())
            self.assertFalse(source.exists())
            self.assertEqual(json.loads(manifest.read_text())[0]["to"],
                             target.relative_to(results).as_posix())

    def test_migration_uses_receipt_for_header_only_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "results/runs/crisismmd_info/abc123"
            source.mkdir(parents=True)
            (source / "summary.tsv").write_text("model\tdataset\ttask\n", encoding="utf-8")
            receipt = {"job_yaml": yaml.safe_dump({"id": "crisismmd_info", "kind": "standard",
                "dataset": "crisismmd", "task": "informativeness", "max_samples": 2,
                "models": ["metaclip2_vith14"]})}
            (source / "run_receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
            moves = migrate_results_layout.plan(root)
            self.assertEqual(len(moves), 1)
            self.assertEqual(moves[0][1].relative_to(root).as_posix(),
                             "results/baseline/crisismmd/informativeness/samples_2/metaclip2_vith14/informativeness_abc123")
            migrate_results_layout.apply(moves, root)
            self.assertFalse(source.exists())
            self.assertTrue(moves[0][1].is_dir())
            self.assertTrue((root / "results/_system/migration_map.json").is_file())

    def test_insert_task_level_updates_existing_migration_map(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "results"
            source = results / "baseline/mmimdb/samples_1000/multi_label_abc123"
            source.mkdir(parents=True)
            (source / "manifest.json").write_text('{"model": "siglip2_so400m"}', encoding="utf-8")
            manifest = results / "_system/migration_map.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text(json.dumps([{"from": "raw_predictions/abc123",
                "to": "baseline/mmimdb/samples_1000/multi_label_abc123/manifest.json"}]),
                encoding="utf-8")
            moves = migrate_results_layout.plan_task_level(root)
            self.assertEqual(len(moves), 1)
            migrate_results_layout.apply_task_level(moves, root)
            target = results / "baseline/mmimdb/genres/samples_1000/multi_label_abc123/manifest.json"
            self.assertTrue(target.is_file())
            self.assertEqual(json.loads(manifest.read_text())[0]["to"],
                             target.relative_to(results).as_posix())
            model_moves = migrate_results_layout.plan_model_level(root)
            self.assertEqual(len(model_moves), 1)
            migrate_results_layout.apply_model_level(model_moves, root)
            final = results / "baseline/mmimdb/genres/samples_1000/siglip2_so400m/multi_label_abc123/manifest.json"
            self.assertTrue(final.is_file())
            self.assertEqual(json.loads(manifest.read_text())[0]["to"],
                             final.relative_to(results).as_posix())
            self.assertEqual(migrate_results_layout.plan_model_level(root), [])


if __name__ == "__main__":
    unittest.main()
