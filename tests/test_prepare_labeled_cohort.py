"""Kiểm tra cohort bằng dữ liệu tổng hợp; không truy cập data/ của dự án."""
import csv
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.error
import socket

from PIL import Image
import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("prepare_labeled_cohort", ROOT / "scripts/prepare_labeled_cohort.py")
cohort = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cohort)
from config import load_config
from run.common import load_dataframe, parse_multilabel, resolve_image_paths, crisismmd_label_column
from prompts.fakeddit_prompts import normalize_labels
promotion_spec = importlib.util.spec_from_file_location("promote_processed_data", ROOT / "scripts/promote_processed_data.py")
promotion = importlib.util.module_from_spec(promotion_spec)
promotion_spec.loader.exec_module(promotion)


class CohortTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        config = yaml.safe_load((ROOT / "configs/experiment_config.yaml").read_text())
        (self.root / "configs").mkdir()
        (self.root / "configs/experiment_config.yaml").write_text(yaml.safe_dump(config))

    def image(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (12, 10), "blue").save(path)

    def tsv(self, path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        cohort.write_tsv(path, rows)

    def args(self, dataset, task, split, samples=3, output="data/processed/cohort", extra=()):
        return cohort.parser().parse_args(["--dataset", dataset, "--task", task, "--split", split,
                                          "--samples", str(samples), "--output-dir", output, *extra])

    def output(self, args):
        with (self.root / args.output_dir / "samples.tsv").open(newline="") as stream:
            return list(csv.DictReader(stream, delimiter="\t"))

    def crisis(self, bad_id=False, overlap=False, bad_image=False):
        folder = self.root / "data/raw/CrisisMMD_v2.0"
        for part, start, length in (("train", 100, 12), ("dev", 200, 2), ("test", 300, 2)):
            rows = []
            for offset in range(length):
                tweet = str(start + offset)
                if overlap and part == "test" and offset == 0:
                    tweet = "100"
                image_id = tweet + ("_1" if overlap and part == "test" and offset == 0 else "_0")
                image = f"data_image/event{offset % 2}/{image_id}.jpg"
                self.image(folder / image)
                rows.append({"tweet_id": "1.0E+2" if bad_id and part == "train" and offset == 0 else tweet,
                             "image_id": image_id, "event_name": f"event{offset % 2}",
                             "tweet_text": "Cứu hộ đang tiếp cận khu vực ngập lụt #relief",
                             "image": image, "label": "informative",
                             "label_text": "informative", "label_image": "not_informative" if offset % 3 == 0 else "informative"})
            self.tsv(folder / f"crisismmd_datasplit_all/task_informative_text_img_{part}.tsv", rows)
        if bad_image:
            (folder / "data_image/event0/100_0.jpg").write_bytes(b"not an image")

    def fakeddit(self):
        rows = []
        for label in range(6):
            for index in range(4):
                key = f"post{label}_{index}"
                rows.append({"id": key, "clean_title": "A sufficiently long genuine title for testing",
                             "image_url": "https://example.invalid/image.jpg", "hasImage": "True",
                             "2_way_label": "1" if label == 0 else "0", "6_way_label": str(label)})
                # Both image locations must work, without changing raw files.
                directory = "raw/fakeddit" if index % 2 else "processed/pro_fakeddit"
                self.image(self.root / f"data/{directory}/images/{key}.jpg")
        rows.append({**rows[0], "id": "too_short", "clean_title": "abc"})
        rows.append({**rows[0], "id": "bad_label", "6_way_label": "unknown"})
        self.tsv(self.root / "data/raw/fakeddit/multimodal_validate.tsv", rows)

    def mmimdb(self):
        folder = self.root / "data/raw/mmimdb"
        (folder / "dataset").mkdir(parents=True)
        splits = {"train": ["0000001"], "dev": ["0000002"],
                  "test": [f"{index:07}" for index in range(3, 16)]}
        (folder / "split.json").write_text(json.dumps(splits))
        for index in range(1, 16):
            raw = {"plot": ["A film plot with enough text to evaluate."],
                   "genres": ["Drama", "Drama", "Short"] if index % 3 == 0 else ["Comedy"], "kind": "movie"}
            if index == 3:
                raw["plot"] = [None]
                raw["plot outline"] = None
            if index == 4:
                raw["genres"] = "Drama"
            (folder / f"dataset/{index:07}.json").write_text(json.dumps(raw))
            self.image(folder / f"dataset/{index:07}.jpeg")

    def test_crisis_targets_preserved_and_config_uses_y_true(self):
        self.crisis(bad_image=True)
        args = self.args("crisismmd", "informativeness", "train", samples=6)
        result = cohort.prepare(args, self.root)
        rows = self.output(args)
        self.assertEqual(len(rows), 6)
        self.assertEqual(len({row["image_id"] for row in rows}), 6)
        for row in rows:
            self.assertEqual(row["y_true"], row["label_image"])
            self.assertEqual(row["y_true"], row["label_informative"])
            self.assertEqual(row["source_split"], "train")
            self.assertEqual(row["label_source"], "image")
            self.assertIn("Cứu hộ", row["clean_text"])
            self.assertTrue(cohort.image_ok(self.root / row["image_path"]))
        self.assertNotIn("100_0", {row["image_id"] for row in rows})
        self.assertEqual(result["rejected"]["missing_or_unreadable_image"], 1)
        profile = yaml.safe_load((self.root / args.output_dir / "experiment.yaml").read_text())
        self.assertEqual(profile["datasets"]["crisismmd"]["label_col_info"], "y_true")
        self.assertEqual(profile["datasets"]["crisismmd"]["id_col"], "image_id")
        self.assertFalse(profile["datasets"]["mmimdb"]["enabled"])
        self.assertEqual(profile["evaluation"]["sample_lock_mode"], "create_or_verify")
        self.assertEqual(result["samples_sha256"], cohort.digest(self.root / args.output_dir / "samples.tsv"))
        task = profile["datasets"]["crisismmd"]
        self.assertEqual(crisismmd_label_column(task), "y_true")

    def test_crisis_bad_ids_and_cross_split_tweets_fail(self):
        self.crisis(bad_id=True)
        args = self.args("crisismmd", "informativeness", "train")
        with self.assertRaisesRegex(ValueError, "ID không nguyên vẹn"):
            cohort.prepare(args, self.root)
        self.assertFalse((self.root / args.output_dir).exists())
        self.crisis(overlap=True)
        with self.assertRaisesRegex(ValueError, "giao nhau"):
            cohort.prepare(args, self.root)

    def test_no_padding_or_overwriting(self):
        self.crisis()
        args = self.args("crisismmd", "informativeness", "test", samples=3)
        with self.assertRaisesRegex(ValueError, "pool chỉ có 2"):
            cohort.prepare(args, self.root)
        self.assertFalse((self.root / args.output_dir).exists())
        args.samples = 2
        cohort.prepare(args, self.root)
        original = (self.root / args.output_dir / "samples.tsv").read_bytes()
        with self.assertRaisesRegex(ValueError, "Không ghi đè"):
            cohort.prepare(args, self.root)
        self.assertEqual((self.root / args.output_dir / "samples.tsv").read_bytes(), original)

    def test_fakeddit_both_image_roots_filter_and_ground_truth(self):
        self.fakeddit()
        args = self.args("fakeddit", "6way", "validate", samples=12)
        result = cohort.prepare(args, self.root)
        rows = self.output(args)
        self.assertEqual(result["selected"]["support"], {str(label): 2 for label in range(6)})
        self.assertEqual(result["rejected"]["short_text"], 1)
        self.assertEqual(result["rejected"]["invalid_label"], 1)
        self.assertTrue(all(row["y_true"] == row["6_way_label"] for row in rows))
        self.assertTrue(all(len(row["clean_title"].replace(" ", "")) > 20 for row in rows))

    def test_config_and_output_compatible_with_existing_loader(self):
        self.fakeddit()
        args = self.args("fakeddit", "2way", "validate", samples=12)
        cohort.prepare(args, self.root)
        config = load_config(self.root / args.output_dir / "experiment.yaml")
        task = config["datasets"]["fakeddit"]
        self.assertEqual(task["label_col_2way"], "y_true")
        # Resolve paths relative to the synthetic root, just as CLI uses repo cwd.
        task["image_dir"] = str(self.root / task["image_dir"])
        frame = load_dataframe(str(self.root / task["data_path"]), max_samples=12,
                               text_col=task["text_col"], min_text_length=task["min_text_length"])
        self.assertEqual(len(frame), 12)
        self.assertEqual(normalize_labels(frame.y_true.tolist(), "2way"),
                         normalize_labels(frame["2_way_label"].tolist(), "2way"))
        self.assertTrue(all(Path(path).is_file() for path in resolve_image_paths(frame, task)))

    def test_fakeddit_cannot_hide_a_missing_class_by_redistribution(self):
        self.fakeddit()
        for path in (self.root / "data").rglob("post5_*.jpg"):
            path.unlink()
        args = self.args("fakeddit", "2way", "validate", samples=12)
        with self.assertRaisesRegex(ValueError, "Thiếu ảnh hợp lệ theo quota"):
            cohort.prepare(args, self.root)
        self.assertFalse((self.root / args.output_dir).exists())
        args.plan_downloads = True
        report = cohort.prepare(args, self.root)
        self.assertEqual(report["status"], "download_plan_only")
        self.assertEqual(report["download_candidates"], 4)
        self.assertFalse((self.root / args.output_dir / "samples.tsv").exists())
        self.assertFalse((self.root / args.output_dir / "experiment.yaml").exists())

    def test_mmimdb_only_requested_split_preserves_multilabel_and_short(self):
        self.mmimdb()
        args = self.args("mmimdb", "genres", "test", samples=7)
        result = cohort.prepare(args, self.root)
        rows = self.output(args)
        self.assertEqual(len(rows), 7)
        self.assertTrue(all(int(row["id"]) >= 5 for row in rows))
        for row in rows:
            self.assertEqual(json.loads(row["y_true"]), row["genres"].split("|"))
            self.assertEqual(parse_multilabel(row["y_true"]), row["genres"].split("|"))
            self.assertEqual(int(row["n_genres"]), len(json.loads(row["y_true"])))
            self.assertEqual(len(row["id"]), 7)
        self.assertIn("Short", result["selected"]["support"])
        self.assertEqual(result["rejected"]["short_text"], 1)
        self.assertEqual(result["rejected"]["invalid_label_list"], 1)
        original_ids = [row["id"] for row in rows]
        args.output_dir = "data/processed/cohort_again"
        second = cohort.prepare(args, self.root)
        self.assertEqual(original_ids, [row["id"] for row in self.output(args)])
        self.assertEqual(result["cohort_id"], second["cohort_id"])

    def test_largest_remainder_and_multilabel_boundary_sizes(self):
        self.assertEqual(sum(cohort.quotas({"a": 1, "b": 2, "c": 10}, 8).values()), 8)
        pool = [{"_key": str(index), "_labels": ["Drama", "Short"] if index % 4 == 0 else ["Comedy"]}
                for index in range(20)]
        for size in (1, 7, 19, 20):
            sample = cohort.multilabel_sample(pool, size, 42)
            self.assertEqual(len(sample), size)
            self.assertEqual(len({row["_key"] for row in sample}), size)
        sample = cohort.multilabel_sample(pool, 8, 42)
        self.assertEqual(sum("Short" in row["_labels"] for row in sample), 2)

    def test_multilabel_three_thousand_synthetic_samples(self):
        pool = [{"_key": f"{index:07}", "_labels": ["Drama", "Short"] if index % 50 == 0 else ["Comedy"]}
                for index in range(5000)]
        selected = cohort.multilabel_sample(pool, 3000, 42)
        self.assertEqual(len(selected), 3000)
        self.assertEqual(len({row["_key"] for row in selected}), 3000)
        self.assertEqual(sum("Short" in row["_labels"] for row in selected), 60)
        reversed_sample = cohort.multilabel_sample(list(reversed(pool)), 3000, 42)
        self.assertEqual([row["_key"] for row in selected], [row["_key"] for row in reversed_sample])

    def test_explicit_crisis_label_source(self):
        self.crisis()
        for source in ("image", "text", "official"):
            args = self.args("crisismmd", "informativeness", "test", samples=2,
                             output=f"data/processed/{source}", extra=("--label-source", source))
            cohort.prepare(args, self.root)
            field = f"label_{source}"
            self.assertTrue(all(row["y_true"] == row[field] for row in self.output(args)))

    def test_humanitarian_keeps_eight_source_labels(self):
        self.crisis()
        folder = self.root / "data/raw/CrisisMMD_v2.0/crisismmd_datasplit_all"
        labels = list(cohort.LABEL_NAMES_HUMANITARIAN)
        for part in ("train", "dev", "test"):
            with (folder / f"task_informative_text_img_{part}.tsv").open(newline="") as stream:
                rows = list(csv.DictReader(stream, delimiter="\t"))
            for index, row in enumerate(rows):
                for column in ("label", "label_text", "label_image"):
                    row[column] = labels[index % len(labels)]
            self.tsv(folder / f"task_humanitarian_text_img_{part}.tsv", rows)
        args = self.args("crisismmd", "humanitarian", "train", samples=12)
        result = cohort.prepare(args, self.root)
        self.assertEqual(set(result["selected"]["support"]), set(labels))
        self.assertTrue(all(row["y_true"] == row["label_humanitarian"] == row["label_image"]
                            for row in self.output(args)))
        config = load_config(self.root / args.output_dir / "experiment.yaml")
        self.assertEqual(crisismmd_label_column(config["datasets"]["crisismmd"]), "y_true")

    def both_crisis(self):
        self.crisis()
        folder = self.root / "data/raw/CrisisMMD_v2.0/crisismmd_datasplit_all"
        for part in ("train", "dev", "test"):
            with (folder / f"task_informative_text_img_{part}.tsv").open(newline="") as stream:
                rows = list(csv.DictReader(stream, delimiter="\t"))
            for index, row in enumerate(rows):
                for column in ("label", "label_text", "label_image"):
                    row[column] = "not_humanitarian" if index % 3 == 0 else "other_relevant_information"
            self.tsv(folder / f"task_humanitarian_text_img_{part}.tsv", rows)

    def test_shared_crisis_has_correct_truth_for_both_tasks(self):
        self.both_crisis()
        args = self.args("crisismmd", "both", "train", samples=8)
        result = cohort.prepare(args, self.root)
        rows = self.output(args)
        self.assertEqual(len(rows), 8)
        for row, entry in zip(rows, result["samples"]):
            truth = json.loads(row["y_true"])
            self.assertEqual(truth["informativeness"], row["label_informative"])
            self.assertEqual(truth["humanitarian"], row["label_humanitarian"])
            self.assertEqual(entry["y_true"], truth)
        config = load_config(self.root / args.output_dir / "experiment.yaml")
        settings = config["datasets"]["crisismmd"]
        for task, field in (("informativeness", "label_informative"), ("humanitarian", "label_humanitarian")):
            self.assertEqual(crisismmd_label_column({**settings, "task": task}), field)

    def test_common_split_excludes_other_task_dev_samples(self):
        self.both_crisis()
        folder = self.root / "data/raw/CrisisMMD_v2.0/crisismmd_datasplit_all"
        path = folder / "task_humanitarian_text_img_train.tsv"
        with path.open(newline="") as stream:
            train = list(csv.DictReader(stream, delimiter="\t"))
        path_dev = folder / "task_humanitarian_text_img_dev.tsv"
        with path_dev.open(newline="") as stream:
            dev = list(csv.DictReader(stream, delimiter="\t"))
        dev.append(train.pop(0))
        self.tsv(path, train)
        self.tsv(path_dev, dev)
        args = self.args("crisismmd", "both", "train", samples=11)
        report = cohort.prepare(args, self.root)
        self.assertNotIn("100_0", {row["image_id"] for row in self.output(args)})
        self.assertEqual(report["metadata_pool"]["samples"], 11)

    def test_download_only_missing_class_quota(self):
        self.fakeddit()
        for path in (self.root / "data").rglob("post5_*.jpg"):
            path.unlink()
        args = self.args("fakeddit", "6way", "validate", samples=12, extra=("--download-missing",))
        def fetch(row, cache, timeout):
            target = cache / f"{row['_key']}.jpg"
            self.image(target)
            return target, None
        with patch.object(cohort, "fetch_image", side_effect=fetch) as fetcher, redirect_stdout(io.StringIO()):
            result = cohort.prepare(args, self.root)
        self.assertEqual(fetcher.call_count, 2)
        self.assertEqual(result["selected"]["support"]["5"], 2)
        self.assertEqual(result["downloads"]["attempts"], 2)

    def test_download_validation_atomicity_and_dns(self):
        path = self.root / "source.jpg"
        self.image(path)
        payload = path.read_bytes()
        class Response(io.BytesIO):
            def geturl(self):
                return "https://example.invalid/real.jpg"
        row = {"_key": "sample", "image_url": "https://example.invalid/real.jpg"}
        cache = self.root / "cache"
        cache.mkdir()
        with patch.object(cohort.urllib.request, "urlopen", return_value=Response(payload)):
            target, error = cohort.fetch_image(row, cache, 1)
        self.assertIsNone(error)
        self.assertEqual(target.read_bytes(), payload)
        with patch.object(cohort.urllib.request, "urlopen", return_value=Response(b"bad image")):
            target, error = cohort.fetch_image(row, cache, 1)
        self.assertIsNone(target)
        self.assertEqual(error, "unreadable_download")
        self.assertEqual((cache / "sample.jpg").read_bytes(), payload)
        self.assertFalse(list(cache.glob("*.part")))
        with patch.object(cohort.urllib.request, "urlopen", side_effect=urllib.error.URLError(socket.gaierror())):
            self.assertEqual(cohort.fetch_image(row, cache, 1)[1], "dns_unavailable")

    def test_decodable_removed_image_placeholder_is_rejected(self):
        path = self.root / "placeholder.jpg"
        self.image(path)
        self.assertTrue(cohort.image_ok(path))
        with patch.object(cohort, "REMOVED_IMAGE_SHA256", {cohort.digest(path)}):
            self.assertFalse(cohort.image_ok(path))

    def promotion_fixture(self):
        primary = self.root / "data/processed/pro_mmimdb"
        primary.mkdir(parents=True)
        (primary / "mmimdb_sample.tsv").write_bytes(b"original dataset bytes\n")
        source = self.root / "data/processed/prepared_mmimdb"
        image = source / "images/shared.jpg"
        self.image(image)
        image_hash = cohort.digest(image)
        rows = [{"id": str(index), "genres": "Drama", "y_true": '["Drama"]',
                 "image_path": str(image.relative_to(self.root))} for index in range(3000)]
        self.tsv(source / "samples.tsv", rows)
        config = yaml.safe_load((self.root / "configs/experiment_config.yaml").read_text())
        (source / "experiment.yaml").write_text(yaml.safe_dump(config))
        report = {"dataset": "mmimdb", "task": "genres", "status": "ready", "source_split": "test",
                  "label_source": "genres", "requested_samples": 3000, "selected": {"samples": 3000},
                  "samples_sha256": cohort.digest(source / "samples.tsv"), "cohort_id": "fixture-cohort",
                  "samples": [{"id": str(index), "image_sha256": image_hash} for index in range(3000)]}
        (source / "manifest.json").write_text(json.dumps(report))
        return primary, source

    def test_promotion_preserves_old_data_updates_paths_and_refuses_v1_overwrite(self):
        primary, source = self.promotion_fixture()
        old = (primary / "mmimdb_sample.tsv").read_bytes()
        original_config = yaml.safe_load((self.root / "configs/experiment_config.yaml").read_text())
        result = promotion.promote("mmimdb", source, self.root)
        self.assertEqual(result["samples"], 3000)
        self.assertEqual((primary.with_name("pro_mmimdb_v1") / "mmimdb_sample.tsv").read_bytes(), old)
        config = load_config(self.root / "configs/experiment_config.yaml")
        self.assertEqual(config["datasets"]["mmimdb"]["max_samples"], 3000)
        self.assertEqual(config["datasets"]["crisismmd"], original_config["datasets"]["crisismmd"])
        with (primary / "mmimdb_sample.tsv").open(newline="") as stream:
            row = next(csv.DictReader(stream, delimiter="\t"))
        self.assertEqual(row["image_path"], "data/processed/pro_mmimdb/images/shared.jpg")
        report = json.loads((primary / "manifest.json").read_text())
        self.assertEqual(report["samples_sha256"], cohort.digest(primary / "mmimdb_sample.tsv"))
        self.assertFalse(source.exists())
        with self.assertRaisesRegex(ValueError, "không ghi đè bản v1"):
            promotion.promote("mmimdb", source, self.root)

    def test_promotion_rolls_back_on_failed_move(self):
        primary, source = self.promotion_fixture()
        old_config = (self.root / "configs/experiment_config.yaml").read_bytes()
        rename = Path.rename
        def fail_source(path, target):
            if path == source.resolve():
                raise OSError("fixture move failure")
            return rename(path, target)
        with patch.object(Path, "rename", fail_source), self.assertRaisesRegex(OSError, "move failure"):
            promotion.promote("mmimdb", source, self.root)
        self.assertEqual((primary / "mmimdb_sample.tsv").read_bytes(), b"original dataset bytes\n")
        self.assertTrue(source.exists())
        self.assertFalse(primary.with_name("pro_mmimdb_v1").exists())
        self.assertEqual((self.root / "configs/experiment_config.yaml").read_bytes(), old_config)


if __name__ == "__main__":
    unittest.main()
