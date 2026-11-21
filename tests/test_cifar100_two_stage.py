import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vlm_project import cifar100_two_stage as experiment
from vlm_project.io import sha256


class FakeBackend:
    metadata = {"backend": "synthetic-test-only"}

    def __init__(self, device):
        self.unloaded = False

    def prototypes(self, bank):
        if len(bank) == 100:
            return np.eye(100)
        return np.asarray([np.isin(np.arange(100), np.arange(g*5, g*5+5)).astype(float) for g in range(20)])

    def encode_images(self, images):
        return np.eye(100)[[im.getpixel((0, 0))[0] for im in images]]

    def unload(self):
        self.unloaded = True


class CifarExperimentTests(unittest.TestCase):
    def test_hard_soft_normalization_and_noncontiguous_mapping(self):
        fine = np.array([[4., 3., 2., 1.]])
        mapping = np.array([1, 0, 1, 0])
        scores, coarse = experiment.predict(fine, [[10., 0.]], mapping)
        self.assertEqual(scores["flat"].argmax(1).item(), 0)
        self.assertEqual(scores["hierarchy_hard"].argmax(1).item(), 1)
        self.assertEqual(scores["hierarchy_soft"].argmax(1).item(), 1)
        for values in scores.values():
            np.testing.assert_allclose(values.sum(1), 1)
        for group in range(2):
            np.testing.assert_allclose(scores["hierarchy_soft"][:, mapping == group].sum(1), coarse[:, group])

    def test_aggregated_flat_coarse_is_identity_control(self):
        fine = np.random.default_rng(9).normal(size=(7, 6))
        mapping = np.array([0, 1, 2, 0, 1, 2])
        group_logits = np.stack([logsumexp(fine[:, mapping == g], axis=1) for g in range(3)], axis=1)
        scores, _ = experiment.predict(fine, group_logits, mapping)
        np.testing.assert_allclose(scores["flat"], scores["hierarchy_soft"], atol=1e-12)

    def test_split_deterministic_and_test_smoke_blocked(self):
        spec = experiment.specification()
        dev = experiment.split_indices(spec, "dev")
        self.assertEqual(len(dev), 10000)
        self.assertEqual(len(np.unique(dev)), 10000)
        self.assertTrue(np.all((dev >= 0) & (dev < 50000)))
        np.testing.assert_array_equal(dev[:5], experiment.split_indices(spec, "dev", True))
        np.testing.assert_array_equal(dev, experiment.split_indices(spec, "dev"))
        with self.assertRaises(ValueError):
            experiment.split_indices(spec, "test", True)
        for mapping in [[0, 0, 0], [0., 1., 0.]]:
            with self.assertRaises(ValueError):
                experiment.predict([[1, 2, 3]], [[1, 2]], mapping)
        with self.assertRaises(ValueError):
            experiment.predict([[np.nan, 1]], [[1, 2]], [0, 1])

    def test_test_requires_complete_full_dev_same_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            protocol = {"specification": {"dev_samples": 10000}}
            run = {"status": "complete", "split": "dev", "mode": "smoke", "samples": 5,
                   "protocol": protocol, "output_sha256": {}}
            def write():
                (folder / "run.json").write_text(json.dumps(run))
                (folder / "completed.json").write_text(json.dumps({"status": "complete", "run_sha256": sha256(folder / "run.json")}))
            write()
            with self.assertRaisesRegex(ValueError, "smoke"):
                experiment.validate_dev(folder, protocol)
            run.update(mode="full", samples=10000)
            write()
            self.assertIn("dev_run_sha256", experiment.validate_dev(folder, protocol))
            with self.assertRaisesRegex(ValueError, "protocol"):
                experiment.validate_dev(folder, {"specification": {"dev_samples": 9999}})
            (folder / "run.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "checksum"):
                experiment.validate_dev(folder, protocol)

    def test_end_to_end_smoke_sparse_synthetic_binary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            root.mkdir()
            metadata = {"dataset": "cifar100", "status": "sources_verified",
                        "splits": {"train": 50000, "test": 10000},
                        "class_names": {"fine": [f"class_{i}" for i in range(100)],
                                        "coarse": [f"group_{i}" for i in range(20)]},
                        "fine_to_coarse": {str(i): i//5 for i in range(100)}}
            (root / "download_manifest.json").write_text(json.dumps(metadata))
            ids = experiment.split_indices(experiment.specification(), "dev", True)
            with (root / "train.bin").open("wb") as stream:
                stream.truncate(50000*3074)  # Sparse fixture; never allocates a real dataset.
                for index in ids:
                    fine = int(index) % 100
                    record = bytes([fine//5, fine]) + bytes([fine])*1024 + bytes(2048)
                    stream.seek(int(index)*3074)
                    stream.write(record)
            instances = []
            def factory(device):
                instances.append(FakeBackend(device))
                return instances[-1]
            output = experiment.run(root, Path(tmp) / "output", backend_factory=factory)
            self.assertTrue(instances[0].unloaded)
            self.assertTrue((output / "completed.json").exists())
            run = json.loads((output / "run.json").read_text())
            self.assertEqual(run["samples"], 5)
            self.assertEqual(run["mode"], "smoke")
            self.assertEqual(len(run["summary"]), 3)
            with np.load(output / "replay_scores.npz", allow_pickle=False) as replay:
                np.testing.assert_array_equal(replay["ids"], ids)
                for method in experiment.METHODS:
                    self.assertEqual(replay[method].shape, (5, 100))
                    np.testing.assert_allclose(replay[method].sum(1), 1)
            with self.assertRaisesRegex(ValueError, "smoke"):
                experiment.validate_dev(output, run["protocol"])

    def test_generated_notebook_contains_only_whitelisted_code(self):
        spec = importlib.util.spec_from_file_location("cifar_job_builder", ROOT / "scripts/prepare_cifar100_two_stage_job.py")
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        with tempfile.TemporaryDirectory() as tmp:
            destination = builder.prepare(output=tmp)
            metadata = json.loads((destination / "kernel-metadata.json").read_text())
            self.assertEqual(metadata["dataset_sources"], ["nguyenhung3105/vlm-cifar100-sources-v1"])
            notebook = json.loads((destination / "run.ipynb").read_text())
            for cell in notebook["cells"]:
                if cell["cell_type"] == "code":
                    compile("".join(cell["source"]), "generated", "exec")
            files = json.loads((destination / "source_manifest.json").read_text())["sha256"]
            self.assertNotIn("src/vlm_project/fakeddit.py", files)
            with self.assertRaises(ValueError):
                builder.prepare(mode="test", output=tmp)


if __name__ == "__main__":
    unittest.main()
