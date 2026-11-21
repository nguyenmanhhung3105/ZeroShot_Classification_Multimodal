import io
import json
import sys
import tempfile
import tarfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vlm_project.shared_data import find_sources, load_samples


class SharedDataTests(unittest.TestCase):
    def test_cifar_planar_pixels_and_hierarchy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Distinct R/G/B planes catch accidental interleaved decoding.
            record = bytes([2, 7]) + bytes([10])*1024 + bytes([20])*1024 + bytes([30])*1024
            (root / "train.bin").write_bytes(record * 5)
            metadata = {"splits": {"train": 5}, "class_names": {"fine": [str(i) for i in range(100)]},
                        "fine_to_coarse": {"7": 2}}
            rows = load_samples("cifar100", root, metadata)
            self.assertEqual(len(rows), 5)
            self.assertEqual(rows[0][1].getpixel((0, 0)), (10, 20, 30))
            self.assertEqual(rows[0][0]["coarse_label"], 2)
            metadata["fine_to_coarse"]["7"] = 3
            with self.assertRaisesRegex(ValueError, "mismatch"):
                load_samples("cifar100", root, metadata)

    def test_flowers_uses_validation_ids_and_zero_based_labels(self):
        import numpy as np
        from PIL import Image
        from scipy.io import savemat
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labels = np.ones(8189, dtype=int)
            labels[9] = 102
            savemat(root / "imagelabels.mat", {"labels": labels})
            savemat(root / "setid.mat", {"valid": np.arange(10, 15), "tstid": [99]})
            for i in range(10, 15):
                Image.new("RGB", (8, 9)).save(root / f"image_{i:05d}.jpg")
            rows = load_samples("flowers102", root, {})
            self.assertEqual(rows[0][0]["id"], "10")
            self.assertEqual(rows[0][0]["label"], 101)
            self.assertEqual(rows[0][0]["split"], "validation")

    def test_aircraft_class_names_with_spaces(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "images").mkdir()
            (root / "images_variant_val.txt").write_text("".join(f"{i:07d} Boeing 737-700\n" for i in range(5)))
            for i in range(5):
                Image.new("RGB", (10, 30)).save(root / "images" / f"{i:07d}.jpg")
            rows = load_samples("fgvc_aircraft", root, {"class_names": {"variant": ["Boeing 737-700"]}})
            self.assertEqual(rows[0][0]["label"], 0)
            self.assertEqual(rows[0][1].height, 30)  # Data smoke preserves raw image.

    def test_flowers_tgz_without_extracted_images(self):
        import numpy as np
        from PIL import Image
        from scipy.io import savemat
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            savemat(root / "imagelabels.mat", {"labels": np.ones(8189, dtype=int)})
            savemat(root / "setid.mat", {"valid": np.arange(10, 15)})
            payload = io.BytesIO()
            Image.new("RGB", (13, 17), color=(25, 50, 75)).save(payload, format="JPEG")
            with tarfile.open(root / "102flowers.tgz", "w:gz") as archive:
                for index in range(9, 16):
                    # Unselected invalid images must not be decoded.
                    data = payload.getvalue() if 10 <= index <= 14 else b"not an image"
                    entry = tarfile.TarInfo(f"jpg/image_{index:05d}.jpg")
                    entry.size = len(data)
                    archive.addfile(entry, io.BytesIO(data))
            rows = load_samples("flowers102", root, {})
            self.assertEqual([r[0]["id"] for r in rows], [str(i) for i in range(10, 15)])
            self.assertEqual(rows[0][1].size, (13, 17))
            self.assertEqual(list(root.rglob("*.jpg")), [])
            # Keep validation selection fixed; never substitute a different image.
            savemat(root / "setid.mat", {"valid": np.arange(100, 105)})
            with self.assertRaisesRegex(ValueError, "Missing Flowers archive images"):
                load_samples("flowers102", root, {})

    def test_cars_decodes_only_requested_rows(self):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest("pyarrow not installed locally")
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = io.BytesIO()
            Image.new("RGB", (11, 12)).save(image, format="PNG")
            names = [str(i) for i in range(196)]
            # Invalid sixth image must never be decoded by a five-image smoke.
            images = [{"bytes": image.getvalue(), "path": None}]*5 + [{"bytes": b"invalid", "path": None}]
            table = pa.table({"image": images, "label": list(range(6))})
            info = {"info": {"features": {"label": {"names": names}}}}
            table = table.replace_schema_metadata({b"huggingface": json.dumps(info).encode()})
            pq.write_table(table, root / "train-00000.parquet")
            rows = load_samples("stanford_cars", root, {"class_names": names})
            self.assertEqual(len(rows), 5)
            self.assertEqual(rows[-1][0]["label"], 4)

    def test_missing_duplicate_and_forbidden_sources_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "Missing"):
                find_sources(root)
            for name in ["a", "b"]:
                folder = root / name
                folder.mkdir()
                (folder / "download_manifest.json").write_text(json.dumps({"dataset": "cifar100", "status": "sources_verified"}))
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                find_sources(root)
        with self.assertRaisesRegex(ValueError, "Forbidden"):
            find_sources("/tmp/dataraw")
        with self.assertRaises(ValueError):
            load_samples("cifar100", "/tmp", {}, limit=6)


if __name__ == "__main__":
    unittest.main()
