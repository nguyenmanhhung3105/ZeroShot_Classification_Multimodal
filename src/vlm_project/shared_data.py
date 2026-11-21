"""Bounded native-source loaders for the four new benchmarks; no model access."""
import hashlib
import io
import json
import tarfile
from itertools import islice
from pathlib import Path

SPLITS = {"cifar100": "train", "flowers102": "validation",
          "stanford_cars": "train", "fgvc_aircraft": "val"}
DATASET_IDS = ["nguyenhung3105/vlm-cifar100-sources-v1",
               "nguyenhung3105/vlm-shared-benchmarks-v1"]


def checked_path(value):
    result = Path(value).resolve()
    if {"dataraw", "data_process"} & set(result.parts):
        raise ValueError("Forbidden data directory")
    return result


def unique_file(root, name):
    candidates = list(checked_path(root).rglob(name))
    candidates = [checked_path(p) for p in candidates if p.is_file()]
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one {name} under {root}, found {len(candidates)}")
    return candidates[0]


def find_sources(input_root):
    roots = {}
    for file in checked_path(input_root).rglob("download_manifest.json"):
        file = checked_path(file)
        # Metadata only; no image or dataset table is read here.
        metadata = json.loads(file.read_text())
        name = metadata.get("dataset")
        if name not in SPLITS:
            continue
        if metadata.get("status") != "sources_verified":
            raise ValueError(f"Unverified source: {file}")
        if name in roots:
            raise ValueError(f"Duplicate mounted source for {name}")
        roots[name] = (file.parent, metadata)
    if set(roots) != set(SPLITS):
        raise ValueError(f"Missing sources: {sorted(set(SPLITS) - set(roots))}")
    return roots


def load_samples(name, root, metadata, limit=5):
    """Return at most five decoded images; use train/validation, never test."""
    from PIL import Image

    if name not in SPLITS or type(limit) is not int or not 1 <= limit <= 5:
        raise ValueError("Data smoke supports only 1..5 samples per dataset")
    root = checked_path(root)
    rows = []

    def append(image_id, label, image, **extra):
        if image.width < 1 or image.height < 1:
            raise ValueError("Invalid image dimensions")
        image.load()
        rows.append(({"id": str(image_id), "label": int(label),
                      "split": SPLITS[name], **extra}, image.convert("RGB")))

    if name == "cifar100":
        names = metadata["class_names"]["fine"]
        mapping = metadata["fine_to_coarse"]
        file = unique_file(root, "train.bin")
        if file.stat().st_size != metadata["splits"]["train"] * 3074:
            raise ValueError("CIFAR train.bin length does not match manifest")
        with file.open("rb") as stream:
            for index in range(limit):
                record = stream.read(3074)
                if len(record) != 3074:
                    raise ValueError("Truncated CIFAR record")
                coarse, fine = record[:2]
                if not 0 <= fine < 100 or not 0 <= coarse < 20 or int(mapping[str(fine)]) != coarse:
                    raise ValueError("CIFAR fine/coarse mismatch")
                # CIFAR binary stores planar R, G, B, not interleaved pixels.
                planes = [Image.frombytes("L", (32, 32), record[2+i*1024:2+(i+1)*1024]) for i in range(3)]
                append(index, fine, Image.merge("RGB", planes), class_name=names[fine], coarse_label=coarse)
    elif name == "flowers102":
        from scipy.io import loadmat

        labels = loadmat(unique_file(root, "imagelabels.mat"))["labels"].ravel()
        # MAT files contain small label/split metadata, not image arrays.
        ids = loadmat(unique_file(root, "setid.mat"))["valid"].ravel()[:limit]
        if len(labels) != 8189:
            raise ValueError("Flowers label count mismatch")
        selected = [int(value) for value in ids]
        expected_images = {f"image_{value:05d}.jpg" for value in selected}
        extracted = {filename: list(root.rglob(filename)) for filename in expected_images}
        if any(len(files) > 1 for files in extracted.values()):
            raise ValueError("Duplicate extracted Flowers images")
        archived_images = {}
        if not all(extracted.values()):
            # Kaggle may preserve .tgz while automatically unpacking .tar.gz.
            # Read only selected JPEG payloads, without writing/extracting sources.
            archive_file = unique_file(root, "102flowers.tgz")
            print(f"Flowers: reading selected images from {archive_file}", flush=True)
            with tarfile.open(archive_file, mode="r|*") as archive:
                for member in archive:
                    name_in_archive = member.name.removeprefix("./")
                    filename = Path(name_in_archive).name
                    if filename not in expected_images or name_in_archive != f"jpg/{filename}":
                        continue
                    if not member.isfile() or filename in archived_images:
                        raise ValueError("Invalid Flowers archive member")
                    with archive.extractfile(member) as stream, Image.open(stream) as image:
                        image.load()
                        archived_images[filename] = image.convert("RGB")
                    if len(archived_images) == len(expected_images):
                        break
            if set(archived_images) != expected_images:
                raise ValueError(f"Missing Flowers archive images: {sorted(expected_images - set(archived_images))}")
        for image_id in ids:
            image_id = int(image_id)
            if not 1 <= image_id <= len(labels) or not 1 <= int(labels[image_id-1]) <= 102:
                raise ValueError("Invalid Flowers ID/label")
            filename = f"image_{image_id:05d}.jpg"
            if archived_images:
                append(image_id, int(labels[image_id-1])-1, archived_images[filename], class_name=None)
            else:
                image_file = checked_path(extracted[filename][0])
                with Image.open(image_file) as image:
                    append(image_id, int(labels[image_id-1])-1, image, class_name=None)
    elif name == "stanford_cars":
        import pyarrow.parquet as pq

        files = sorted(root.glob("train-*.parquet"))
        if not files:
            raise ValueError("Cars train parquet shards missing")
        names = metadata["class_names"]
        for file in files:
            parquet = pq.ParquetFile(checked_path(file))
            info = json.loads(parquet.schema_arrow.metadata[b"huggingface"])
            if info["info"]["features"]["label"]["names"] != names:
                raise ValueError("Cars class order mismatch")
            offset = 0
            for batch in parquet.iter_batches(batch_size=limit, columns=["image", "label"]):
                for index, record in enumerate(batch.to_pylist()):
                    label = record["label"]
                    if type(label) is not int or not 0 <= label < 196:
                        raise ValueError("Invalid Cars label")
                    content = record["image"]["bytes"]
                    if not content:
                        raise ValueError("Cars requires embedded image bytes")
                    with Image.open(io.BytesIO(content)) as image:
                        append(f"{file.name}:{offset+index}", label, image, class_name=names[label])
                    if len(rows) == limit:
                        return rows
                offset += batch.num_rows
    else:
        annotation = unique_file(root, "images_variant_val.txt")
        classes = metadata["class_names"]["variant"]
        with annotation.open() as stream:
            for line in islice(stream, limit):
                image_id, class_name = line.rstrip().split(" ", 1)
                if len(image_id) != 7 or not image_id.isdigit() or class_name not in classes:
                    raise ValueError("Invalid Aircraft annotation")
                image_file = checked_path(annotation.parent / "images" / f"{image_id}.jpg")
                with Image.open(image_file) as image:
                    append(image_id, classes.index(class_name), image, class_name=class_name,
                           preprocessing="raw image; copyright banner retained for data smoke")
    if len(rows) != limit:
        raise ValueError(f"Not enough samples for {name}")
    return rows


def smoke(input_root="/kaggle/input", output="/kaggle/working/shared_data_smoke"):
    """Export only five PNG images per source plus auditable JSON metadata."""
    destination = checked_path(output)
    sources = find_sources(input_root)
    if destination == checked_path(input_root) or checked_path(input_root) in destination.parents:
        raise ValueError("Output must be separate from input")
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / "summary.json"
    marker.unlink(missing_ok=True)
    report = {"status": "running", "scope": "data decoding only, no inference", "datasets": {}}
    for name, (root, metadata) in sources.items():
        folder = destination / name
        folder.mkdir(exist_ok=True)
        records = []
        for index, (record, image) in enumerate(load_samples(name, root, metadata)):
            file = folder / f"sample_{index}.png"
            image.save(file)
            record.update(image=str(file.relative_to(destination)), width=image.width,
                          height=image.height, sha256=hashlib.sha256(file.read_bytes()).hexdigest())
            records.append(record)
        report["datasets"][name] = {"source_root": str(root), "source_splits": metadata["splits"],
                                    "samples": records}
        print(f"SMOKE OK {name}: {len(records)} images, split={SPLITS[name]}", flush=True)
    report["status"] = "data_smoke_passed"
    report["limitations"] = ["No model predictions or quality metrics",
                              "Flowers class names still need a verified mapping",
                              "Only sampled images are decoded; full image integrity not rechecked"]
    marker.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return destination
