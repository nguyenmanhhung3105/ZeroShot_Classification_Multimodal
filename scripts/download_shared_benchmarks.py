"""Download the three shared paper benchmarks on Kaggle CPU, without inference."""
import csv
import json
import shutil
import tarfile
from pathlib import Path
from urllib.request import urlopen

from download_two_stage_datasets import FILES, checksum, download, validate_flowers

CARS_REVISION = "9abf6cf7d6dfa7b95152a0d6e791ea9435b47a40"
CARS_FILES = {
    "test-00000-of-00002.parquet": "ddb15e6a44573203cc1be170be2f05d1e34b22532d37579a6d53cfd05c21485a",
    "test-00001-of-00002.parquet": "22c54046e4beaeb01d4a6daa72a39c5af50feda6850dbf2b0f784cabf865ccaa",
    "train-00000-of-00002.parquet": "b35f21fdeda40973f3b3e1d17bd7a0d2baf8ef249777fc0a0ea7c33c63fa718e",
    "train-00001-of-00002.parquet": "b8fc512aac709672c09f17a1b5ee7ae47585894ca52be753cd4e3af726fdef77",
}
AIRCRAFT = (
    "fgvc-aircraft-2013b.tar.gz",
    "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz",
    # Oxford checksum page and datasets-mila archive checksum respectively.
    ("32eca553f897e747706144f3bb444df0", "d4acdd33327262359767eeaa97a4f732"),
)


def reuse_sources(root, name, input_root="/kaggle/input"):
    """Copy checksum-verified sources from an attached previous notebook Output."""
    inputs = Path(input_root)
    if not inputs.exists():
        return
    if name == "stanford_cars":
        entries = [(file, "sha256", {sha}) for file, sha in CARS_FILES.items()]
    else:
        bank = FILES["flowers102"] if name == "flowers102" else [AIRCRAFT]
        entries = [(file, "md5", {md5} if isinstance(md5, str) else set(md5)) for file, _, md5 in bank]
    candidates = [inputs / "shared_benchmarks"]
    candidates.extend(child / "shared_benchmarks" for child in inputs.iterdir() if child.is_dir())
    for filename, algorithm, accepted in entries:
        target = root / filename
        if target.exists():
            continue
        for base in candidates:
            found = False
            for source in [base / name / filename, base / name / (filename + ".part")]:
                if source.is_file() and checksum(source, algorithm) in accepted:
                    print(f"Reusing verified source: {source}", flush=True)
                    shutil.copyfile(source, target)
                    found = True
                    break
            if found:
                break


def fetch_cars(root):
    import pyarrow.parquet as pq

    sources, counts, class_names = [], {}, None
    for filename, expected in CARS_FILES.items():
        target = root / filename
        url = f"https://huggingface.co/datasets/tanganke/stanford_cars/resolve/{CARS_REVISION}/data/{filename}"
        if not target.exists():
            partial = target.with_suffix(".part")
            print(f"Downloading Cars {filename}", flush=True)
            with urlopen(url, timeout=120) as response, partial.open("wb") as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
            if checksum(partial, "sha256") != expected:
                raise ValueError(f"Cars SHA256 mismatch: {partial}")
            partial.replace(target)
        if checksum(target, "sha256") != expected:
            raise ValueError(f"Cars SHA256 mismatch: {target}")
        parquet = pq.ParquetFile(target)
        if not {"image", "label"} <= set(parquet.schema_arrow.names):
            raise ValueError("Cars parquet lacks image/label columns")
        metadata = json.loads(parquet.schema_arrow.metadata[b"huggingface"])
        names = metadata["info"]["features"]["label"]["names"]
        if len(names) != 196 or len(set(names)) != 196 or (class_names is not None and names != class_names):
            raise ValueError("Cars class order mismatch")
        class_names = names
        split = filename.split("-", 1)[0]
        count = 0
        # Only read labels, in bounded batches, never load the embedded images.
        for batch in parquet.iter_batches(batch_size=256, columns=["label"]):
            labels = batch.column(0).to_pylist()
            if any(type(label) is not int or not 0 <= label < 196 for label in labels):
                raise ValueError("Cars invalid label")
            count += len(labels)
        counts[split] = counts.get(split, 0) + count
        sources.append({"file": filename, "url": url, "sha256": expected, "bytes": target.stat().st_size})
    if counts != {"train": 8144, "test": 8041}:
        raise ValueError(f"Cars incorrect splits: {counts}")
    return {"sources": sources, "splits": counts, "classes": 196, "label_base": 0,
            "class_names": class_names, "mirror": "tanganke/stanford_cars",
            "revision": CARS_REVISION, "original_equivalence": "mirror; not independently compared to original images"}


def validate_aircraft(root):
    prefix = "fgvc-aircraft-2013b/data/"
    counts, seen, taxonomy = {}, set(), {}
    with tarfile.open(root / AIRCRAFT[0]) as archive:
        def lines(name):
            with archive.extractfile(prefix + name) as stream:
                return stream.read().decode().splitlines()

        classes = {level: lines(file) for level, file in [
            ("variant", "variants.txt"), ("family", "families.txt"),
            ("manufacturer", "manufacturers.txt")]}
        if [len(classes[level]) for level in ["variant", "family", "manufacturer"]] != [100, 70, 30]:
            raise ValueError("Aircraft unexpected taxonomy")
        image_paths = {m.name for m in archive.getmembers() if m.isfile() and m.name.endswith(".jpg")}
        for split, expected in [("train", 3334), ("val", 3333), ("test", 3333)]:
            labels = {}
            for level in classes:
                records = [line.split(" ", 1) for line in lines(f"images_{level}_{split}.txt")]
                labels[level] = dict(records)
                if len(records) != expected or len(labels[level]) != expected:
                    raise ValueError("Aircraft duplicate/missing annotation")
                if set(labels[level].values()) - set(classes[level]):
                    raise ValueError("Aircraft unknown class")
            ids = set(labels["variant"])
            if ids != set(labels["family"]) or ids != set(labels["manufacturer"]) or seen & ids:
                raise ValueError("Aircraft misaligned/overlapping splits")
            for image_id in ids:
                if prefix + f"images/{image_id}.jpg" not in image_paths:
                    raise ValueError("Aircraft missing image")
                variant = labels["variant"][image_id]
                parent = (labels["family"][image_id], labels["manufacturer"][image_id])
                if variant in taxonomy and taxonomy[variant] != parent:
                    raise ValueError("Aircraft inconsistent hierarchy")
                taxonomy[variant] = parent
            seen.update(ids)
            counts[split] = len(ids)
        if len(image_paths) != 10000 or len(seen) != 10000:
            raise ValueError("Aircraft incorrect image count")
    return {"splits": counts, "class_names": classes, "variant_to_parents": taxonomy,
            "preprocessing_note": "Keep source images; remove bottom 20-pixel copyright banner during evaluation."}


def main(output="/kaggle/working/shared_benchmarks", datasets=("flowers102", "stanford_cars", "fgvc_aircraft")):
    for name in datasets:
        if name not in {"flowers102", "stanford_cars", "fgvc_aircraft"}:
            raise ValueError(f"Unknown dataset: {name}")
        root = Path(output) / name
        root.mkdir(parents=True, exist_ok=True)
        reuse_sources(root, name)
        marker = root / "download_manifest.json"
        marker.unlink(missing_ok=True)
        if name == "flowers102":
            sources = [download(root, *entry) for entry in FILES[name]]
            details = {"sources": sources, **validate_flowers(root)}
        elif name == "stanford_cars":
            details = fetch_cars(root)
        else:
            source = download(root, *AIRCRAFT)
            details = {"sources": [source], **validate_aircraft(root)}
        manifest = {"dataset": name, "status": "sources_verified", "runner_ready": False, **details}
        marker.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Verified {name}: {details['splits']}", flush=True)
    print("Save successful Output as a PRIVATE Kaggle Dataset: vlm-shared-benchmarks-v1", flush=True)


if __name__ == "__main__":
    main()
