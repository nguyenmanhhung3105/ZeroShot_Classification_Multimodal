"""Run in a Kaggle CPU notebook; download official archives, no inference."""
import hashlib
import json
import tarfile
import time
from pathlib import Path
from urllib.request import urlopen


FILES = {
    "cifar100": [
        ("cifar-100-binary.tar.gz", "https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz",
         "03b5dce01913d631647c71ecec9e9cb8"),
    ],
    "flowers102": [
        ("102flowers.tgz", "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz",
         "52808999861908f626f3c1f4e79d11fa"),
        ("imagelabels.mat", "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat",
         "e0620be6f572b9609742df49c70aed4d"),
        ("setid.mat", "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat",
         "a5357ecc9cb78c4bef273ce3793fc85c"),
    ],
}


def checksum(file, algorithm):
    value = hashlib.new(algorithm)
    with file.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def download(root, filename, url, expected_md5):
    accepted = {expected_md5} if isinstance(expected_md5, str) else set(expected_md5)
    target = root / filename
    if not target.exists():
        partial = target.with_name(target.name + ".part")
        # A previous job may have downloaded a complete archive before failing
        # against an outdated published checksum.
        if not partial.exists() or checksum(partial, "md5") not in accepted:
            for attempt in range(1, 4):
                try:
                    print(f"Downloading {filename} (attempt {attempt}/3)", flush=True)
                    with urlopen(url, timeout=120) as response, partial.open("wb") as output:
                        length = response.headers.get("Content-Length")
                        for chunk in iter(lambda: response.read(1024 * 1024), b""):
                            output.write(chunk)
                    actual = checksum(partial, "md5")
                    size = partial.stat().st_size
                    print(f"Downloaded {filename}: bytes={size}, md5={actual}, Content-Length={length}", flush=True)
                    if length is not None and size != int(length):
                        raise ValueError(f"Incomplete download: {size} != {length}")
                    if actual not in accepted:
                        raise ValueError(f"Checksum mismatch: got {actual}; expected one of {sorted(accepted)}")
                    break
                except (OSError, ValueError) as error:
                    print(f"Download failed: {error}", flush=True)
                    if attempt == 3:
                        raise
                    time.sleep(2)
        partial.replace(target)
    actual_md5 = checksum(target, "md5")
    if actual_md5 not in accepted:
        raise ValueError(f"Checksum mismatch: {target}; use a fresh output directory")
    return {"file": filename, "url": url, "md5": actual_md5,
            "sha256": checksum(target, "sha256"), "bytes": target.stat().st_size}


def validate_cifar(root):
    mapping = {}
    counts = {}
    with tarfile.open(root / "cifar-100-binary.tar.gz") as archive:
        for split, expected in [("train", 50000), ("test", 10000)]:
            member = archive.getmember(f"cifar-100-binary/{split}.bin")
            if member.size != expected * 3074:
                raise ValueError(f"CIFAR invalid size: {split}")
            count = 0
            with archive.extractfile(member) as stream:
                while row := stream.read(3074):
                    if len(row) != 3074 or row[0] >= 20 or row[1] >= 100:
                        raise ValueError("CIFAR invalid binary record")
                    coarse, fine = row[:2]
                    if fine in mapping and mapping[fine] != coarse:
                        raise ValueError("CIFAR inconsistent hierarchy")
                    mapping[fine] = coarse
                    count += 1
            if count != expected:
                raise ValueError("CIFAR incorrect count")
            counts[split] = count
        names = {}
        for level in ["coarse", "fine"]:
            with archive.extractfile(f"cifar-100-binary/{level}_label_names.txt") as stream:
                names[level] = stream.read().decode().splitlines()
    if len(mapping) != 100 or any(list(mapping.values()).count(c) != 5 for c in range(20)):
        raise ValueError("CIFAR incomplete taxonomy")
    return {"splits": counts, "label_base": 0, "class_names": names,
            "fine_to_coarse": mapping, "official_validation_split": False}


def validate_flowers(root):
    import numpy as np
    from scipy.io import loadmat

    labels = loadmat(root / "imagelabels.mat")["labels"].ravel()
    split_data = loadmat(root / "setid.mat")
    if len(labels) != 8189 or set(labels.tolist()) != set(range(1, 103)):
        raise ValueError("Flowers invalid labels")
    counts, seen = {}, set()
    for name, key, expected in [("train", "trnid", 1020),
                                 ("validation", "valid", 1020),
                                 ("test", "tstid", 6149)]:
        ids = split_data[key].ravel().astype(int).tolist()
        if len(ids) != expected or len(set(ids)) != expected or seen.intersection(ids):
            raise ValueError("Flowers invalid or overlapping splits")
        seen.update(ids)
        counts[name] = len(ids)
    if seen != set(range(1, 8190)):
        raise ValueError("Flowers incomplete splits")
    with tarfile.open(root / "102flowers.tgz") as archive:
        images = {m.name for m in archive.getmembers() if m.isfile() and m.name.endswith(".jpg")}
    if images != {f"jpg/image_{i:05d}.jpg" for i in range(1, 8190)}:
        raise ValueError("Flowers incomplete image archive")
    return {"splits": counts, "classes": int(np.unique(labels).size),
            "label_base": 1, "image_id_base": 1, "taxonomy": "flat",
            "class_names_included": False}


def main(output="/kaggle/working/two_stage_sources", datasets=("cifar100", "flowers102")):
    for name in datasets:
        root = Path(output) / name
        root.mkdir(parents=True, exist_ok=True)
        # Prevent a failed rerun from leaving an old ready marker.
        marker = root / "download_manifest.json"
        if marker.exists():
            marker.unlink()
        sources = [download(root, *entry) for entry in FILES[name]]
        details = validate_cifar(root) if name == "cifar100" else validate_flowers(root)
        manifest = {"dataset": name, "status": "sources_verified", "sources": sources,
                    **details, "runner_ready": False}
        marker.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Verified {name}: {details['splits']}", flush=True)
    print(f"Output: {output}. Save notebook version to persist these files.")


if __name__ == "__main__":
    main()
