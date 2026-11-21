"""Hàm Fakeddit giữ nguyên từ prepare_labeled_cohort.py; không đọc dữ liệu khi import."""
import csv
import hashlib
import html
from pathlib import Path
import unicodedata
from PIL import Image

REMOVED_IMAGE_SHA256 = {"9b5936f4006146e4e1e9025b474c02863c0b5614132ad40db4b925a10e8bfbb9"}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def text_value(value):
    if not isinstance(value, str):
        return ""
    return " ".join(unicodedata.normalize("NFC", html.unescape(value)).split())


def quotas(counts, size):
    """Largest remainder, quota nguyên đúng tổng và không vượt sức chứa."""
    total = sum(counts.values())
    if size < 1 or size > total:
        raise ValueError(f"Yêu cầu {size} mẫu, pool chỉ có {total} mẫu hợp lệ.")
    result = {key: size * n // total for key, n in counts.items()}
    ranked = sorted(counts, key=lambda key: (-(size * counts[key] % total), str(key)))
    for key in ranked[:size - sum(result.values())]:
        result[key] += 1
    return result


def image_ok(path):
    try:
        if Path(path).stat().st_size <= 64 * 1024 and digest(path) in REMOVED_IMAGE_SHA256:
            return False
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.convert("RGB").load()
        return True
    except (OSError, ValueError, Image.DecompressionBombError):
        return False


def write_tsv(path, rows):
    fields = list(rows[0])
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
