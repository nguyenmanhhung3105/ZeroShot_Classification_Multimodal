"""Đưa cohort đã xác thực vào pro_*, lưu nguyên dữ liệu cũ ở pro_*_v1."""
import argparse
from copy import deepcopy
import csv
import json
from pathlib import Path
import re
import sys
import tempfile

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_labeled_cohort import digest, load_config

FILES = {"fakeddit": "fakeddit_sample.tsv", "crisismmd": "pre_crisismmd.tsv", "mmimdb": "mmimdb_sample.tsv"}


def patch_block(text, block, fields, indent):
    lines = text.splitlines(keepends=True)
    start = next(index for index, line in enumerate(lines) if line.rstrip() == " " * indent + block + ":")
    end = next((index for index in range(start + 1, len(lines))
                if lines[index].strip() and not lines[index].lstrip().startswith("#")
                and len(lines[index]) - len(lines[index].lstrip()) <= indent), len(lines))
    remaining = dict(fields)
    for index in range(start + 1, end):
        match = re.match(r"^" + " " * (indent + 2) + r"([a-zA-Z0-9_]+):", lines[index])
        if match and match.group(1) in remaining:
            key = match.group(1)
            value = remaining.pop(key)
            comment = re.search(r"\s+#.*", lines[index])
            encoded = json.dumps(value, ensure_ascii=False)
            lines[index] = " " * (indent + 2) + f"{key}: {encoded}" + (comment.group() if comment else "") + "\n"
    lines[end:end] = [" " * (indent + 2) + f"{key}: {json.dumps(value, ensure_ascii=False)}\n"
                      for key, value in remaining.items()]
    return "".join(lines)


def promote(dataset, source, root=ROOT):
    root = Path(root).resolve()
    source = (root / source).resolve()
    primary = root / "data/processed" / f"pro_{dataset}"
    backup = primary.with_name(primary.name + "_v1")
    if not source.is_relative_to(root / "data/processed") or source in {primary, backup}:
        raise ValueError("Nguồn phải là thư mục cohort mới trong data/processed, khác primary/v1.")
    if source.is_relative_to(primary) or primary.is_relative_to(source):
        raise ValueError("Nguồn và primary không được lồng nhau.")
    if backup.exists():
        raise ValueError(f"Đã có {backup}; không ghi đè bản v1.")
    if not primary.is_dir():
        raise ValueError(f"Thiếu thư mục dữ liệu cũ: {primary}")
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest.get("dataset") != dataset or manifest.get("status") != "ready":
        raise ValueError("Manifest không đúng dataset hoặc chưa ready.")
    if manifest["selected"]["samples"] != 3000 or manifest["requested_samples"] != 3000:
        raise ValueError("Chỉ đưa cohort đã đủ 3.000 mẫu làm dữ liệu chính.")
    if dataset == "crisismmd" and manifest.get("task") != "both":
        raise ValueError("CrisisMMD chính phải chuẩn bị --task both để có nhãn cho cả hai task.")
    if digest(source / "samples.tsv") != manifest["samples_sha256"]:
        raise ValueError("samples.tsv khác checksum manifest.")
    config_path = root / "configs/experiment_config.yaml"
    config_text = config_path.read_text()
    profile = yaml.safe_load((source / "experiment.yaml").read_text())
    relative = primary.relative_to(root).as_posix()
    fields = {"data_path": f"{relative}/{FILES[dataset]}", "image_dir": f"{relative}/images",
              "max_samples": 3000, "cohort_manifest": f"{relative}/manifest.json",
              "source_split": manifest["source_split"], "label_source": manifest["label_source"]}
    if dataset == "crisismmd":
        fields.update(id_col="image_id", label_col_info="label_informative", label_col_human="label_humanitarian")
    elif dataset == "fakeddit":
        fields.update(label_col_2way="2_way_label", label_col_6way="6_way_label",
                      min_text_length=profile["datasets"][dataset]["min_text_length"])
    else:
        fields.update(label_col="genres")
    updated_text = patch_block(config_text, dataset, fields, 2)
    updated_text = patch_block(updated_text, "evaluation", {"split": "per_dataset",
                               "sample_lock_mode": "create_or_verify",
                               "sample_lock_dir": "results/_system/sample_locks/processed_v2"}, 0)
    profile["datasets"][dataset].update(fields)
    manifest = deepcopy(manifest)
    manifest["promotion"] = {"primary": relative, "previous_version": backup.relative_to(root).as_posix(),
                             "previous_tsv_sha256": digest(primary / FILES[dataset])}
    with tempfile.TemporaryDirectory(prefix=".promote-", dir=primary.parent) as folder:
        stage = Path(folder)
        table = stage / FILES[dataset]
        image_prefix = (source / "images").resolve()
        entries = {item["id"]: item for item in manifest["samples"]}
        seen = set()
        with (source / "samples.tsv").open(newline="", encoding="utf-8") as inp, table.open("w", newline="", encoding="utf-8") as out:
            reader = csv.DictReader(inp, delimiter="\t")
            writer = csv.DictWriter(out, fieldnames=reader.fieldnames, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            for row in reader:
                key = row["image_id" if dataset == "crisismmd" else "id"]
                if key not in entries or key in seen:
                    raise ValueError("ID không khớp manifest hoặc bị lặp.")
                seen.add(key)
                image = (root / row["image_path"]).resolve()
                if not image.is_relative_to(image_prefix) or digest(image) != entries[key]["image_sha256"]:
                    raise ValueError(f"Ảnh không khớp manifest: {key}")
                row["image_path"] = f"{relative}/images/{image.relative_to(image_prefix).as_posix()}"
                writer.writerow(row)
        if len(seen) != 3000:
            raise ValueError("Số ID trong TSV khác 3.000.")
        manifest["samples_sha256"] = digest(table)
        profile_file = stage / "experiment.yaml"
        profile_file.write_text(yaml.safe_dump(profile, sort_keys=False, allow_unicode=True))
        manifest["experiment_sha256"] = digest(profile_file)
        (stage / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        updated = stage / "main.yaml"
        updated.write_text(updated_text)
        load_config(updated)
        load_config(profile_file)
        # Preserve source metadata for rollback, and every byte of the old data.
        originals = {name: (source / name).read_bytes() for name in ("samples.tsv", "manifest.json", "experiment.yaml")}
        archived_config = root / "configs/experiment_config_v1.yaml"
        if not archived_config.exists():
            text = config_text
            for name in FILES:
                text = text.replace(f"data/processed/pro_{name}/", f"data/processed/pro_{name}_v1/")
            archived_config.write_text(text)
        primary.rename(backup)
        try:
            source.rename(primary)
            (primary / "samples.tsv").rename(primary / FILES[dataset])
            for name in (FILES[dataset], "manifest.json", "experiment.yaml"):
                (stage / name).replace(primary / name)
            updated.replace(config_path)
        except Exception:
            if primary.exists():
                canonical = primary / FILES[dataset]
                if canonical.exists():
                    canonical.rename(primary / "samples.tsv")
                for name, data in originals.items():
                    (primary / name).write_bytes(data)
                primary.rename(source)
            backup.rename(primary)
            raise
    return {"dataset": dataset, "samples": 3000, "primary": relative,
            "v1": backup.relative_to(root).as_posix(), "cohort_id": manifest["cohort_id"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=list(FILES), required=True)
    parser.add_argument("--source", required=True)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(promote(args.dataset, args.source), ensure_ascii=False))
    except (ValueError, OSError, KeyError) as error:
        print(f"Không chuyển dữ liệu: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
