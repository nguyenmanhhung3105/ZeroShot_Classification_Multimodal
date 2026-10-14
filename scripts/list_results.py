"""Tra cứu kết quả theo dataset/model/phương pháp thay vì nhớ run ID."""

import argparse
import csv
from datetime import datetime
import os
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
FIELDS = ("saved_at", "dataset", "task", "job", "kind", "model", "method",
          "output", "seed", "n_samples", "accuracy", "macro_f1",
          "balanced_accuracy", "run_id", "summary_path")


def result_rows(root=ROOT):
    """Đọc các summary đã tải về; mỗi model/phương pháp là một dòng."""
    results = Path(root) / "results"
    rows = []
    for family in ("runs", "experiments"):
        for summary in (results / family).glob("*/*/summary.tsv"):
            with summary.open(newline="", encoding="utf-8") as stream:
                for source in csv.DictReader(stream, delimiter="\t"):
                    row = {key: "" for key in FIELDS}
                    row.update({key: source.get(key, "") or "" for key in
                                ("dataset", "task", "model", "n_samples", "accuracy",
                                 "macro_f1", "balanced_accuracy")})
                    row.update(saved_at=datetime.fromtimestamp(summary.stat().st_mtime).astimezone().isoformat(timespec="seconds"),
                               job=summary.parent.parent.name,
                               kind="standard" if family == "runs" else "experiment",
                               method=source.get("image_method") or source.get("method") or "standard",
                               output=source.get("output_variant") or source.get("prompt_variant") or "",
                               seed=source.get("research_seed") or source.get("seed") or "",
                               run_id=summary.parent.name,
                               summary_path=summary.relative_to(root).as_posix())
                    rows.append(row)
    return sorted(rows, key=lambda row: (row["saved_at"], row["run_id"], row["model"]), reverse=True)


def write_index(root=ROOT):
    """Cập nhật bảng tra cứu; không đổi tên hay di chuyển kết quả gốc."""
    results = Path(root) / "results"
    results.mkdir(exist_ok=True)
    target = results / "INDEX.tsv"
    with tempfile.NamedTemporaryFile("w", newline="", encoding="utf-8", dir=results,
                                     prefix=".INDEX_", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            writer = csv.DictWriter(stream, fieldnames=FIELDS, delimiter="\t")
            writer.writeheader()
            writer.writerows(result_rows(root))
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, target)
    return target


def matching_rows(rows, args):
    for row in rows:
        if any(getattr(args, key) and getattr(args, key).casefold() not in row[key].casefold()
               for key in ("dataset", "task", "job", "model", "method")):
            continue
        if args.samples is not None and row["n_samples"] != str(args.samples):
            continue
        yield row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("dataset", "task", "job", "model", "method"):
        parser.add_argument("--" + key, help=f"Lọc theo {key}")
    parser.add_argument("--samples", type=int, help="Số mẫu chính xác")
    parser.add_argument("--limit", type=int, default=30, help="Số dòng hiển thị; 0 là tất cả")
    parser.add_argument("--write-index", action="store_true", help="Ghi lại results/INDEX.tsv")
    args = parser.parse_args(argv)
    if (args.samples is not None and args.samples < 1) or args.limit < 0:
        parser.error("--samples phải dương và --limit không âm")
    if args.write_index:
        print(f"[INDEX] {write_index().relative_to(ROOT)}")
    rows = list(matching_rows(result_rows(), args))
    if args.limit:
        rows = rows[:args.limit]
    columns = ("saved_at", "dataset", "task", "model", "method", "n_samples",
               "macro_f1", "summary_path")
    writer = csv.DictWriter(sys.stdout, fieldnames=columns, delimiter="\t",
                            extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)


if __name__ == "__main__":
    main()
