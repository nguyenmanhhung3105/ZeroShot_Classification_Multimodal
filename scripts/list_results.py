"""In kết quả hiện có; thư mục theo phương pháp/dataset/task/số mẫu/model."""

import argparse
import csv
from datetime import datetime
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
FIELDS = ("saved_at", "method", "dataset", "task", "model", "variant", "n_samples",
          "accuracy", "macro_f1", "run_id", "summary_path")
MODEL_NAMES = {"openai_clip_vit_b_32", "metaclip2_vith14", "dfn5b_vith14", "siglip2_so400m"}


def result_rows(root=ROOT):
    """Đọc summary, không đọc ảnh/dữ liệu gốc hoặc tạo file chỉ mục."""
    rows = []
    for summary in (Path(root) / "results").glob("*/*/*/samples_*/*/*/summary.tsv"):
        method, dataset, task, sample_folder, model_folder, run_id = summary.relative_to(
            Path(root) / "results").parts[:6]
        with summary.open(newline="", encoding="utf-8-sig") as stream:
            for source in csv.DictReader(stream, delimiter="\t"):
                model = source.get("model") or ""
                variant = source.get("image_method") or source.get("method") or source.get("prompt_variant") or ""
                source_task = source.get("task") or task
                if (source.get("dataset") or dataset) == "mmimdb" and source_task == "multi_label":
                    source_task = "genres"
                if not model and variant in MODEL_NAMES:
                    model, variant = variant, "base_model"
                elif not model and variant:
                    model = ("ensemble" if method == "ensemble" else
                             "multi_model" if model_folder == "multi_model" else "")
                if not model and model_folder != "multi_model":
                    model = model_folder
                rows.append({
                    "saved_at": datetime.fromtimestamp(summary.stat().st_mtime).astimezone().isoformat(timespec="seconds"),
                    "method": method, "dataset": source.get("dataset") or dataset,
                    "task": source_task, "model": model, "variant": variant,
                    "n_samples": source.get("n_samples") or sample_folder.removeprefix("samples_"),
                    "accuracy": source.get("accuracy") or "", "macro_f1": source.get("macro_f1") or "",
                    "run_id": run_id, "summary_path": summary.relative_to(root).as_posix()})
    return sorted(rows, key=lambda row: (row["saved_at"], row["run_id"], row["model"]), reverse=True)


def matching_rows(rows, args):
    for row in rows:
        if any(getattr(args, key, None) and getattr(args, key).casefold() not in row[key].casefold()
               for key in ("method", "dataset", "task", "model", "variant")):
            continue
        if getattr(args, "samples", None) is not None and row["n_samples"] != str(args.samples):
            continue
        yield row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("method", "dataset", "task", "model", "variant"):
        parser.add_argument("--" + key, help=f"Lọc theo {key}")
    parser.add_argument("--samples", type=int, help="Số mẫu chính xác")
    parser.add_argument("--limit", type=int, default=30, help="Số dòng hiển thị; 0 là tất cả")
    args = parser.parse_args(argv)
    if (args.samples is not None and args.samples < 1) or args.limit < 0:
        parser.error("--samples phải dương và --limit không âm")
    rows = list(matching_rows(result_rows(), args))
    if args.limit:
        rows = rows[:args.limit]
    writer = csv.DictWriter(sys.stdout, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)


if __name__ == "__main__":
    main()
