"""Thử cờ needs_review trên log single-label đã lưu; không chạy lại model.

Cờ chỉ dùng điểm hai nhánh, không dùng nhãn thật. Nhãn thật chỉ phục vụ báo cáo
hậu nghiệm; báo cáo này không phải kiểm chứng trên một tập test độc lập.
"""

import argparse
from collections import Counter
import csv
import io
import json
import math
from pathlib import Path
import shutil
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[2]
POLICY = "branch_disagreement_v1"
FLAG_FIELDS = ("id", "y_true", "y_pred", "correct", "image_pred", "text_pred",
               "margin", "needs_review", "review_reason")


def _prediction_member(archive, model=None):
    members = [name for name in archive.namelist() if name.endswith("/raw_predictions.tsv")]
    if model:
        members = [name for name in members if Path(name).parent.name.startswith(model + "__")]
    if len(members) != 1:
        raise ValueError("Cần đúng một raw_predictions.tsv; dùng --model nếu archive có nhiều model")
    return members[0]


def _labels(columns):
    labels = [name.removeprefix("score_") for name in columns
              if name.startswith("score_") and not name.startswith(("score_image_", "score_text_"))]
    if len(labels) < 2 or any(f"score_{branch}_{label}" not in columns
                              for branch in ("image", "text") for label in labels):
        raise ValueError("Log thiếu điểm ảnh/text của bài toán single-label")
    return labels


def _branch_pred(row, labels, branch):
    values = [row[f"score_{branch}_{label}"] for label in labels]
    if any(value == "" for value in values):
        return None
    scores = [float(value) for value in values]
    if not all(math.isfinite(value) for value in scores):
        raise ValueError("Log chứa điểm nhánh không hữu hạn")
    return labels[max(range(len(labels)), key=scores.__getitem__)]


def analyze(run_dir, output_dir=None, model=None):
    run_dir = Path(run_dir).resolve()
    archive_path = run_dir / "run_artifacts.zip"
    receipt_path = run_dir / "run_receipt.json"
    if not archive_path.is_file() or not receipt_path.is_file():
        raise FileNotFoundError("Lượt chạy cần run_artifacts.zip và run_receipt.json")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("run_id") != run_dir.name:
        raise ValueError("run_receipt.json không khớp thư mục lượt chạy")
    target = (Path(output_dir).resolve() if output_dir else
              ROOT / "results/analysis/uncertainty" / f"{run_dir.name}_{POLICY}")
    if target.exists():
        raise FileExistsError(f"Kết quả phân tích đã tồn tại: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".review_", dir=target.parent))
    try:
        counts = Counter()
        per_class = Counter()
        with zipfile.ZipFile(archive_path) as archive:
            member = _prediction_member(archive, model)
            manifest_name = str(Path(member).parent / "manifest.json")
            manifest = json.loads(archive.read(manifest_name))
            with archive.open(member) as source, (temporary / "review_flags.tsv").open(
                    "w", newline="", encoding="utf-8") as destination:
                reader = csv.DictReader(io.TextIOWrapper(source, encoding="utf-8"), delimiter="\t")
                labels = _labels(reader.fieldnames or [])
                writer = csv.DictWriter(destination, fieldnames=FLAG_FIELDS, delimiter="\t",
                                        lineterminator="\n")
                writer.writeheader()
                for row in reader:
                    truth, predicted = row["y_true"], row["y_pred"]
                    if truth not in labels or predicted not in labels:
                        raise ValueError("Nhãn trong log không khớp cột điểm")
                    image_pred = _branch_pred(row, labels, "image")
                    text_pred = _branch_pred(row, labels, "text")
                    missing = image_pred is None or text_pred is None
                    review = missing or image_pred != text_pred
                    reason = "missing_branch" if missing else "branch_disagreement" if review else ""
                    correct = truth == predicted
                    group = "review" if review else "automatic"
                    counts["total"] += 1
                    counts[group] += 1
                    counts[f"{group}_errors"] += not correct
                    per_class[(truth, group, "total")] += 1
                    per_class[(truth, group, "errors")] += not correct
                    writer.writerow(dict(id=row["id"], y_true=truth, y_pred=predicted,
                                         correct=str(correct).lower(), image_pred=image_pred or "",
                                         text_pred=text_pred or "", margin=row.get("margin", ""),
                                         needs_review=str(review).lower(), review_reason=reason))
        if counts["total"] != manifest.get("n_total_samples"):
            raise ValueError("Số dòng dự đoán không khớp manifest")
        n = counts["total"]
        if not n:
            raise ValueError("Log dự đoán rỗng")
        errors = counts["automatic_errors"] + counts["review_errors"]
        summary = {
            "source_run_id": run_dir.name,
            "model": manifest.get("model", model or ""),
            "dataset": manifest.get("dataset", ""),
            "task": manifest.get("task", ""),
            "policy": POLICY,
            "n_samples": n,
            "n_automatic": counts["automatic"],
            "n_review": counts["review"],
            "coverage": counts["automatic"] / n,
            "n_errors_total": errors,
            "n_errors_automatic": counts["automatic_errors"],
            "n_errors_review": counts["review_errors"],
            "automatic_error_rate": (counts["automatic_errors"] / counts["automatic"]
                                     if counts["automatic"] else None),
            "review_error_rate": (counts["review_errors"] / counts["review"]
                                  if counts["review"] else None),
            "error_capture_rate": counts["review_errors"] / errors if errors else None,
        }
        with (temporary / "summary.json").open("w", encoding="utf-8") as stream:
            json.dump(summary, stream, ensure_ascii=False, indent=2)
        with (temporary / "by_class.tsv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=("y_true", "group", "n_samples", "n_errors",
                                                        "error_rate"), delimiter="\t", lineterminator="\n")
            writer.writeheader()
            for label in labels:
                for group in ("automatic", "review"):
                    size = per_class[(label, group, "total")]
                    wrong = per_class[(label, group, "errors")]
                    writer.writerow(dict(y_true=label, group=group, n_samples=size,
                                         n_errors=wrong, error_rate=wrong / size if size else ""))
        (temporary / "analysis_manifest.json").write_text(json.dumps({
            "source_run_id": run_dir.name,
            "source_job_id": receipt.get("job_id"),
            "source_code_sha256": receipt.get("code_sha256"),
            "source_data_sha256": receipt.get("data_sha256"),
            "sample_set_id": manifest.get("sample_set_id"),
            "source_archive_member": member,
            "policy": POLICY,
            "uses_true_labels_for_flag": False,
            "split_verified": manifest.get("split_verified", False),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.rename(target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return target, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Thư mục kết quả Kaggle đã tải về")
    parser.add_argument("--model", help="Chọn model nếu archive có nhiều kết quả")
    parser.add_argument("--output-dir", help="Thư mục kết quả phân tích; mặc định dưới results/analysis/")
    args = parser.parse_args(argv)
    target, summary = analyze(args.run_dir, args.output_dir, args.model)
    print(f"[RESULT] {target}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
