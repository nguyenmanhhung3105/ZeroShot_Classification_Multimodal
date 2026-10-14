"""Đánh giá hậu nghiệm: dùng model phụ chỉ cho mẫu được gắn needs_review.

Không chạy inference. Ba file dự đoán/cờ phải là cùng mẫu và cùng thứ tự;
nhãn thật chỉ được dùng để tính chỉ số sau khi quyết định chuyển model.
"""

import argparse
from collections import Counter
import csv
from contextlib import ExitStack
import io
from itertools import zip_longest
import json
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.uncertainty.analyze_review import ROOT, _prediction_member


def _macro_f1(truth_pred, labels):
    values = []
    for label in labels:
        tp = truth_pred[(label, label)]
        fp = sum(count for (truth, predicted), count in truth_pred.items()
                 if predicted == label and truth != label)
        fn = sum(count for (truth, predicted), count in truth_pred.items()
                 if truth == label and predicted != label)
        denominator = 2 * tp + fp + fn
        values.append(2 * tp / denominator if denominator else 0.0)
    return sum(values) / len(values)


def compare(primary_dir, flags_dir, candidate_dir, candidate_model, output_dir=None,
            primary_model=None):
    primary_dir, flags_dir, candidate_dir = map(
        lambda path: Path(path).resolve(), (primary_dir, flags_dir, candidate_dir))
    flag_manifest = json.loads((flags_dir / "analysis_manifest.json").read_text(encoding="utf-8"))
    if flag_manifest.get("source_run_id") != primary_dir.name:
        raise ValueError("Cờ needs_review không thuộc lượt chạy chính")
    if flag_manifest.get("uses_true_labels_for_flag") is not False:
        raise ValueError("Quy tắc gắn cờ phải độc lập với nhãn thật")
    if not candidate_model:
        raise ValueError("Cần chỉ rõ --candidate-model")
    target = (Path(output_dir).resolve() if output_dir else ROOT / "results/analysis/fallback" /
              f"{primary_dir.name}_{candidate_dir.name}_{candidate_model}")
    if target.exists():
        raise FileExistsError(f"Kết quả đã tồn tại: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".fallback_", dir=target.parent))
    try:
        counts = Counter()
        confusion = {key: Counter() for key in ("primary", "candidate", "gated")}
        per_class = Counter()
        labels = set()
        with ExitStack() as stack:
            primary_zip = stack.enter_context(zipfile.ZipFile(primary_dir / "run_artifacts.zip"))
            candidate_zip = stack.enter_context(zipfile.ZipFile(candidate_dir / "run_artifacts.zip"))
            primary_member = _prediction_member(primary_zip, primary_model)
            candidate_member = _prediction_member(candidate_zip, candidate_model)
            primary_info = json.loads(primary_zip.read(str(Path(primary_member).parent / "manifest.json")))
            candidate_info = json.loads(candidate_zip.read(str(Path(candidate_member).parent / "manifest.json")))
            if primary_info.get("dataset") != candidate_info.get("dataset") or \
                    primary_info.get("task") != candidate_info.get("task"):
                raise ValueError("Hai lượt chạy không cùng dataset/task")
            primary_source = stack.enter_context(primary_zip.open(primary_member))
            candidate_source = stack.enter_context(candidate_zip.open(candidate_member))
            flag_source = stack.enter_context((flags_dir / "review_flags.tsv").open(
                newline="", encoding="utf-8"))
            primary_rows = csv.DictReader(io.TextIOWrapper(primary_source, encoding="utf-8"), delimiter="\t")
            candidate_rows = csv.DictReader(io.TextIOWrapper(candidate_source, encoding="utf-8"), delimiter="\t")
            flag_rows = csv.DictReader(flag_source, delimiter="\t")
            change_file = stack.enter_context((temporary / "switches.tsv").open(
                "w", newline="", encoding="utf-8"))
            writer = csv.DictWriter(change_file,
                                    fieldnames=("id", "y_true", "primary_pred", "candidate_pred",
                                                "gated_pred", "outcome"), delimiter="\t",
                                    lineterminator="\n")
            writer.writeheader()
            for index, (primary, candidate, flag) in enumerate(
                    zip_longest(primary_rows, candidate_rows, flag_rows), start=1):
                if primary is None or candidate is None or flag is None:
                    raise ValueError("Ba nguồn khác số mẫu")
                if any(primary[field] != candidate[field] for field in ("id", "y_true", "text")) \
                        or Path(primary["image_path"]).name != Path(candidate["image_path"]).name:
                    raise ValueError(f"Hai lượt chạy khác mẫu/thứ tự tại dòng {index}")
                if any(primary[field] != flag[field] for field in ("id", "y_true", "y_pred")):
                    raise ValueError(f"Cờ không khớp dự đoán chính tại dòng {index}")
                if flag["needs_review"] not in ("true", "false"):
                    raise ValueError(f"Cờ không hợp lệ tại dòng {index}")
                review = flag["needs_review"] == "true"
                truth, original, alternate = primary["y_true"], primary["y_pred"], candidate["y_pred"]
                selected = alternate if review else original
                labels.add(truth)
                counts["n_samples"] += 1
                counts["n_review"] += review
                counts["n_primary_correct"] += original == truth
                counts["n_candidate_correct"] += alternate == truth
                counts["n_gated_correct"] += selected == truth
                for name, prediction in (("primary", original), ("candidate", alternate), ("gated", selected)):
                    confusion[name][(truth, prediction)] += 1
                if review:
                    counts["n_review_changed"] += original != alternate
                    fixed = original != truth and alternate == truth
                    broken = original == truth and alternate != truth
                    counts["n_fixed"] += fixed
                    counts["n_broken"] += broken
                    per_class[(truth, "fixed")] += fixed
                    per_class[(truth, "broken")] += broken
                    if original != alternate:
                        outcome = "fixed" if fixed else "broken" if broken else "wrong_to_wrong"
                        writer.writerow(dict(id=primary["id"], y_true=truth, primary_pred=original,
                                             candidate_pred=alternate, gated_pred=selected,
                                             outcome=outcome))
        if not counts["n_samples"]:
            raise ValueError("Không có mẫu nào")
        if counts["n_samples"] != primary_info.get("n_total_samples") or \
                counts["n_samples"] != candidate_info.get("n_total_samples"):
            raise ValueError("Số mẫu không khớp manifest")
        n = counts["n_samples"]
        summary = {
            "primary_run_id": primary_dir.name,
            "primary_model": primary_info.get("model"),
            "candidate_run_id": candidate_dir.name,
            "candidate_model": candidate_info.get("model"),
            "dataset": primary_info.get("dataset"),
            "task": primary_info.get("task"),
            "review_policy": flag_manifest.get("policy"),
            "n_samples": n,
            "n_review": counts["n_review"],
            "n_review_changed": counts["n_review_changed"],
            "n_fixed": counts["n_fixed"],
            "n_broken": counts["n_broken"],
            "net_correct_change": counts["n_fixed"] - counts["n_broken"],
            "primary_accuracy": counts["n_primary_correct"] / n,
            "candidate_accuracy": counts["n_candidate_correct"] / n,
            "gated_accuracy": counts["n_gated_correct"] / n,
            "primary_macro_f1": _macro_f1(confusion["primary"], sorted(labels)),
            "candidate_macro_f1": _macro_f1(confusion["candidate"], sorted(labels)),
            "gated_macro_f1": _macro_f1(confusion["gated"], sorted(labels)),
            "per_class": {label: {"fixed": per_class[(label, "fixed")],
                                  "broken": per_class[(label, "broken")]}
                          for label in sorted(labels)},
        }
        (temporary / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.rename(target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return target, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-dir", required=True)
    parser.add_argument("--flags-dir", required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--candidate-model", required=True)
    parser.add_argument("--primary-model")
    parser.add_argument("--output-dir")
    args = parser.parse_args(argv)
    target, summary = compare(args.primary_dir, args.flags_dir, args.candidate_dir,
                              args.candidate_model, args.output_dir, args.primary_model)
    print(f"[RESULT] {target}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
