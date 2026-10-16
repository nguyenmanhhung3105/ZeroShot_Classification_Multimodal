"""Chẩn đoán nhánh text Fakeddit 2way từ log đã lưu; không sửa dự đoán.

Ghép nhãn con của log 6way theo ID, text và tên ảnh. Nhãn thật chỉ được dùng
để báo metric; script không chọn prompt, dấu, trọng số hay ngưỡng.
"""

import argparse
import csv
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


CLASS_NAMES = {
    0: "True", 1: "Satire/Parody", 2: "False Connection",
    3: "Imposter Content", 4: "Manipulated Content", 5: "Misleading Content",
}


def _read_tsv(stream):
    return list(csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8"), delimiter="\t"))


def _read_six_way(path):
    path = Path(path)
    with path.open("rb") as stream:
        rows = _read_tsv(stream)
    if not rows or not {"id", "text", "image_path", "y_true"}.issubset(rows[0]):
        raise ValueError("Log 6way thiếu cột bắt buộc")
    by_id = {}
    for row in rows:
        if row["id"] in by_id or int(row["y_true"]) not in CLASS_NAMES:
            raise ValueError("Log 6way có ID trùng hoặc nhãn không hợp lệ")
        by_id[row["id"]] = row
    return by_id


def _auc(labels, scores):
    return float(roc_auc_score(labels, scores)) if len(set(labels)) == 2 else None


def _correlation(first, second):
    if len(first) < 3 or len(set(first)) < 2 or len(set(second)) < 2:
        return None
    result = spearmanr(first, second)
    value = float(result.statistic)
    return value if np.isfinite(value) else None


def analyze(run_dir, six_way_log):
    run_dir = Path(run_dir)
    by_id = _read_six_way(six_way_log)
    archive_path = run_dir / "run_artifacts.zip"
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    reports = {}
    with zipfile.ZipFile(archive_path) as archive:
        members = [name for name in archive.namelist() if name.endswith("/raw_predictions.tsv")]
        if not members:
            raise ValueError("Archive không có raw_predictions.tsv")
        for member in members:
            folder = Path(member).parent
            manifest = json.loads(archive.read(str(folder / "manifest.json")))
            if manifest.get("dataset") != "fakeddit" or manifest.get("task") != "2way":
                raise ValueError("Chỉ nhận log Fakeddit 2way")
            model = manifest.get("model")
            if not model or model in reports:
                raise ValueError("Model thiếu tên hoặc trùng trong archive")
            with archive.open(member) as stream:
                rows = _read_tsv(stream)
            required = {"id", "text", "image_path", "y_true", "score_text_1"}
            if not rows or not required.issubset(rows[0]):
                raise ValueError(f"Log {model} thiếu cột bắt buộc")
            if len(rows) != manifest.get("n_total_samples") or len(rows) != len(by_id):
                raise ValueError(f"Log {model} không cùng số mẫu với log 6way")
            ids = [row["id"] for row in rows]
            if len(set(ids)) != len(ids) or set(ids) != set(by_id):
                raise ValueError(f"Log {model} không cùng ID với log 6way")
            for row in rows:
                match = by_id[row["id"]]
                if (row["text"] != match["text"] or
                        Path(row["image_path"]).name != Path(match["image_path"]).name or
                        int(row["y_true"]) != int(int(match["y_true"]) == 0)):
                    raise ValueError(f"Mẫu {row['id']} không khớp text/ảnh/nhãn giữa 2way và 6way")
            truth = np.array([int(row["y_true"]) for row in rows])
            subclass = np.array([int(by_id[row["id"]]["y_true"]) for row in rows])
            scores = np.array([float(row["score_text_1"]) for row in rows])
            titles = [row["text"] for row in rows]
            lengths = np.array([len(title) for title in titles])
            if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
                raise ValueError(f"Log {model} có score_text_1 không hợp lệ")
            style = {
                "title_characters": lengths,
                "title_words": np.array([len(title.split()) for title in titles]),
                "uppercase_letter_share": np.array([
                    sum(char.isupper() for char in title) /
                    max(1, sum(char.isalpha() for char in title)) for title in titles]),
                "digit_character_share": np.array([
                    sum(char.isdigit() for char in title) / max(1, len(title)) for title in titles]),
                "contains_question_mark": np.array([int("?" in title) for title in titles]),
                "contains_exclamation_mark": np.array([int("!" in title) for title in titles]),
            }
            style_associations = {}
            for feature, values in style.items():
                style_associations[feature] = {
                    "n_nonzero": int(np.count_nonzero(values)),
                    "spearman_all": _correlation(values, scores),
                    "spearman_real": _correlation(values[truth == 1], scores[truth == 1]),
                    "spearman_fake": _correlation(values[truth == 0], scores[truth == 0]),
                }
            by_subclass = {}
            for label in range(1, 6):
                subset = (subclass == 0) | (subclass == label)
                by_subclass[CLASS_NAMES[label]] = {
                    "n_fake": int(np.sum(subclass == label)),
                    "auc_real_vs_subclass": _auc(truth[subset], scores[subset]),
                    "median_title_characters": float(np.median(lengths[subclass == label])),
                }
            reports[model] = {
                "n_samples": len(rows), "n_real": int(np.sum(truth == 1)),
                "n_fake": int(np.sum(truth == 0)),
                "text_auc_real_positive": _auc(truth, scores),
                "spearman_title_characters_vs_text_score": _correlation(lengths, scores),
                "title_style_associations": style_associations,
                "by_fake_subclass": by_subclass,
                "split_declared": manifest.get("split"),
                "split_verified": bool(manifest.get("split_verified", False)),
                "per_prompt_scores_available": False,
                "subreddit_available": False,
            }
    return {
        "source_2way_run": str(run_dir), "source_6way_log": str(six_way_log),
        "n_six_way_rows": len(by_id), "models": reports,
        "independent_dev_verified": False,
        "exploratory_only": True,
        "note": "Cần xác minh dev/test độc lập trước khi dùng để kết luận hoặc chọn phương pháp.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--six-way-log", required=True)
    parser.add_argument("--output", help="File JSON kết quả; mặc định chỉ in ra stdout")
    args = parser.parse_args()
    report = analyze(args.run_dir, args.six_way_log)
    output = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(output, encoding="utf-8")
        print(f"[RESULT] {target}")
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
