"""Read-only label inventory. No model, inference, image decoding or data edits.

Run from the repository root:
    venv/bin/python src/run/test_label_distribution.py
Use --max-rows 5 for a bounded preview (first five records per source file).
Full reports describe processed FILES, not the filtered inference cohort.
"""
import argparse
import ast
from collections import Counter
import csv
from datetime import datetime
import html
import itertools
import json
from pathlib import Path
import sys
import tempfile

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prompts import fakeddit_prompts, crisismmd_prompts, mmimdb_prompts


def canon(value):
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


def task_specs(dataset, config):
    if dataset == "fakeddit":
        return [(task, config.get(f"label_col_{task}", f"{task[0]}_way_label"),
                 fakeddit_prompts.get_label_names(task), False) for task in ("2way", "6way")]
    if dataset == "crisismmd":
        specs = []
        for task, key, default in (("informativeness", "label_col_info", "label_informative"),
                                   ("humanitarian", "label_col_human", "label_humanitarian")):
            legacy = config.get("label_col")
            fallback = default if legacy in (None, "label", "label_informative", "label_humanitarian") else legacy
            specs.append((task, config.get(key, fallback), crisismmd_prompts.get_label_names(task), False))
        return specs
    return [("genres", config.get("label_col", "genres"), mmimdb_prompts.get_label_names(), True)]


def read_rows(path, limit=None):
    """Return schema and streaming records, bounded before any data loading."""
    path = Path(path)
    if any(p.lower() in ("raw", "dataraw") for p in path.resolve().parts):
        raise ValueError("Refusing raw/dataraw paths; use processed data only")
    if limit is not None and limit < 1:
        raise ValueError("max_rows must be positive")
    if path.suffix.lower() in (".tsv", ".csv"):
        stream = path.open(encoding="utf-8-sig", newline="")
        try:
            reader = csv.DictReader(stream, delimiter="\t" if path.suffix.lower() == ".tsv" else ",")
            yield list(reader.fieldnames or []), itertools.islice(reader, limit)
        finally:
            stream.close()
    elif path.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq
        with pq.ParquetFile(path) as source:
            def records():
                remaining = limit
                for batch in source.iter_batches(batch_size=min(limit or 1024, 1024)):
                    rows = batch.to_pylist()
                    for row in rows[:remaining] if remaining is not None else rows:
                        yield row
                    if remaining is not None:
                        remaining -= len(rows)
                        if remaining <= 0:
                            break
            yield source.schema.names, records()
    else:
        raise ValueError("Only TSV, CSV and Parquet are supported")


def parse_genres(value):
    if isinstance(value, (list, tuple, set)):
        return [str(x).strip() for x in value]
    if value is None or not str(value).strip():
        return []
    text = str(value).strip()
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
        except (ValueError, SyntaxError):
            continue
        if isinstance(parsed, (list, tuple, set)):
            return [str(x).strip() for x in parsed]
    return [x.strip() for x in text.split("|" if "|" in text else ",")]


def summarize(config, max_rows=None):
    counts, overview, issues = [], [], []
    for dataset in ("fakeddit", "crisismmd", "mmimdb"):
        settings = config.get("datasets", {}).get(dataset, {})
        path = settings.get("data_path")
        specs = task_specs(dataset, settings)
        states = []
        for task, column, names, multi in specs:
            mapping = {canon(k): k for k in names}
            mapping.update({canon(v): k for k, v in names.items()})
            states.append(dict(task=task, column=column, names=names, multi=multi, mapping=mapping,
                               count=Counter(), unknown=Counter(), missing=0, invalid=0, valid=0, duplicates=0))
        total, columns, error = 0, [], None
        try:
            if not path:
                raise ValueError("data_path not configured")
            for columns, rows in read_rows(path, max_rows):
                for row in rows:
                    total += 1
                    for state in states:
                        if state["column"] not in columns:
                            continue
                        value = row.get(state["column"])
                        values = parse_genres(value) if state["multi"] else [value]
                        if not values or all(v is None or not str(v).strip() for v in values):
                            state["missing"] += 1
                            continue
                        labels, unknown = set(), set()
                        for v in values:
                            if not state["multi"] and isinstance(v, float) and v.is_integer():
                                v = int(v)
                            # Genres are case-sensitive, matching the MM-IMDb evaluator.
                            label = v if state["multi"] and v in state["names"] else (
                                state["mapping"].get(canon(v)) if not state["multi"] else None)
                            if label is None:
                                unknown.add(str(v))
                            else:
                                labels.add(label)
                        state["duplicates"] += int(state["multi"] and len(values) != len(set(values)))
                        state["count"].update(labels)
                        state["unknown"].update(unknown)
                        state["invalid"] += bool(unknown)
                        state["valid"] += not bool(unknown)
        except (OSError, ValueError, ImportError, csv.Error) as exc:
            error = str(exc)
        for state in states:
            problem = error or (f"Missing column: {state['column']}" if state["column"] not in columns else "")
            status = "error" if problem else "ok"
            common = dict(dataset=dataset, task=state["task"], status=status)
            overview.append({**common, "data_path": path, "label_column": state["column"],
                             "scope": "first_rows_preview" if max_rows else "full_processed_file",
                             "n_rows": total, "n_valid_rows": state["valid"],
                             "n_missing_labels": state["missing"], "n_unknown_label_rows": state["invalid"],
                             "n_duplicate_label_rows": state["duplicates"],
                             "known_label_assignments": sum(state["count"].values()),
                             "error": problem})
            for label, name in state["names"].items():
                n = state["count"][label]
                counts.append({**common, "label_id": label, "label_name": name,
                               "n_samples": n if not problem else None,
                               "percent_of_rows": round(100 * n / total, 4) if total and not problem else None})
            for value, n in state["unknown"].items():
                issues.append({"dataset": dataset, "task": state["task"], "unknown_label": value, "n_rows": n})
    return counts, overview, issues


def save_report(counts, overview, issues, output_root):
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix=datetime.now().strftime("labels_%Y%m%d_%H%M%S_"), dir=root))
    tables = [("class_counts.tsv", counts, list(counts[0])),
              ("dataset_overview.tsv", overview, list(overview[0])),
              ("unknown_labels.tsv", issues, ["dataset", "task", "unknown_label", "n_rows"])]
    for filename, rows, fields in tables:
        with (folder / filename).open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
    sections = []
    esc = lambda v: html.escape(str(v))
    for item in overview:
        rows = [r for r in counts if (r["dataset"], r["task"]) == (item["dataset"], item["task"])]
        body = "".join(f"<tr><td>{esc(r['label_id'])}</td><td>{esc(r['label_name'])}</td>"
                       f"<td>{r['n_samples'] if r['n_samples'] is not None else 'N/A'}</td>"
                       f"<td>{r['percent_of_rows'] if r['percent_of_rows'] is not None else 'N/A'}%</td>"
                       f"<td><progress max='100' value='{r['percent_of_rows'] or 0}'></progress></td></tr>" for r in rows)
        sections.append(f"<section><h2>{esc(item['dataset'])} — {esc(item['task'])}</h2>"
                        f"<p>Nguồn: {esc(item['data_path'])}<br>Cột: {esc(item['label_column'])} | Phạm vi: {esc(item['scope'])}</p>"
                        f"<p>Số dòng: {item['n_rows']} | Thiếu nhãn: {item['n_missing_labels']} | "
                        f"Dòng có nhãn lạ: {item['n_unknown_label_rows']} | Nhãn lặp trong cùng mẫu: {item['n_duplicate_label_rows']}</p>"
                        f"<p class='error'>{esc(item['error'])}</p><table><tr><th>ID</th><th>Lớp</th><th>Số mẫu</th>"
                        f"<th>% số dòng</th><th>Phân bố</th></tr>{body}</table></section>")
    document = """<!doctype html><html lang="vi"><meta charset="utf-8"><title>Thống kê nhãn</title>
<style>body{font:16px system-ui;max-width:1100px;margin:30px auto;padding:0 20px;background:#f4f6fa;color:#172033}
section{background:white;padding:22px;margin:20px 0;border-radius:12px}table{border-collapse:collapse;width:100%}
td,th{text-align:left;padding:9px;border-bottom:1px solid #ddd}progress{width:100%;accent-color:#2864ce}
.error{color:#b42318}th{background:#edf2fa}</style><h1>Phân bố nhãn của 3 dataset</h1>
<p>Thống kê file processed trước lọc text/ảnh; không phải cohort inference. Không kiểm tra ảnh,
không tự cân bằng hay sửa dữ liệu. Bỏ qua enabled/max_samples của thí nghiệm; --max-rows giới hạn bản xem trước.
Tỷ lệ = số dòng mang nhãn / tổng số dòng đã đọc. Nhãn thiếu/lạ được báo riêng.
MM-IMDb: mỗi phim được tính một lần cho từng thể loại, tổng % có thể vượt 100%.
Nhãn hợp lệ trong dòng chứa thêm nhãn lạ vẫn được đếm. N/A là lỗi đọc/cột, không phải lớp có 0 mẫu.</p>
<p>Tải bảng: <a href="class_counts.tsv">Số mẫu từng lớp</a> ·
<a href="dataset_overview.tsv">Tổng quan dataset</a> · <a href="unknown_labels.tsv">Nhãn lạ</a></p>
""" + "".join(sections) + "</html>"
    (folder / "report.html").write_text(document, encoding="utf-8")
    return folder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment_config.yaml")
    parser.add_argument("--max-rows", type=int, help="First N records per file; omit for full processed files")
    parser.add_argument("--output-dir", default="results/label_distribution")
    args = parser.parse_args()
    if args.max_rows is not None and args.max_rows < 1:
        parser.error("--max-rows must be positive")
    with open(args.config, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    counts, overview, issues = summarize(config, args.max_rows)
    folder = save_report(counts, overview, issues, args.output_dir)
    for row in overview:
        print(f"{row['dataset']}/{row['task']}: {row['n_rows']} rows, {row['status']} {row['error']}")
    print(f"Mở báo cáo: {folder / 'report.html'}")
    if any(row["status"] == "error" for row in overview):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
