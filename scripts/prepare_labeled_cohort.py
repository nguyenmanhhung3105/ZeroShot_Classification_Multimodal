"""Tạo cohort có nhãn nguồn, ảnh hợp lệ và manifest; không tải model.

Chạy từ thư mục gốc. Không ghi đè đầu ra đã tồn tại. --plan-downloads chỉ tạo
danh sách URL Fakeddit cho downloader hiện có, không truy cập mạng.
"""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import csv
import hashlib
import html
import json
from pathlib import Path
import random
import re
import shutil
import socket
import sys
import tempfile
import time
import unicodedata
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import urllib.error
import urllib.parse
import urllib.request

from PIL import Image
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prompts.mmimdb_prompts import GENRES
from prompts.crisismmd_prompts import LABEL_NAMES_HUMANITARIAN, LABEL_NAMES_INFORMATIVENESS
from config import load_config

# Exact bytes of the Imgur "image no longer available" placeholder observed
# in the previous local Fakeddit image pool. Not an annotated photograph.
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


def tsv_rows(path, required):
    # Stream from the beginning; no DataFrame loading of the entire source.
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        missing = set(required) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: thiếu cột {sorted(missing)}")
        for number, row in enumerate(reader, start=2):
            yield number, {key: row.get(key) or "" for key in required}


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


def proportional_sample(rows, size, key, seed):
    groups = defaultdict(list)
    for row in sorted(rows, key=lambda row: row["_key"]):
        groups[key(row)].append(row)
    wanted = quotas({group: len(items) for group, items in groups.items()}, size)
    rng = random.Random(seed)
    selected = []
    for group in sorted(groups, key=str):
        rng.shuffle(groups[group])
        selected.extend(groups[group][:wanted[group]])
    rng.shuffle(selected)
    return selected


def multilabel_sample(rows, size, seed):
    """Phân tầng lặp bậc một vào hai phần, giữ gần support từng genre.

    Không tối ưu trực tiếp cặp genre; manifest báo cả sai lệch từng nhãn,
    số nhãn/phim và đồng xuất hiện để kiểm tra giới hạn đó.
    """
    quotas({"pool": len(rows)}, size)
    rows = sorted(rows, key=lambda row: row["_key"])
    rng = random.Random(seed)
    order = list(range(len(rows)))
    rng.shuffle(order)
    rank = {index: pos for pos, index in enumerate(order)}
    members = {label: set() for label in GENRES}
    for index, row in enumerate(rows):
        for label in row["_labels"]:
            members[label].add(index)
    remaining = [size, len(rows) - size]
    desired = [{label: len(ids) * capacity / len(rows) for label, ids in members.items()}
               for capacity in remaining]
    selected = []
    unassigned = set(order)
    while unassigned:
        active = [label for label, ids in members.items() if ids]
        if not active:
            raise ValueError("Mẫu nhiều nhãn có tập nhãn rỗng.")
        rare = min(active, key=lambda label: (len(members[label]), label))
        for index in sorted(members[rare], key=rank.get):
            bins = [part for part in (0, 1) if remaining[part] > 0]
            best = max((desired[part][rare], remaining[part]) for part in bins)
            tied = [part for part in bins if (desired[part][rare], remaining[part]) == best]
            part = rng.choice(tied)
            if part == 0:
                selected.append(rows[index])
            remaining[part] -= 1
            unassigned.remove(index)
            for label in rows[index]["_labels"]:
                desired[part][label] -= 1
                members[label].discard(index)
    rng.shuffle(selected)
    assert len(selected) == size
    return selected


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


def add_record(rows, seen, row, key, text, labels, paths, source, number):
    if key in seen:
        raise ValueError(f"ID lặp trong nguồn: {key}")
    seen.add(key)
    row.update(_key=key, _text=text, _labels=list(labels), _paths=list(paths),
               source_file=str(source), source_row=str(number))
    rows.append(row)


def read_fakeddit(root, args, rejected):
    if args.split not in {"validate", "test_public"} or args.task not in {"2way", "6way"}:
        raise ValueError("Fakeddit cần --split validate|test_public và --task 2way|6way.")
    path = root / f"data/raw/fakeddit/multimodal_{args.split}.tsv"
    columns = ["id", "clean_title", "image_url", "hasImage", "2_way_label", "6_way_label"]
    rows, seen = [], set()
    roots = [root / p for p in args.image_dir] if args.image_dir else [
        root / "data/processed/pro_fakeddit/images", root / "data/raw/fakeddit/images"]
    for number, raw in tsv_rows(path, columns):
        key = raw["id"].strip()
        text = text_value(raw["clean_title"])
        a, b = raw["2_way_label"], raw["6_way_label"]
        if a not in {"0", "1"} or b not in set("012345") or (a == "1") != (b == "0"):
            rejected["invalid_label"] += 1
            continue
        if not re.fullmatch(r"[A-Za-z0-9_-]+", key):
            rejected["invalid_id"] += 1
            continue
        if len(text.replace(" ", "")) <= args.min_text_length:
            rejected["short_text"] += 1
            continue
        if raw["hasImage"].lower() != "true" or not raw["image_url"].startswith(("http://", "https://")):
            rejected["missing_image_metadata"] += 1
            continue
        raw.update(id=key, clean_title=text, hasImage="True",
                   y_true=a if args.task == "2way" else b)
        paths = [directory / f"{key}{ext}" for directory in roots for ext in (".jpg", ".jpeg", ".JPG", ".JPEG")]
        add_record(rows, seen, raw, key, text, [raw["y_true"]], paths, path, number)
    return rows, [path], "proportional_6way_source_quota"


def read_crisismmd(root, args, rejected):
    if args.task == "both":
        task_rows, sources = {}, []
        for task in ("informativeness", "humanitarian"):
            single = deepcopy(args)
            single.task = task
            records, paths, _ = read_crisismmd(root, single, rejected)
            task_rows[task] = {row["_key"]: row for row in records}
            sources.extend(paths)
        info, human = task_rows["informativeness"], task_rows["humanitarian"]
        common = sorted(info.keys() & human.keys())
        rejected["outside_common_info_split"] += len(info) - len(common)
        rejected["outside_common_human_split"] += len(human) - len(common)
        rows = []
        for key in common:
            a, b = info[key], human[key]
            if a["_text"] != b["_text"] or a["event"] != b["event"] or a["tweet_id"] != b["tweet_id"]:
                raise ValueError(f"Hai annotation không khớp text/event/tweet cho image_id {key}.")
            row = {k: v for k, v in a.items() if k.startswith("_") or k in
                   {"tweet_id", "image_id", "clean_text", "event", "source_file", "source_row"}}
            truth = {"informativeness": a["y_true"], "humanitarian": b["y_true"]}
            row.update(label_informative=a["y_true"], label_humanitarian=b["y_true"],
                       y_true=json.dumps(truth, sort_keys=True),
                       y_true_informative=a["y_true"], y_true_humanitarian=b["y_true"],
                       source_file_humanitarian=b["source_file"], source_row_humanitarian=b["source_row"])
            for task, record in (("informative", a), ("humanitarian", b)):
                for field in ("label_text", "label_image", "label_official", "text_image_agree"):
                    row[f"{field}_{task}"] = record[field]
            row["_labels"] = [f"informativeness:{a['y_true']}", f"humanitarian:{b['y_true']}"]
            rows.append(row)
        return rows, sources, "common_task_split_proportional_joint_labels_then_event"
    if args.split not in {"train", "dev", "test"} or args.task not in {"informativeness", "humanitarian"}:
        raise ValueError("CrisisMMD cần split train|dev|test và task informativeness|humanitarian.")
    name = "informative" if args.task == "informativeness" else "humanitarian"
    folder = root / "data/raw/CrisisMMD_v2.0/crisismmd_datasplit_all"
    columns = ["tweet_id", "image_id", "event_name", "tweet_text", "image", "label", "label_text", "label_image"]
    rows, seen, tweet_splits, image_splits, sources = [], set(), {}, {}, []
    allowed = set(LABEL_NAMES_INFORMATIVENESS if name == "informative" else LABEL_NAMES_HUMANITARIAN)
    label_column = f"label_{args.label_source}" if args.label_source != "official" else "label"
    target = "label_informative" if name == "informative" else "label_humanitarian"
    for part in ("train", "dev", "test"):
        path = folder / f"task_{name}_text_img_{part}.tsv"
        sources.append(path)
        for number, raw in tsv_rows(path, columns):
            key, tweet = raw["image_id"], raw["tweet_id"]
            if not re.fullmatch(r"[0-9]+_[0-9]+", key) or tweet != key.rsplit("_", 1)[0]:
                raise ValueError(f"{path}:{number}: ID không nguyên vẹn; không khôi phục bằng float.")
            if key in image_splits or (tweet in tweet_splits and tweet_splits[tweet] != part):
                raise ValueError(f"{path}:{number}: trùng ảnh hoặc tweet giao nhau giữa split.")
            image_splits[key], tweet_splits[tweet] = part, part
            if part != args.split:
                continue
            if any(raw[field] not in allowed for field in ("label", "label_text", "label_image")):
                rejected["invalid_label"] += 1
                continue
            text = text_value(raw["tweet_text"])
            if not text:
                rejected["empty_text"] += 1
                continue
            image = (root / "data/raw/CrisisMMD_v2.0" / raw["image"]).resolve()
            if not image.is_relative_to((root / "data/raw/CrisisMMD_v2.0/data_image").resolve()):
                raise ValueError(f"{path}:{number}: đường dẫn ảnh ngoài data_image.")
            row = {"tweet_id": tweet, "image_id": key, "clean_text": text, "event": raw["event_name"],
                   target: raw[label_column], "y_true": raw[label_column],
                   "label_text": raw["label_text"], "label_image": raw["label_image"],
                   "label_official": raw["label"],
                   "text_image_agree": str(raw["label_text"] == raw["label_image"])}
            add_record(rows, seen, row, key, text, [row["y_true"]], [image], path, number)
    return rows, sources, "proportional_label_then_event"


def read_mmimdb(root, args, rejected):
    if args.split not in {"train", "dev", "test"} or args.task != "genres":
        raise ValueError("MM-IMDb cần split train|dev|test và --task genres.")
    path = root / "data/raw/mmimdb/split.json"
    with path.open(encoding="utf-8") as stream:
        splits = json.load(stream)
    seen_split = set()
    for part in ("train", "dev", "test"):
        ids = splits[part]
        if not isinstance(ids, list) or any(not isinstance(key, str) or not key.isdigit() for key in ids):
            raise ValueError("split.json cần danh sách ID dạng chuỗi chữ số, giữ số 0 đầu.")
        if len(set(ids)) != len(ids) or seen_split.intersection(ids):
            raise ValueError("split.json chứa ID lặp/giao nhau.")
        seen_split.update(ids)
    folder = root / "data/raw/mmimdb/dataset"
    images = {}
    for item in sorted(folder.iterdir(), key=lambda p: (p.suffix.lower() != ".jpeg", p.name)):
        if item.suffix.lower() in {".jpg", ".jpeg"}:
            images.setdefault(item.stem, item)
    rows, seen = [], set()
    for key in sorted(splits[args.split]):
        source = folder / f"{key}.json"
        try:
            with source.open(encoding="utf-8") as stream:
                raw = json.load(stream)
        except (OSError, ValueError):
            rejected["missing_or_invalid_json"] += 1
            continue
        if not isinstance(raw, dict):
            rejected["invalid_json_object"] += 1
            continue
        plot = raw.get("plot")
        choices = plot if isinstance(plot, list) else [plot]
        text = next((value for candidate in choices if (value := text_value(candidate))), "")
        text = text or text_value(raw.get("plot outline"))
        if len(text) < 3:
            rejected["short_text"] += 1
            continue
        genres = raw.get("genres")
        if not isinstance(genres, list) or any(not isinstance(label, str) for label in genres):
            rejected["invalid_label_list"] += 1
            continue
        labels = sorted(set(genres) & set(GENRES))
        if not labels:
            rejected["empty_valid_label_list"] += 1
            continue
        row = {"id": key, "plot": text, "genres": "|".join(labels), "n_genres": str(len(labels)),
               "y_true": json.dumps(labels, ensure_ascii=False)}
        add_record(rows, seen, row, key, text, labels, [images[key]] if key in images else [], source, "")
    return rows, [path], "iterative_multilabel_order1"


def summary(rows):
    support = Counter(label for row in rows for label in row["_labels"])
    pairs = Counter()
    for row in rows:
        labels = sorted(row["_labels"])
        pairs.update(f"{a}|{b}" for pos, a in enumerate(labels) for b in labels[pos + 1:])
    return {"samples": len(rows), "support": dict(sorted(support.items())),
            "events": dict(sorted(Counter(row["event"] for row in rows if "event" in row).items())),
            "labels_per_sample": dict(sorted(Counter(str(len(row["_labels"])) for row in rows).items())),
            "label_pairs": dict(sorted(pairs.items()))}


def fetch_image(row, cache, timeout):
    """Download atomically; reject broken responses and known removal redirects."""
    target = cache / f"{row['_key']}.jpg"
    url = row["image_url"]
    if url.startswith("http://"):
        url = "https://" + url[len("http://"):]
    temp_path = None
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if Path(urllib.parse.urlparse(response.geturl()).path).name.lower() in {"removed.png", "removed.jpg"}:
                return None, "removed_image"
            with tempfile.NamedTemporaryFile(prefix=".download-", suffix=".part", dir=cache, delete=False) as stream:
                temp_path = Path(stream.name)
                size = 0
                while chunk := response.read(64 * 1024):
                    size += len(chunk)
                    if size > 20 * 1024 * 1024:
                        return None, "image_too_large"
                    stream.write(chunk)
        if not image_ok(temp_path):
            return None, "unreadable_download"
        temp_path.replace(target)
        return target, None
    except urllib.error.HTTPError as error:
        return None, f"http_{error.code}"
    except urllib.error.URLError as error:
        if isinstance(error.reason, socket.gaierror):
            return None, "dns_unavailable"
        return None, "connection_error"
    except (OSError, ValueError):
        return None, "connection_error"
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def download_missing(rows, available, args, root):
    wanted = quotas(Counter(row["6_way_label"] for row in rows), args.samples)
    present = Counter(row["6_way_label"] for row in available)
    known = {row["_key"] for row in available}
    cache = root / "data/raw/fakeddit/images"
    cache.mkdir(parents=True, exist_ok=True)
    attempts, errors = 0, Counter()
    started = time.monotonic()
    print(f"Ảnh hiện có hợp lệ: {len(available)}; quota 6way: {wanted}", flush=True)
    for label in sorted(wanted):
        needed = max(0, wanted[label] - present[label])
        if not needed:
            continue
        candidates = [row for row in rows if row["6_way_label"] == label and row["_key"] not in known]
        candidates.sort(key=lambda row: row["_key"])
        random.Random(args.seed).shuffle(candidates)
        candidates = iter(candidates)
        pending = {}
        def submit(executor):
            nonlocal attempts
            row = next(candidates, None)
            if row is None or attempts >= args.max_download_attempts:
                return False
            attempts += 1
            pending[executor.submit(fetch_image, row, cache, args.download_timeout)] = row
            return True
        with ThreadPoolExecutor(max_workers=args.download_workers) as executor:
            for _ in range(min(needed, args.download_workers)):
                submit(executor)
            while pending:
                done, _ = wait(pending, timeout=10, return_when=FIRST_COMPLETED)
                for future in done:
                    row = pending.pop(future)
                    path, error = future.result()
                    if error == "dns_unavailable":
                        raise ConnectionError("DNS/network không khả dụng; cần chạy lệnh với quyền truy cập mạng.")
                    if path is not None:
                        row["_image"] = path
                        available.append(row)
                        known.add(row["_key"])
                        present[label] += 1
                    else:
                        errors[error] += 1
                while len(pending) < min(args.download_workers, max(0, wanted[label] - present[label])):
                    if not submit(executor):
                        break
                if done and (attempts % 20 < args.download_workers or present[label] >= wanted[label]):
                    print(f"Lớp {label}: {present[label]}/{wanted[label]}; thử {attempts}, "
                          f"lỗi {sum(errors.values())}, {time.monotonic() - started:.0f}s", flush=True)
        print(f"Kết thúc lớp {label}: {present[label]}/{wanted[label]}", flush=True)
    return {"attempts": attempts, "errors": dict(errors), "seconds": round(time.monotonic() - started, 2),
            "cache": "data/raw/fakeddit/images"}


def select_rows(rows, available, args):
    if args.dataset == "fakeddit":
        # Targets come from the raw metadata pool, not the biased local image subset.
        desired = quotas(Counter(row["6_way_label"] for row in rows), args.samples)
        groups = defaultdict(list)
        for row in available:
            groups[row["6_way_label"]].append(row)
        deficits = {label: need - len(groups[label]) for label, need in desired.items()
                    if len(groups[label]) < need}
        if deficits:
            raise ValueError(f"Thiếu ảnh hợp lệ theo quota 6way: {deficits}. "
                             "Dùng --plan-downloads rồi tải/bổ sung ảnh; không xuất cohort thiếu mẫu.")
        rng = random.Random(args.seed)
        selected = []
        for label in sorted(desired):
            group = sorted(groups[label], key=lambda row: row["_key"])
            rng.shuffle(group)
            selected.extend(group[:desired[label]])
        rng.shuffle(selected)
        return selected
    quotas({"available": len(available)}, args.samples)
    if args.dataset == "mmimdb":
        return multilabel_sample(available, args.samples, args.seed)
    # Allocate class totals before event quotas so rare labels are not rounded
    # away separately in every event. Do not force equal event/class counts.
    desired = quotas(Counter(row["y_true"] for row in available), args.samples)
    selected = []
    for label in sorted(desired):
        if desired[label]:
            group = [row for row in available if row["y_true"] == label]
            selected.extend(proportional_sample(group, desired[label], lambda row: row["event"], args.seed))
    random.Random(args.seed).shuffle(selected)
    return selected


def write_tsv(path, rows):
    fields = list(rows[0])
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def experiment_config(base, args, output):
    config = deepcopy(base)
    for settings in config["datasets"].values():
        settings["enabled"] = False
    task = config["datasets"][args.dataset]
    task.update(enabled=True, task="informativeness" if args.task == "both" else args.task, max_samples=args.samples,
                data_path=str(output / "samples.tsv"), image_dir=str(output / "images"))
    if args.dataset == "fakeddit":
        task.update(id_col="id", text_col="clean_title", min_text_length=args.min_text_length)
        task[f"label_col_{args.task}"] = "y_true"
    elif args.dataset == "crisismmd":
        task.update(id_col="image_id", text_col="clean_text")
        if args.task == "both":
            task.update(label_col_info="label_informative", label_col_human="label_humanitarian")
        else:
            task["label_col_info" if args.task == "informativeness" else "label_col_human"] = "y_true"
    else:
        task.update(id_col="id", text_col="plot", label_col="y_true")
    # Avoid an inherited unrelated generic label_col for CrisisMMD.
    if args.dataset == "crisismmd":
        task.pop("label_col", None)
    task.update(cohort_manifest=str(output / "manifest.json"), label_source=args.label_source
                if args.dataset == "crisismmd" else "dataset_annotation", source_split=args.split)
    config.setdefault("evaluation", {}).update(split=f"{args.split}_cohort_{args.samples}",
                                               sample_lock_mode="create_or_verify")
    config.setdefault("output", {}).update(results_dir="auto", raw_predictions_dir="auto", summary_table="auto")
    return config


def prepare(args, root=ROOT):
    root = Path(root).resolve()
    output = (root / args.output_dir).resolve()
    if output.exists():
        raise ValueError(f"Không ghi đè: {output}. Chọn --output-dir mới.")
    if args.samples < 1 or args.min_text_length < 0 or args.candidate_multiplier < 1:
        raise ValueError("samples và candidate-multiplier phải >0; min-text-length phải >=0.")
    if args.plan_downloads and args.dataset != "fakeddit":
        raise ValueError("--plan-downloads chỉ dùng cho Fakeddit.")
    if args.download_missing and (args.dataset != "fakeddit" or args.plan_downloads):
        raise ValueError("--download-missing chỉ dùng khi tạo cohort Fakeddit, không cùng --plan-downloads.")
    if args.download_workers < 1 or args.download_timeout <= 0 or args.max_download_attempts < 1:
        raise ValueError("Các giới hạn download phải dương.")
    rejected = Counter()
    reader = {"fakeddit": read_fakeddit, "crisismmd": read_crisismmd, "mmimdb": read_mmimdb}[args.dataset]
    rows, sources, method = reader(root, args, rejected)
    quotas({"metadata": len(rows)}, args.samples)
    label_source = (args.label_source if args.dataset == "crisismmd" else
                    (f"{args.task[0]}_way_label" if args.dataset == "fakeddit" else "genres"))
    manifest = {"schema_version": 1, "dataset": args.dataset, "task": args.task,
                "generator_sha256": digest(Path(__file__)),
                "source_split": args.split, "label_source": label_source,
                "requested_samples": args.samples, "seed": args.seed, "sampling": method,
                "text_policy": "HTML unescape, Unicode NFC, collapse whitespace; preserve content",
                "min_text_length_exclusive_no_spaces": args.min_text_length if args.dataset == "fakeddit" else None,
                "sources": [{"path": str(path.relative_to(root)), "sha256": digest(path)} for path in sources],
                "metadata_pool": summary(rows)}
    if args.plan_downloads:
        # More candidates than the final target, with the same six-way quota rule.
        candidates = proportional_sample(rows, min(len(rows), args.samples * args.candidate_multiplier),
                                         lambda row: row["6_way_label"], args.seed)
        exported = [{key: row[key] for key in ("id", "image_url", "hasImage", "2_way_label", "6_way_label")}
                    for row in candidates if not any(path.is_file() for path in row["_paths"])]
        if not exported:
            raise ValueError("Danh sách URL trống; thử tạo cohort trực tiếp hoặc kiểm tra ảnh local hỏng.")
        manifest.update(status="download_plan_only", download_candidates=len(exported),
                        note="URL có thể thất bại; kế hoạch này chưa bảo đảm đủ ảnh hợp lệ.")
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".cohort-", dir=output.parent) as folder:
            stage = Path(folder) / "prepared"
            stage.mkdir()
            write_tsv(stage / "download_candidates.tsv", exported)
            (stage / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            stage.rename(output)
        return manifest
    available = []
    for row in rows:
        found = next((path for path in row["_paths"] if path.is_file() and image_ok(path)), None)
        if found is None:
            rejected["missing_or_unreadable_image"] += 1
        else:
            row["_image"] = found
            available.append(row)
    if args.download_missing:
        manifest["downloads"] = download_missing(rows, available, args, root)
    selected = select_rows(rows, available, args)
    if len(selected) != args.samples or len({row["_key"] for row in selected}) != args.samples:
        raise ValueError("Cohort không đủ N mẫu riêng biệt.")
    with (root / args.base_config).open(encoding="utf-8") as stream:
        base = yaml.safe_load(stream)
    config = experiment_config(base, args, output.relative_to(root) if output.is_relative_to(root) else output)
    manifest.update(status="ready", readable_pool=summary(available), selected=summary(selected),
                    rejected=dict(rejected), samples=[])
    all_labels = set(GENRES) if args.dataset == "mmimdb" else {label for row in rows for label in row["_labels"]}
    support = manifest["selected"]["support"]
    manifest["low_support_labels"] = {label: support.get(label, 0) for label in sorted(all_labels) if support.get(label, 0) < 10}
    manifest["label_prevalence_delta"] = {
        label: support.get(label, 0) / len(selected) - manifest["readable_pool"]["support"].get(label, 0) / len(available)
        for label in sorted(all_labels)}
    manifest["metadata_label_prevalence_delta"] = {
        label: support.get(label, 0) / len(selected) - manifest["metadata_pool"]["support"].get(label, 0) / len(rows)
        for label in sorted(all_labels)}
    if args.dataset == "fakeddit":
        manifest["six_way_support"] = {
            name: dict(sorted(Counter(row["6_way_label"] for row in part).items()))
            for name, part in (("metadata_pool", rows), ("readable_pool", available), ("selected", selected))}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".cohort-", dir=output.parent) as folder:
        stage = Path(folder) / "prepared"
        (stage / "images").mkdir(parents=True)
        exported = []
        for row in selected:
            filename = row["_key"] + row["_image"].suffix.lower()
            destination = stage / "images" / filename
            shutil.copy2(row["_image"], destination)
            clean = {key: value for key, value in row.items() if not key.startswith("_")}
            clean.update(image_path=str((output / "images" / filename).relative_to(root))
                         if output.is_relative_to(root) else str(output / "images" / filename),
                         source_split=args.split, label_source=label_source)
            clean["source_file"] = str(Path(clean["source_file"]).relative_to(root))
            if "source_file_humanitarian" in clean:
                clean["source_file_humanitarian"] = str(Path(clean["source_file_humanitarian"]).relative_to(root))
            exported.append(clean)
            truth = (json.loads(row["y_true"]) if args.task == "both" else
                     (row["_labels"] if args.dataset == "mmimdb" else row["y_true"]))
            manifest["samples"].append({"id": row["_key"], "y_true": truth,
                                        "source_file": clean["source_file"], "source_row": clean["source_row"],
                                        "text_sha256": hashlib.sha256(row["_text"].encode()).hexdigest(),
                                        "image_sha256": digest(destination)})
        write_tsv(stage / "samples.tsv", exported)
        manifest["samples_sha256"] = digest(stage / "samples.tsv")
        manifest["cohort_id"] = hashlib.sha256(json.dumps(manifest["samples"], sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:20]
        (stage / "experiment.yaml").write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
        load_config(stage / "experiment.yaml")
        manifest["experiment_sha256"] = digest(stage / "experiment.yaml")
        (stage / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        stage.rename(output)
    return manifest


def parser():
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--dataset", required=True, choices=["fakeddit", "crisismmd", "mmimdb"])
    command.add_argument("--task", required=True, choices=["2way", "6way", "informativeness", "humanitarian", "genres", "both"])
    command.add_argument("--split", required=True, choices=["train", "dev", "test", "validate", "test_public"])
    command.add_argument("--samples", type=int, default=3000)
    command.add_argument("--seed", type=int, default=42)
    command.add_argument("--output-dir", required=True)
    command.add_argument("--base-config", default="configs/experiment_config.yaml")
    command.add_argument("--label-source", choices=["image", "text", "official"], default="image")
    command.add_argument("--min-text-length", type=int, default=20,
                         help="Fakeddit: giữ len(text.replace(' ', '')) > ngưỡng này.")
    command.add_argument("--image-dir", action="append", default=[], help="Fakeddit: có thể truyền nhiều thư mục ảnh.")
    command.add_argument("--plan-downloads", action="store_true")
    command.add_argument("--candidate-multiplier", type=int, default=2,
                         help="Kế hoạch tải: tối đa samples × hệ số ứng viên, mặc định 2.")
    command.add_argument("--download-missing", action="store_true", help="Fakeddit: tải và xác thực ảnh còn thiếu theo quota.")
    command.add_argument("--download-workers", type=int, default=4)
    command.add_argument("--download-timeout", type=float, default=8)
    command.add_argument("--max-download-attempts", type=int, default=6000)
    return command


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        result = prepare(args)
    except (OSError, ValueError, KeyError) as error:
        print(f"Không tạo cohort: {error}", file=sys.stderr)
        return 1
    # Do not print raw text, full sample IDs, or every label pair to the console.
    print(json.dumps({key: result[key] for key in ("status", "dataset", "task", "source_split", "requested_samples")}, ensure_ascii=False))
    if result["status"] == "ready":
        print(json.dumps({"support": result["selected"]["support"], "low_support_labels": result["low_support_labels"],
                          "cohort_id": result["cohort_id"]}, ensure_ascii=False))
    print(f"Đầu ra: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
