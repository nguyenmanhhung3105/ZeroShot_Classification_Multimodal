"""Một task, một backbone, một biến thể Waffle, một seed mỗi lượt.

--prepare-only không đọc dataset, không tải checkpoint. Mặc định 5 mẫu.
"""
import argparse
from copy import deepcopy
import csv
import hashlib
from itertools import islice
import json
from pathlib import Path
import sys
import uuid

LAB = Path(__file__).resolve().parent
ROOT = LAB.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import yaml

from experiments.waffle_lab.core import VARIANTS, REQUIRED_BANKS, attach_semantics, combine
from prompts.crisismmd_prompts import get_label_names
from research.upstream import (author_api, generate_prompts, verify_sources,
                               OpenAIClipBackend, choose_device)
from results_layout import run_path, model_folder

BACKBONES = ("ViT-B/32", "ViT-L/14", "RN50")


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def project_path(value):
    path = Path(value)
    path = (path if path.is_absolute() else ROOT / path).resolve()
    if not path.is_relative_to(ROOT.resolve()) or "dataraw" in path.parts or "raw" in path.parts:
        raise ValueError("Chỉ dùng file trong dự án, không truy cập dữ liệu thô")
    return path


def validate(settings):
    if not isinstance(settings, dict) or settings.get("dataset") != "crisismmd":
        raise ValueError("Waffle lab giai đoạn đầu chỉ hỗ trợ CrisisMMD")
    if settings.get("task") not in ("informativeness", "humanitarian"):
        raise ValueError("Task CrisisMMD không hợp lệ")
    if settings.get("variant") not in VARIANTS or settings.get("backbone") not in BACKBONES:
        raise ValueError("Biến thể hoặc backbone không hợp lệ")
    for key in ("waffle_pairs", "batch_size", "max_samples"):
        if type(settings.get(key)) is not int or settings[key] < 1:
            raise ValueError(f"{key} phải là số nguyên dương")
    if type(settings.get("seed")) is not int or not 0 <= settings["seed"] < 2**32:
        raise ValueError("seed phải trong [0, 2**32)")
    cap = settings.get("noise_cap")
    if type(cap) not in (int, float) or not np.isfinite(cap) or not 0 <= cap <= 1:
        raise ValueError("noise_cap phải trong [0, 1]")
    if settings.get("device", "auto") not in ("auto", "cpu", "cuda", "mps"):
        raise ValueError("device không hợp lệ")


def freeze_plan(settings, directory, api, sources):
    """Cố định cả đối chứng và biến thể trước khi có quyền truy cập cohort."""
    validate(settings)
    base = yaml.safe_load(project_path(settings["base_config"]).read_text(encoding="utf-8"))
    dataset_config = deepcopy(base["datasets"]["crisismmd"])
    names = get_label_names(settings["task"])
    descriptor_file = project_path(settings["descriptor_files"][settings["task"]])
    descriptor = json.loads(descriptor_file.read_text(encoding="utf-8"))
    if not descriptor.get("provenance"):
        raise ValueError("Thiếu nguồn descriptor")
    descriptions = descriptor["descriptors"]
    converted = directory / "author_descriptors.json"
    if set(descriptions) != set(names):
        raise ValueError("Descriptor thiếu/thừa lớp")
    write_json(converted, {name: descriptions[key] for key, name in names.items()})
    original, _ = generate_prompts(api, "waffle", names.values(), converted,
                                   settings["waffle_pairs"], settings["seed"])
    baseline, _ = generate_prompts(api, "clip", names.values(), converted,
                                   settings["waffle_pairs"], 0)
    clean, noisy = attach_semantics(names, descriptions, original, settings["waffle_pairs"])
    banks = {"clip": {key: baseline[name] for key, name in names.items()},
             "waffle": {key: original[name] for key, name in names.items()},
             "semantic": clean, "semantic_waffle": noisy}
    for bank in banks.values():
        for prompts in bank.values():
            # Từ chối prompt quá 77 token; không cắt ngầm mất class hoặc suffix.
            api.clip.tokenize(prompts, truncate=False)
    source_files = [LAB / "core.py", LAB / "run.py",
                    LAB.parents[1] / "src/research/upstream.py",
                    LAB.parents[1] / "src/prompts/crisismmd_prompts.py"]
    hashes = {str(path.relative_to(LAB.parents[1])): file_hash(path) for path in source_files}
    common = {key: settings[key] for key in ("dataset", "task", "backbone", "waffle_pairs", "noise_cap")}
    common.update(sources=sources, source_hashes=hashes, labels=names, label_order=list(names), dataset_config=dataset_config,
                  descriptor_sha256=file_hash(descriptor_file),
                  word_list_sha256=file_hash(LAB.parents[1] / "third_party/WaffleCLIP/word_list.pkl"),
                  aggregation="mean_cosine_except_explicit_waffle_unit",
                  target="image_labels", inference="image_only_frozen_no_fit")
    plan = {"settings": deepcopy(settings), "protocol": common,
            "protocol_fingerprint": digest(common), "prompt_banks": banks,
            "prompt_fingerprint": digest(banks), "label_order": list(names),
            "descriptor_provenance": descriptor["provenance"], "dataset_config": dataset_config,
            "scope": "exploratory_zero_shot_no_label_fit",
            "status": "prepared_no_inference"}
    write_json(directory / "frozen_plan.json", plan)
    return plan


def read_cohort(settings, base):
    """Đọc có giới hạn từ đầu; không lọc theo nhãn/text hoặc bỏ ảnh lỗi im lặng."""
    task = base["datasets"]["crisismmd"]
    source = project_path(task["data_path"])
    image_root = project_path(task["image_dir"])
    if not source.is_relative_to((ROOT / "data/processed").resolve()):
        raise ValueError("TSV phải nằm trong data/processed/")
    label_column = task["label_col_info" if settings["task"] == "informativeness" else "label_col_human"]
    id_column = task.get("id_col", "image_id")
    labels = get_label_names(settings["task"])
    rows, seen = [], set()
    with source.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        required = {id_column, label_column, "image_path", "source_split", "label_source"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"TSV thiếu cột: {sorted(required - set(reader.fieldnames or []))}")
        for row in islice(reader, settings["max_samples"]):
            key, truth = row[id_column].strip(), row[label_column].strip()
            if not key or key in seen or truth not in labels:
                raise ValueError("ID rỗng/lặp hoặc nhãn ngoài taxonomy; không thay mẫu")
            if row["label_source"] != "image" or row["source_split"] != task.get("source_split"):
                raise ValueError("Nguồn nhãn/split không đúng cấu hình nhãn ảnh")
            path = project_path(row["image_path"])
            if not path.is_relative_to(image_root) or not path.is_file():
                raise ValueError(f"Ảnh thiếu/ngoài thư mục processed: {key}")
            # Metadata đã chọn; caption và nhãn text không được chuyển tới scorer.
            rows.append({"id": key, "y_true": truth, "image_path": str(path.relative_to(ROOT.resolve())),
                         "image_sha256": file_hash(path), "source_split": row["source_split"],
                         "label_source": row["label_source"], "event": row.get("event", "")})
            seen.add(key)
    if len(rows) != settings["max_samples"]:
        raise ValueError("Không đủ số mẫu yêu cầu; không báo kết quả dưới samples_N sai")
    return rows


def lock_cohort(settings, rows):
    """Khóa cả ID, thứ tự, ảnh và nhãn; độc lập prompt/seed/backbone."""
    fingerprint = digest(rows)
    directory = ROOT / "results/_system/sample_locks/waffle_lab"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"crisismmd_{settings['task']}_samples_{len(rows)}.json"
    value = {"cohort_fingerprint": fingerprint, "samples": len(rows), "rows": rows}
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
    except FileExistsError:
        if json.loads(path.read_text(encoding="utf-8"))["cohort_fingerprint"] != fingerprint:
            raise ValueError("Cohort khác khóa Waffle lab; không ghép kết quả hai cohort")
    return value


def prompt_scores(backend, prompts, labels, image_vectors, batch_size):
    """Tất cả lớp cùng thứ tự; embedding prompt unit, prototype không renormalize."""
    arrays, norms = [], []
    if list(prompts) != labels:
        raise ValueError("Prompt bank thay thứ tự lớp")
    for key in labels:
        blocks = []
        for start in range(0, len(prompts[key]), batch_size):
            blocks.append(backend.encode_texts(prompts[key][start:start + batch_size])
                          .detach().float().cpu().numpy())
        vectors = np.concatenate(blocks)
        vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
        arrays.append(image_vectors @ vectors.T)
        norms.append(float(np.linalg.norm(vectors.mean(axis=0))))
    return np.stack(arrays, axis=2), np.asarray(norms)


def evaluate(truth, predicted, scores, labels):
    """Nhãn chỉ xuất hiện sau quyết định, dùng để tính metrics."""
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
    metrics = {"accuracy": float(accuracy_score(truth, predicted)),
               "macro_f1": float(f1_score(truth, predicted, labels=labels, average="macro", zero_division=0)),
               "per_class": classification_report(truth, predicted, labels=labels, output_dict=True, zero_division=0),
               "confusion_matrix": confusion_matrix(truth, predicted, labels=labels).tolist(),
               "label_order": labels}
    metrics["auroc"] = None
    if len(labels) == 2 and len(set(truth)) == 2:
        positive = labels.index("informative")
        metrics["positive_label"] = "informative"
        metrics["auroc"] = float(roc_auc_score([y == "informative" for y in truth],
                                               scores[:, positive] - scores[:, 1 - positive]))
    return metrics


def run(settings, prepare_only=False):
    validate(settings)
    api, sources = author_api(), verify_sources()
    if settings.get("output_dir", "auto") == "auto":
        if prepare_only:
            directory = LAB / "checks/prepared" / f"{settings['task']}_{settings['variant']}_seed_{settings['seed']}"
        else:
            directory = ROOT / run_path("waffle_lab_" + settings["variant"], "crisismmd", settings["task"],
                settings["max_samples"], model_folder(backbones=[settings["backbone"]]), uuid.uuid4().hex[:12])
    else:
        directory = project_path(settings["output_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "summary.tsv").exists():
        raise ValueError("Lượt đã có summary.tsv; không ghi đè")
    plan = freeze_plan(settings, directory, api, sources)
    print(f"[PLAN] {directory / 'frozen_plan.json'}", flush=True)
    if prepare_only:
        print("[PREPARED] Chưa đọc dataset, tải model hoặc inference", flush=True)
        return directory
    backend = None
    manifest = {"status": "running", "inference_backend": "not_loaded",
                "task": settings["task"], "variant": settings["variant"], "seed": settings["seed"],
                "model": model_folder(backbones=[settings["backbone"]]),
                "protocol_fingerprint": plan["protocol_fingerprint"], "samples": settings["max_samples"]}
    write_json(directory / "run_manifest.json", manifest)
    try:
        base = {"datasets": {"crisismmd": plan["dataset_config"]}}
        rows = read_cohort(settings, base)
        cohort = lock_cohort(settings, rows)
        write_json(directory / "cohort.json", cohort)
        manifest["cohort_fingerprint"] = cohort["cohort_fingerprint"]
        print(f"[COHORT] {len(rows)} ảnh; {cohort['cohort_fingerprint']}", flush=True)
        device = choose_device(api.torch, settings.get("device", "auto"))
        print(f"[MODEL] Nạp duy nhất {settings['backbone']} trên {device}", flush=True)
        backend = OpenAIClipBackend(api, settings["backbone"], device)
        backend_type = type(backend)
        manifest["inference_backend"] = ("openai_clip_real" if
            (backend_type.__module__, backend_type.__name__) == ("research.upstream", "OpenAIClipBackend")
            else "synthetic_or_custom")
        manifest.update(device=device, batch_size=settings["batch_size"],
                        compute_dtype=str(getattr(backend.model, "dtype", "synthetic_or_unspecified")))
        manifest["logit_scale"] = backend.logit_scale
        from PIL import Image
        blocks = []
        for start in range(0, len(rows), settings["batch_size"]):
            images = []
            try:
                for row in rows[start:start + settings["batch_size"]]:
                    with Image.open(project_path(row["image_path"])) as image:
                        images.append(image.convert("RGB"))
                blocks.append(backend.encode_images(images).detach().float().cpu().numpy())
            finally:
                for image in images:
                    image.close()
            print(f"[IMAGE] {min(start + settings['batch_size'], len(rows))}/{len(rows)}", flush=True)
        image_vectors = np.concatenate(blocks)
        image_vectors /= np.maximum(np.linalg.norm(image_vectors, axis=1, keepdims=True), 1e-12)
        banks, norms = {}, {}
        for bank in REQUIRED_BANKS[settings["variant"]]:
            print(f"[PROMPTS] {bank}", flush=True)
            banks[bank], norms[bank] = prompt_scores(backend, plan["prompt_banks"][bank],
                                                    plan["label_order"], image_vectors, settings["batch_size"])
        combined = combine(settings["variant"], banks, settings["noise_cap"], norms.get("waffle"))
        scores, labels = combined["scores"], plan["label_order"]
        predicted = [labels[index] for index in scores.argmax(axis=1)]
        metrics = evaluate([row["y_true"] for row in rows], predicted, scores, labels)
        logits = backend.logit_scale * scores
        probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        with (directory / "predictions.tsv").open("w", encoding="utf-8", newline="") as stream:
            fields = ["id", "y_true", "y_pred", "event", "cosine_scores", "probabilities",
                      "margin_cosine", "margin_probability", "prompt_agreement", "noise_weight"]
            writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            for index, row in enumerate(rows):
                ordered, ordered_p = np.sort(scores[index]), np.sort(probabilities[index])
                agreement = combined["prompt_agreement"][index]
                writer.writerow({"id": row["id"], "y_true": row["y_true"], "y_pred": predicted[index],
                    "event": row["event"], "cosine_scores": json.dumps(scores[index].tolist()),
                    "probabilities": json.dumps(probabilities[index].tolist()),
                    "margin_cosine": ordered[-1] - ordered[-2],
                    "margin_probability": ordered_p[-1] - ordered_p[-2],
                    "prompt_agreement": float(agreement) if np.isfinite(agreement) else "",
                    "noise_weight": combined["noise_weight"][index]})
        write_json(directory / "metrics.json", metrics)
        summary = {**{key: manifest[key] for key in ("task", "variant", "seed", "model", "samples")},
                   **{key: metrics[key] for key in ("accuracy", "macro_f1", "auroc")},
                   "cohort_fingerprint": cohort["cohort_fingerprint"]}
        with (directory / "summary.tsv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summary), delimiter="\t")
            writer.writeheader()
            writer.writerow(summary)
        manifest.update(status="complete", probabilities_calibrated=False, label_fit=False,
                        prototype_norms={key: values.tolist() for key, values in norms.items()})
        write_json(directory / "run_manifest.json", manifest)
        print(f"[RESULT] {directory / 'summary.tsv'}", flush=True)
    except Exception as exc:
        manifest.update(status="failed", error=str(exc))
        write_json(directory / "run_manifest.json", manifest)
        raise
    finally:
        if backend is not None:
            backend.unload()
            print("[MODEL] Đã giải phóng model", flush=True)
    return directory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(LAB / "config.yaml"))
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--task", choices=("informativeness", "humanitarian"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)
    settings = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    for key in ("variant", "task", "seed", "max_samples", "output_dir"):
        value = getattr(args, key)
        if value is not None:
            settings[key] = value
    return run(settings, args.prepare_only)


if __name__ == "__main__":
    main()
