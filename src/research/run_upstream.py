"""Chạy repo tác giả với OpenAI CLIP; thay loader bằng dữ liệu của dự án.

--prepare-only chỉ cố định prompt và cấu hình, không đọc dataset/tải checkpoint.
Nhánh ảnh gốc và phần mở rộng ảnh + text được báo cáo thành các dòng riêng.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import yaml

from config import load_config, TASKS
from evaluation_audit import prepare_audit, finish_audit
from evaluate import evaluate_single_label, evaluate_multi_label
from inference import (build_class_embeddings, multilabel_settings, _progress_iter,
                       _softmax, _sigmoid, _weighted_fuse, _record_cosine,
                       relative_multilabel_scores)
from logging_utils import save_single_label_log, save_multi_label_log
from prompts import fakeddit_prompts
from research.run_adapted_methods import prompt_spec, baseline_prompt_set, write_json, git_revision
from research.upstream import (ROOT, verify_sources, author_api, generate_prompts,
                               description_encodings, image_scores, OpenAIClipBackend, choose_device)
from run.common import (load_dataframe, validate_columns, crisismmd_label_column,
                        normalize_single_labels, parse_multilabel, resolve_image_paths,
                        load_images_safe, get_image_weight, resolve_override, compact_summary)
from text_chunking import encode_chunked_texts, encode_chunked_max_sim
from text_scale import text_scale_settings
from pair_relation import (relation_settings, relation_evidence, apply_class_relation,
                           attach_relation, research_relation_config)


def validate_settings(settings):
    """Từ chối lựa chọn sai trước khi sinh prompt hoặc tải model."""
    if not isinstance(settings, dict):
        raise ValueError("Cấu hình nghiên cứu phải là một ánh xạ")
    choices = {"methods": {"clip", "waffle", "dclip_labels", "dclip"},
               "outputs": {"original_image", "multimodal"}}
    for key, allowed in choices.items():
        values = settings.get(key)
        if (not isinstance(values, list) or not values or any(not isinstance(v, str) or v not in allowed for v in values)
                or len(values) != len(set(values))):
            raise ValueError(f"{key} phải là danh sách không lặp thuộc {sorted(allowed)}")
    for key in ("batch_size", "waffle_pairs"):
        if type(settings.get(key)) is not int or settings[key] < 1:
            raise ValueError(f"{key} phải là số nguyên dương")
    seeds = settings.get("seeds")
    if (not isinstance(seeds, list) or not seeds or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("seeds phải là các số nguyên không lặp trong [0, 2**32)")
    backbones = settings.get("backbones")
    if (not isinstance(backbones, list) or not backbones or any(not isinstance(b, str) for b in backbones)
            or len(backbones) != len(set(backbones))):
        raise ValueError("backbones phải là danh sách tên checkpoint không lặp")
    if settings.get("device", "auto") not in ("auto", "cpu", "cuda", "mps"):
        raise ValueError("device phải là auto, cpu, cuda hoặc mps")
    datasets = settings.get("datasets")
    if (not isinstance(datasets, dict) or any(not isinstance(d, dict) for d in datasets.values())
            or not any(d.get("enabled") is True for d in datasets.values())):
        raise ValueError("Cần bật ít nhất một dataset")
    for dataset, options in datasets.items():
        relation_task = {**options, "image_text_relation": research_relation_config({}, settings, options)}
        relation_settings(relation_task, dataset)
        if dataset not in TASKS or options.get("task") not in TASKS[dataset]:
            raise ValueError(f"Dataset/task không hợp lệ: {dataset}/{options.get('task')}")
        if type(options.get("enabled")) is not bool:
            raise ValueError(f"{dataset}: enabled phải là true/false")
        if type(options.get("max_samples")) is not int or options["max_samples"] < 1:
            raise ValueError(f"{dataset}: max_samples phải là số nguyên dương")


def prepare_plan(settings, base, api, root, sources):
    """Sinh toàn bộ prompt bằng code gốc trước khi tiếp xúc với mẫu đánh giá."""
    validate_settings(settings)
    unknown = set(settings["backbones"]) - set(api.clip.available_models())
    if unknown:
        raise ValueError(f"Checkpoint không thuộc OpenAI CLIP: {sorted(unknown)}")
    cohorts, trials = {}, []
    for dataset, options in settings["datasets"].items():
        if not options["enabled"]:
            continue
        task = {**deepcopy(base["datasets"][dataset]), **options}
        task["image_text_relation"] = research_relation_config(base["datasets"][dataset], settings, options)
        relation_settings(task, dataset, base)
        for plural, singular, default in (("prompt_variants", "prompt_variant", "current"),
                                          ("prompt_aggregations", "prompt_aggregation", "mean_embed")):
            values = task.pop(plural, [task.get(singular, default)])
            if not isinstance(values, list) or len(values) != 1:
                raise ValueError(f"Giữ đúng một {plural} trong base_config để đối chứng")
            task[singular] = values[0]
        if task["prompt_aggregation"] not in ("mean_embed", "mean_sim"):
            raise ValueError("prompt_aggregation phải là mean_embed hoặc mean_sim")
        if dataset == "mmimdb":
            multilabel_settings(task, task.get("threshold", 0.22))
            if task.get("decision_mode", "relative_z") not in ("relative_z", "fixed_threshold"):
                raise ValueError("decision_mode phải là relative_z hoặc fixed_threshold")
            cutoff = task.get("relative_z_threshold", 1.0) if task.get("decision_mode", "relative_z") == "relative_z" else task.get("threshold", 0.22)
            if not isinstance(cutoff, (int, float)) or isinstance(cutoff, bool) or not np.isfinite(cutoff) or cutoff < 0:
                raise ValueError("Ngưỡng quyết định phải là số hữu hạn không âm")
            if task.get("decision_mode") == "fixed_threshold" and cutoff > 1:
                raise ValueError("threshold xác suất phải nằm trong [0, 1]")
        text_prompts = baseline_prompt_set(dataset, task)
        names = prompt_spec(dataset, task["task"])
        if list(names) != list(text_prompts) or len(set(names.values())) != len(names):
            raise ValueError("Thứ tự hoặc tên nhãn không khớp giữa các nhánh")
        descriptor_path = settings["descriptor_files"][dataset][task["task"]]
        descriptor = json.loads(Path(descriptor_path).read_text(encoding="utf-8"))
        values = descriptor.get("descriptors", {})
        if not descriptor.get("provenance") or set(values) != {str(k) for k in names}:
            raise ValueError(f"Descriptor thiếu nguồn hoặc thiếu/thừa nhãn: {descriptor_path}")
        for key, items in values.items():
            if not isinstance(items, list) or not items or any(not isinstance(s, str) or not s.strip() for s in items):
                raise ValueError(f"Descriptor rỗng/sai kiểu cho nhãn {key}")
        # Loader tác giả nhận tên lớp, còn file của dự án dùng ID; ánh xạ một lần.
        converted = {name: values[str(key)] for key, name in names.items()}
        path = root / f"descriptors_{dataset}_{task['task']}.json"
        write_json(path, converted)
        cohorts[dataset] = {"task_config": task, "names": names, "text_prompts": text_prompts,
                            "descriptor_provenance": descriptor["provenance"],
                            "descriptor_sha256": hashlib.sha256(Path(descriptor_path).read_bytes()).hexdigest()}
        for method in settings["methods"]:
            for seed in settings["seeds"] if method == "waffle" else [None]:
                prompts, author_options = generate_prompts(api, method, names.values(), path,
                                                          settings["waffle_pairs"], seed or 0)
                if list(prompts) != list(names.values()):
                    raise ValueError("Loader tác giả thay đổi thứ tự nhãn")
                # CLIP gốc từ chối prompt quá dài; phát hiện khi chuẩn bị, không cắt ngầm.
                for items in prompts.values():
                    api.clip.tokenize(items)
                trials.append({"dataset": dataset, "method": method, "seed": seed,
                               "prompts": prompts, "author_options": author_options})
    snapshot = {"settings": settings, "base_config": base, "cohorts": cohorts,
                "trials": [{**t, "author_options": vars(t["author_options"]) if isinstance(t["author_options"], argparse.Namespace)
                            else t["author_options"]} for t in trials],
                "sources": sources, "git_hash": git_revision(ROOT),
                "protocol": "frozen_no_fit_exploratory",
                "source_hashes": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted((ROOT / "src").rglob("*.py"))}}
    write_json(root / "frozen_plan.json", snapshot)
    return cohorts, trials


def load_cohort(dataset, spec):
    """Dùng đúng bộ lọc, cách bỏ ảnh lỗi và ánh xạ nhãn của runner chính."""
    task = spec["task_config"]
    id_col = task.get("id_col", "id")
    text_col = task.get("text_col", "plot" if dataset == "mmimdb" else "tweet_text" if dataset == "crisismmd" else "clean_title")
    label_col = (crisismmd_label_column(task) if dataset == "crisismmd" else
                 task.get("label_col", "genres") if dataset == "mmimdb" else
                 task.get(f"label_col_{task['task']}", "6_way_label" if task["task"] == "6way" else "2_way_label"))
    frame = load_dataframe(task["data_path"], max_samples=task["max_samples"], text_col=text_col,
                           min_text_length=task.get("min_text_length", 20) if dataset == "fakeddit" else 0)
    validate_columns(frame, [id_col, text_col, label_col], dataset)
    paths = resolve_image_paths(frame, task)
    images, indices = load_images_safe(paths)
    if not indices:
        raise RuntimeError(f"{dataset}: không có ảnh hợp lệ")
    selected = frame.iloc[indices].reset_index(drop=True)
    selected["_resolved_image_path"] = [paths[i] for i in indices]
    labels = list(spec["names"])
    truth = ([parse_multilabel(v) for v in selected[label_col]] if dataset == "mmimdb" else
             fakeddit_prompts.normalize_labels(selected[label_col].tolist(), task["task"]) if dataset == "fakeddit" else
             normalize_single_labels(selected[label_col].tolist(), spec["names"], labels))
    unknown = ({x for row in truth for x in row} if dataset == "mmimdb" else set(truth)) - set(labels)
    if unknown:
        raise ValueError(f"Nhãn không thuộc taxonomy: {unknown}")
    return {"frame": selected, "images": images, "truth": truth, "labels": labels,
            "texts": selected[text_col].fillna("").astype(str).tolist(),
            "id_col": id_col, "text_col": text_col, "label_col": label_col,
            "n_skipped_images": len(frame) - len(indices)}


def encode_inputs(vlm, dataset, spec, cohort, config, settings):
    """Mã hóa một lần mỗi model/bộ mẫu; cache embedding CPU, không giữ ảnh PIL."""
    batch_size = settings["batch_size"]
    relation, _ = relation_settings(spec["task_config"], dataset, config)
    blocks = []
    for start in _progress_iter(range(0, len(cohort["images"]), batch_size), desc="Mã hóa ảnh", unit="batch"):
        batch = cohort["images"][start:start + batch_size]
        try:
            blocks.append(vlm.encode_images(batch).detach().float().cpu())
        finally:
            for image in batch:
                image.close()
    image = vlm.api.torch.cat(blocks)
    texts = [t.strip() for t in cohort["texts"]]
    empty = np.array([not t for t in texts], dtype=bool)
    text_scores = np.zeros((len(texts), len(cohort["labels"])))
    result = {"image": image, "text_scores": text_scores, "empty": empty}
    if relation["enabled"]:
        result["relation_embeddings"] = {"image": image.numpy(),
                                         "text": np.zeros((len(texts), image.shape[1]))}
    if empty.all() or (not relation["enabled"] and
            ("multimodal" not in settings["outputs"] or get_image_weight(config, dataset) == 1)):
        return result
    classes, labels = build_class_embeddings(vlm, spec["text_prompts"],
                                            aggregation=spec["task_config"]["prompt_aggregation"])
    if labels != cohort["labels"]:
        raise ValueError("Vector text và nhánh ảnh khác thứ tự lớp")
    task = spec["task_config"]
    chunking = dataset == "mmimdb" and task.get("chunk_long_text", False)
    use_max = task.get("chunk_aggregation", "mean_embed") == "max_sim"
    counts = [0] * len(texts)
    nonempty = np.flatnonzero(~empty)
    for start in _progress_iter(range(0, len(nonempty), batch_size), desc="Mã hóa text", unit="batch"):
        indices = nonempty[start:start + batch_size]
        batch = [texts[i] for i in indices]
        if chunking and use_max:
            mean_buffer = {} if relation["enabled"] else None
            scores, sizes = encode_chunked_max_sim(vlm, batch, batch_size, classes, mean_embeddings=mean_buffer)
            if mean_buffer is not None:
                vectors = mean_buffer["text"]
        elif chunking:
            vectors, sizes = encode_chunked_texts(vlm, batch, batch_size)
            scores = vectors @ classes.T
        else:
            vectors = vlm.encode_texts(batch).detach().float().cpu().numpy()
            scores, sizes = vectors @ classes.T, [1] * len(batch)
        text_scores[indices] = scores
        if relation["enabled"]:
            result["relation_embeddings"]["text"][indices] = vectors
        for i, size in zip(indices, sizes):
            counts[i] = size
    if chunking:
        result["chunking"] = {"enabled": True, "aggregation": task.get("chunk_aggregation", "mean_embed"),
                               "counts": counts, "total_chunks": sum(counts),
                               "n_chunked_texts": sum(n > 1 for n in counts)}
    return result


def score_predictions(vlm, image, text, empty, labels, weight, scale, *, multilabel, decision, native):
    """Giữ scoring ảnh gốc; fusion/nhiều nhãn là phần mở rộng có ghi rõ nguồn."""
    relative = multilabel and decision["score_type"] == "relative_z"
    if relative:
        fused, p_image, p_text = relative_multilabel_scores(image, text, weight, empty)
    else:
        transform = _sigmoid if multilabel else _softmax
        # Paper gốc chọn argmax cosine; softmax cosine chỉ phục vụ log, không đổi nhãn.
        image_scale = 1.0 if native and not multilabel else vlm.logit_scale
        p_image = transform(image * image_scale)
        p_text = transform(text * (vlm.logit_scale if scale is None else scale))
        fused = _weighted_fuse(p_image, p_text, weight, empty)
    if not multilabel:
        return [labels[i] for i in fused.argmax(axis=1)], fused, p_image, p_text, np.zeros(len(image), bool)
    cutoff = decision["relative_z_threshold"] if relative else decision["threshold"]
    selected = fused > cutoff if relative else fused >= cutoff
    missing = ~selected.any(axis=1)
    if decision["fallback_top1"]:
        selected[np.flatnonzero(missing), fused[missing].argmax(axis=1)] = True
    predictions = [[label for label, keep in zip(labels, row) if keep] for row in selected]
    return predictions, fused, p_image, p_text, missing & decision["fallback_top1"]


def evaluate_trial(vlm, dataset, spec, cohort, cache, trial, image, output, base, root):
    task = spec["task_config"]
    native, multi = output == "original_image", dataset == "mmimdb"
    relation, relation_meta = relation_settings(task, dataset, base)
    effective_relation = {**relation, "mode": "diagnostic" if native else relation["mode"]}
    relation_meta["relation_mode"] = effective_relation["mode"]
    config = deepcopy(base)
    config["datasets"][dataset] = task
    config.setdefault("evaluation", {}).update(enabled=True, sample_lock_mode="off")
    weight = 1.0 if native else get_image_weight(config, dataset)
    _, decision = multilabel_settings(task, resolve_override(task.get("threshold", 0.22), task.get("thresholds"), vlm.model_name)) if multi else ({}, {})
    decision["fallback_top1"] = task.get("fallback_top1", True)
    scale, scale_meta = text_scale_settings(config, dataset, vlm, task, weight)
    audit = prepare_audit(cohort["frame"], task, config, dataset, task["task"],
                          cohort["id_col"], cohort["text_col"], cohort["label_col"],
                          {"image": trial["prompts"], "text": spec["text_prompts"]}, scope="upstream")
    text = cache["text_scores"]
    predictions, fused, p_image, p_text, fallback = score_predictions(
        vlm, image, text, cache["empty"], cohort["labels"], weight, scale,
        multilabel=multi, decision=decision, native=native)
    relation_scores = {}
    if relation["enabled"]:
        vectors = cache["relation_embeddings"]
        relation_scores = relation_evidence(vectors["image"], vectors["text"], p_image, p_text,
                                            cache["empty"], single_label=not multi)
        fused = apply_class_relation(fused, cohort["labels"], relation_scores, effective_relation, weight)
        if not multi:
            predictions = [cohort["labels"][i] for i in fused.argmax(axis=1)]
        if audit is not None:
            audit["image_text_relation"] = {**effective_relation, **relation_scores}
    _record_cosine(audit, image, text, weight, cache["empty"])
    if not native and "chunking" in cache:
        audit["chunking"] = deepcopy(cache["chunking"])
    frame = cohort["frame"].copy()
    frame["_text_was_empty"] = cache["empty"]
    attach_relation(frame, relation_scores)
    if multi:
        frame["_used_top1_fallback"] = fallback
    result = (evaluate_multi_label(cohort["truth"], predictions, cohort["labels"]) if multi else
              evaluate_single_label(cohort["truth"], predictions, cohort["labels"]))
    result.update(finish_audit(audit, vlm, frame, cohort["texts"], cohort["truth"], cohort["labels"],
                               fused, p_image, p_text, cache["empty"], weight,
                               multilabel=multi, decision_meta=decision, fallback=decision["fallback_top1"]))
    metadata = {**decision, **scale_meta, **relation_meta, "method": trial["method"], "seed": trial["seed"],
                "output_variant": output, "backbone": vlm.backbone, "device": vlm.device,
                "implementation": "author_functions_openai_clip_v1",
                "classification_scope": "multi_label_extension" if multi else "paper_image" if native else "multimodal_extension",
                "image_logit_scale_effective": None if multi and decision["score_type"] == "relative_z" else 1.0 if native and not multi else vlm.logit_scale,
                "descriptor_provenance": spec["descriptor_provenance"], "descriptor_sha256": spec["descriptor_sha256"],
                "protocol": "frozen_no_fit_exploratory", "text_prompt_aggregation": task["prompt_aggregation"],
                "n_skipped_images": cohort["n_skipped_images"], "frozen_plan": str(root / "frozen_plan.json")}
    if native and multi:
        metadata.update(chunk_long_text=False, chunk_aggregation=None)
    log = save_multi_label_log if multi else save_single_label_log
    result["prediction_log_dir"] = log(frame, cohort["truth"], predictions, fused, cohort["labels"],
        id_col=cohort["id_col"], text_col=cohort["text_col"], model_name=vlm.model_name,
        dataset_name=dataset, task_name=task["task"], output_dir=str(root / "raw_predictions"),
        prompt_version=f"{trial['method']}_{output}_s{trial['seed']}", sim_image=p_image, sim_text=p_text,
        extra_manifest=metadata, evaluation_audit=audit)
    result.update(metadata)
    return {"model": vlm.model_name, "dataset": dataset, "task": task["task"], **result}


def seed_summary(rows):
    """Tổng hợp riêng từng bộ mẫu/variant; không chọn seed có kết quả tốt nhất."""
    # Metric audit mang tiền tố fusion_; dùng cùng cách chọn metric với summary.tsv.
    frame = compact_summary(rows)
    keys = ["model", "dataset", "task", "method", "output_variant"]
    metrics = [m for m in ("macro_f1", "micro_f1", "balanced_accuracy", "samples_f1") if m in frame]
    output = []
    for identity, group in frame.groupby(keys, sort=False):
        row = {**dict(zip(keys, identity)), "n_runs": len(group)}
        for metric in metrics:
            row[metric + "_mean"] = group[metric].mean()
            row[metric + "_std"] = group[metric].std(ddof=0) if len(group) > 1 else None
        output.append(row)
    return pd.DataFrame(output)


def experiment_summary(rows):
    """Lưu metric và cấu hình đối chiếu cần đọc; chi tiết còn lại nằm trong manifest."""
    frame = compact_summary(rows)
    columns = ["dataset", "task", "model", "backbone", "device", "method", "seed", "output_variant",
               "classification_scope", "protocol", "n_samples", "accuracy", "macro_f1", "micro_f1",
               "balanced_accuracy", "samples_f1", "decision_mode", "score_type",
               "relative_z_threshold", "threshold", "image_weight", "text_logit_scale_requested",
               "text_logit_scale_effective", "text_logit_scale_status", "chunk_long_text",
               "chunk_aggregation", "n_skipped_images", "image_macro_f1", "text_macro_f1", "fusion_macro_f1_gain",
               "fusion_improves_macro_f1", "relation_enabled", "relation_mode", "relation_weight", "prediction_log_dir"]
    native = frame["output_variant"].eq("original_image")
    for key in ("fusion_macro_f1_gain", "fusion_improves_macro_f1", "text_macro_f1"):
        if key in frame:
            frame[key] = frame[key].where(~native)
    if "image_macro_f1" in frame:
        frame.loc[native, "image_macro_f1"] = frame.loc[native, "macro_f1"]
    return frame[[key for key in columns if key in frame.columns]]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/research/upstream.yaml")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)
    settings = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    validate_settings(settings)
    base = load_config(settings["base_config"])
    sources, api = verify_sources(), author_api()
    output = Path(settings["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="experiment_", dir=output))
    rows, vlm = [], None
    try:
        specs, trials = prepare_plan(settings, base, api, root, sources)
        print(f"[PLAN] {root / 'frozen_plan.json'}; {len(trials)} phương pháp/seed mỗi model")
        if args.prepare_only:
            print("Chỉ chuẩn bị: chưa đọc dataset, tải model hoặc suy luận.")
            return root
        device = choose_device(api.torch, settings.get("device", "auto"))
        # Loader chỉ chạy sau khi toàn bộ prompt đã được cố định vào plan.
        cohorts = {dataset: load_cohort(dataset, spec) for dataset, spec in specs.items()}
        for backbone in settings["backbones"]:
            vlm = OpenAIClipBackend(api, backbone, device)
            vlm.backbone = backbone
            try:
                for dataset, spec in specs.items():
                    cfg = deepcopy(base)
                    cfg["datasets"][dataset] = spec["task_config"]
                    cache = encode_inputs(vlm, dataset, spec, cohorts[dataset], cfg, settings)
                    for trial in (t for t in trials if t["dataset"] == dataset):
                        descriptions = description_encodings(api, vlm.model, trial["method"], trial["prompts"],
                                                             trial["author_options"], device)
                        descriptions = {name: values.detach().float().cpu() for name, values in descriptions.items()}
                        if list(descriptions) != list(spec["names"].values()):
                            raise ValueError("Hàm mã hóa tác giả thay đổi thứ tự nhãn")
                        image = image_scores(api, cache["image"], descriptions, trial["method"]).numpy()
                        for variant in settings["outputs"]:
                            row = evaluate_trial(vlm, dataset, spec, cohorts[dataset], cache, trial,
                                                 image, variant, base, root)
                            rows.append(row)
                            experiment_summary(rows).to_csv(root / "summary.tsv", sep="\t", index=False)
                    del cache
            finally:
                vlm.unload()
                vlm = None
        seed_summary(rows).to_csv(root / "seed_summary.tsv", sep="\t", index=False)
        print(f"[RESULTS] {root}; summary.tsv + seed_summary.tsv + raw_predictions/")
    except Exception as exc:
        write_json(root / "failure.json", {"error": str(exc), "completed_runs": len(rows)})
        raise
    return root


if __name__ == "__main__":
    main()
