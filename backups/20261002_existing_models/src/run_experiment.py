"""
Main experiment loop: model x dataset.

Nguyên tắc:
- Mỗi model chỉ load 1 lần.
- Chạy toàn bộ dataset được enable.
- Sau đó unload trước khi chuyển sang model tiếp theo.
- Dataset, đường dẫn, batch size và output đều điều khiển từ experiment_config.yaml.
"""

import ast
import json
import os

import pandas as pd
import yaml
from PIL import Image

from run import common as shared
from models.model_registry import load_model
from evaluation_audit import prepare_audit, finish_audit
from text_scale import text_scale_settings
from modality_branches import modality_settings
from nli_text import make_text_scorer, backend_summary, backend_manifest
from inference import build_class_embeddings, predict_single_label, predict_multi_label, multilabel_settings
from evaluate import evaluate_single_label, evaluate_binary, evaluate_multi_label
from logging_utils import save_single_label_log, save_multi_label_log
from prompts import fakeddit_prompts, crisismmd_prompts, mmimdb_prompts


def load_config(path="configs/experiment_config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f: return yaml.safe_load(f)


def load_dataframe(path: str, max_samples=None, text_col=None, min_text_length=0) -> pd.DataFrame:
    return shared.load_dataframe(path, max_samples, text_col, min_text_length)


def resolve_override(default_value, overrides: dict, key: str):
    """Tra cứu giá trị override theo key (model_name hoặc dataset_name).
    Dùng chung cho image_weight (theo dataset) và threshold (theo model),
    cùng một pattern nên gộp lại 1 hàm thay vì lặp code."""
    if not overrides:
        return default_value
    return overrides.get(key, default_value)


def resolve_image_paths(df: pd.DataFrame, task_config: dict) -> list:
    return shared.resolve_image_paths(df, task_config)


def load_images_safe(image_paths: list) -> tuple:
    return shared.load_images_safe(image_paths)


def validate_columns(df: pd.DataFrame, required_columns: list, dataset_name: str) -> None:
    missing = [column for column in required_columns if column not in df.columns]

    if missing:
        raise ValueError(
            f"{dataset_name}: thiếu các cột {missing}. "
            f"Các cột hiện có: {df.columns.tolist()}"
        )


def parse_multilabel(value) -> list:
    """Chuyển genres từ TSV thành list label."""
    if isinstance(value, (list, tuple, set)): return list(value)
    if value is None or (isinstance(value, float) and pd.isna(value)): return []

    text = str(value).strip()
    if not text: return []

    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
            if isinstance(parsed, (list, tuple, set)): return [str(x).strip() for x in parsed]
        except Exception:
            pass

    if "|" in text: return [x.strip() for x in text.split("|") if x.strip()]
    return [x.strip() for x in text.split(",") if x.strip()]


def normalize_single_labels(values: list, label_names, label_order: list) -> list:
    return shared.normalize_single_labels(values, label_names, label_order)


def get_image_weight(config: dict, dataset_name: str) -> float:
    inference_cfg = config.get("inference", {})
    return resolve_override(
        inference_cfg.get("image_weight", 0.5),
        inference_cfg.get("image_weight_overrides"),
        dataset_name,
    )


def run_fakeddit(vlm, task_config: dict, config: dict) -> dict:
    task = task_config["task"]
    print(f"\n{'=' * 70}\nDATASET: Fakeddit | TASK: {task}\n{'=' * 70}")

    df = load_dataframe(task_config["data_path"], max_samples=task_config.get("max_samples"),
                        text_col=task_config.get("text_col", "clean_title"),
                        min_text_length=task_config.get("min_text_length", 20))

    id_col = task_config.get("id_col", "id")
    text_col = task_config.get("text_col", "clean_title")
    label_col = task_config.get("label_col_6way", "6_way_label") if task == "6way" else task_config.get("label_col_2way", "2_way_label")

    validate_columns(df, [id_col, text_col, label_col], "Fakeddit")

    prompt_set = fakeddit_prompts.get_prompt_set(task, task_config.get("prompt_variant", "current"))
    prompt_meta = fakeddit_prompts.prompt_metadata(task_config, prompt_set)
    class_embeds, label_order = build_class_embeddings(vlm, prompt_set)
    text_scorer = make_text_scorer(task_config, "fakeddit", label_order)

    image_paths = resolve_image_paths(df, task_config)
    images, valid_idx = load_images_safe(image_paths)

    if not images: raise RuntimeError("Fakeddit: không load được ảnh hợp lệ nào.")

    df_valid = df.iloc[valid_idx].reset_index(drop=True)
    df_valid["_resolved_image_path"] = [image_paths[i] for i in valid_idx]

    audit = prepare_audit(df_valid, task_config, config, "fakeddit", task_config.get("task", "genres"),
                          id_col, text_col, label_col, prompt_set)

    texts = df_valid[text_col].fillna("").astype(str).tolist()

    batch_size = config.get("inference", {}).get("batch_size", 16)
    image_weight = get_image_weight(config, "fakeddit")
    image_weight, branch_meta = modality_settings(task_config, image_weight)
    text_scale, scale_meta = text_scale_settings(config, "fakeddit", vlm, task_config, image_weight)
    scale_meta.update(branch_meta)

    predictions, p, p_img, p_text, text_empty_mask = predict_single_label(
        vlm, images, texts, class_embeds, label_order,
        batch_size=batch_size, image_weight=image_weight, diagnostics=audit, text_logit_scale=text_scale,
        **({"text_scorer": text_scorer} if text_scorer is not None else {}),
    )
    # Gắn thẳng vào df_valid để lọt vào log mà không cần sửa logging_utils.py
    df_valid["_text_was_empty"] = text_empty_mask

    y_true = fakeddit_prompts.normalize_labels(df_valid[label_col].tolist(), task)

    if task == "2way" and task_config.get("positive_label") is not None:
        positive_label = fakeddit_prompts.normalize_labels([task_config["positive_label"]], task)[0]
        positive_idx = label_order.index(positive_label)
        result = evaluate_binary(y_true, predictions, p[:, positive_idx], positive_label, label_order=label_order)
    else:
        result = evaluate_single_label(y_true, predictions, label_order)

    result.update(scale_meta)
    result.update(prompt_meta)
    result.update(backend_summary(text_scorer))
    result.update(finish_audit(audit, vlm, df_valid, texts, y_true, label_order,
                               p, p_img, p_text, text_empty_mask, image_weight))

    log_dir = save_single_label_log(
        df_valid, y_true, predictions, p, label_order,
        id_col=id_col,
        text_col=text_col,
        model_name=vlm.model_name,
        dataset_name="fakeddit",
        task_name=task,
        output_dir=config["output"]["raw_predictions_dir"],
        prompt_version=task_config.get("prompt_version", "v1"),
        sim_image=p_img, sim_text=p_text,
        evaluation_audit=audit,
        extra_manifest={
            **scale_meta,
            **prompt_meta, "rendered_prompts": prompt_set,
            **backend_manifest(text_scorer),
            "batch_size": batch_size, "image_weight": image_weight,
            "max_samples": task_config.get("max_samples"),
            "min_text_length": task_config.get("min_text_length", 20),
            "logit_scale": getattr(vlm, "logit_scale", 100.0),
            "logit_bias": getattr(vlm, "logit_bias", 0.0),
        },
    )

    result["prediction_log_dir"] = log_dir
    return result


def run_crisismmd(vlm, task_config: dict, config: dict) -> dict:
    task = task_config["task"]
    print(f"\n{'=' * 70}\nDATASET: CrisisMMD | TASK: {task}\n{'=' * 70}")

    df = load_dataframe(task_config["data_path"], max_samples=task_config.get("max_samples"))

    id_col = task_config.get("id_col", "id")
    text_col = task_config.get("text_col", "tweet_text")
    label_col = shared.crisismmd_label_column(task_config)

    validate_columns(df, [id_col, text_col, label_col], "CrisisMMD")

    prompt_set = crisismmd_prompts.get_prompt_set(task)
    label_names = crisismmd_prompts.get_label_names(task)
    class_embeds, label_order = build_class_embeddings(vlm, prompt_set)

    image_paths = resolve_image_paths(df, task_config)
    images, valid_idx = load_images_safe(image_paths)

    if not images: raise RuntimeError("CrisisMMD: không load được ảnh hợp lệ nào.")

    df_valid = df.iloc[valid_idx].reset_index(drop=True)
    df_valid["_resolved_image_path"] = [image_paths[i] for i in valid_idx]

    audit = prepare_audit(df_valid, task_config, config, "crisismmd", task_config.get("task", "genres"),
                          id_col, text_col, label_col, prompt_set)

    texts = df_valid[text_col].fillna("").astype(str).tolist()

    batch_size = config.get("inference", {}).get("batch_size", 16)
    image_weight = get_image_weight(config, "crisismmd")
    image_weight, branch_meta = modality_settings(task_config, image_weight)
    text_scale, scale_meta = text_scale_settings(config, "crisismmd", vlm, task_config, image_weight)
    scale_meta.update(branch_meta)

    predictions, p, p_img, p_text, text_empty_mask = predict_single_label(
        vlm, images, texts, class_embeds, label_order,
        batch_size=batch_size, image_weight=image_weight, diagnostics=audit, text_logit_scale=text_scale,
    )
    df_valid["_text_was_empty"] = text_empty_mask

    y_true = normalize_single_labels(df_valid[label_col].tolist(), label_names, label_order)
    low_sample = crisismmd_prompts.LOW_SAMPLE_WARNING_CLASSES if task == "humanitarian" else None

    if task == "informativeness" and task_config.get("positive_label") is not None:
        positive_label = normalize_single_labels([task_config["positive_label"]], label_names, label_order)[0]
        positive_idx = label_order.index(positive_label)
        result = evaluate_binary(y_true, predictions, p[:, positive_idx], positive_label, label_order=label_order)
    else:
        result = evaluate_single_label(y_true, predictions, label_order, low_sample_classes=low_sample)

    result.update(scale_meta)
    result.update(finish_audit(audit, vlm, df_valid, texts, y_true, label_order,
                               p, p_img, p_text, text_empty_mask, image_weight))

    save_single_label_log(
        df_valid, y_true, predictions, p, label_order,
        id_col=id_col,
        text_col=text_col,
        model_name=vlm.model_name,
        dataset_name="crisismmd",
        task_name=task,
        output_dir=config["output"]["raw_predictions_dir"],
        prompt_version=task_config.get("prompt_version", "v1"),
        sim_image=p_img, sim_text=p_text,
        evaluation_audit=audit,
        extra_manifest={
            "batch_size": batch_size, "image_weight": image_weight,
            **scale_meta,
            "max_samples": task_config.get("max_samples"),
            "min_text_length": 0,
            "logit_scale": getattr(vlm, "logit_scale", 100.0),
            "logit_bias": getattr(vlm, "logit_bias", 0.0),
        },
    )

    return result


def run_mmimdb(vlm, task_config: dict, config: dict) -> dict:
    print(f"\n{'=' * 70}\nDATASET: MM-IMDb | TASK: multi-label\n{'=' * 70}")

    df = load_dataframe(task_config["data_path"], max_samples=task_config.get("max_samples"))

    id_col = task_config.get("id_col", "id")
    text_col = task_config.get("text_col", "plot")
    label_col = task_config.get("label_col", "genres")

    validate_columns(df, [id_col, text_col, label_col], "MM-IMDb")

    prompt_set = mmimdb_prompts.get_prompt_set()
    class_embeds, label_order = build_class_embeddings(vlm, prompt_set)

    image_paths = resolve_image_paths(df, task_config)
    images, valid_idx = load_images_safe(image_paths)

    if not images: raise RuntimeError("MM-IMDb: không load được ảnh hợp lệ nào.")

    df_valid = df.iloc[valid_idx].reset_index(drop=True)
    df_valid["_resolved_image_path"] = [image_paths[i] for i in valid_idx]

    audit = prepare_audit(df_valid, task_config, config, "mmimdb", task_config.get("task", "genres"),
                          id_col, text_col, label_col, prompt_set)

    texts = df_valid[text_col].fillna("").astype(str).tolist()

    threshold = resolve_override(
        task_config.get("threshold", 0.22),
        task_config.get("thresholds"),
        vlm.model_name,
    )
    fallback_top1 = task_config.get("fallback_top1", True)
    decision_options, decision_meta = multilabel_settings(task_config, threshold)
    batch_size = config.get("inference", {}).get("batch_size", 16)
    image_weight = get_image_weight(config, "mmimdb")
    image_weight, branch_meta = modality_settings(task_config, image_weight)
    text_scale, scale_meta = text_scale_settings(config, "mmimdb", vlm, task_config, image_weight)
    scale_meta.update(branch_meta)

    predictions, probs, p_img, p_text, used_fallback, text_empty_mask = predict_multi_label(
        vlm, images, texts, class_embeds, label_order,
        threshold=threshold,
        fallback_top1=fallback_top1,
        **decision_options, diagnostics=audit, text_logit_scale=text_scale,
        batch_size=batch_size,
        image_weight=image_weight,
    )
    df_valid["_text_was_empty"] = text_empty_mask
    df_valid["_used_top1_fallback"] = used_fallback

    y_true = [parse_multilabel(value) for value in df_valid[label_col].tolist()]
    result = evaluate_multi_label(y_true, predictions, label_order)
    result.update(decision_meta)
    result.update(scale_meta)

    result.update(finish_audit(audit, vlm, df_valid, texts, y_true, label_order,
                               probs, p_img, p_text, text_empty_mask, image_weight,
                               multilabel=True, decision_meta=decision_meta, fallback=fallback_top1))

    save_multi_label_log(
        df_valid, y_true, predictions, probs, label_order,
        id_col=id_col,
        text_col=text_col,
        model_name=vlm.model_name,
        dataset_name="mmimdb",
        task_name="multi_label",
        output_dir=config["output"]["raw_predictions_dir"],
        prompt_version=task_config.get("prompt_version", "v1"),
        sim_image=p_img, sim_text=p_text,
        evaluation_audit=audit,
        extra_manifest={
            "batch_size": batch_size, "image_weight": image_weight,
            "max_samples": task_config.get("max_samples"),
            "min_text_length": 0,
            "logit_scale": getattr(vlm, "logit_scale", 100.0),
            "logit_bias": getattr(vlm, "logit_bias", 0.0),
            **decision_meta, "fallback_top1": fallback_top1,
            **scale_meta,
        },
    )

    return result


DATASET_RUNNERS = {
    "fakeddit": run_fakeddit,
    "crisismmd": run_crisismmd,
    "mmimdb": run_mmimdb,
}


def main():
    config = load_config()
    all_results = []

    os.makedirs(config["output"]["results_dir"], exist_ok=True)
    os.makedirs(config["output"]["raw_predictions_dir"], exist_ok=True)

    for model_name in config["models"]:
        print(f"\n{'#' * 70}\nMODEL: {model_name}\n{'#' * 70}")

        vlm = None

        try:
            vlm = load_model(model_name, device=config.get("device"))

            for dataset_name, dataset_config in config["datasets"].items():
                if not dataset_config.get("enabled", False):
                    print(f"Bỏ qua '{dataset_name}' vì enabled=false")
                    continue

                runner = DATASET_RUNNERS.get(dataset_name)

                if runner is None:
                    print(f"Chưa có runner cho dataset '{dataset_name}'")
                    continue

                all_results.extend(shared.run_prompt_trials(
                    lambda active: runner(vlm, active, config), model_name, dataset_name, dataset_config))

        finally:
            if vlm is not None: vlm.unload()

    summary_df = pd.DataFrame(all_results)
    summary_paths = shared.save_summary_tables(all_results, config["output"]["summary_table"])

    print(f"\n{'=' * 70}")
    for summary_path in summary_paths:
        print(f"Đã lưu summary tại: {summary_path}")
    print(f"{'=' * 70}")
    print(shared.compact_summary(summary_df).to_string(index=False))


if __name__ == "__main__":
    main()
