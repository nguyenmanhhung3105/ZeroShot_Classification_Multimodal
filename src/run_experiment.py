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

from models.model_registry import load_model
from inference import build_class_embeddings, predict_single_label, predict_multi_label
from evaluate import evaluate_single_label, evaluate_binary, evaluate_multi_label
from logging_utils import save_single_label_log, save_multi_label_log
from prompts import fakeddit_prompts, crisismmd_prompts, mmimdb_prompts


def load_config(path="configs/experiment_config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f: return yaml.safe_load(f)


def load_dataframe(path: str) -> pd.DataFrame:
    """Tự nhận diện TSV / CSV / Parquet."""
    if not os.path.exists(path): raise FileNotFoundError(f"Không tìm thấy dataset: {path}")

    if path.endswith(".tsv"): return pd.read_csv(path, sep="\t")
    if path.endswith(".csv"): return pd.read_csv(path)
    if path.endswith(".parquet"): return pd.read_parquet(path)

    raise ValueError(f"Định dạng dataset chưa hỗ trợ: {path}")


def resolve_override(default_value, overrides: dict, key: str):
    """Tra cứu giá trị override theo key (model_name hoặc dataset_name).
    Dùng chung cho image_weight (theo dataset) và threshold (theo model),
    cùng một pattern nên gộp lại 1 hàm thay vì lặp code."""
    if not overrides:
        return default_value
    return overrides.get(key, default_value)


def resolve_image_paths(df: pd.DataFrame, task_config: dict) -> list:
    """
    Xác định path ảnh.

    Ưu tiên:
    1. image_col trong dataframe nếu tồn tại và path đó chạy được.
    2. image_dir + basename(image_col).
    3. image_dir + id + image_ext.
    """
    image_dir = task_config.get("image_dir", "")
    image_col = task_config.get("image_col", "image_path")
    id_col = task_config.get("id_col", "id")
    image_ext = task_config.get("image_ext", ".jpg")

    paths = []

    for _, row in df.iterrows():
        candidates = []

        if image_col in df.columns and pd.notna(row[image_col]):
            raw_path = str(row[image_col])
            candidates.append(raw_path)

            if image_dir:
                candidates.append(os.path.join(image_dir, raw_path))
                candidates.append(os.path.join(image_dir, os.path.basename(raw_path)))

        if id_col in df.columns and image_dir:
            sample_id = str(row[id_col])
            candidates.append(os.path.join(image_dir, sample_id))
            candidates.append(os.path.join(image_dir, f"{sample_id}{image_ext}"))

        resolved = next((path for path in candidates if os.path.exists(path)), candidates[0] if candidates else "")
        paths.append(resolved)

    return paths


def load_images_safe(image_paths: list) -> tuple:
    """Load ảnh và bỏ qua ảnh lỗi thay vì crash toàn bộ experiment."""
    images, valid_indices = [], []

    for i, path in enumerate(image_paths):
        try:
            with Image.open(path) as img: images.append(img.convert("RGB"))
            valid_indices.append(i)
        except Exception as e:
            print(f"[image] Bỏ qua ảnh lỗi: {path} -> {e}")

    return images, valid_indices


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
    """
    Map label số trong dataset sang tên class trong prompt.
    """
    if label_names is None: return values

    mapping = {}

    if isinstance(label_names, dict):
        mapping.update(label_names)
        mapping.update({str(k): v for k, v in label_names.items()})

    elif isinstance(label_names, (list, tuple)):
        mapping.update({i: label for i, label in enumerate(label_names)})
        mapping.update({str(i): label for i, label in enumerate(label_names)})

    normalized = []

    for value in values:
        if isinstance(value, float) and value.is_integer(): value = int(value)
        normalized.append(mapping.get(value, mapping.get(str(value), value)))

    return normalized


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

    df = load_dataframe(task_config["data_path"])

    id_col = task_config.get("id_col", "id")
    text_col = task_config.get("text_col", "clean_title")
    label_col = task_config.get("label_col_6way", "6_way_label") if task == "6way" else task_config.get("label_col_2way", "2_way_label")

    validate_columns(df, [id_col, text_col, label_col], "Fakeddit")

    prompt_set = fakeddit_prompts.get_prompt_set(task)
    label_names = fakeddit_prompts.get_label_names(task)
    class_embeds, label_order = build_class_embeddings(vlm, prompt_set)

    image_paths = resolve_image_paths(df, task_config)
    images, valid_idx = load_images_safe(image_paths)

    if not images: raise RuntimeError("Fakeddit: không load được ảnh hợp lệ nào.")

    df_valid = df.iloc[valid_idx].reset_index(drop=True)
    df_valid["_resolved_image_path"] = [image_paths[i] for i in valid_idx]

    texts = df_valid[text_col].fillna("").astype(str).tolist()

    batch_size = config.get("inference", {}).get("batch_size", 16)
    image_weight = get_image_weight(config, "fakeddit")

    predictions, p, p_img, p_text, text_empty_mask = predict_single_label(
        vlm, images, texts, class_embeds, label_order,
        batch_size=batch_size, image_weight=image_weight
    )
    # Gắn thẳng vào df_valid để lọt vào log mà không cần sửa logging_utils.py
    df_valid["_text_was_empty"] = text_empty_mask

    y_true = normalize_single_labels(df_valid[label_col].tolist(), label_names, label_order)

    if task == "2way" and task_config.get("positive_label"):
        positive_label = task_config["positive_label"]
        positive_idx = label_order.index(positive_label)
        result = evaluate_binary(y_true, predictions, p[:, positive_idx], positive_label)
    else:
        result = evaluate_single_label(y_true, predictions, label_order)

    save_single_label_log(
        df_valid, y_true, predictions, p, label_order,
        id_col=id_col,
        text_col=text_col,
        model_name=vlm.model_name,
        dataset_name="fakeddit",
        task_name=task,
        output_dir=config["output"]["raw_predictions_dir"],
        prompt_version=task_config.get("prompt_version", "v1"),
        # THÊM: sim_image/sim_text riêng để chẩn đoán prompt sau này.
        # CẦN cập nhật save_single_label_log để nhận và ghi 2 tham số này —
        # file logging_utils.py không nằm trong review nên chưa sửa được trực tiếp.
        sim_image=p_img,
        sim_text=p_text,
        extra_manifest={
            "batch_size": batch_size,
            "image_weight": image_weight,
            "logit_scale": getattr(vlm, "logit_scale", 100.0),
            "logit_bias": getattr(vlm, "logit_bias", 0.0),
        }
    )

    return result


def run_crisismmd(vlm, task_config: dict, config: dict) -> dict:
    task = task_config["task"]
    print(f"\n{'=' * 70}\nDATASET: CrisisMMD | TASK: {task}\n{'=' * 70}")

    df = load_dataframe(task_config["data_path"])

    id_col = task_config.get("id_col", "id")
    text_col = task_config.get("text_col", "tweet_text")
    label_col = task_config.get("label_col", "label")

    validate_columns(df, [id_col, text_col, label_col], "CrisisMMD")

    prompt_set = crisismmd_prompts.get_prompt_set(task)
    label_names = crisismmd_prompts.get_label_names(task)
    class_embeds, label_order = build_class_embeddings(vlm, prompt_set)

    image_paths = resolve_image_paths(df, task_config)
    images, valid_idx = load_images_safe(image_paths)

    if not images: raise RuntimeError("CrisisMMD: không load được ảnh hợp lệ nào.")

    df_valid = df.iloc[valid_idx].reset_index(drop=True)
    df_valid["_resolved_image_path"] = [image_paths[i] for i in valid_idx]

    texts = df_valid[text_col].fillna("").astype(str).tolist()

    batch_size = config.get("inference", {}).get("batch_size", 16)
    image_weight = get_image_weight(config, "crisismmd")

    predictions, p, p_img, p_text, text_empty_mask = predict_single_label(
        vlm, images, texts, class_embeds, label_order,
        batch_size=batch_size, image_weight=image_weight
    )
    df_valid["_text_was_empty"] = text_empty_mask

    y_true = normalize_single_labels(df_valid[label_col].tolist(), label_names, label_order)
    low_sample = crisismmd_prompts.LOW_SAMPLE_WARNING_CLASSES if task == "humanitarian" else None

    if task == "informativeness" and task_config.get("positive_label"):
        positive_label = task_config["positive_label"]
        positive_idx = label_order.index(positive_label)
        result = evaluate_binary(y_true, predictions, p[:, positive_idx], positive_label)
    else:
        result = evaluate_single_label(y_true, predictions, label_order, low_sample_classes=low_sample)

    save_single_label_log(
        df_valid, y_true, predictions, p, label_order,
        id_col=id_col,
        text_col=text_col,
        model_name=vlm.model_name,
        dataset_name="crisismmd",
        task_name=task,
        output_dir=config["output"]["raw_predictions_dir"],
        prompt_version=task_config.get("prompt_version", "v1"),
        sim_image=p_img,
        sim_text=p_text,
        extra_manifest={
            "batch_size": batch_size,
            "image_weight": image_weight,
            "logit_scale": getattr(vlm, "logit_scale", 100.0),
            "logit_bias": getattr(vlm, "logit_bias", 0.0),
        }
    )

    return result


def run_mmimdb(vlm, task_config: dict, config: dict) -> dict:
    print(f"\n{'=' * 70}\nDATASET: MM-IMDb | TASK: multi-label\n{'=' * 70}")

    df = load_dataframe(task_config["data_path"])

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

    texts = df_valid[text_col].fillna("").astype(str).tolist()

    threshold = resolve_override(
        task_config.get("threshold", 0.22),
        task_config.get("thresholds"),
        vlm.model_name,
    )
    fallback_top1 = task_config.get("fallback_top1", True)
    batch_size = config.get("inference", {}).get("batch_size", 16)
    image_weight = get_image_weight(config, "mmimdb")

    predictions, probs, p_img, p_text, used_fallback, text_empty_mask = predict_multi_label(
        vlm, images, texts, class_embeds, label_order,
        threshold=threshold,
        fallback_top1=fallback_top1,
        batch_size=batch_size,
        image_weight=image_weight,
    )
    df_valid["_text_was_empty"] = text_empty_mask
    df_valid["_used_top1_fallback"] = used_fallback

    y_true = [parse_multilabel(value) for value in df_valid[label_col].tolist()]
    result = evaluate_multi_label(y_true, predictions, label_order)

    save_multi_label_log(
        df_valid, y_true, predictions, probs, label_order,
        id_col=id_col,
        text_col=text_col,
        model_name=vlm.model_name,
        dataset_name="mmimdb",
        task_name="multi_label",
        output_dir=config["output"]["raw_predictions_dir"],
        prompt_version=task_config.get("prompt_version", "v1"),
        sim_image=p_img,
        sim_text=p_text,
        extra_manifest={
            "threshold": threshold,
            "fallback_top1": fallback_top1,
            "batch_size": batch_size,
            "image_weight": image_weight,
            "logit_scale": getattr(vlm, "logit_scale", 100.0),
            "logit_bias": getattr(vlm, "logit_bias", 0.0),
        }
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

                try:
                    metrics = runner(vlm, dataset_config, config)
                except Exception as e:
                    print(f"Lỗi {model_name} x {dataset_name}: {e}")
                    metrics = {"error": str(e)}

                row = {
                    "model": model_name,
                    "dataset": dataset_name,
                    "task": dataset_config.get("task", "-"),
                    "task_type": dataset_config.get("task_type", "-"),
                }

                row.update(metrics)
                all_results.append(row)

        finally:
            if vlm is not None: vlm.unload()

    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv(config["output"]["summary_table"], index=False)

    print(f"\n{'=' * 70}")
    print(f"Đã lưu summary tại: {config['output']['summary_table']}")
    print(f"{'=' * 70}")
    print(summary_df)


if __name__ == "__main__":
    main()