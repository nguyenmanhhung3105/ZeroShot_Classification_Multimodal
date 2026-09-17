"""
Chạy zero-shot classification cho MM-IMDb (multi-label genres).

Tách riêng từ run_experiment.py để chạy độc lập từng dataset. Logic dùng
chung nằm ở common.py.

Chạy:
    python3 experiments/run_mmimdb.py
"""

import os
import sys

# PHẢI đứng trước các import bên dưới — xem giải thích trong run_crisismmd.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference import build_class_embeddings, predict_multi_label
from evaluate import evaluate_multi_label
from logging_utils import save_multi_label_log
from prompts import mmimdb_prompts

from common import (
    load_config,
    load_dataframe,
    validate_columns,
    parse_multilabel,
    resolve_image_paths,
    load_images_safe,
    get_image_weight,
    resolve_override,
    call_with_supported_kwargs,
    run_all_models_for_dataset,
)


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

    if not images:
        raise RuntimeError("MM-IMDb: không load được ảnh hợp lệ nào.")

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

    log_dir = call_with_supported_kwargs(
        save_multi_label_log,
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
        },
    )
    print(f"[LOG] {log_dir}")

    return result


def main():
    config = load_config()
    run_all_models_for_dataset("mmimdb", run_mmimdb, config)


if __name__ == "__main__":
    main()