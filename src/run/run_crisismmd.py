"""
Chạy CrisisMMD theo tác vụ informativeness hoặc humanitarian.

Chạy: python3 src/run/run_crisismmd.py --config configs/experiment_config.yaml
Tự chọn cột nhãn tương ứng theo task.
"""

import os
import sys

# Thêm src trước import để chạy trực tiếp script từ thư mục gốc.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation_audit import prepare_audit, finish_audit
from text_scale import text_scale_settings
from research.adapted_methods import research_embeddings
from inference import build_class_embeddings, predict_single_label
from inference import prompt_aggregation_settings
from evaluate import evaluate_single_label, evaluate_binary
from logging_utils import save_single_label_log
from prompts import crisismmd_prompts

from run.common import (
    load_config,
    crisismmd_label_column,
    load_dataframe,
    validate_columns,
    normalize_single_labels,
    resolve_image_paths,
    load_images_safe,
    get_image_weight,
    call_with_supported_kwargs,
    run_all_models_for_dataset,
)


def run_crisismmd(vlm, task_config: dict, config: dict) -> dict:
    task = task_config["task"]
    print(f"\n{'=' * 70}\nDATASET: CrisisMMD | TASK: {task}\n{'=' * 70}")

    df = load_dataframe(task_config["data_path"], max_samples=task_config.get("max_samples"))

    id_col = task_config.get("id_col", "id")
    text_col = task_config.get("text_col", "tweet_text")
    label_col = crisismmd_label_column(task_config)

    validate_columns(df, [id_col, text_col, label_col], "CrisisMMD")

    prompt_set = crisismmd_prompts.get_prompt_set(task)
    label_names = crisismmd_prompts.get_label_names(task)
    aggregation = prompt_aggregation_settings(task_config, "crisismmd")
    class_embeds, label_order = build_class_embeddings(vlm, prompt_set, aggregation=aggregation)
    image_classes, research_meta, audit_prompts = research_embeddings(vlm, task_config, prompt_set, label_order)

    image_paths = resolve_image_paths(df, task_config)
    images, valid_idx = load_images_safe(image_paths)

    if not images:
        raise RuntimeError("CrisisMMD: không load được ảnh hợp lệ nào.")

    df_valid = df.iloc[valid_idx].reset_index(drop=True)
    df_valid["_resolved_image_path"] = [image_paths[i] for i in valid_idx]

    audit = prepare_audit(df_valid, task_config, config, "crisismmd", task_config.get("task", "genres"),
                          id_col, text_col, label_col, audit_prompts,
                          scope=task_config.get("_audit_scope", "main"))

    texts = df_valid[text_col].fillna("").astype(str).tolist()

    batch_size = config.get("inference", {}).get("batch_size", 16)
    image_weight = get_image_weight(config, "crisismmd")
    text_scale, scale_meta = text_scale_settings(config, "crisismmd", vlm, task_config, image_weight)
    scale_meta["prompt_aggregation"] = aggregation

    predictions, p, p_img, p_text, text_empty_mask = predict_single_label(
        vlm, images, texts, class_embeds, label_order,
        batch_size=batch_size, image_weight=image_weight, diagnostics=audit, text_logit_scale=text_scale,
        image_class_embeds=image_classes,
    )
    df_valid["_text_was_empty"] = text_empty_mask

    y_true = normalize_single_labels(df_valid[label_col].tolist(), label_names, label_order)
    low_sample = crisismmd_prompts.LOW_SAMPLE_WARNING_CLASSES if task == "humanitarian" else None

    positive_label = task_config.get("positive_label")
    if task == "informativeness" and positive_label is not None:
        positive_label = normalize_single_labels([positive_label], label_names, label_order)[0]
        positive_idx = label_order.index(positive_label)
        result = evaluate_binary(y_true, predictions, p[:, positive_idx], positive_label, label_order=label_order)
    else:
        result = evaluate_single_label(y_true, predictions, label_order, low_sample_classes=low_sample)

    result.update(scale_meta)
    result.update(research_meta)
    result.update(finish_audit(audit, vlm, df_valid, texts, y_true, label_order,
                               p, p_img, p_text, text_empty_mask, image_weight))

    log_dir = call_with_supported_kwargs(
        save_single_label_log,
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
        evaluation_audit=audit,
        extra_manifest={
            **research_meta,
            "research_image_prompts": task_config.get("_research_image_prompts"),
            **scale_meta,
            "max_samples": task_config.get("max_samples"),
            "min_text_length": 0,
            "batch_size": batch_size,
            "image_weight": image_weight,
            "logit_scale": getattr(vlm, "logit_scale", 100.0),
            "logit_bias": getattr(vlm, "logit_bias", 0.0),
        },
    )
    print(f"[LOG] {log_dir}")
    result["prediction_log_dir"] = log_dir

    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment_config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    run_all_models_for_dataset("crisismmd", run_crisismmd, config)


if __name__ == "__main__":
    main()
