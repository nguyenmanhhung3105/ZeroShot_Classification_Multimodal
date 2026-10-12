"""
Chạy MM-IMDb với 23 nhãn thể loại độc lập.

Chạy: python3 src/run/run_mmimdb.py --config configs/experiment_config.yaml
Quy tắc quyết định nhãn và chunking lấy từ cấu hình MM-IMDb.
"""

import os
import sys

# Thêm src trước import để chạy trực tiếp script từ thư mục gốc.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation_audit import prepare_audit, finish_audit
from text_scale import text_scale_settings
from pair_relation import relation_settings, attach_relation
from research.adapted_methods import research_embeddings
from inference import build_class_embeddings, predict_multi_label, multilabel_settings
from inference import prompt_aggregation_settings
from evaluate import evaluate_multi_label
from logging_utils import save_multi_label_log
from prompts import mmimdb_prompts

from run.common import (
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
    relation, relation_meta = relation_settings(task_config, "mmimdb", config)
    relation_scores = {}
    print(f"\n{'=' * 70}\nDATASET: MM-IMDb | TASK: multi-label\n{'=' * 70}")

    df = load_dataframe(task_config["data_path"], max_samples=task_config.get("max_samples"))

    id_col = task_config.get("id_col", "id")
    text_col = task_config.get("text_col", "plot")
    label_col = task_config.get("label_col", "genres")

    validate_columns(df, [id_col, text_col, label_col], "MM-IMDb")

    prompt_set = mmimdb_prompts.get_prompt_set()
    aggregation = prompt_aggregation_settings(task_config, "mmimdb")
    class_embeds, label_order = build_class_embeddings(vlm, prompt_set, aggregation=aggregation)
    image_classes, research_meta, audit_prompts = research_embeddings(vlm, task_config, prompt_set, label_order)

    image_paths = resolve_image_paths(df, task_config)
    images, valid_idx = load_images_safe(image_paths)

    if not images:
        raise RuntimeError("MM-IMDb: không load được ảnh hợp lệ nào.")

    df_valid = df.iloc[valid_idx].reset_index(drop=True)
    df_valid["_resolved_image_path"] = [image_paths[i] for i in valid_idx]

    audit = prepare_audit(df_valid, task_config, config, "mmimdb", task_config.get("task", "genres"),
                          id_col, text_col, label_col, audit_prompts,
                          scope=task_config.get("_audit_scope", "main"))

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
    text_scale, scale_meta = text_scale_settings(config, "mmimdb", vlm, task_config, image_weight)
    scale_meta["prompt_aggregation"] = aggregation

    predictions, probs, p_img, p_text, used_fallback, text_empty_mask = predict_multi_label(
        vlm, images, texts, class_embeds, label_order,
        threshold=threshold,
        fallback_top1=fallback_top1,
        batch_size=batch_size,
        image_weight=image_weight,
        **decision_options, diagnostics=audit, text_logit_scale=text_scale,
        image_class_embeds=image_classes,
        image_text_relation=relation, relation_scores=relation_scores,
    )
    df_valid["_text_was_empty"] = text_empty_mask
    df_valid["_used_top1_fallback"] = used_fallback
    attach_relation(df_valid, relation_scores)

    y_true = [parse_multilabel(value) for value in df_valid[label_col].tolist()]
    result = evaluate_multi_label(y_true, predictions, label_order)
    result.update(decision_meta)
    result.update(scale_meta)
    result.update(research_meta)
    result.update(relation_meta)

    result.update(finish_audit(audit, vlm, df_valid, texts, y_true, label_order,
                               probs, p_img, p_text, text_empty_mask, image_weight,
                               multilabel=True, decision_meta=decision_meta, fallback=fallback_top1))

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
        evaluation_audit=audit,
        extra_manifest={
            **relation_meta,
            **research_meta,
            "research_image_prompts": task_config.get("_research_image_prompts"),
            **scale_meta,
            **decision_meta,
            "fallback_top1": fallback_top1,
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
    run_all_models_for_dataset("mmimdb", run_mmimdb, config)


if __name__ == "__main__":
    main()
