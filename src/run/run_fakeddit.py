"""
Chạy zero-shot classification cho Fakeddit (2way / 6way).

Tách riêng từ run_experiment.py để chạy độc lập từng dataset. Logic dùng
chung (đọc dataset, resolve ảnh, override, logging an toàn với signature
thật, vòng lặp model) nằm ở common.py — sửa ở đó sẽ áp dụng cho cả
run_crisismmd.py và run_mmimdb.py.

Chạy:
    python3 experiments/run_fakeddit.py
"""

import os
import sys

# PHẢI đứng trước các import bên dưới — xem giải thích trong run_crisismmd.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference import build_class_embeddings, predict_single_label
from evaluate import evaluate_single_label, evaluate_binary
from logging_utils import save_single_label_log
from prompts import fakeddit_prompts

from common import (
    load_config,
    load_dataframe,
    validate_columns,
    normalize_single_labels,
    coerce_labels_to_prompt_format,
    resolve_image_paths,
    load_images_safe,
    get_image_weight,
    call_with_supported_kwargs,
    run_all_models_for_dataset,
)


def run_fakeddit(vlm, task_config: dict, config: dict) -> dict:
    task = task_config["task"]
    print(f"\n{'=' * 70}\nDATASET: Fakeddit | TASK: {task}\n{'=' * 70}")

    df = load_dataframe(task_config["data_path"])

    id_col = task_config.get("id_col", "id")
    text_col = task_config.get("text_col", "clean_title")
    label_col = (
        task_config.get("label_col_6way", "6_way_label") if task == "6way"
        else task_config.get("label_col_2way", "2_way_label")
    )

    validate_columns(df, [id_col, text_col, label_col], "Fakeddit")
    
    # df = df[df[text_col].fillna("").astype(str).str.replace(" ", "", regex=False).str.len() > 10].reset_index(drop=True)     #sử dụng để dùng những mẫu có text lớn hơn
    df = df[df[text_col].fillna("").astype(str).str.replace(" ", "", regex=False).str.len() > 20].reset_index(drop=True)     #sử dụng để dùng những mẫu có text lớn hơn

    prompt_set = fakeddit_prompts.get_prompt_set(task)
    label_names = fakeddit_prompts.get_label_names(task)
    class_embeds, label_order = build_class_embeddings(vlm, prompt_set)

    image_paths = resolve_image_paths(df, task_config)
    images, valid_idx = load_images_safe(image_paths)

    if not images:
        raise RuntimeError("Fakeddit: không load được ảnh hợp lệ nào.")

    df_valid = df.iloc[valid_idx].reset_index(drop=True)
    df_valid["_resolved_image_path"] = [image_paths[i] for i in valid_idx]

    texts = df_valid[text_col].fillna("").astype(str).tolist()

    batch_size = config.get("inference", {}).get("batch_size", 16)
    image_weight = get_image_weight(config, "fakeddit")

    predictions, p, p_img, p_text, text_empty_mask = predict_single_label(
        vlm, images, texts, class_embeds, label_order,
        batch_size=batch_size, image_weight=image_weight,
    )
    
    # df_valid["_text_was_empty"] = text_empty_mask

    # y_true = normalize_single_labels(df_valid[label_col].tolist(), label_names, label_order)
    # # Fallback case/khoảng-trắng-insensitive, phòng khi nhãn thật khác định
    # # dạng với khoá prompt (đã gặp ở CrisisMMD) — vô hại nếu Fakeddit đã khớp sẵn.
    # y_true = coerce_labels_to_prompt_format(y_true, label_order)

    # positive_label = task_config.get("positive_label")
    
    df_valid["_text_was_empty"] = text_empty_mask

    if task == "2way":
        y_true = [int(x) for x in df_valid[label_col].tolist()]
    else:
        y_true = normalize_single_labels(df_valid[label_col].tolist(), label_names, label_order)
        y_true = coerce_labels_to_prompt_format(y_true, label_order)

    print(f"[DEBUG] label_order : {label_order}")
    print(f"[DEBUG] y_true       : {sorted(set(y_true))}")
    print(f"[DEBUG] predictions  : {sorted(set(predictions))}")

    positive_label = task_config.get("positive_label")
    
    if task == "2way" and positive_label and positive_label in label_order:
        positive_idx = label_order.index(positive_label)
        result = evaluate_binary(y_true, predictions, p[:, positive_idx], positive_label)
    else:
        result = evaluate_single_label(y_true, predictions, label_order)

    log_dir = call_with_supported_kwargs(
        save_single_label_log,
        df_valid, y_true, predictions, p, label_order,
        id_col=id_col,
        text_col=text_col,
        model_name=vlm.model_name,
        dataset_name="fakeddit",
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
        },
    )
    print(f"[LOG] {log_dir}")

    return result


def main():
    config = load_config()
    run_all_models_for_dataset("fakeddit", run_fakeddit, config)


if __name__ == "__main__":
    main()