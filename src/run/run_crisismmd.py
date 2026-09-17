"""
Chạy zero-shot classification cho CrisisMMD (informativeness / humanitarian).

Tách riêng từ run_experiment.py để chạy độc lập từng dataset. Logic dùng
chung nằm ở common.py.

Chạy:
    python3 experiments/run_crisismmd.py
"""

import os
import sys

# PHẢI đứng trước các import bên dưới (inference/evaluate/logging_utils/prompts
# nằm ở src/, còn file này nằm trong src/run/ hoặc src/experiments/) — nếu đặt
# sau, các import bên dưới sẽ fail trước khi sys.path kịp được sửa.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference import build_class_embeddings, predict_single_label
from evaluate import evaluate_single_label, evaluate_binary
from logging_utils import save_single_label_log
from prompts import crisismmd_prompts

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


def run_crisismmd(vlm, task_config: dict, config: dict) -> dict:
    task = task_config["task"]
    print(f"\n{'=' * 70}\nDATASET: CrisisMMD | TASK: {task}\n{'=' * 70}")

    df = load_dataframe(task_config["data_path"])

    id_col = task_config.get("id_col", "id")
    text_col = task_config.get("text_col", "tweet_text")
    # SỬA: tự chọn cột nhãn theo task nếu yaml không set "label_col" tường
    # minh. Bản gốc default cứng "label" cho mọi task -> đổi task phải nhớ
    # sửa label_col bằng tay trong yaml, dễ quên và sai âm thầm (không lỗi,
    # chỉ đọc nhầm cột).
    default_label_col = "label_informative" if task == "informativeness" else "label_humanitarian"
    label_col = task_config.get("label_col", default_label_col)

    validate_columns(df, [id_col, text_col, label_col], "CrisisMMD")

    prompt_set = crisismmd_prompts.get_prompt_set(task)
    label_names = crisismmd_prompts.get_label_names(task)
    class_embeds, label_order = build_class_embeddings(vlm, prompt_set)

    image_paths = resolve_image_paths(df, task_config)
    images, valid_idx = load_images_safe(image_paths)

    if not images:
        raise RuntimeError("CrisisMMD: không load được ảnh hợp lệ nào.")

    df_valid = df.iloc[valid_idx].reset_index(drop=True)
    df_valid["_resolved_image_path"] = [image_paths[i] for i in valid_idx]

    texts = df_valid[text_col].fillna("").astype(str).tolist()

    batch_size = config.get("inference", {}).get("batch_size", 16)
    image_weight = get_image_weight(config, "crisismmd")

    predictions, p, p_img, p_text, text_empty_mask = predict_single_label(
        vlm, images, texts, class_embeds, label_order,
        batch_size=batch_size, image_weight=image_weight,
    )
    df_valid["_text_was_empty"] = text_empty_mask

    y_true = normalize_single_labels(df_valid[label_col].tolist(), label_names, label_order)
    # Fallback case/khoảng-trắng-insensitive: dataset thật ghi "Informative" /
    # "Not Informative" trong khi prompt dùng "informative" / "not_informative",
    # và LABEL_NAMES_INFORMATIVENESS trong crisismmd_prompts.py map sai chiều
    # nên normalize_single_labels không tự xử lý được -> cần lớp fallback này.
    # LƯU Ý: với task "humanitarian", nếu LABEL_NAMES_HUMANITARIAN cũng map
    # sai chiều VÀ nhãn thật khác cả từ ngữ (không chỉ hoa/thường) so với khoá
    # prompt, fallback này sẽ KHÔNG cứu được — phải sửa đúng chiều dict đó.
    y_true = coerce_labels_to_prompt_format(y_true, label_order)

    low_sample = crisismmd_prompts.LOW_SAMPLE_WARNING_CLASSES if task == "humanitarian" else None

    positive_label = task_config.get("positive_label")
    if task == "informativeness" and positive_label and positive_label in label_order:
        positive_idx = label_order.index(positive_label)
        result = evaluate_binary(y_true, predictions, p[:, positive_idx], positive_label)
    else:
        result = evaluate_single_label(y_true, predictions, label_order, low_sample_classes=low_sample)

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
    run_all_models_for_dataset("crisismmd", run_crisismmd, config)


if __name__ == "__main__":
    main()