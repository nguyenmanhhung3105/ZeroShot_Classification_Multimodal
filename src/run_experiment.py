"""
Vòng lặp thí nghiệm chính: model x dataset.

QUAN TRỌNG (đúng nguyên tắc đã thống nhất):
- Mỗi model chỉ load 1 LẦN DUY NHẤT (vòng ngoài cùng).
- Với mỗi model, chạy hết tất cả dataset rồi mới unload, load model tiếp theo.
- Không có bước train/fine-tune nào -> không có "xung đột" giữa các lần gọi.
"""

import os
import yaml
import pandas as pd
from PIL import Image

from models.model_registry import load_model
from inference import build_class_embeddings, predict_single_label, predict_multi_label
from evaluate import evaluate_single_label, evaluate_multi_label

from prompts import fakeddit_prompts, crisismmd_prompts, mmimdb_prompts


def load_config(path="configs/experiment_config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_images_safe(image_paths: list) -> tuple:
    """
    Load ảnh, bỏ qua ảnh lỗi/không tồn tại thay vì crash toàn bộ pipeline.
    Trả về (list ảnh hợp lệ, list index tương ứng trong dataframe gốc).
    """
    images, valid_indices = [], []
    for i, path in enumerate(image_paths):
        try:
            img = Image.open(path).convert("RGB")
            images.append(img)
            valid_indices.append(i)
        except Exception as e:
            print(f" Bỏ qua ảnh lỗi tại {path}: {e}")
    return images, valid_indices


def run_fakeddit(vlm, task_config: dict) -> dict:
    print(f"\n{'='*60}\nDATASET: Fakeddit (task={task_config['task']})\n{'='*60}")

    if not os.path.exists(task_config["data_path"]):
        print(f" Không tìm thấy file {task_config['data_path']}, bỏ qua dataset này")
        return {}

    df = pd.read_parquet(task_config["data_path"])
    label_col = "6_way_label" if task_config["task"] == "6way" else "2_way_label"

    prompt_set = fakeddit_prompts.get_prompt_set(task_config["task"])
    label_names = fakeddit_prompts.get_label_names(task_config["task"])
    class_embeds, label_order = build_class_embeddings(vlm, prompt_set)

    images, valid_idx = load_images_safe(df["image_path"].tolist() if "image_path" in df.columns
                                          else [f"data/raw/fakeddit/images/{i}.jpg" for i in df["id"]])
    df_valid = df.iloc[valid_idx].reset_index(drop=True)

    predictions, sims = predict_single_label(vlm, images, class_embeds, label_order)
    y_true = df_valid[label_col].tolist()

    result = evaluate_single_label(y_true, predictions, label_order)
    return result


def run_crisismmd(vlm, task_config: dict) -> dict:
    print(f"\n{'='*60}\nDATASET: CrisisMMD (task={task_config['task']})\n{'='*60}")

    if not os.path.exists(task_config["data_path"]):
        print(f" Không tìm thấy file {task_config['data_path']}, bỏ qua dataset này")
        return {}

    df = pd.read_parquet(task_config["data_path"])

    prompt_set = crisismmd_prompts.get_prompt_set(task_config["task"])
    label_names = crisismmd_prompts.get_label_names(task_config["task"])
    class_embeds, label_order = build_class_embeddings(vlm, prompt_set)

    images, valid_idx = load_images_safe(df["image_path"].tolist())
    df_valid = df.iloc[valid_idx].reset_index(drop=True)

    predictions, sims = predict_single_label(vlm, images, class_embeds, label_order)
    y_true = df_valid["label"].tolist()

    low_sample = crisismmd_prompts.LOW_SAMPLE_WARNING_CLASSES if task_config["task"] == "humanitarian" else None
    result = evaluate_single_label(y_true, predictions, label_order, low_sample_classes=low_sample)
    return result


def run_mmimdb(vlm, task_config: dict) -> dict:
    print(f"\n{'='*60}\nDATASET: MM-IMDb (multi-label)\n{'='*60}")

    if not os.path.exists(task_config["data_path"]):
        print(f" Không tìm thấy file {task_config['data_path']}, bỏ qua dataset này")
        return {}

    df = pd.read_parquet(task_config["data_path"])

    prompt_set = mmimdb_prompts.get_prompt_set()
    class_embeds, label_order = build_class_embeddings(vlm, prompt_set)

    images, valid_idx = load_images_safe(df["poster_path"].tolist())
    df_valid = df.iloc[valid_idx].reset_index(drop=True)

    predictions, sims = predict_multi_label(vlm, images, class_embeds, label_order,
                                             threshold=task_config.get("threshold", 0.22))
    y_true = df_valid["genres"].tolist()  # kỳ vọng mỗi phần tử là list, ví dụ ["Comedy","Family"]

    result = evaluate_multi_label(y_true, predictions, label_order)
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

    # ============================================================
    # VÒNG NGOÀI CÙNG: MODEL — chỉ load 1 lần, chạy hết dataset rồi mới đổi model
    # ============================================================
    for model_name in config["models"]:
        vlm = load_model(model_name)

        for dataset_name, dataset_config in config["datasets"].items():
            if not dataset_config.get("enabled", False):
                print(f"⏭  Bỏ qua dataset '{dataset_name}' (enabled=false trong config)")
                continue

            runner = DATASET_RUNNERS.get(dataset_name)
            if runner is None:
                print(f" Chưa có runner cho dataset '{dataset_name}', bỏ qua")
                continue

            try:
                metrics = runner(vlm, dataset_config)
            except Exception as e:
                print(f" Lỗi khi chạy {model_name} x {dataset_name}: {e}")
                metrics = {"error": str(e)}

            row = {"model": model_name, "dataset": dataset_name,
                   "task": dataset_config.get("task", "-")}
            row.update(metrics)
            all_results.append(row)

        vlm.unload()  # giải phóng GPU trước khi load model tiếp theo

    # ============================================================
    # LƯU BẢNG TỔNG HỢP CUỐI CÙNG
    # ============================================================
    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv(config["output"]["summary_table"], index=False)
    print(f"\n Đã lưu bảng tổng hợp tại: {config['output']['summary_table']}")
    print(summary_df)


if __name__ == "__main__":
    main()
