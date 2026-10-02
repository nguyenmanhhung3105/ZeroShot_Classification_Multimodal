"""
Demo zero-shot classification cho:
- Fakeddit: 2way / 6way
- CrisisMMD: informativeness / humanitarian
- MMIMDb: multi-label genres

Chạy:
    python3 src/demo.py

Ghi chú:
Bản này được đồng bộ lại với file main experiment loop (model x dataset):
- Đọc dataset qua `load_dataframe` (tự nhận diện TSV/CSV/Parquet) thay vì
  hardcode `pd.read_csv(..., sep="\\t")`.
- Resolve/ load ảnh bằng `resolve_image_paths` + `load_images_safe` (nhiều
  candidate path, bỏ qua ảnh lỗi) thay vì hàm `load_images` cũ chỉ thử 2 path.
- Không tự re-implement encode/predict nữa — dùng thẳng `build_class_embeddings`,
  `predict_single_label`, `predict_multi_label` từ module `inference` (module
  dùng chung với pipeline chính) để tránh 2 nơi tính fusion image/text lệch nhau.
- Lấy id_col/text_col/label_col từ `task_config` trong YAML (có default),
  thay vì dict `DATASETS` hardcode cứng trong code — nếu yaml đổi tên cột thì
  demo tự theo, không còn lệch với pipeline chính.
- Map nhãn số -> tên class bằng `normalize_single_labels` (bug cũ: so sánh
  thẳng "0"/"1" với tên class dạng "true"/"fake" nên accuracy luôn sai).
- Áp dụng `resolve_override`/`get_image_weight` cho threshold & image_weight
  theo từng model/dataset, giống pipeline chính, thay vì 1 giá trị global.
- Dùng `evaluate_single_label` / `evaluate_binary` / `evaluate_multi_label`
  để in ra bộ metric giống hệt báo cáo của pipeline chính (không chỉ accuracy
  thô).
- Log cả nhánh multi-label (MM-IMDb) bằng `save_multi_label_log` — bản cũ chỉ
  log cho single-label.

Lưu ý: các hàm tiện ích (load_dataframe, resolve_image_paths, ...) được copy
lại nguyên logic từ file main experiment loop để demo chạy độc lập, không phụ
thuộc vào tên file/module đó (không rõ từ ngữ cảnh). Nếu trong repo các hàm
này đã nằm trong một module dùng chung (vd. `data_utils.py`), nên xoá phần copy
ở đây và import trực tiếp để tránh 2 bản logic trôi dạt (drift) theo thời gian
như đã từng xảy ra.
"""

import os
import sys
import ast
import json
import math
import inspect
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run import common as shared
from models.model_registry import load_model
from inference import build_class_embeddings, predict_single_label, predict_multi_label, multilabel_settings
from inference import prompt_aggregation_settings
from evaluation_audit import prepare_audit, finish_audit
from text_scale import text_scale_settings
from evaluate import evaluate_single_label, evaluate_binary, evaluate_multi_label
from logging_utils import save_single_label_log, save_multi_label_log
from prompts import fakeddit_prompts, crisismmd_prompts, mmimdb_prompts


# ============================================================
# CONFIG
# ============================================================

CONFIG_PATH = "configs/experiment_config.yaml"  # SỬA: khớp với main.py (trước là "config/..." — sai tên thư mục)
N_SAMPLES = 20
BATCH_SIZE = 4
LOG_DIR = "results/demo_logs"
OUTPUT_DIR = "results/demo"
PROMPT_VERSION = "v1"
IMAGE_WEIGHT = 0.5  # fallback mặc định nếu config không set inference.image_weight

PROMPT_MODULES = {
    "fakeddit": fakeddit_prompts,
    "crisismmd": crisismmd_prompts,
    "mmimdb": mmimdb_prompts,
}


# ============================================================
# HÀM TIỆN ÍCH DÙNG CHUNG (đồng bộ với main.py)
# ============================================================

def load_config(path: str = CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_dataframe(path: str, max_samples=None, text_col=None, min_text_length=0) -> pd.DataFrame:
    return shared.load_dataframe(path, max_samples, text_col, min_text_length)


def resolve_override(default_value, overrides: dict, key: str):
    """Tra cứu override theo key (model_name hoặc dataset_name)."""
    if not overrides:
        return default_value
    return overrides.get(key, default_value)


def get_image_weight(config: dict, dataset_name: str) -> float:
    inference_cfg = config.get("inference", {})
    return resolve_override(
        inference_cfg.get("image_weight", IMAGE_WEIGHT),
        inference_cfg.get("image_weight_overrides"),
        dataset_name,
    )


def validate_columns(df: pd.DataFrame, required_columns: list, dataset_name: str) -> None:
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(
            f"{dataset_name}: thiếu các cột {missing}. "
            f"Các cột hiện có: {df.columns.tolist()}"
        )


def parse_multilabel(value) -> list:
    """Chuyển genres từ TSV thành list label."""
    if isinstance(value, (list, tuple, set)):
        return list(value)
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []

    text = str(value).strip()
    if not text:
        return []

    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
            if isinstance(parsed, (list, tuple, set)):
                return [str(x).strip() for x in parsed]
        except Exception:
            pass

    if "|" in text:
        return [x.strip() for x in text.split("|") if x.strip()]
    return [x.strip() for x in text.split(",") if x.strip()]


def normalize_single_labels(values: list, label_names, label_order: list) -> list:
    return shared.normalize_single_labels(values, label_names, label_order)


def resolve_image_paths(df: pd.DataFrame, task_config: dict) -> list:
    return shared.resolve_image_paths(df, task_config)


def _canonical_label_lookup(label_order: list) -> dict:
    """canon(nhãn) -> nhãn gốc trong label_order. canon = lowercase, bỏ
    khoảng trắng/gạch ngang thừa -> so khớp được dù dataset ghi
    'Not Informative' còn prompt ghi 'not_informative'."""
    def canon(s):
        return str(s).strip().lower().replace(" ", "_").replace("-", "_")
    return {canon(label): label for label in label_order}


def coerce_labels_to_prompt_format(values: list, label_order: list) -> list:
    """Fallback chuẩn hoá case/space-insensitive: nếu normalize_single_labels
    (map theo get_label_names) không phủ hết biến thể viết của dataset thật
    (vd. 'Informative' vs 'informative'), vẫn cố khớp bằng canonical form
    trước khi đành chịu giữ nguyên giá trị gốc."""
    lookup = _canonical_label_lookup(label_order)

    def canon(s):
        return str(s).strip().lower().replace(" ", "_").replace("-", "_")

    return [lookup.get(canon(v), v) for v in values]


def call_with_supported_kwargs(func, *args, **kwargs):
    """Chỉ truyền các kwargs mà `func` THẬT SỰ khai báo — tránh crash kiểu
    'unexpected keyword argument' khi logging_utils.py (không nằm trong review,
    theo đúng ghi chú ở file main) chưa/không hỗ trợ đúng những gì demo.py và
    main.py giả định (đã gặp với sim_image/sim_text, rồi extra_manifest).
    Kwarg nào bị bỏ sẽ được cảnh báo ra console để biết mà bổ sung logging_utils.py
    sau, thay vì âm thầm mất dữ liệu."""
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return func(*args, **kwargs)

    params = sig.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return func(*args, **kwargs)  # hàm có **kwargs -> nhận hết được

    accepted = {k: v for k, v in kwargs.items() if k in params}
    dropped = sorted(set(kwargs) - set(accepted))
    if dropped:
        print(f"[logging] {func.__name__} chưa hỗ trợ kwargs {dropped} -> bỏ qua, không ghi vào log.")

    return func(*args, **accepted)


def load_images_safe(image_paths: list) -> tuple:
    return shared.load_images_safe(image_paths)


def visualize(df, images, text_col, true_col, dataset_name, task, model_name):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    n_cols = min(5, len(df))
    n_rows = math.ceil(len(df) / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 5 * n_rows), squeeze=False)
    axes = axes.flatten()

    for ax, (_, row), img in zip(axes, df.iterrows(), images):
        text = str(row[text_col])
        if len(text) > 60:
            text = text[:60] + "..."

        ax.imshow(img)
        if hasattr(images, "paths"):
            img.close()
        ax.axis("off")
        ax.set_title(f"{text}\nTrue: {row[true_col]}\nPred: {row['predicted']}", fontsize=9)

    for ax in axes[len(df):]:
        ax.axis("off")

    plt.suptitle(f"{dataset_name} — {task} — {model_name}")
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, f"demo_{dataset_name}_{task}_{model_name}.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[OUTPUT] {output_path}")


# ============================================================
# RUN DATASET
# ============================================================

def run_dataset(vlm, model_name: str, dataset_name: str, task_config: dict, image_weight: float, config=None) -> dict:
    prompt_module = PROMPT_MODULES[dataset_name]
    task_type = task_config["task_type"]
    task = task_config.get("task", "genres")

    print("\n" + "=" * 80)
    print(f"{dataset_name.upper()} | {task} | {model_name} | image_weight={image_weight}")
    print("=" * 80)

    # df = load_dataframe(task_config["data_path"])
    # df = df.head(N_SAMPLES).copy()

    # id_col = task_config.get("id_col", "id")

    # # --- cột theo từng dataset: lấy từ task_config (yaml) trước, default sau,
    # #     giống hệt pattern trong main.py, thay vì dict DATASETS hardcode cứng ---
    # if dataset_name == "fakeddit":
    #     text_col = task_config.get("text_col", "clean_title")
    #     label_col = (
    #         task_config.get("label_col_6way", "6_way_label") if task == "6way"
    #         else task_config.get("label_col_2way", "2_way_label")
    #     )
    #     label_names = prompt_module.get_label_names(task)
    # elif dataset_name == "crisismmd":
    #     text_col = task_config.get("text_col", "tweet_text")
    #     label_col = task_config.get("label_col", "label")
    #     label_names = prompt_module.get_label_names(task)
    # else:  # mmimdb
    #     text_col = task_config.get("text_col", "plot")
    #     label_col = task_config.get("label_col", "genres")
    #     label_names = None

    # validate_columns(df, [id_col, text_col, label_col], dataset_name)
    
    limit = min(N_SAMPLES, task_config.get("max_samples") or N_SAMPLES)
    df = load_dataframe(task_config["data_path"], max_samples=limit,
                        text_col=task_config.get("text_col", "clean_title"),
                        min_text_length=task_config.get("min_text_length", 20) if dataset_name == "fakeddit" else 0)

    id_col = task_config.get("id_col", "id")

    if dataset_name == "fakeddit":
        text_col = task_config.get("text_col", "clean_title")
        label_col = (
            task_config.get("label_col_6way", "6_way_label") if task == "6way"
            else task_config.get("label_col_2way", "2_way_label")
        )
        label_names = prompt_module.get_label_names(task)

    elif dataset_name == "crisismmd":
        text_col = task_config.get("text_col", "tweet_text")
        label_col = shared.crisismmd_label_column(task_config)
        label_names = prompt_module.get_label_names(task)

    else:  # mmimdb
        text_col = task_config.get("text_col", "plot")
        label_col = task_config.get("label_col", "genres")
        label_names = None

    validate_columns(df, [id_col, text_col, label_col], dataset_name)

    # Lọc text và giới hạn mẫu được áp dụng ngay trong lúc đọc.

#-------------------------------------------------------------------------

    prompt_meta = {}
    if dataset_name == "fakeddit":
        prompt_set = prompt_module.get_prompt_set(task, task_config.get("prompt_variant", "current"))
        prompt_meta = prompt_module.prompt_metadata(task_config, prompt_set)
    else:
        try:
            prompt_set = prompt_module.get_prompt_set(task)
        except TypeError:
            prompt_set = prompt_module.get_prompt_set()

    aggregation = prompt_aggregation_settings(task_config, dataset_name)
    if dataset_name == "fakeddit":
        prompt_meta["prompt_aggregation"] = aggregation
    class_embeds, label_order = build_class_embeddings(vlm, prompt_set, aggregation=aggregation)

    # --- ảnh: image_dir mặc định suy ra từ data_path nếu yaml không set,
    #     giữ tiện lợi của bản demo cũ nhưng resolve bằng logic robust hơn ---
    image_dir = task_config.get("image_dir") or os.path.join(
        os.path.dirname(task_config["data_path"]), "images"
    )
    resolved_task_config = {**task_config, "image_dir": image_dir, "id_col": id_col}
    image_paths = resolve_image_paths(df, resolved_task_config)
    images, valid_idx = load_images_safe(image_paths)

    if not images:
        print("[WARNING] Không có ảnh hợp lệ.")
        return {"error": "no_valid_images"}

    df_valid = df.iloc[valid_idx].reset_index(drop=True)
    df_valid["_resolved_image_path"] = [image_paths[i] for i in valid_idx]

    texts = df_valid[text_col].fillna("").astype(str).tolist()
    audit = prepare_audit(df_valid, task_config, config or {}, dataset_name, task,
                          id_col, text_col, label_col, prompt_set, scope="demo", limit=limit)

    print(f"[DATA] Samples : {len(df_valid)}")
    print(f"[MODEL] Classes: {label_order}")
    text_scale, scale_meta = text_scale_settings(config, dataset_name, vlm, task_config, image_weight)
    scale_meta["prompt_aggregation"] = aggregation

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ========================================================
    # SINGLE LABEL — Fakeddit / CrisisMMD
    # ========================================================
    if task_type == "single_label":
        predictions, p, p_img, p_text, text_empty_mask = predict_single_label(
            vlm, images, texts, class_embeds, label_order,
            batch_size=BATCH_SIZE, image_weight=image_weight, diagnostics=audit, text_logit_scale=text_scale,
        )
        df_valid["_text_was_empty"] = text_empty_mask
        df_valid["predicted"] = predictions

        if dataset_name == "fakeddit":
            y_true = fakeddit_prompts.normalize_labels(df_valid[label_col].tolist(), task)
        else:
            y_true = normalize_single_labels(df_valid[label_col].tolist(), label_names, label_order)
            y_true = coerce_labels_to_prompt_format(y_true, label_order)

        print(f"[DEBUG] label_order (từ prompt) : {label_order}")
        print(f"[DEBUG] y_true thật (sau map)   : {sorted(set(map(str, y_true)))}")
        print(f"[DEBUG] predictions              : {sorted(set(map(str, predictions)))}")

        positive_label = task_config.get("positive_label")
        if positive_label is not None:
            positive_label = normalize_single_labels([positive_label], label_names, label_order)[0]
            positive_idx = label_order.index(positive_label)
            result = evaluate_binary(y_true, predictions, p[:, positive_idx], positive_label, label_order=label_order)
        else:
            low_sample = (
                crisismmd_prompts.LOW_SAMPLE_WARNING_CLASSES
                if dataset_name == "crisismmd" and task == "humanitarian"
                else None
            )
            result = evaluate_single_label(y_true, predictions, label_order, low_sample_classes=low_sample)

        result.update(scale_meta)
        result.update(prompt_meta)
        result.update(finish_audit(audit, vlm, df_valid, texts, y_true, label_order,
                                   p, p_img, p_text, text_empty_mask, image_weight))
        log_dir = call_with_supported_kwargs(
            save_single_label_log,
            df_valid, y_true, predictions, p, label_order,
            id_col=id_col,
            text_col=text_col,
            model_name=model_name,
            dataset_name=dataset_name,
            task_name=task,
            output_dir=LOG_DIR,
            prompt_version=PROMPT_VERSION,
            sim_image=p_img,
            sim_text=p_text,
            evaluation_audit=audit,
            extra_manifest={
                **scale_meta,
                **prompt_meta,
                **({"rendered_prompts": prompt_set} if dataset_name == "fakeddit" else {}),
                "max_samples": limit,
                "min_text_length": task_config.get("min_text_length", 20) if dataset_name == "fakeddit" else 0,
                "batch_size": BATCH_SIZE,
                "image_weight": image_weight,
                "logit_scale": getattr(vlm, "logit_scale", 100.0),
                "logit_bias": getattr(vlm, "logit_bias", 0.0),
            },
        )

        if dataset_name == "fakeddit":
            result["prediction_log_dir"] = log_dir
        for i, row in df_valid.iterrows():
            print(f"\n--- Mẫu {i + 1} | {row[id_col]} ---")
            print(f"Text : {row[text_col]}")
            print(f"True : {y_true[i]}")
            print(f"Pred : {row['predicted']}")

        print(f"\n[EVAL] {result}")
        print(f"[LOG] {log_dir}")

        visualize(df_valid, images, text_col, label_col, dataset_name, task, model_name)
        return result

    # ========================================================
    # MULTI LABEL — MM-IMDb
    # ========================================================
    threshold = resolve_override(
        task_config.get("threshold", 0.22),
        task_config.get("thresholds"),
        model_name,
    )
    fallback_top1 = task_config.get("fallback_top1", True)
    decision_options, decision_meta = multilabel_settings(task_config, threshold)

    predictions, probs, p_img, p_text, used_fallback, text_empty_mask = predict_multi_label(
        vlm, images, texts, class_embeds, label_order,
        threshold=threshold,
        fallback_top1=fallback_top1,
        **decision_options, diagnostics=audit, text_logit_scale=text_scale,
        batch_size=BATCH_SIZE,
        image_weight=image_weight,
    )
    df_valid["_text_was_empty"] = text_empty_mask
    df_valid["_used_top1_fallback"] = used_fallback
    df_valid["predicted"] = [str(x) for x in predictions]

    y_true = [parse_multilabel(value) for value in df_valid[label_col].tolist()]
    result = evaluate_multi_label(y_true, predictions, label_order)
    result.update(decision_meta)
    result.update(scale_meta)
    result.update(finish_audit(audit, vlm, df_valid, texts, y_true, label_order,
                               probs, p_img, p_text, text_empty_mask, image_weight,
                               multilabel=True, decision_meta=decision_meta, fallback=fallback_top1))

    log_dir = call_with_supported_kwargs(
        save_multi_label_log,
        df_valid, y_true, predictions, probs, label_order,
        id_col=id_col,
        text_col=text_col,
        model_name=model_name,
        dataset_name=dataset_name,
        task_name="multi_label",
        output_dir=LOG_DIR,
        prompt_version=PROMPT_VERSION,
        sim_image=p_img,
        sim_text=p_text,
        evaluation_audit=audit,
        extra_manifest={
            **scale_meta,
            "max_samples": limit,
            **decision_meta,
            "fallback_top1": fallback_top1,
            "batch_size": BATCH_SIZE,
            "image_weight": image_weight,
            "logit_scale": getattr(vlm, "logit_scale", 100.0),
            "logit_bias": getattr(vlm, "logit_bias", 0.0),
        },
    )

    for i, row in df_valid.iterrows():
        print(f"\n--- Mẫu {i + 1} | {row[id_col]} ---")
        print(f"Plot : {str(row[text_col])[:150]}")
        print(f"True : {y_true[i]}")
        print(f"Pred : {predictions[i]}")

    print(f"\n[EVAL] {result}")
    print(f"[LOG] {log_dir}")

    visualize(df_valid, images, text_col, label_col, dataset_name, task, model_name)
    return result


# ============================================================
# MAIN
# ============================================================

def main():
    global N_SAMPLES, BATCH_SIZE, LOG_DIR, OUTPUT_DIR, PROMPT_VERSION
    config = load_config()
    demo_config = config.get("demo", {})
    N_SAMPLES = demo_config.get("n_samples", 20)
    BATCH_SIZE = demo_config.get("batch_size", 4)
    LOG_DIR = demo_config.get("log_dir", "results/demo_logs")
    OUTPUT_DIR = demo_config.get("visualization_dir", "results/demo")
    PROMPT_VERSION = demo_config.get("prompt_version", "v1")

    models = config["models"]
    datasets = {name: cfg for name, cfg in config["datasets"].items() if cfg.get("enabled", False)}

    print(f"[CONFIG] Models   : {models}")
    print(f"[CONFIG] Datasets : {list(datasets.keys())}")

    all_results = []

    for model_name in models:
        print(f"\n[MODEL] Loading: {model_name}")

        vlm = None
        try:
            vlm = load_model(model_name, device=config.get("device"))

            for dataset_name, task_config in datasets.items():
                if dataset_name not in PROMPT_MODULES:
                    print(f"[WARNING] Chưa hỗ trợ demo cho dataset '{dataset_name}'")
                    continue

                image_weight = get_image_weight(config, dataset_name)

                all_results.extend(shared.run_prompt_trials(
                    lambda active: run_dataset(vlm, model_name, dataset_name, active, image_weight, config=config),
                    model_name, dataset_name, task_config))
        finally:
            if vlm is not None:
                vlm.unload()

    if all_results:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        summary_df = pd.DataFrame(all_results)
        summary_paths = shared.save_summary_tables(
            all_results, os.path.join(OUTPUT_DIR, "demo_summary.tsv"))

        print("\n" + "=" * 80)
        for summary_path in summary_paths:
            print(f"[OUTPUT] Summary: {summary_path}")
        print(shared.compact_summary(summary_df).to_string(index=False))


if __name__ == "__main__":
    main()
