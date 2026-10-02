"""Tiện ích dùng chung cho các entry point: dữ liệu, nhãn và chạy model."""

import os
import sys
import ast
import json
import inspect
import re

import numpy as np
import pandas as pd
import yaml
from PIL import Image

# Các file run_*.py nằm trong thư mục con (experiments/), còn models/,
# inference.py, evaluate.py, logging_utils.py, prompts/ nằm ở thư mục cha
# (src/) — cần add path này vào sys.path thì mới import được.
# Nếu layout thư mục thật khác đi, chỉ cần sửa dòng insert bên dưới.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model_registry import load_model  # noqa: E402  (import sau khi sửa sys.path)


# ============================================================
# CONFIG
# ============================================================

def load_config(path: str = "configs/experiment_config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ============================================================
# ĐỌC DATASET
# ============================================================

def load_dataframe(path: str, max_samples=None, text_col=None, min_text_length=0) -> pd.DataFrame:
    """Đọc có giới hạn; nếu lọc text, lấy tối đa max_samples dòng sau lọc."""
    if max_samples is not None and (not isinstance(max_samples, int) or max_samples < 1):
        raise ValueError("max_samples phải là số nguyên dương hoặc None")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy dataset: {path}")

    def select(frame):
        if min_text_length:
            validate_columns(frame, [text_col], "dataset")
            length = frame[text_col].fillna("").astype(str).str.replace(" ", "", regex=False).str.len()
            frame = frame[length > min_text_length]
        return frame

    if path.endswith((".tsv", ".csv")):
        kwargs = dict(sep="\t" if path.endswith(".tsv") else ",",
                      dtype=str, keep_default_na=False)
        if not min_text_length:
            return pd.read_csv(path, nrows=max_samples, **kwargs)
        chunks = pd.read_csv(path, chunksize=min(max_samples or 1024, 1024), **kwargs)
    elif path.endswith(".parquet"):
        # iter_batches tránh nạp toàn bộ Parquet rồi mới head().
        import pyarrow.parquet as pq
        chunks = (batch.to_pandas() for batch in pq.ParquetFile(path).iter_batches(
            batch_size=min(max_samples or 1024, 1024)))
    else:
        raise ValueError(f"Định dạng dataset chưa hỗ trợ: {path}")

    parts, total = [], 0
    try:
        for chunk in chunks:
            selected = select(chunk)
            if max_samples is not None:
                selected = selected.iloc[:max_samples - total]
            parts.append(selected)
            total += len(selected)
            if max_samples is not None and total >= max_samples:
                break
    finally:
        if hasattr(chunks, "close"):
            chunks.close()
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def crisismmd_label_column(task_config: dict) -> str:
    default = "label_informative" if task_config["task"] == "informativeness" else "label_humanitarian"
    configured = task_config.get("label_col")
    # Hai cột chuẩn đi theo task; vẫn cho phép tên cột tùy chỉnh.
    return default if configured in (None, "label", "label_informative", "label_humanitarian") else configured


def validate_columns(df: pd.DataFrame, required_columns: list, dataset_name: str) -> None:
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(
            f"{dataset_name}: thiếu các cột {missing}. "
            f"Các cột hiện có: {df.columns.tolist()}"
        )


# ============================================================
# OVERRIDE THEO MODEL / DATASET
# ============================================================

def resolve_override(default_value, overrides: dict, key: str):
    """Tra cứu override theo key (model_name hoặc dataset_name). Dùng chung
    cho image_weight (theo dataset) và threshold (theo model)."""
    if not overrides:
        return default_value
    return overrides.get(key, default_value)


def get_image_weight(config: dict, dataset_name: str) -> float:
    inference_cfg = config.get("inference", {})
    return resolve_override(
        inference_cfg.get("image_weight", 0.5),
        inference_cfg.get("image_weight_overrides"),
        dataset_name,
    )


# ============================================================
# NHÃN
# ============================================================

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
    """Ánh xạ ID hoặc tên hiển thị về đúng khóa prompt; từ chối nhãn lạ."""
    def canon(value):
        return str(value).strip().lower().replace(" ", "_").replace("-", "_")
    mapping = {canon(label): label for label in label_order}
    if isinstance(label_names, dict):
        for key, display in label_names.items():
            if key in label_order:
                mapping[canon(display)] = key
            elif display in label_order:
                mapping[canon(key)] = display
    elif isinstance(label_names, (list, tuple)):
        for index, label in enumerate(label_names):
            if label in label_order:
                mapping[canon(index)] = label
    result = []
    for value in values:
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        key = canon(value)
        if key not in mapping:
            raise ValueError(f"Nhãn không thuộc prompt: {value!r}; cần {label_order}")
        result.append(mapping[key])
    return result


def _canonical_label_lookup(label_order: list) -> dict:
    def canon(s):
        return str(s).strip().lower().replace(" ", "_").replace("-", "_")
    return {canon(label): label for label in label_order}


def coerce_labels_to_prompt_format(values: list, label_order: list) -> list:
    """Fallback case/khoảng-trắng-insensitive: nếu normalize_single_labels
    (map theo get_label_names khai báo trong file prompts) không phủ hết biến
    thể viết của dataset thật (vd. 'Not Informative' vs 'not_informative'),
    vẫn cố khớp bằng canonical form trước khi đành giữ nguyên giá trị gốc.
    Không cứu được các trường hợp khác nhau cả về từ ngữ (không chỉ hoa/thường
    hay khoảng trắng) — những trường hợp đó phải sửa đúng ở file prompts."""
    lookup = _canonical_label_lookup(label_order)

    def canon(s):
        return str(s).strip().lower().replace(" ", "_").replace("-", "_")

    return [lookup.get(canon(v), v) for v in values]


# ============================================================
# ẢNH
# ============================================================

def resolve_image_paths(df: pd.DataFrame, task_config: dict) -> list:
    """Xác định path ảnh.

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

        raw_value = row[image_col] if image_col in df.columns else None
        if raw_value is not None and pd.notna(raw_value) and str(raw_value).strip():
            raw_path = str(raw_value)
            candidates.append(raw_path)
            if image_dir:
                candidates.append(os.path.join(image_dir, raw_path))
                candidates.append(os.path.join(image_dir, os.path.basename(raw_path)))

        if id_col in df.columns and image_dir:
            sample_id = str(row[id_col])
            candidates.append(os.path.join(image_dir, sample_id))
            candidates.append(os.path.join(image_dir, f"{sample_id}{image_ext}"))
            for ext in (".jpeg", ".jpg", ".JPEG", ".JPG"):
                candidates.append(os.path.join(image_dir, f"{sample_id}{ext}"))

        resolved = next((path for path in candidates if os.path.exists(path)), candidates[0] if candidates else "")
        paths.append(resolved)

    return paths


class ImagePathSequence:
    """Chỉ giữ đường dẫn; giải mã ảnh khi inference lấy từng batch."""
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        with Image.open(self.paths[index]) as image:
            return image.convert("RGB")


def load_images_safe(image_paths: list) -> tuple:
    """Kiểm tra từng ảnh, bỏ ảnh lỗi, chỉ giữ đường dẫn của ảnh hợp lệ."""
    paths, valid_indices = [], []
    for i, path in enumerate(image_paths):
        try:
            with Image.open(path) as img:
                img.load()
            paths.append(path)
            valid_indices.append(i)
        except Exception as e:
            print(f"[image] Bỏ qua ảnh lỗi: {path} -> {e}")
    return ImagePathSequence(paths), valid_indices


def call_with_supported_kwargs(func, *args, **kwargs):
    """Chỉ truyền các kwargs mà `func` THẬT SỰ khai báo — tránh crash kiểu
    'unexpected keyword argument' khi logging_utils.py (không nằm trong review)
    chưa/không hỗ trợ đúng những gì các file run_*.py giả định (đã gặp với
    sim_image/sim_text/extra_manifest). Kwarg nào bị bỏ sẽ in cảnh báo ra
    console để biết mà bổ sung logging_utils.py sau, không âm thầm mất dữ liệu."""
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


# ============================================================
# VÒNG LẶP MODEL DÙNG CHUNG (thay cho main() lặp lại 3 lần)
# ============================================================

def compact_summary(rows):
    """Presentation only: detailed branch metrics/config stay in per-run logs."""
    frame = (rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows))
    comparison = {"image_macro_f1", "text_macro_f1", "fusion_macro_f1",
                  "image_n_samples", "text_n_samples", "fusion_n_samples"}
    if comparison.issubset(frame.columns):
        comparable = ((frame["fusion_n_samples"] > 0) &
                      frame["image_n_samples"].eq(frame["fusion_n_samples"]) &
                      frame["text_n_samples"].eq(frame["fusion_n_samples"]) &
                      frame[["image_macro_f1", "text_macro_f1", "fusion_macro_f1"]].notna().all(axis=1))
        best = frame[["image_macro_f1", "text_macro_f1"]].max(axis=1)
        frame["fusion_macro_f1_gain"] = (frame["fusion_macro_f1"] - best).where(comparable)
        frame["fusion_improves_macro_f1"] = frame["fusion_macro_f1_gain"].gt(1e-12).astype("boolean").where(comparable)
    if "fusion_n_samples" in frame:
        frame["n_samples"] = frame["fusion_n_samples"]
    for metric in ("micro_f1", "macro_f1", "samples_f1", "balanced_accuracy"):
        source = f"fusion_{metric}"
        if source in frame:
            frame[metric] = frame[source].combine_first(frame[metric]) if metric in frame else frame[source]
    details = {"task_type", "score_type", "model_scale_applied", "apply_bias_to_text",
               "sample_set_id", "prompt_id", "config_id", "hamming_accuracy"}
    # Drop only known audit fields, not arbitrary caller-defined result columns.
    scale_columns = {"text_logit_scale_effective", "text_logit_scale_status",
                     "text_logit_scale_requested", "text_scale_override_applied",
                     "fusion_enabled", "image_weight", "text_weight",
                     "image_macro_f1", "text_macro_f1",
                     "fusion_macro_f1_gain", "fusion_improves_macro_f1"}
    drop = [c for c in frame if c not in scale_columns and
            (c in details or c.startswith(("image_", "text_", "fusion_")))]
    frame = frame.drop(columns=drop)
    for column in ("threshold", "relative_z_threshold"):
        if column in frame and frame[column].isna().all():
            frame = frame.drop(columns=[column])
    return frame


def save_numbered_summary(rows, base_path: str, dataset_name: str, task_name: str) -> str:
    """Lưu TSV có số thứ tự; chấp nhận cả base_path .csv của config cũ."""
    root, _ = os.path.splitext(os.fspath(base_path))
    dataset = re.sub(r"[^a-zA-Z0-9_-]", "_", str(dataset_name))
    task = re.sub(r"[^a-zA-Z0-9_-]", "_", str(task_name))
    prefix = f"{root}_{dataset}_{task}_"
    ext = ".tsv"
    directory = os.path.dirname(prefix) or "."
    os.makedirs(directory, exist_ok=True)
    pattern = re.compile(re.escape(os.path.basename(prefix)) + r"(\d+)" + re.escape(ext))
    last_number = 0
    with os.scandir(directory) as entries:
        for entry in entries:
            match = pattern.fullmatch(entry.name)
            if match:
                last_number = max(last_number, int(match.group(1)))
    number = last_number + 1
    frame = compact_summary(rows)
    while True:
        summary_path = f"{prefix}{number}{ext}"
        try:
            # Exclusive create bảo vệ cả trường hợp hai tiến trình chạy đồng thời.
            stream = open(summary_path, "x", encoding="utf-8", newline="")
        except FileExistsError:
            number += 1
            continue
        with stream:
            frame.to_csv(stream, sep="\t", index=False)
        return summary_path


def save_summary_tables(all_results: list, base_path: str) -> list:
    """Một file cho mỗi dataset/task; giữ các model của cùng lần chạy chung file."""
    if not all_results:
        return []
    frame = pd.DataFrame(all_results)
    return [
        save_numbered_summary(group, base_path, dataset, task)
        for (dataset, task), group in frame.groupby(["dataset", "task"], sort=False)
    ]


def save_dataset_summary(all_results: list, config: dict, dataset_name: str) -> str:
    base_path = config["output"].get(
        "summary_table", os.path.join(config["output"]["results_dir"], "summary_table.tsv"))
    task_name = config["datasets"][dataset_name].get("task", "-")
    return save_numbered_summary(all_results, base_path, dataset_name, task_name)


def run_all_models_for_dataset(dataset_name: str, runner_fn, config: dict) -> pd.DataFrame:
    """Vòng lặp dùng chung cho cả 3 script: load từng model ĐÚNG 1 LẦN, chạy
    `runner_fn(vlm, task_config, config)` cho dataset hiện tại, unload trước
    khi sang model tiếp theo — giữ đúng nguyên tắc gốc của run_experiment.py,
    chỉ khác là mỗi lần chỉ chạy 1 dataset thay vì lặp qua cả 3."""
    task_config = config["datasets"].get(dataset_name)
    if task_config is None:
        raise KeyError(f"Không tìm thấy cấu hình cho dataset '{dataset_name}' trong config['datasets'].")

    if not task_config.get("enabled", False):
        print(f"[SKIP] Dataset '{dataset_name}' có enabled=false trong config -> không chạy.")
        return pd.DataFrame()

    os.makedirs(config["output"]["results_dir"], exist_ok=True)
    os.makedirs(config["output"]["raw_predictions_dir"], exist_ok=True)

    all_results = []

    for model_name in config["models"]:
        print(f"\n{'#' * 70}\nMODEL: {model_name}\n{'#' * 70}")

        vlm = None
        try:
            vlm = load_model(model_name, device=config.get("device"))

            all_results.extend(run_prompt_trials(
                lambda active: runner_fn(vlm, active, config), model_name, dataset_name, task_config))
        finally:
            if vlm is not None:
                vlm.unload()

    summary_df = pd.DataFrame(all_results)
    summary_path = save_dataset_summary(all_results, config, dataset_name)

    print(f"\n{'=' * 70}")
    print(f"Đã lưu summary tại: {summary_path}")
    print(f"{'=' * 70}")
    print(compact_summary(summary_df).to_string(index=False))

    return summary_df


def prompt_trial_configs(dataset_name, task_config):
    """Expand prompt/aggregation choices; never change model, data, scale or fusion."""
    variants = task_config.get("prompt_variants")
    if variants is None:
        variants = [task_config.get("prompt_variant", "current")]
    if not isinstance(variants, list) or not variants or any(not isinstance(v, str) for v in variants):
        raise ValueError("prompt_variants must be a nonempty list of variant names")
    if len(variants) != len(set(variants)):
        raise ValueError("Duplicate prompt variants")
    from prompts.fakeddit_prompts import PROMPT_VARIANTS
    for variant in variants:
        if variant not in PROMPT_VARIANTS:
            raise ValueError(f"Unknown prompt variant: {variant}")
        if variant != "current" and (dataset_name != "fakeddit" or task_config.get("task") != "6way"):
            raise ValueError("Alternative prompt variants support only Fakeddit 6way")
    modes = task_config.get("prompt_aggregations", [task_config.get("prompt_aggregation", "mean_embed")])
    if (not isinstance(modes, list) or not modes or any(not isinstance(m, str) for m in modes)
            or len(set(modes)) != len(modes)):
        raise ValueError("prompt_aggregations must be a nonempty list of unique mode names")
    from inference import prompt_aggregation_settings
    trials = [{**task_config, "prompt_variant": variant, "prompt_aggregation": mode}
              for variant in variants for mode in modes]
    for active in trials:
        prompt_aggregation_settings(active, dataset_name)
    return trials


def run_prompt_trials(run_one, model_name, dataset_name, task_config):
    """One model already loaded; each variant gets an independent result/log."""
    rows = []
    for active in prompt_trial_configs(dataset_name, task_config):
        variant = active["prompt_variant"]
        aggregation = active["prompt_aggregation"]
        print(f"[PROMPT] {dataset_name}/{active.get('task')}: {variant}; aggregation={aggregation}")
        try:
            metrics = run_one(active)
        except Exception as exc:
            print(f"Lỗi {model_name} x {dataset_name} x {variant} x {aggregation}: {exc}")
            metrics = {"error": str(exc)}
        row = {"model": model_name, "dataset": dataset_name,
               "task": active.get("task", "-"), "task_type": active.get("task_type", "-")}
        if dataset_name == "fakeddit":
            row["prompt_variant"] = variant
        row["prompt_aggregation"] = aggregation
        row.update(metrics or {})
        rows.append(row)
    return rows
