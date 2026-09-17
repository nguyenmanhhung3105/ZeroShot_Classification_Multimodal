"""
Hàm dùng chung cho run_fakeddit.py / run_crisismmd.py / run_mmimdb.py.

Đây là bản tách ra từ run_experiment.py (file main gộp cả 3 dataset), kèm
một vài fix bug đã phát hiện khi chạy thật với demo.py (cùng dùng chung logic
với run_experiment.py nên bị lỗi giống hệt):

1. `save_single_label_log`/`save_multi_label_log` thật (trong logging_utils.py)
   KHÔNG nhận `sim_image`, `sim_text`, `extra_manifest` như run_experiment.py
   giả định -> crash "unexpected keyword argument". Thêm
   `call_with_supported_kwargs()` để chỉ truyền đúng kwargs hàm thật hỗ trợ,
   dư ra thì bỏ qua kèm cảnh báo, không phải sửa lại mỗi lần phát hiện thêm
   1 tham số bị thiếu.
2. CrisisMMD lưu nhãn dạng "Informative" / "Not Informative" (viết hoa, có
   khoảng trắng) trong khi prompt set dùng "informative" / "not_informative"
   -> `normalize_single_labels` (map theo get_label_names) không phủ được vì
   dict LABEL_NAMES_INFORMATIVENESS trong crisismmd_prompts.py map SAI CHIỀU.
   Thêm `coerce_labels_to_prompt_format()` làm lớp fallback case/khoảng-trắng
   -insensitive, không cần sửa file prompts.
   LƯU Ý: fallback này chỉ xử lý được khác biệt hoa/thường + khoảng trắng/gạch
   ngang. Với task "humanitarian", nếu nhãn thật là dạng câu khác hẳn từ ngữ
   so với khoá prompt (vd. "Rescue/Volunteering/Donation" vs
   "rescue_volunteering_or_donation_effort") thì fallback này KHÔNG cứu được
   — phải sửa đúng chiều dict LABEL_NAMES_HUMANITARIAN trong
   crisismmd_prompts.py.
"""

import os
import sys
import ast
import json
import inspect

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

def load_dataframe(path: str) -> pd.DataFrame:
    """Tự nhận diện TSV/CSV/Parquet. Giữ dtype=str + keep_default_na=False cho
    TSV/CSV để pandas không tự ép kiểu id/label (số hoá id, hoặc ô trống ->
    NaN thay vì chuỗi rỗng) — an toàn hơn bản gốc trong run_experiment.py."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy dataset: {path}")

    if path.endswith(".tsv"):
        return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    if path.endswith(".csv"):
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    if path.endswith(".parquet"):
        return pd.read_parquet(path)

    raise ValueError(f"Định dạng dataset chưa hỗ trợ: {path}")


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
    """Map label số/tên trong dataset sang tên class trong prompt, theo
    mapping do từng file prompts khai báo (get_label_names)."""
    if label_names is None:
        return values

    mapping = {}
    if isinstance(label_names, dict):
        mapping.update(label_names)
        mapping.update({str(k): v for k, v in label_names.items()})
    elif isinstance(label_names, (list, tuple)):
        mapping.update({i: label for i, label in enumerate(label_names)})
        mapping.update({str(i): label for i, label in enumerate(label_names)})

    normalized = []
    for value in values:
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        normalized.append(mapping.get(value, mapping.get(str(value), value)))
    return normalized


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

        resolved = next((path for path in candidates if os.path.exists(path)), candidates[0] if candidates else "")
        paths.append(resolved)

    return paths


def load_images_safe(image_paths: list) -> tuple:
    """Load ảnh và bỏ qua ảnh lỗi thay vì crash toàn bộ experiment."""
    images, valid_indices = [], []
    for i, path in enumerate(image_paths):
        try:
            with Image.open(path) as img:
                images.append(img.convert("RGB"))
            valid_indices.append(i)
        except Exception as e:
            print(f"[image] Bỏ qua ảnh lỗi: {path} -> {e}")
    return images, valid_indices


# ============================================================
# LOGGING AN TOÀN VỚI SIGNATURE THẬT
# ============================================================

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

def save_dataset_summary(all_results: list, config: dict, dataset_name: str) -> str:
    """Mỗi script run_*.py giờ chỉ chạy 1 dataset -> summary tách riêng theo
    tên dataset thay vì gộp 1 file summary_table.csv duy nhất như bản gốc."""
    results_dir = config["output"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    base_path = config["output"].get("summary_table", os.path.join(results_dir, "summary_table.csv"))
    root, ext = os.path.splitext(base_path)
    summary_path = f"{root}_{dataset_name}{ext or '.csv'}"

    pd.DataFrame(all_results).to_csv(summary_path, index=False)
    return summary_path


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

            try:
                metrics = runner_fn(vlm, task_config, config)
            except Exception as e:
                print(f"Lỗi {model_name} x {dataset_name}: {e}")
                metrics = {"error": str(e)}

            row = {
                "model": model_name,
                "dataset": dataset_name,
                "task": task_config.get("task", "-"),
                "task_type": task_config.get("task_type", "-"),
            }
            row.update(metrics or {})
            all_results.append(row)
        finally:
            if vlm is not None:
                vlm.unload()

    summary_df = pd.DataFrame(all_results)
    summary_path = save_dataset_summary(all_results, config, dataset_name)

    print(f"\n{'=' * 70}")
    print(f"Đã lưu summary tại: {summary_path}")
    print(f"{'=' * 70}")
    print(summary_df)

    return summary_df