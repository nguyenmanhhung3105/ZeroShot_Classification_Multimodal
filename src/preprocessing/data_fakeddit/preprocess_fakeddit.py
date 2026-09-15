"""
preprocess_fakeddit.py

Tiền xử lý Fakeddit cho VLM Zero-shot Classification.

Input : data/raw/fakeddit/multimodal_validate.tsv
Output: data/processed/pro_fakeddit/fakeddit_sample.tsv

Output columns:
- id
- clean_title
- image_path
- hasImage
- 2_way_label
- 6_way_label

image_path:
data/processed/pro_fakeddit/images/{id}.jpg
"""

import os
import pandas as pd
import numpy as np

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

np.random.seed(42)

# ============================================================
# CONFIG
# ============================================================

CONFIG_PATH = "configs/experiment_config.yaml"
DEFAULT_CONFIG = {"task": "2way", "output_path": "data/processed/pro_fakeddit/fakeddit_sample.tsv"}

def load_fakeddit_config():
    cfg = DEFAULT_CONFIG.copy()
    if HAS_YAML and os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f: full_cfg = yaml.safe_load(f) or {}
        fk_cfg = (full_cfg.get("datasets", {}) or {}).get("fakeddit", {}) or {}
        cfg["task"] = fk_cfg.get("task", cfg["task"])
        cfg["output_path"] = fk_cfg.get("data_path", cfg["output_path"])
    return cfg

CFG = load_fakeddit_config()
TASK = CFG["task"]
LABEL_COL = "2_way_label" if TASK == "2way" else "6_way_label"
OUTPUT_PATH = CFG["output_path"]
RAW_PATH = "data/raw/fakeddit/multimodal_validate.tsv"
IMAGE_DIR = "data/processed/pro_fakeddit/images"

RAW_REQUIRED_COLS = ["id", "clean_title", "image_url", "hasImage", "2_way_label", "6_way_label"]
OUTPUT_COLS = ["id", "clean_title", "image_path", "hasImage", "2_way_label", "6_way_label"]

TARGET_TOTAL = 3000
MIN_PER_CLASS = 200

print("=" * 60)
print(f"TIỀN XỬ LÝ FAKEDDIT — task={TASK} | label={LABEL_COL}")
print("=" * 60)

# ============================================================
# BƯỚC 1: LOAD RAW
# ============================================================

print("\n" + "=" * 60)
print("BƯỚC 1: LOAD DỮ LIỆU RAW")
print("=" * 60)

if not os.path.exists(RAW_PATH): raise FileNotFoundError(f"Không tìm thấy file: {RAW_PATH}")

df = pd.read_csv(RAW_PATH, sep="\t", on_bad_lines="warn", engine="python")

print(f"Tổng số dòng: {len(df)}")
print(f"Tổng số cột: {len(df.columns)}")

missing_cols = [c for c in RAW_REQUIRED_COLS if c not in df.columns]
if missing_cols: raise ValueError(f"Thiếu các cột bắt buộc: {missing_cols}")

print("Đầy đủ tất cả cột cần thiết.")

# ============================================================
# BƯỚC 2: LOẠI NULL
# ============================================================

print("\n" + "=" * 60)
print("BƯỚC 2: LOẠI DÒNG NULL / LỖI")
print("=" * 60)

null_before = df[RAW_REQUIRED_COLS].isnull().sum()

print("Số null theo cột:")
print(null_before)

if null_before.sum() > 0:
    before = len(df)
    df = df.dropna(subset=RAW_REQUIRED_COLS)
    print(f"Đã loại {before - len(df)} dòng null, còn lại {len(df)} dòng")
else:
    print("Không có dòng null.")

df["2_way_label"] = df["2_way_label"].astype(int)
df["6_way_label"] = df["6_way_label"].astype(int)

# ============================================================
# BƯỚC 3: LỌC ẢNH + TẠO image_path + LOẠI ẢNH KHÔNG TỒN TẠI
# ============================================================

print("\n" + "=" * 60)
print("BƯỚC 3: LỌC ẢNH + TẠO image_path + LOẠI ẢNH THIẾU")
print("=" * 60)

before = len(df)

df = df[df["hasImage"] == True]
df = df[df["image_url"].notna()]
df = df[df["image_url"].astype(str).str.startswith(("http://", "https://"))]

print(f"Trước: {before} -> Sau khi lọc image_url hợp lệ: {len(df)}")

df["image_path"] = df["id"].astype(str).apply(lambda image_id: f"data/processed/pro_fakeddit/images/{image_id}.jpg")

before_image_check = len(df)
df = df[df["image_path"].apply(os.path.exists)].copy().reset_index(drop=True)

print(f"Kiểm tra ảnh local: {before_image_check} -> còn {len(df)} mẫu")
print(f"Đã loại {before_image_check - len(df)} hàng do ảnh không tồn tại.")

print("\nVí dụ image_path:")
print(df[["id", "image_path"]].head().to_string(index=False))

# ============================================================
# BƯỚC 4: LỌC TEXT
# ============================================================

print("\n" + "=" * 60)
print("BƯỚC 4: LỌC TEXT QUÁ NGẮN")
print("=" * 60)

before = len(df)

df = df[df["clean_title"].astype(str).str.strip() != ""]
df = df[df["clean_title"].astype(str).str.len() >= 3]

print(f"Trước: {before} -> Sau khi lọc text ngắn: {len(df)}")

# ============================================================
# BƯỚC 5: LOẠI DUPLICATE
# ============================================================

print("\n" + "=" * 60)
print("BƯỚC 5: LOẠI TRÙNG LẶP id")
print("=" * 60)

before = len(df)
df = df.drop_duplicates(subset="id")

print(f"Trước: {before} -> Sau khi loại trùng id: {len(df)}")

# ============================================================
# BƯỚC 6: SAMPLE CÂN BẰNG
# ============================================================

print("\n" + "=" * 60)
print(f"BƯỚC 6: SAMPLE CÂN BẰNG THEO '{LABEL_COL}'")
print("=" * 60)

class_counts = df[LABEL_COL].value_counts()
n_classes = len(class_counts)

print("Phân bố gốc:")
print(class_counts.sort_index())

sampled_dfs = []

for label, count in class_counts.items():
    n_sample = min(max(MIN_PER_CLASS, TARGET_TOTAL // n_classes), count)
    class_df = df[df[LABEL_COL] == label].sample(n=n_sample, random_state=42)
    sampled_dfs.append(class_df)
    print(f"Lớp {label}: lấy {n_sample}/{count} mẫu")

df_sample = pd.concat(sampled_dfs).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nTổng số mẫu sau sample: {len(df_sample)}")

print("\nPhân bố 2_way_label:")
print(df_sample["2_way_label"].value_counts().sort_index())

print("\nPhân bố 6_way_label:")
print(df_sample["6_way_label"].value_counts().sort_index())

# ============================================================
# BƯỚC 7: SAFETY CHECK
# ============================================================

print("\n" + "=" * 60)
print("BƯỚC 7: KIỂM TRA CUỐI")
print("=" * 60)

null_final = df_sample[OUTPUT_COLS].isnull().sum()
dup_final = df_sample["id"].duplicated().sum()

assert null_final.sum() == 0, "Vẫn còn NULL sau xử lý."
assert dup_final == 0, "Vẫn còn ID bị trùng."

print("Dữ liệu đạt chuẩn — không còn null, không còn duplicate.")

# ============================================================
# BƯỚC 8: SAVE
# ============================================================

print("\n" + "=" * 60)
print("BƯỚC 8: LƯU KẾT QUẢ")
print("=" * 60)

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
df_sample[OUTPUT_COLS].to_csv(OUTPUT_PATH, sep="\t", index=False)

print(f"Đã lưu {len(df_sample)} dòng tại: {OUTPUT_PATH}")

# ============================================================
# KIỂM TRA ẢNH LOCAL
# ============================================================

existing_images = df_sample["image_path"].apply(os.path.exists).sum()
missing_images = len(df_sample) - existing_images

print("\n" + "=" * 60)
print("THỐNG KÊ ẢNH LOCAL")
print("=" * 60)

print(f"Ảnh tồn tại: {existing_images}/{len(df_sample)}")
print(f"Ảnh thiếu   : {missing_images}/{len(df_sample)}")

if missing_images > 0: print(f"LƯU Ý: Có {missing_images} ảnh chưa tồn tại trong '{IMAGE_DIR}'.")

print("\nHOÀN TẤT TIỀN XỬ LÝ.")