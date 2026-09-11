"""
preprocess_fakeddit.py
========================================================================
Tiền xử lý dữ liệu Fakeddit cho bài toán VLM Zero-shot Content
Classification.

Gộp lại từ 2 file cũ (check_fakeddit.py + sample_fakeddit.py):
  - Đọc raw tsv, kiểm tra cột / null / duplicate / url hợp lệ
  - Lọc các dòng "sạch" (có ảnh thật, text đủ dài, nhãn hợp lệ)
  - Sample subset cân bằng theo nhãn (dựa theo task trong config.yaml)
  - Lưu kết quả cuối cùng, chỉ giữ các cột cần thiết cho pipeline

Input : data/raw/fakeddit/multimodal_validate.tsv
Output: data/processed/pro_fakeddit/fakeddit_sample.tsv

Việc kiểm tra chất lượng bản sample cuối cùng (đọc lại ảnh, xem mẫu,...)
được tách sang file riêng: check_fakeddit_ready.py
========================================================================
"""

import os
import pandas as pd
import numpy as np

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

np.random.seed(42)  # để sample có thể tái lập lại

# ============================================================
# CONFIG
# Ưu tiên đọc từ configs/config.yaml (mục datasets.fakeddit) — nếu
# không tìm thấy file config hoặc thiếu thư viện pyyaml thì dùng giá
# trị mặc định khớp với config đã cung cấp trong đề bài.
# ============================================================
CONFIG_PATH = "configs/experiment_config.yaml"

DEFAULT_CONFIG = {
    "task": "2way",  # "2way" hoặc "6way" -> quyết định cột nhãn dùng để cân bằng khi sample
    "output_path": "data/processed/pro_fakeddit/fakeddit_sample.tsv",
}


def load_fakeddit_config():
    cfg = DEFAULT_CONFIG.copy()
    if HAS_YAML and os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            full_cfg = yaml.safe_load(f) or {}
        fk_cfg = (full_cfg.get("datasets", {}) or {}).get("fakeddit", {}) or {}
        cfg["task"] = fk_cfg.get("task", cfg["task"])
        cfg["output_path"] = fk_cfg.get("data_path", cfg["output_path"])
    return cfg


CFG = load_fakeddit_config()
TASK = CFG["task"]                                  # "2way" hoặc "6way"
LABEL_COL = "2_way_label" if TASK == "2way" else "6_way_label"
OUTPUT_PATH = CFG["output_path"]

RAW_PATH = "data/raw/fakeddit/multimodal_validate.tsv"

# Các cột cần thiết cho pipeline VLM zero-shot (text + ảnh + 2 loại nhãn,
# giữ cả hai để dataset có thể dùng lại cho cả task 2way lẫn 6way sau này)
REQUIRED_COLS = ["id", "clean_title", "image_url", "hasImage", "2_way_label", "6_way_label"]

TARGET_TOTAL = 3000    # tổng số mẫu mong muốn cho cả subset
MIN_PER_CLASS = 200    # số mẫu tối thiểu / lớp, tránh lớp bị quá ít mẫu

print("=" * 60)
print(f"TIỀN XỬ LÝ FAKEDDIT — task = {TASK} (cân bằng theo cột '{LABEL_COL}')")
print("=" * 60)

# ============================================================
# BƯỚC 1: LOAD DỮ LIỆU RAW
# ============================================================
print("\n" + "=" * 60)
print("BƯỚC 1: LOAD DỮ LIỆU RAW")
print("=" * 60)

if not os.path.exists(RAW_PATH):
    raise FileNotFoundError(f"Không tìm thấy file tại {RAW_PATH} — kiểm tra lại đường dẫn")

df = pd.read_csv(RAW_PATH, sep="\t", on_bad_lines="warn", engine="python")
print(f"Tổng số dòng: {len(df)}")
print(f"Tổng số cột: {len(df.columns)}")

missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
if missing_cols:
    raise ValueError(f"Thiếu các cột bắt buộc: {missing_cols}")
print("Đầy đủ tất cả cột cần thiết.")

# ============================================================
# BƯỚC 2: LOẠI DÒNG NULL / LỆCH CỘT
# ============================================================
print("\n" + "=" * 60)
print("BƯỚC 2: LOẠI DÒNG NULL / LỖI")
print("=" * 60)

null_before = df[REQUIRED_COLS].isnull().sum()
print("Số null theo cột (trước xử lý):")
print(null_before)

if null_before.sum() > 0:
    before = len(df)
    df = df.dropna(subset=REQUIRED_COLS)
    print(f"Đã loại {before - len(df)} dòng null, còn lại {len(df)} dòng")
else:
    print("Không có dòng null nào.")

# Ép kiểu lại nhãn về int (sau dropna có thể vẫn là float do trước đó lẫn NaN)
df["2_way_label"] = df["2_way_label"].astype(int)
df["6_way_label"] = df["6_way_label"].astype(int)

# ============================================================
# BƯỚC 3: LỌC ẢNH — hasImage=True, image_url hợp lệ & không rỗng
# (loại bỏ mismatch hasImage vs image_url)
# ============================================================
print("\n" + "=" * 60)
print("BƯỚC 3: LỌC hasImage / image_url")
print("=" * 60)

before = len(df)
df = df[df["hasImage"] == True]
df = df[df["image_url"].notna()]
df = df[df["image_url"].astype(str).str.startswith(("http://", "https://"))]
print(f"Trước: {before} -> Sau khi lọc ảnh hợp lệ: {len(df)}")

# ============================================================
# BƯỚC 4: LỌC TEXT QUÁ NGẮN
# ============================================================
print("\n" + "=" * 60)
print("BƯỚC 4: LỌC TEXT QUÁ NGẮN")
print("=" * 60)

before = len(df)
df = df[df["clean_title"].astype(str).str.strip() != ""]
df = df[df["clean_title"].astype(str).str.len() >= 3]
print(f"Trước: {before} -> Sau khi lọc text ngắn: {len(df)}")

# ============================================================
# BƯỚC 5: LOẠI TRÙNG LẶP id
# ============================================================
print("\n" + "=" * 60)
print("BƯỚC 5: LOẠI TRÙNG LẶP id")
print("=" * 60)

before = len(df)
df = df.drop_duplicates(subset="id")
print(f"Trước: {before} -> Sau khi loại trùng id: {len(df)}")

# ============================================================
# BƯỚC 6: SAMPLE SUBSET CÂN BẰNG THEO LABEL_COL (theo task trong config)
# ============================================================
print("\n" + "=" * 60)
print(f"BƯỚC 6: SAMPLE CÂN BẰNG THEO '{LABEL_COL}'")
print("=" * 60)

class_counts = df[LABEL_COL].value_counts()
n_classes = len(class_counts)
print("Phân bố gốc trước khi sample:")
print(class_counts.sort_index())

sampled_dfs = []
for label, count in class_counts.items():
    n_sample = min(max(MIN_PER_CLASS, TARGET_TOTAL // n_classes), count)
    class_df = df[df[LABEL_COL] == label].sample(n=n_sample, random_state=42)
    sampled_dfs.append(class_df)
    print(f"  Lớp {label}: lấy {n_sample}/{count} mẫu")

df_sample = pd.concat(sampled_dfs).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nTổng số mẫu sau khi sample: {len(df_sample)}")
print(f"Phân bố '2_way_label' sau sample:")
print(df_sample["2_way_label"].value_counts().sort_index())
print(f"Phân bố '6_way_label' sau sample:")
print(df_sample["6_way_label"].value_counts().sort_index())

# ============================================================
# BƯỚC 7: SAFETY NET — kiểm tra lần cuối trước khi lưu
# ============================================================
print("\n" + "=" * 60)
print("BƯỚC 7: KIỂM TRA CUỐI CÙNG TRƯỚC KHI LƯU")
print("=" * 60)

null_final = df_sample[REQUIRED_COLS].isnull().sum()
dup_final = df_sample["id"].duplicated().sum()

assert null_final.sum() == 0, "VẪN CÒN NULL sau xử lý — kiểm tra lại file raw"
assert dup_final == 0, "VẪN CÒN TRÙNG LẶP id"
print("Dữ liệu đạt chuẩn — không còn null, không còn trùng lặp.")

# ============================================================
# BƯỚC 8: LƯU KẾT QUẢ (chỉ giữ các cột cần thiết)
# ============================================================
print("\n" + "=" * 60)
print("BƯỚC 8: LƯU KẾT QUẢ")
print("=" * 60)

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
df_sample[REQUIRED_COLS].to_csv(OUTPUT_PATH, sep="\t", index=False)

print(f"Đã lưu subset ({len(df_sample)} dòng) tại: {OUTPUT_PATH}")
print("\nHOÀN TẤT TIỀN XỬ LÝ. Chạy tiếp check_fakeddit_ready.py để kiểm tra chất lượng cuối cùng.")
