"""
check_fakeddit_ready.py
========================================================================
Kiểm tra bản sample Fakeddit cuối cùng (output của preprocess_fakeddit.py)
có đạt chuẩn để dùng cho vlm_zeroshot_content_classification hay không.

Gộp lại từ check_fakeddit_sample.py cũ, gồm:
  - Kiểm tra cột / null / duplicate
  - Kiểm tra phân bố nhãn (2-way & 6-way) có cân bằng không
  - Kiểm tra độ dài text
  - Xem vài mẫu thực tế từng lớp
  - Kiểm tra thử ảnh có tải được không (link Reddit cũ hay chết)

Input: data/processed/pro_fakeddit/fakeddit_sample.tsv
========================================================================
"""

import os
import pandas as pd
import requests
from io import BytesIO
from PIL import Image

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

# ============================================================
# CONFIG (đồng bộ với preprocess_fakeddit.py)
# ============================================================
CONFIG_PATH = "configs/config.yaml"

DEFAULT_CONFIG = {
    "task": "2way",
    "data_path": "data/processed/pro_fakeddit/fakeddit_sample.tsv",
}


def load_fakeddit_config():
    cfg = DEFAULT_CONFIG.copy()
    if HAS_YAML and os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            full_cfg = yaml.safe_load(f) or {}
        fk_cfg = (full_cfg.get("datasets", {}) or {}).get("fakeddit", {}) or {}
        cfg["task"] = fk_cfg.get("task", cfg["task"])
        cfg["data_path"] = fk_cfg.get("data_path", cfg["data_path"])
    return cfg


CFG = load_fakeddit_config()
TASK = CFG["task"]
LABEL_COL = "2_way_label" if TASK == "2way" else "6_way_label"
FILE_PATH = CFG["data_path"]

REQUIRED_COLS = ["id", "clean_title", "image_url", "hasImage", "2_way_label", "6_way_label"]

MIN_TOTAL_SAMPLES = 500      # số mẫu tối thiểu để coi là "đủ" cho zero-shot eval
N_URL_CHECK = 10             # số url ảnh sẽ thử tải để kiểm tra sống/chết
MAX_FAILED_URL_RATIO = 0.3   # tỷ lệ url chết tối đa chấp nhận được

ready = True  # cờ tổng, sẽ bật False nếu có bước nào fail

print("=" * 60)
print(f"KIỂM TRA BẢN SAMPLE FAKEDDIT — task = {TASK}")
print("=" * 60)

# ============================================================
# BƯỚC 1: LOAD & THÔNG TIN TỔNG QUAN
# ============================================================
if not os.path.exists(FILE_PATH):
    raise FileNotFoundError(f"Không tìm thấy file tại {FILE_PATH} — chạy preprocess_fakeddit.py trước")

df = pd.read_csv(FILE_PATH, sep="\t")
print(f"Tổng số dòng: {len(df)}")
print(f"Tổng số cột: {len(df.columns)}")

missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
if missing_cols:
    ready = False
    print(f"❌ THIẾU CỘT: {missing_cols}")
else:
    print("✅ Đầy đủ tất cả cột cần thiết")

if len(df) < MIN_TOTAL_SAMPLES:
    ready = False
    print(f"❌ Số mẫu ({len(df)}) ít hơn mức tối thiểu ({MIN_TOTAL_SAMPLES}) cho zero-shot eval")
else:
    print(f"✅ Số mẫu ({len(df)}) đủ cho zero-shot eval")

# ============================================================
# BƯỚC 2: KIỂM TRA NULL
# ============================================================
print("\n" + "=" * 60)
print("BƯỚC 2: KIỂM TRA NULL")
print("=" * 60)
null_counts = df[REQUIRED_COLS].isnull().sum()
print(null_counts)
if null_counts.sum() != 0:
    ready = False
    print("❌ VẪN CÒN NULL")
else:
    print("✅ Không có null")

# ============================================================
# BƯỚC 3: KIỂM TRA TRÙNG LẶP id
# ============================================================
print("\n" + "=" * 60)
print("BƯỚC 3: KIỂM TRA TRÙNG LẶP id")
print("=" * 60)
dup_ids = df["id"].duplicated().sum()
print(f"Số dòng trùng 'id': {dup_ids}")
if dup_ids != 0:
    ready = False
    print("❌ CÓ TRÙNG LẶP id")
else:
    print("✅ Không có trùng lặp id")

# ============================================================
# BƯỚC 4: KIỂM TRA hasImage vs image_url
# ============================================================
print("\n" + "=" * 60)
print("BƯỚC 4: KIỂM TRA MISMATCH hasImage vs image_url")
print("=" * 60)
mismatch = df[(df["hasImage"] != True) | (df["image_url"].isna())]
print(f"Số dòng hasImage != True hoặc image_url rỗng: {len(mismatch)}")
if len(mismatch) > 0:
    ready = False
    print("❌ VẪN CÒN DÒNG KHÔNG CÓ ẢNH HỢP LỆ")
else:
    print("✅ Toàn bộ dòng đều có ảnh hợp lệ (đạt yêu cầu multimodal)")

# ============================================================
# BƯỚC 5: KIỂM TRA PHÂN BỐ NHÃN
# ============================================================
print("\n" + "=" * 60)
print(f"BƯỚC 5: PHÂN BỐ NHÃN (theo '{LABEL_COL}' — dùng cho task '{TASK}')")
print("=" * 60)

print("\n2_way_label:")
print(df["2_way_label"].value_counts().sort_index())
print("\n6_way_label:")
print(df["6_way_label"].value_counts().sort_index())

label_counts = df[LABEL_COL].value_counts()
ratio = label_counts.max() / label_counts.min()
print(f"\nTỷ lệ lớp nhiều nhất / lớp ít nhất (trên '{LABEL_COL}'): {ratio:.2f}x")
if ratio > 3:
    ready = False
    print("❌ Lệch lớp khá nhiều, cân nhắc chạy lại preprocess_fakeddit.py với MIN_PER_CLASS lớn hơn")
else:
    print("✅ Phân bố nhãn tương đối cân bằng")

# ============================================================
# BƯỚC 6: KIỂM TRA TEXT
# ============================================================
print("\n" + "=" * 60)
print("BƯỚC 6: KIỂM TRA TEXT (clean_title)")
print("=" * 60)
too_short = (df["clean_title"].astype(str).str.len() < 3).sum()
print(f"Số dòng text quá ngắn (<3 ký tự): {too_short}")
if too_short != 0:
    ready = False
    print("❌ VẪN CÒN TEXT QUÁ NGẮN")
else:
    print("✅ Không còn text quá ngắn")

print(f"\nĐộ dài text - trung bình: {df['clean_title'].str.len().mean():.1f} ký tự")
print(f"Độ dài text - min: {df['clean_title'].str.len().min()}, max: {df['clean_title'].str.len().max()}")

# ============================================================
# BƯỚC 7: XEM VÀI MẪU THỰC TẾ TỪNG LỚP (theo LABEL_COL của task)
# ============================================================
print("\n" + "=" * 60)
print("BƯỚC 7: VÍ DỤ MẪU THỰC TẾ TỪNG LỚP")
print("=" * 60)

label_names_6way = {
    0: "True (thật)",
    1: "Satire/Parody",
    2: "False Connection",
    3: "Imposter Content",
    4: "Manipulated Content",
    5: "Misleading Content",
}
label_names_2way = {0: "True (thật)", 1: "Fake (giả)"}
label_names = label_names_2way if TASK == "2way" else label_names_6way

for label, name in label_names.items():
    subset = df[df[LABEL_COL] == label]
    if len(subset) > 0:
        print(f"\n--- Lớp {label} ({name}) — {len(subset)} mẫu ---")
        print(subset[["clean_title"]].sample(min(2, len(subset)), random_state=1).to_string(index=False))

# ============================================================
# BƯỚC 8: KIỂM TRA THỬ ẢNH CÓ TẢI ĐƯỢC KHÔNG
# ============================================================
print("\n" + "=" * 60)
print(f"BƯỚC 8: KIỂM TRA THỬ ẢNH (sample {N_URL_CHECK} url)")
print("=" * 60)

n_check = min(N_URL_CHECK, len(df))
sample_urls = df["image_url"].sample(n_check, random_state=1).tolist()
success, failed = 0, 0

for url in sample_urls:
    try:
        resp = requests.get(url, timeout=5)
        img = Image.open(BytesIO(resp.content))
        img.verify()
        success += 1
    except Exception as e:
        failed += 1
        print(f"  Lỗi với url: {url[:60]}... -> {type(e).__name__}")

print(f"\nKết quả: {success}/{n_check} ảnh tải được, {failed}/{n_check} lỗi")
failed_ratio = failed / n_check if n_check else 0
if failed_ratio > MAX_FAILED_URL_RATIO:
    ready = False
    print(f"❌ Tỷ lệ lỗi ({failed_ratio:.0%}) vượt ngưỡng {MAX_FAILED_URL_RATIO:.0%} — cần lọc thêm ảnh chết")
else:
    print("✅ Tỷ lệ ảnh sống chấp nhận được")

# ============================================================
# KẾT LUẬN
# ============================================================
print("\n" + "=" * 60)
if ready:
    print(f"✅ DỮ LIỆU ĐẠT CHUẨN — SẴN SÀNG CHO vlm_zeroshot_content_classification (task={TASK})")
else:
    print("❌ DỮ LIỆU CHƯA ĐẠT CHUẨN — xem lại các mục ❌ ở trên trước khi chạy thí nghiệm")
print("=" * 60)
