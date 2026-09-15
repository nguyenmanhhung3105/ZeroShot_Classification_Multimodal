# ============================================================
# check_mmimdb_ready.py
# Đặt tại: src/preprocessing/data_mmimdb/check_mmimdb_ready.py
# ------------------------------------------------------------
# Kiểm tra bản sample mmimdb cuối cùng (output của preprocess_mmimdb.py)
# có đạt chuẩn để dùng cho vlm_zeroshot_content_classification không.
#
# Khác biệt so với check_fakeddit_ready.py:
#   - Không kiểm tra "phân bố nhãn" bằng value_counts() đơn giản mà phải
#     "explode" cột genres (multi-label) trước khi đếm
#   - Kiểm tra ảnh bằng cách đọc trực tiếp file local, KHÔNG cần
#     requests.get() như Fakeddit (vì ảnh đã copy sẵn về máy)
#   - Không có khái niệm "tỷ lệ lớp nhiều nhất/ít nhất phải <=3x" chặt như
#     single-label, vì multi-label vốn dĩ luôn lệch (Drama luôn nhiều
#     hơn hẳn Film-Noir) — chỉ cảnh báo nếu có thể loại gần như không
#     có mẫu nào
# ============================================================

import os
import pandas as pd
from PIL import Image

# ============================================================
# CONFIG
# ============================================================
OUTPUT_DIR = "data/processed/pro_mmimdb"
FILE_PATH = os.path.join(OUTPUT_DIR, "mmimdb_sample.tsv")
IMG_DIR = OUTPUT_DIR   # image_path trong tsv là đường dẫn tương đối tính từ đây

REQUIRED_COLS = ["id", "plot", "genres", "n_genres", "image_path"]

MIN_TOTAL_SAMPLES = 500        # số mẫu tối thiểu để coi là đủ cho zero-shot eval
MIN_TEXT_LEN = 3
MIN_SAMPLES_PER_GENRE = 5      # thể loại có ít hơn ngưỡng này -> cảnh báo, khó đánh giá đáng tin cậy
N_IMAGE_CHECK = 15             # số ảnh sẽ mở thử để kiểm tra file không hỏng

ready = True

print("=" * 60)
print("KIỂM TRA BẢN SAMPLE MMIMDB")
print("=" * 60)

# ============================================================
# BƯỚC 1: LOAD & THÔNG TIN TỔNG QUAN
# ============================================================
if not os.path.exists(FILE_PATH):
    raise FileNotFoundError(f"Không tìm thấy file tại {FILE_PATH} — chạy preprocess_mmimdb.py trước")

df = pd.read_csv(FILE_PATH, sep="\t")
print(f"Tổng số dòng: {len(df)}")
print(f"Tổng số cột: {len(df.columns)}")

missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
if missing_cols:
    ready = False
    print(f" THIẾU CỘT: {missing_cols}")
else:
    print(" Đầy đủ tất cả cột cần thiết")

if len(df) < MIN_TOTAL_SAMPLES:
    ready = False
    print(f" Số mẫu ({len(df)}) ít hơn mức tối thiểu ({MIN_TOTAL_SAMPLES})")
else:
    print(f" Số mẫu ({len(df)}) đủ cho zero-shot eval")

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
    print(" VẪN CÒN NULL")
else:
    print(" Không có null")

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
    print(" CÓ TRÙNG LẶP id")
else:
    print(" Không có trùng lặp id")

# ============================================================
# BƯỚC 4: KIỂM TRA TEXT (plot)
# ============================================================
print("\n" + "=" * 60)
print("BƯỚC 4: KIỂM TRA TEXT (plot)")
print("=" * 60)
too_short = (df["plot"].astype(str).str.len() < MIN_TEXT_LEN).sum()
print(f"Số dòng plot quá ngắn (<{MIN_TEXT_LEN} ký tự): {too_short}")
if too_short != 0:
    ready = False
    print(" VẪN CÒN TEXT QUÁ NGẮN")
else:
    print(" Không còn text quá ngắn")
print(f"Độ dài text - trung bình: {df['plot'].str.len().mean():.1f}, "
      f"min: {df['plot'].str.len().min()}, max: {df['plot'].str.len().max()}")

# ============================================================
# BƯỚC 5: KIỂM TRA GENRES (multi-label) — KHÔNG rỗng, n_genres khớp thực tế
# ============================================================
print("\n" + "=" * 60)
print("BƯỚC 5: KIỂM TRA CỘT genres (multi-label)")
print("=" * 60)

empty_genres = (df["genres"].astype(str).str.strip() == "").sum()
print(f"Số dòng genres rỗng: {empty_genres}")
if empty_genres != 0:
    ready = False
    print(" VẪN CÒN DÒNG KHÔNG CÓ GENRE NÀO")
else:
    print(" Mọi dòng đều có ít nhất 1 genre")

# kiểm tra n_genres có khớp với số lượng thực tế trong chuỗi genres không
actual_n = df["genres"].astype(str).str.split("|").apply(len)
mismatch_n = (actual_n != df["n_genres"]).sum()
print(f"Số dòng n_genres không khớp thực tế: {mismatch_n}")
if mismatch_n != 0:
    ready = False
    print(" CỘT n_genres BỊ SAI LỆCH SO VỚI genres")
else:
    print(" Cột n_genres khớp với dữ liệu genres")

# ============================================================
# BƯỚC 6: PHÂN BỐ THỂ LOẠI (explode multi-label rồi mới value_counts)
# ============================================================
print("\n" + "=" * 60)
print("BƯỚC 6: PHÂN BỐ THỂ LOẠI (multi-label)")
print("=" * 60)

genre_counts = df["genres"].astype(str).str.split("|").explode().value_counts()
print(genre_counts)

rare_genres = genre_counts[genre_counts < MIN_SAMPLES_PER_GENRE]
if len(rare_genres) > 0:
    print(f"\n  Cảnh báo: {len(rare_genres)} thể loại có ít hơn {MIN_SAMPLES_PER_GENRE} mẫu:")
    print(rare_genres)
    print("   (Không nhất thiết fail toàn bộ, nhưng đánh giá F1 cho các lớp này sẽ kém tin cậy)")
else:
    print(f" Mọi thể loại đều có ít nhất {MIN_SAMPLES_PER_GENRE} mẫu")

avg_genres_per_sample = df["n_genres"].mean()
print(f"\nSố genre trung bình / mẫu: {avg_genres_per_sample:.2f}")

# ============================================================
# BƯỚC 7: XEM VÀI MẪU THỰC TẾ
# ============================================================
print("\n" + "=" * 60)
print("BƯỚC 7: VÍ DỤ MẪU THỰC TẾ")
print("=" * 60)
print(df[["id", "plot", "genres"]].sample(min(5, len(df)), random_state=1).to_string(index=False))

# ============================================================
# BƯỚC 8: KIỂM TRA ẢNH LOCAL CÓ MỞ ĐƯỢC KHÔNG
# (khác Fakeddit: đọc file local trực tiếp, không cần requests.get)
# ============================================================
print("\n" + "=" * 60)
print(f"BƯỚC 8: KIỂM TRA ẢNH LOCAL (sample {N_IMAGE_CHECK} file)")
print("=" * 60)

n_check = min(N_IMAGE_CHECK, len(df))
sample_rows = df.sample(n_check, random_state=1)

success, failed, missing = 0, 0, 0
for _, row in sample_rows.iterrows():
    full_path = os.path.join(IMG_DIR, row["image_path"])
    if not os.path.exists(full_path):
        missing += 1
        print(f"  Không tìm thấy file: {full_path}")
        continue
    try:
        with Image.open(full_path) as img:
            img.verify()
        success += 1
    except Exception as e:
        failed += 1
        print(f"  Lỗi mở ảnh {row['image_path']}: {type(e).__name__}")

print(f"\nKết quả: {success}/{n_check} ảnh OK, {failed}/{n_check} lỗi, {missing}/{n_check} không tìm thấy file")
if failed + missing > 0:
    ready = False
    print(" CÓ ẢNH BỊ LỖI/THIẾU — kiểm tra lại bước copy ở preprocess_mmimdb.py")
else:
    print(" Toàn bộ ảnh kiểm tra đều mở được")

# ============================================================
# KẾT LUẬN
# ============================================================
print("\n" + "=" * 60)
if ready:
    print(" DỮ LIỆU MMIMDB ĐẠT CHUẨN — SẴN SÀNG CHO vlm_zeroshot_content_classification")
else:
    print(" DỮ LIỆU CHƯA ĐẠT CHUẨN — xem lại các mục  ở trên")
print("=" * 60)