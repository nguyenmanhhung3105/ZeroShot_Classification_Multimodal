# ============================================================
# preprocess_mmimdb.py
# Đặt tại: src/preprocessing/data_mmimdb/preprocess_mmimdb.py
# ------------------------------------------------------------
# Tiền xử lý dữ liệu mmimdb (multi-label) cho vlm_zeroshot_content_classification.
#
# Input : data/raw/mmimdb/dataset/  (chứa các cặp file <id>.json + <id>.jpeg)
# Output:
#   - data/processed/pro_mmimdb/mmimdb_sample.tsv
#   - data/processed/pro_mmimdb/images/<id>.jpeg   (copy từ raw, chỉ những mẫu được chọn)
#
# Khác biệt so với Fakeddit:
#   - Nhãn là MULTI-LABEL (list thể loại), không phải 1 số nguyên
#   - Ảnh đã có sẵn dạng file local, chỉ cần copy chứ không tải qua URL
#   - Sample tạm thời dùng random đơn giản (sẽ nâng cấp iterative stratification sau)
# ============================================================

import os
import json
import shutil
import pandas as pd
import numpy as np
from PIL import Image

np.random.seed(42)

# ============================================================
# CONFIG
# ============================================================
RAW_DIR = "data/raw/mmimdb/dataset"
OUTPUT_DIR = "data/processed/pro_mmimdb"
OUTPUT_TSV = os.path.join(OUTPUT_DIR, "mmimdb_sample.tsv")
OUTPUT_IMG_DIR = os.path.join(OUTPUT_DIR, "images")

TARGET_TOTAL = 3000   # tổng số mẫu mong muốn
MIN_TEXT_LEN = 3      # độ dài tối thiểu của plot

# 23 thể loại nội dung chuẩn của MM-IMDb gốc (Arevalo et al.)
# — dùng để lọc bỏ các nhãn không phải "thể loại nội dung"
# (ví dụ Short, News, Talk-Show, Reality-TV, Game-Show, Adult...)
VALID_GENRES = {
    "Drama", "Comedy", "Romance", "Thriller", "Crime", "Action", "Adventure",
    "Horror", "Documentary", "Mystery", "Sci-Fi", "Fantasy", "Family",
    "Biography", "War", "History", "Music", "Animation", "Musical",
    "Western", "Sport", "Short", "Film-Noir"
}
# Ghi chú: nếu muốn loại "Short" khỏi danh sách vì nó ám chỉ ĐỊNH DẠNG
# (phim ngắn) hơn là NỘI DUNG, có thể bỏ "Short" ra khỏi VALID_GENRES.

REQUIRED_JSON_FIELDS = ["plot", "genres"]

print("=" * 60)
print("TIỀN XỬ LÝ MMIMDB")
print("=" * 60)

if not os.path.exists(RAW_DIR):
    raise FileNotFoundError(f"Không tìm thấy thư mục raw tại {RAW_DIR}")

# ============================================================
# BƯỚC 1: QUÉT THƯ MỤC RAW, TÌM CẶP (json, jpeg) HỢP LỆ
# ============================================================
print("\n" + "=" * 60)
print("BƯỚC 1: QUÉT CẶP FILE json/jpeg")
print("=" * 60)

all_files = os.listdir(RAW_DIR)
json_ids = {os.path.splitext(f)[0] for f in all_files if f.endswith(".json")}
jpeg_ids = {os.path.splitext(f)[0] for f in all_files if f.lower().endswith((".jpeg", ".jpg"))}

paired_ids = sorted(json_ids & jpeg_ids)
only_json = json_ids - jpeg_ids
only_jpeg = jpeg_ids - json_ids

print(f"Số file .json: {len(json_ids)}")
print(f"Số file .jpeg: {len(jpeg_ids)}")
print(f"Số cặp json+jpeg khớp nhau: {len(paired_ids)}")
if only_json:
    print(f"  -> {len(only_json)} json không có ảnh tương ứng (sẽ bị loại)")
if only_jpeg:
    print(f"  -> {len(only_jpeg)} ảnh không có json tương ứng (sẽ bị loại)")

# ============================================================
# BƯỚC 2: ĐỌC TỪNG JSON, TRÍCH TRƯỜNG CẦN THIẾT + LỌC HỢP LỆ
# ============================================================
print("\n" + "=" * 60)
print("BƯỚC 2: ĐỌC JSON & LỌC MẪU HỢP LỆ")
print("=" * 60)

records = []
skipped_no_field = 0
skipped_no_genre = 0
skipped_short_text = 0
skipped_bad_image = 0

for sample_id in paired_ids:
    json_path = os.path.join(RAW_DIR, f"{sample_id}.json")
    jpeg_path = os.path.join(RAW_DIR, f"{sample_id}.jpeg")

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        skipped_no_field += 1
        continue

    # Chỉ giữ phim điện ảnh nếu có field "kind" (một số json có thể không có field này)
    if meta.get("kind") and meta.get("kind") != "movie":
        continue

    # --- Trích plot text: ưu tiên "plot", fallback "plot outline" ---
    plot_field = meta.get("plot")
    if isinstance(plot_field, list) and len(plot_field) > 0:
        text = str(plot_field[0])          # lấy bản mô tả đầu tiên
    elif isinstance(plot_field, str):
        text = plot_field
    else:
        text = meta.get("plot outline", "")

    text = str(text).strip()
    if len(text) < MIN_TEXT_LEN:
        skipped_short_text += 1
        continue

    # --- Trích genres, lọc theo VALID_GENRES ---
    raw_genres = meta.get("genres", [])
    genres = [g for g in raw_genres if g in VALID_GENRES]
    if len(genres) == 0:
        skipped_no_genre += 1
        continue

    # --- Kiểm tra ảnh mở được không (loại file jpeg hỏng/rỗng) ---
    try:
        with Image.open(jpeg_path) as img:
            img.verify()
    except Exception:
        skipped_bad_image += 1
        continue

    records.append({
        "id": sample_id,
        "plot": text,
        "genres": "|".join(genres),   # nối bằng "|" để tránh xung đột dấu phẩy trong tên thể loại
        "n_genres": len(genres),
        "src_image_path": jpeg_path,
    })

df = pd.DataFrame(records)

print(f"Số mẫu ban đầu (có cặp json+jpeg): {len(paired_ids)}")
print(f"  - Bị loại do lỗi đọc json: {skipped_no_field}")
print(f"  - Bị loại do text quá ngắn (<{MIN_TEXT_LEN} ký tự): {skipped_short_text}")
print(f"  - Bị loại do không còn genre hợp lệ sau lọc: {skipped_no_genre}")
print(f"  - Bị loại do ảnh hỏng/không mở được: {skipped_bad_image}")
print(f"Số mẫu HỢP LỆ còn lại: {len(df)}")

assert len(df) > 0, "Không còn mẫu nào hợp lệ — kiểm tra lại đường dẫn RAW_DIR hoặc VALID_GENRES"

# ============================================================
# BƯỚC 3: KIỂM TRA TRÙNG LẶP id (an toàn, vì id lấy từ tên file)
# ============================================================
print("\n" + "=" * 60)
print("BƯỚC 3: KIỂM TRA TRÙNG LẶP id")
print("=" * 60)
dup_ids = df["id"].duplicated().sum()
print(f"Số id trùng lặp: {dup_ids}")
assert dup_ids == 0, "CÓ TRÙNG LẶP id — không nên xảy ra vì id lấy trực tiếp từ tên file"

# ============================================================
# BƯỚC 4: THỐNG KÊ PHÂN BỐ THỂ LOẠI TRƯỚC KHI SAMPLE
# ============================================================
print("\n" + "=" * 60)
print("BƯỚC 4: PHÂN BỐ THỂ LOẠI TRƯỚC KHI SAMPLE")
print("=" * 60)
genre_counts = df["genres"].str.split("|").explode().value_counts()
print(genre_counts)

# ============================================================
# BƯỚC 5: SAMPLE TỔNG SỐ TARGET_TOTAL MẪU
# (Tạm thời: random sample đơn giản, giữ nguyên phân bố tự nhiên.
#  TODO sau này: nâng cấp iterative stratification để cân bằng
#  đồng thời tất cả 23 thể loại thay vì chỉ random.)
# ============================================================
print("\n" + "=" * 60)
print("BƯỚC 5: SAMPLE MẪU")
print("=" * 60)

if len(df) > TARGET_TOTAL:
    df_sample = df.sample(n=TARGET_TOTAL, random_state=42).reset_index(drop=True)
else:
    df_sample = df.reset_index(drop=True)
    print(f"Lưu ý: số mẫu hợp lệ ({len(df)}) ít hơn TARGET_TOTAL ({TARGET_TOTAL}), giữ nguyên toàn bộ")

print(f"Số mẫu sau khi sample: {len(df_sample)}")
print("\nPhân bố thể loại sau khi sample:")
print(df_sample["genres"].str.split("|").explode().value_counts())

# ============================================================
# BƯỚC 6: COPY ẢNH TỪ RAW SANG PROCESSED (chỉ những mẫu được chọn)
# ============================================================
print("\n" + "=" * 60)
print("BƯỚC 6: COPY ẢNH ĐÃ CHỌN SANG THƯ MỤC PROCESSED")
print("=" * 60)

os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)

image_paths = []
for _, row in df_sample.iterrows():
    dst_filename = f"{row['id']}.jpeg"
    dst_path = os.path.join(OUTPUT_IMG_DIR, dst_filename)
    shutil.copy2(row["src_image_path"], dst_path)
    # Lưu đường dẫn TƯƠNG ĐỐI (so với OUTPUT_DIR) vào tsv để dễ mang theo cả thư mục
    image_paths.append(os.path.join("images", dst_filename))

df_sample["image_path"] = image_paths
df_sample = df_sample.drop(columns=["src_image_path"])

print(f"Đã copy {len(df_sample)} ảnh sang: {OUTPUT_IMG_DIR}")

# ============================================================
# BƯỚC 7: SAFETY NET TRƯỚC KHI LƯU
# ============================================================
print("\n" + "=" * 60)
print("BƯỚC 7: KIỂM TRA CUỐI CÙNG TRƯỚC KHI LƯU")
print("=" * 60)

null_final = df_sample[["id", "plot", "genres", "image_path"]].isnull().sum()
dup_final = df_sample["id"].duplicated().sum()
empty_genre_final = (df_sample["genres"].str.len() == 0).sum()

assert null_final.sum() == 0, "VẪN CÒN NULL sau xử lý"
assert dup_final == 0, "VẪN CÒN TRÙNG LẶP id"
assert empty_genre_final == 0, "VẪN CÒN MẪU RỖNG GENRE"
print("Dữ liệu đạt chuẩn — không null, không trùng lặp, không rỗng genre.")

# ============================================================
# BƯỚC 8: LƯU KẾT QUẢ
# ============================================================
print("\n" + "=" * 60)
print("BƯỚC 8: LƯU KẾT QUẢ")
print("=" * 60)

os.makedirs(OUTPUT_DIR, exist_ok=True)
FINAL_COLS = ["id", "plot", "genres", "n_genres", "image_path"]
df_sample[FINAL_COLS].to_csv(OUTPUT_TSV, sep="\t", index=False)

print(f"Đã lưu {len(df_sample)} dòng tại: {OUTPUT_TSV}")
print(f"Ảnh đã lưu tại: {OUTPUT_IMG_DIR}")
print("\nHOÀN TẤT TIỀN XỬ LÝ MMIMDB. Chạy tiếp check_mmimdb_ready.py để kiểm định.")