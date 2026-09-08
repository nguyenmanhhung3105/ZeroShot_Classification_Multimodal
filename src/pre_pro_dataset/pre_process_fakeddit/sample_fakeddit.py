import pandas as pd
import numpy as np

np.random.seed(42)  # để kết quả sample có thể tái lập lại (reproducible)

REQUIRED_COLS = ["id", "clean_title", "image_url", "hasImage", "2_way_label", "6_way_label"]

# ============================================================
# BƯỚC 1: LOAD BẢN ĐÃ LÀM SẠCH TỪ BƯỚC TRƯỚC (đọc an toàn hơn)
# ============================================================
print("="*60)
print("BƯỚC 1: LOAD DỮ LIỆU")
print("="*60)

# on_bad_lines="warn" giúp Pandas báo động nếu có dòng bị lệch cột khi đọc,
# thay vì âm thầm đọc sai rồi để lại lỗi null vô hình như đã gặp
df = pd.read_csv(
    "data/processed/fakeddit_clean.csv",
    on_bad_lines="warn",
    engine="python"  # engine python xử lý tốt hơn với text có ký tự đặc biệt
)
print(f"Số dòng đọc được: {len(df)}")

# ============================================================
# BƯỚC 2: KIỂM TRA & LOẠI BỎ NGAY CÁC DÒNG BỊ LỖI LỆCH CỘT
# (đây chính là fix tận gốc cho lỗi null xuất hiện ở lần trước)
# ============================================================
print("\n" + "="*60)
print("BƯỚC 2: KIỂM TRA & LOẠI DÒNG LỖI TRƯỚC KHI SAMPLE")
print("="*60)

null_before = df[REQUIRED_COLS].isnull().sum()
print("Số null theo cột (trước khi xử lý):")
print(null_before)

if null_before.sum() > 0:
    bad_rows = df[df[REQUIRED_COLS].isnull().any(axis=1)]
    print(f"\n Phát hiện {len(bad_rows)} dòng lỗi (null bất thường do lệch cột khi đọc CSV)")
    print("Ví dụ vài dòng lỗi:")
    print(bad_rows.head(3).to_string())

    df = df.dropna(subset=REQUIRED_COLS)
    print(f"\n Đã loại bỏ, còn lại {len(df)} dòng hợp lệ")
else:
    print(" Không có dòng lỗi nào")

# Ép kiểu lại cho 2 cột label — vì sau khi dropna, dtype có thể vẫn là float
# do trước đó lẫn NaN, cần ép về int để tránh lỗi so sánh ở bước sau
df["2_way_label"] = df["2_way_label"].astype(int)
df["6_way_label"] = df["6_way_label"].astype(int)

# ============================================================
# BƯỚC 3: LỌC TEXT QUÁ NGẮN
# ============================================================
print("\n" + "="*60)
print("BƯỚC 3: LỌC TEXT QUÁ NGẮN")
print("="*60)

before = len(df)
df = df[df["clean_title"].astype(str).str.len() >= 3]
print(f"Trước: {before} -> Sau khi lọc text ngắn: {len(df)}")

# ============================================================
# BƯỚC 4: SAMPLE SUBSET CÂN BẰNG THEO 6_way_label
# ============================================================
print("\n" + "="*60)
print("BƯỚC 4: SAMPLE SUBSET CÂN BẰNG")
print("="*60)

TARGET_TOTAL = 3000
MIN_PER_CLASS = 200

class_counts = df["6_way_label"].value_counts()
n_classes = len(class_counts)
print("Phân bố gốc trước khi sample:")
print(class_counts.sort_index())

sampled_dfs = []
for label, count in class_counts.items():
    n_sample = min(max(MIN_PER_CLASS, TARGET_TOTAL // n_classes), count)
    class_df = df[df["6_way_label"] == label].sample(n=n_sample, random_state=42)
    sampled_dfs.append(class_df)
    print(f"  Lớp {label}: lấy {n_sample}/{count} mẫu")

df_sample = pd.concat(sampled_dfs).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\n Tổng số mẫu sau khi sample: {len(df_sample)}")
print("Phân bố sau khi sample:")
print(df_sample["6_way_label"].value_counts().sort_index())

# ============================================================
# BƯỚC 5: KIỂM TRA LẠI LẦN CUỐI TRƯỚC KHI LƯU (safety net)
# — đảm bảo tuyệt đối không còn null/duplicate nào lọt qua
# ============================================================
print("\n" + "="*60)
print("BƯỚC 5: KIỂM TRA CUỐI CÙNG TRƯỚC KHI LƯU")
print("="*60)

null_final = df_sample[REQUIRED_COLS].isnull().sum()
dup_final = df_sample["id"].duplicated().sum()

print("Null theo cột (phải toàn bộ = 0):")
print(null_final)
print(f"Số id trùng lặp (phải = 0): {dup_final}")

assert null_final.sum() == 0, " VẪN CÒN NULL sau toàn bộ quy trình xử lý — cần kiểm tra lại file gốc"
assert dup_final == 0, " VẪN CÒN TRÙNG LẶP id"
print("\n Dữ liệu đạt chuẩn — không còn null, không còn trùng lặp")

# ============================================================
# BƯỚC 6: LƯU KẾT QUẢ DẠNG PARQUET (thay vì CSV để tránh lỗi lệch cột)
# ============================================================
print("\n" + "="*60)
print("BƯỚC 6: LƯU KẾT QUẢ")
print("="*60)

df_sample.to_csv(
    "data/processed/fakeddit_sample.tsv",
    sep="\t",
    index=False
)

print("Đã lưu subset tại: data/processed/fakeddit_sample.tsv")