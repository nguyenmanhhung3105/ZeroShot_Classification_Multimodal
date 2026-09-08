import pandas as pd
import os

FILE_PATH = "data/raw/fakeddit/multimodal_train.tsv"

# FILE_PATH = "data/processed/fakeddit_sample.csv"

# ============================================================
# BƯỚC 1: LOAD DỮ LIỆU
# ============================================================
print("="*60)
print("BƯỚC 1: LOAD DỮ LIỆU")
print("="*60)

if not os.path.exists(FILE_PATH):
    raise FileNotFoundError(f"Không tìm thấy file tại {FILE_PATH} — kiểm tra lại đường dẫn")

df = pd.read_csv(FILE_PATH, sep="\t")
# df = pd.read_csv(FILE_PATH, sep=";")
print(f"Tổng số dòng: {len(df)}")
print(f"Tổng số cột: {len(df.columns)}")
print(f"Danh sách cột: {df.columns.tolist()}")

# ============================================================
# BƯỚC 2: KIỂM TRA CỘT CẦN THIẾT CÓ TỒN TẠI KHÔNG
# ============================================================
print("\n" + "="*60)
print("BƯỚC 2: KIỂM TRA CỘT CẦN THIẾT")
print("="*60)

required_cols = ["id", "clean_title", "image_url", "hasImage", "2_way_label", "6_way_label"]
missing_cols = [c for c in required_cols if c not in df.columns]

if missing_cols:
    print(f" CẢNH BÁO: Thiếu các cột sau: {missing_cols}")
else:
    print(" Đầy đủ tất cả cột cần thiết")

# ============================================================
# BƯỚC 3: KIỂM TRA GIÁ TRỊ NULL / MISSING
# ============================================================
print("\n" + "="*60)
print("BƯỚC 3: KIỂM TRA NULL / MISSING")
print("="*60)

null_counts = df[required_cols].isnull().sum() if not missing_cols else df.isnull().sum()
print(null_counts)

null_pct = (null_counts / len(df) * 100).round(2)
print("\nTỷ lệ % null theo từng cột:")
print(null_pct)

# ============================================================
# BƯỚC 4: KIỂM TRA DUPLICATE (TRÙNG LẶP)
# ============================================================
print("\n" + "="*60)
print("BƯỚC 4: KIỂM TRA TRÙNG LẶP")
print("="*60)

if "id" in df.columns:
    dup_ids = df["id"].duplicated().sum()
    print(f"Số dòng trùng 'id': {dup_ids}")

dup_rows = df.duplicated().sum()
print(f"Số dòng trùng lặp hoàn toàn (mọi cột): {dup_rows}")

# ============================================================
# BƯỚC 5: KIỂM TRA TEXT RỖNG / QUÁ NGẮN (clean_title)
# ============================================================
print("\n" + "="*60)
print("BƯỚC 5: KIỂM TRA TEXT (clean_title)")
print("="*60)

if "clean_title" in df.columns:
    empty_text = df["clean_title"].isna().sum()
    blank_text = (df["clean_title"].astype(str).str.strip() == "").sum()
    too_short = (df["clean_title"].astype(str).str.len() < 3).sum()

    print(f"Số dòng clean_title là NaN: {empty_text}")
    print(f"Số dòng clean_title rỗng (chuỗi trắng): {blank_text}")
    print(f"Số dòng clean_title quá ngắn (<3 ký tự): {too_short}")

    # In vài ví dụ text bị lỗi để xem thực tế
    if too_short > 0:
        print("\nVí dụ vài dòng text quá ngắn:")
        print(df[df["clean_title"].astype(str).str.len() < 3][["id", "clean_title"]].head())

# ============================================================
# BƯỚC 6: KIỂM TRA NHÃN CÓ HỢP LỆ KHÔNG (đúng khoảng giá trị)
# ============================================================
print("\n" + "="*60)
print("BƯỚC 6: KIỂM TRA TÍNH HỢP LỆ CỦA NHÃN")
print("="*60)

if "2_way_label" in df.columns:
    invalid_2way = df[~df["2_way_label"].isin([0, 1])]
    print(f"Số dòng có 2_way_label KHÔNG thuộc {{0,1}}: {len(invalid_2way)}")
    print("Phân bố 2_way_label:")
    print(df["2_way_label"].value_counts(dropna=False))

if "6_way_label" in df.columns:
    invalid_6way = df[~df["6_way_label"].isin(range(6))]
    print(f"\nSố dòng có 6_way_label KHÔNG thuộc {{0..5}}: {len(invalid_6way)}")
    print("Phân bố 6_way_label:")
    print(df["6_way_label"].value_counts(dropna=False))

# ============================================================
# BƯỚC 7: KIỂM TRA MISMATCH — hasImage vs image_url
# (đây là lỗi hay gặp nhất: cột nói CÓ ảnh nhưng url lại rỗng, hoặc ngược lại)
# ============================================================
print("\n" + "="*60)
print("BƯỚC 7: KIỂM TRA MISMATCH hasImage vs image_url")
print("="*60)

if "hasImage" in df.columns and "image_url" in df.columns:
    # Trường hợp 1: nói có ảnh (True) nhưng url lại rỗng/NaN
    case1 = df[(df["hasImage"] == True) & (df["image_url"].isna())]
    print(f"Trường hợp hasImage=True nhưng image_url rỗng: {len(case1)}")

    # Trường hợp 2: nói không có ảnh (False) nhưng url lại có giá trị
    case2 = df[(df["hasImage"] == False) & (df["image_url"].notna())]
    print(f"Trường hợp hasImage=False nhưng image_url có giá trị: {len(case2)}")

    if len(case1) > 0:
        print("\nVí dụ vài dòng bị case 1 (cần loại bỏ khỏi tập multimodal):")
        print(case1[["id", "hasImage", "image_url"]].head())

# ============================================================
# BƯỚC 8: KIỂM TRA URL ẢNH CÓ ĐÚNG ĐỊNH DẠNG KHÔNG
# ============================================================
print("\n" + "="*60)
print("BƯỚC 8: KIỂM TRA ĐỊNH DẠNG image_url")
print("="*60)

if "image_url" in df.columns:
    valid_url_pattern = df["image_url"].astype(str).str.startswith(("http://", "https://"))
    invalid_urls = (~valid_url_pattern) & df["image_url"].notna()
    print(f"Số dòng có image_url không đúng định dạng http(s)://: {invalid_urls.sum()}")

    if invalid_urls.sum() > 0:
        print("\nVí dụ vài url lỗi:")
        print(df[invalid_urls][["id", "image_url"]].head())

# ============================================================
# BƯỚC 9: TÍNH SỐ MẪU "SẠCH" CUỐI CÙNG DÙNG ĐƯỢC
# (Sau khi loại bỏ null, mismatch, url lỗi)
# ============================================================
print("\n" + "="*60)
print("BƯỚC 9: TỔNG KẾT SỐ MẪU SẠCH CÓ THỂ DÙNG")
print("="*60)

df_clean = df.copy()

if "clean_title" in df_clean.columns:
    df_clean = df_clean[df_clean["clean_title"].notna()]
    df_clean = df_clean[df_clean["clean_title"].astype(str).str.strip() != ""]

if "hasImage" in df_clean.columns and "image_url" in df_clean.columns:
    df_clean = df_clean[df_clean["hasImage"] == True]
    df_clean = df_clean[df_clean["image_url"].notna()]
    df_clean = df_clean[df_clean["image_url"].astype(str).str.startswith(("http://", "https://"))]

if "id" in df_clean.columns:
    df_clean = df_clean.drop_duplicates(subset="id")

print(f"Số dòng gốc: {len(df)}")
print(f"Số dòng SẠCH (dùng được cho zero-shot): {len(df_clean)}")
print(f"Tỷ lệ giữ lại: {len(df_clean)/len(df)*100:.2f}%")

# Lưu lại bản đã làm sạch để dùng cho bước sau
os.makedirs("data/processed", exist_ok=True)
# df_clean.to_csv("data/processed/fakeddit_clean.csv", index=False)

df_clean.to_csv("data/processed/pro_fakeddit/fakeddit_clean.tsv",
    sep="\t",
    index=False
)

print("\n Đã lưu bản sạch tại: data/processed/pro_fakeddit/fakeddit_clean.tsv")