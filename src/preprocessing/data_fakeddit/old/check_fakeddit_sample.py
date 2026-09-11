import pandas as pd

FILE_PATH = "data/processed/fakeddit_sample.tsv"


# ============================================================
# BƯỚC 1: LOAD & THÔNG TIN TỔNG QUAN
# ============================================================
print("="*60)
print("KIỂM TRA LẠI BẢN SAMPLE")
print("="*60)

df = pd.read_csv(FILE_PATH, sep="\t")
print(f"Tổng số dòng: {len(df)}")
print(f"Tổng số cột: {len(df.columns)}")

# ============================================================
# BƯỚC 2: KIỂM TRA NULL (phải là 0 hết, vì đã lọc từ bản clean)
# ============================================================
print("\n" + "="*60)
print("KIỂM TRA NULL")
print("="*60)
null_counts = df[["id", "clean_title", "image_url", "hasImage", "2_way_label", "6_way_label"]].isnull().sum()
print(null_counts)
assert null_counts.sum() == 0, " VẪN CÒN NULL — kiểm tra lại bước sample"

# ============================================================
# BƯỚC 3: KIỂM TRA TRÙNG LẶP id (đặc biệt quan trọng sau khi sample
# nhiều lớp rồi concat lại — dễ bị trùng nếu code sample sai)
# ============================================================
print("\n" + "="*60)
print("KIỂM TRA TRÙNG LẶP")
print("="*60)
dup_ids = df["id"].duplicated().sum()
print(f"Số dòng trùng 'id': {dup_ids}")
assert dup_ids == 0, " CÓ TRÙNG LẶP — có thể do sample chồng lấn giữa các lớp"

# ============================================================
# BƯỚC 4: KIỂM TRA PHÂN BỐ NHÃN — đây là điều quan trọng NHẤT
# cần xác nhận sau khi sample (đã cân bằng đúng như mong muốn chưa)
# ============================================================
print("\n" + "="*60)
print("PHÂN BỐ NHÃN SAU KHI SAMPLE")
print("="*60)

print("\n6_way_label:")
print(df["6_way_label"].value_counts().sort_index())

print("\n2_way_label:")
print(df["2_way_label"].value_counts().sort_index())

# Kiểm tra không có lớp nào bị "biến mất" hoặc quá lệch
min_class_count = df["6_way_label"].value_counts().min()
max_class_count = df["6_way_label"].value_counts().max()
ratio = max_class_count / min_class_count
print(f"\nTỷ lệ lớp nhiều nhất / lớp ít nhất: {ratio:.2f}x")
if ratio > 3:
    print(" CẢNH BÁO: vẫn còn lệch khá nhiều giữa các lớp, cân nhắc tăng MIN_PER_CLASS")
else:
    print(" Phân bố tương đối cân bằng")

# ============================================================
# BƯỚC 5: KIỂM TRA TEXT — xác nhận không còn text quá ngắn
# (đã lọc ở bước sample, giờ verify lại)
# ============================================================
print("\n" + "="*60)
print("KIỂM TRA TEXT SAU KHI SAMPLE")
print("="*60)
too_short = (df["clean_title"].astype(str).str.len() < 3).sum()
print(f"Số dòng text quá ngắn (<3 ký tự): {too_short}")
assert too_short == 0, " VẪN CÒN TEXT QUÁ NGẮN — bước lọc trước đó có vấn đề"

# Thống kê độ dài text để hiểu thêm về dữ liệu
print(f"\nĐộ dài text - trung bình: {df['clean_title'].str.len().mean():.1f} ký tự")
print(f"Độ dài text - min: {df['clean_title'].str.len().min()}, max: {df['clean_title'].str.len().max()}")

# ============================================================
# BƯỚC 6: XEM VÀI DÒNG MẪU THỰC TẾ TỪ MỖI LỚP
# (kiểm tra bằng mắt xem nội dung có hợp lý không, không chỉ dựa vào số liệu)
# ============================================================
print("\n" + "="*60)
print("VÍ DỤ MẪU THỰC TẾ TỪ TỪNG LỚP")
print("="*60)

label_names = {
    0: "True (thật)",
    1: "Satire/Parody",
    2: "False Connection",
    3: "Imposter Content",
    4: "Manipulated Content",
    5: "Misleading Content"
}

for label, name in label_names.items():
    subset = df[df["6_way_label"] == label]
    if len(subset) > 0:
        print(f"\n--- Lớp {label} ({name}) — {len(subset)} mẫu ---")
        print(subset[["clean_title"]].sample(min(2, len(subset)), random_state=1).to_string(index=False))

# ============================================================
# BƯỚC 7: KIỂM TRA ẢNH CÓ THẬT SỰ TẢI ĐƯỢC KHÔNG (sample thử vài url)
# (bước này quan trọng vì Bước 8 trước chỉ check ĐỊNH DẠNG url,
# chưa chắc url đó còn sống — link Reddit cũ rất hay bị chết)
# ============================================================
print("\n" + "="*60)
print("KIỂM TRA THỬ ẢNH CÓ TẢI ĐƯỢC KHÔNG (sample 10 url)")
print("="*60)

import requests
from io import BytesIO
from PIL import Image

sample_urls = df["image_url"].sample(10, random_state=1).tolist()
success, failed = 0, 0

for url in sample_urls:
    try:
        resp = requests.get(url, timeout=5)
        img = Image.open(BytesIO(resp.content))
        img.verify()  # kiểm tra file có phải ảnh hợp lệ không, không chỉ là status code 200
        success += 1
    except Exception as e:
        failed += 1
        print(f"   Lỗi với url: {url[:60]}... -> {type(e).__name__}")

print(f"\nKết quả: {success}/10 ảnh tải được, {failed}/10 lỗi")
if failed > 3:
    print(" Tỷ lệ lỗi khá cao — link Reddit cũ thường bị chết theo thời gian, cần lọc thêm bước kiểm tra ảnh sống trước khi dùng toàn bộ subset")

print("\n" + "="*60)
print(" HOÀN TẤT KIỂM TRA BẢN SAMPLE")
print("="*60)