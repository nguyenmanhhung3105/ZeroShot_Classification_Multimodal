"""
Demo test — CrisisMMD, task Informativeness.

Lấy 5 dòng đầu tiên trong data, với mỗi dòng:
- In ra text (clean_text)
- Hiển thị ảnh (dùng matplotlib)
- In ra nhãn thật (label_informative)
- In ra nhãn model zero-shot dự đoán

Chạy: python3 demo_crisismmd.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from models.model_registry import load_model
from inference import build_class_embeddings, predict_single_label
from prompts import crisismmd_prompts

# ============================================================
# CẤU HÌNH — chỉnh lại cho khớp đúng máy bạn
# ============================================================
DATA_PATH = "data/processed/pro_CrisisMMD/crisismmd_multimodal.tsv"       # đổi path/tên file cho khớp máy bạn
CRISISMMD_IMAGE_ROOT = "data/raw/CrisisMMD_v2.0"             # thư mục gốc chứa data_image/
N_SAMPLES = 10
# LABEL_COL = "label_informative"                          # đang test task Informativeness
LABEL_COL = "label_humanitarian"                         # đang test task Humanitarian
# TASK = "informativeness"
TASK = "humanitarian"


# ============================================================
# BƯỚC 1: LOAD DATA (đọc TSV, không phải Parquet)
# ============================================================
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Không tìm thấy {DATA_PATH}. Kiểm tra lại đường dẫn.")

# engine="python" + on_bad_lines="warn" giúp phát hiện sớm nếu có dòng bị lệch cột
# (lỗi từng gặp với text chứa ký tự đặc biệt khi lưu/đọc dạng phân tách bằng delimiter)
df = pd.read_csv(DATA_PATH, sep="\t", engine="python", on_bad_lines="warn")
print(f"✅ Load {len(df)} dòng từ {DATA_PATH}")

if LABEL_COL not in df.columns:
    raise ValueError(f"Không tìm thấy cột '{LABEL_COL}'. Các cột hiện có: {df.columns.tolist()}")

df_demo = df.head(N_SAMPLES).copy()

# Nối đường dẫn ảnh đầy đủ
df_demo["full_image_path"] = df_demo["image_path"].apply(
    lambda p: p if os.path.isabs(p) else os.path.join(CRISISMMD_IMAGE_ROOT, p)
)

# ============================================================
# BƯỚC 2: LOAD MODEL + PROMPT SET
# ============================================================
print("\nĐang load model CLIP (ViT-B/32) ...")
vlm = load_model("clip_vitb32")

prompt_set = crisismmd_prompts.get_prompt_set(TASK)
class_embeds, label_order = build_class_embeddings(vlm, prompt_set)
print(f"Các lớp trong task '{TASK}': {label_order}")

# ============================================================
# BƯỚC 3: LOAD ẢNH THẬT + DỰ ĐOÁN
# ============================================================
images = []
for p in df_demo["full_image_path"]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Ảnh không tồn tại: {p} — kiểm tra lại CRISISMMD_IMAGE_ROOT")
    images.append(Image.open(p).convert("RGB"))

predictions, sims = predict_single_label(vlm, images, class_embeds, label_order)
df_demo["predicted"] = predictions

vlm.unload()

# ============================================================
# BƯỚC 4: IN RA TEXT + NHÃN THẬT + NHÃN DỰ ĐOÁN
# ============================================================
print("\n" + "="*70)
print("KẾT QUẢ DEMO — 5 DÒNG ĐẦU TIÊN")
print("="*70)

for i, row in df_demo.iterrows():
    correct = "✅ ĐÚNG" if row[LABEL_COL] == row["predicted"] else "❌ SAI"
    print(f"\n--- Dòng {i+1} (tweet_id: {row['tweet_id']}) ---")
    print(f"Text        : {row['clean_text']}")
    print(f"Ảnh         : {row['full_image_path']}")
    print(f"Nhãn thật   : {row[LABEL_COL]}")
    print(f"Model đoán  : {row['predicted']}   {correct}")

# ============================================================
# BƯỚC 5: HIỂN THỊ ẢNH KÈM TEXT + NHÃN (dùng matplotlib)
# ============================================================
fig, axes = plt.subplots(1, N_SAMPLES, figsize=(4 * N_SAMPLES, 5))
if N_SAMPLES == 1:
    axes = [axes]

for ax, (i, row), img in zip(axes, df_demo.iterrows(), images):
    ax.imshow(img)
    ax.axis("off")

    title_text = row["clean_text"]
    if len(title_text) > 50:
        title_text = title_text[:50] + "..."

    match_symbol = "✅" if row[LABEL_COL] == row["predicted"] else "❌"
    ax.set_title(
        f"{title_text}\nThật: {row[LABEL_COL]}\nĐoán: {row['predicted']} {match_symbol}",
        fontsize=9
    )

plt.tight_layout()
output_path = "demo_crisismmd_result.png"
plt.savefig(output_path, dpi=120)
print(f"\n✅ Đã lưu ảnh minh hoạ tại: {output_path}")
plt.show()