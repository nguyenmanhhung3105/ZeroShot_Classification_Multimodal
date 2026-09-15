"""
Demo zero-shot classification — CrisisMMD.

Columns:
tweet_id | image_id | clean_text | image_path | image_url |
label_informative | label_humanitarian | event

Chạy:
    python3 src/demo_crisismmd.py
"""

import sys
import os
import math
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from models.model_registry import load_model
from prompts import crisismmd_prompts
from logging_utils import save_single_label_log


# ============================================================
# CONFIG
# ============================================================

# DATA_PATH = "data/processed/pro_fakeddit/fakeddit_sample.tsv"
# IMAGE_DIR = "data/processed/pro_fakeddit/images"

DATA_PATH = "data/processed/pro_crisismmd/pre_crisismmd.tsv"
IMAGE_DIR = "data/processed/pro_crisismmd/images"

# DATA_PATH = "data/processed/pro_mmimdb/mmimdb_sample.tsv"
# IMAGE_DIR = "data/processed/pro_mmimdb/images"

MODEL_NAME = "metaclip2_vith14"
# MODEL_NAME = "openclip_vith14_laion2b"
# MODEL_NAME = "siglip2_so400m"

N_SAMPLES = 10
BATCH_SIZE = 4

TASK = "informativeness"
LABEL_COL = "label_informative"

LOG_DIR = "results/demo_logs"
PROMPT_VERSION = "v1"

# Nếu muốn test humanitarian:
# TASK = "humanitarian"
# LABEL_COL = "label_humanitarian"

ID_COL = "tweet_id"
TEXT_COL = "clean_text"
IMAGE_COL = "image_path"

OUTPUT_DIR = "results/demo"


# ============================================================
# BUILD CLASS EMBEDDINGS
# ============================================================

@torch.inference_mode()
def build_class_embeddings(vlm, prompt_set: dict) -> tuple:
    label_order = list(prompt_set.keys())
    class_embeds = []

    for label in label_order:
        prompts = prompt_set[label]
        if isinstance(prompts, str): prompts = [prompts]

        text_embeds = vlm.encode_texts(prompts)
        mean_embed = text_embeds.mean(dim=0)
        mean_embed = mean_embed / mean_embed.norm().clamp_min(1e-12)

        class_embeds.append(mean_embed)

    return torch.stack(class_embeds, dim=0), label_order


# ============================================================
# SINGLE-LABEL INFERENCE
# ============================================================

@torch.inference_mode()
def predict_single_label(vlm, images: list, class_embeds: torch.Tensor, label_order: list, batch_size: int = 4) -> tuple:
    predictions = []
    all_sims = []

    for start in range(0, len(images), batch_size):
        batch_images = images[start:start + batch_size]

        image_embeds = vlm.encode_images(batch_images)
        sims = vlm.similarity(image_embeds, class_embeds)

        pred_indices = sims.argmax(dim=1).cpu().tolist()

        predictions.extend([label_order[i] for i in pred_indices])
        all_sims.append(sims.cpu())

    sims = torch.cat(all_sims, dim=0).numpy()

    return predictions, sims


# ============================================================
# LOAD DATA
# ============================================================

if not os.path.exists(DATA_PATH): raise FileNotFoundError(f"Không tìm thấy dataset: {DATA_PATH}")

df = pd.read_csv(DATA_PATH, sep="\t", engine="python", on_bad_lines="warn")

required_cols = [ID_COL, TEXT_COL, IMAGE_COL, LABEL_COL]
missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols: raise ValueError(f"Thiếu cột: {missing_cols}. Các cột hiện có: {df.columns.tolist()}")

print(f"[DATA] Tổng số mẫu: {len(df)}")
print(f"[DATA] Columns: {df.columns.tolist()}")

df_demo = df.head(N_SAMPLES).copy()


# ============================================================
# LOAD IMAGES
# ============================================================

images = []
valid_indices = []
full_image_paths = []

for i, row in df_demo.iterrows():
    image_path = os.path.join(IMAGE_DIR, str(row[IMAGE_COL]))

    if not os.path.exists(image_path):
        print(f"[WARNING] Không tìm thấy ảnh: {image_path}")
        continue

    try:
        with Image.open(image_path) as img: images.append(img.convert("RGB"))
        valid_indices.append(i)
        full_image_paths.append(image_path)
    except Exception as e:
        print(f"[WARNING] Không đọc được ảnh {image_path}: {e}")

df_demo = df_demo.loc[valid_indices].copy().reset_index(drop=True)
df_demo["full_image_path"] = full_image_paths

if not images: raise RuntimeError("Không có ảnh hợp lệ để chạy demo.")

print(f"[DATA] Ảnh hợp lệ: {len(images)}/{N_SAMPLES}")


# ============================================================
# LOAD MODEL
# ============================================================

print(f"\n[MODEL] Loading: {MODEL_NAME}")

vlm = load_model(MODEL_NAME)

prompt_set = crisismmd_prompts.get_prompt_set(TASK)
class_embeds, label_order = build_class_embeddings(vlm, prompt_set)

print(f"[MODEL] Task: {TASK}")
print(f"[MODEL] Classes: {label_order}")
print(f"[MODEL] Class embeddings shape: {class_embeds.shape}")


# ============================================================
# INFERENCE
# ============================================================

predictions, sims = predict_single_label(vlm, images, class_embeds, label_order, batch_size=BATCH_SIZE)

df_demo["predicted"] = predictions

y_true = df_demo[LABEL_COL].astype(str).tolist()

model_name = vlm.model_name


# ============================================================
# LOGGING
# ============================================================

os.makedirs(LOG_DIR, exist_ok=True)

log_dir = save_single_label_log(
    df_valid=df_demo,
    y_true=y_true,
    y_pred=predictions,
    sims=sims,
    label_order=label_order,
    id_col=ID_COL,
    text_col=TEXT_COL,
    model_name=vlm.model_name,
    dataset_name="crisismmd",
    task_name=TASK,
    output_dir=LOG_DIR,
    prompt_version=PROMPT_VERSION
)

print(f"[LOG] Đã lưu tại: {log_dir}")


# ============================================================
# PRINT RESULT
# ============================================================

print("\n" + "=" * 80)
print(f"KẾT QUẢ DEMO — CrisisMMD / {TASK} / {model_name}")
print("=" * 80)

n_correct = 0

for i, row in df_demo.iterrows():
    true_label = str(row[LABEL_COL])
    pred_label = str(row["predicted"])
    correct = true_label == pred_label

    if correct: n_correct += 1

    print(f"\n--- Mẫu {i + 1} | tweet_id: {row[ID_COL]} ---")
    print(f"Text      : {row[TEXT_COL]}")
    print(f"Ảnh       : {row['full_image_path']}")
    print(f"Nhãn thật : {true_label}")
    print(f"Dự đoán   : {pred_label} {'✓' if correct else '✗'}")

    print("Scores    :", end=" ")
    for j, label in enumerate(label_order): print(f"{label}={sims[i, j]:.4f}", end="  ")
    print()

accuracy = n_correct / len(df_demo)

print("\n" + "-" * 80)
print(f"Demo Accuracy: {n_correct}/{len(df_demo)} = {accuracy:.2%}")
print("-" * 80)


# ============================================================
# VISUALIZE
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

n_cols = min(5, len(df_demo))
n_rows = math.ceil(len(df_demo) / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 5 * n_rows), squeeze=False)
axes = axes.flatten()

for ax, (_, row), img in zip(axes, df_demo.iterrows(), images):
    true_label = str(row[LABEL_COL])
    pred_label = str(row["predicted"])
    correct = true_label == pred_label

    text = str(row[TEXT_COL])
    if len(text) > 60: text = text[:60] + "..."

    ax.imshow(img)
    ax.axis("off")
    ax.set_title(f"{text}\nTrue: {true_label}\nPred: {pred_label} {'✓' if correct else '✗'}", fontsize=9)

for ax in axes[len(df_demo):]: ax.axis("off")

plt.suptitle(f"CrisisMMD — {TASK} — {model_name}", fontsize=14)
plt.tight_layout()

output_path = os.path.join(OUTPUT_DIR, f"demo_crisismmd_{TASK}_{model_name}.png")

plt.savefig(output_path, dpi=150, bbox_inches="tight")

print(f"\n[OUTPUT] Đã lưu: {output_path}")

vlm.unload()

plt.show()