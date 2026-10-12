"""
Kiểm tra TSV CrisisMMD đã xử lý; có lựa chọn copy ảnh sang thư mục processed.

Kiểm tra cột, nhãn, text đã làm sạch, cặp ID và đường dẫn ảnh.
Mặc định không đọc/copy ảnh từ nguồn thô. File này không tạo TSV từ annotation
gốc; chỉ kiểm tra bản processed đã có. Không chạy khi rà soát mã nguồn.
"""

import os
import re
import sys
import shutil
import pandas as pd

# =============================================================================
# CẤU HÌNH: đồng bộ với preprocess_crisismmd.py
# =============================================================================

IN_FILE = "data/processed/pro_crisismmd/pre_crisismmd.tsv"
IMAGE_ROOT = "data/raw/CrisisMMD_v2.0"  # Thư mục chứa "data_image/..."
CHECK_IMAGE_EXISTS = False  # Ảnh thiếu/hỏng được bỏ qua tại inference.

# cấu hình cho bước trích xuất/copy ảnh
COPY_IMAGES = False  # Ảnh processed đã được chuẩn bị; chỉ copy khi chủ động bật.
OUT_IMAGE_DIR = "data/processed/pro_crisismmd/images"

EXPECTED_COLUMNS = [
    "tweet_id", "image_id", "clean_text", "image_path", "image_url",
    "label_informative", "label_humanitarian", "event",
]

INFORMATIVE_CLASSES = {"informative", "not_informative"}

HUMANITARIAN_CLASSES = {
    "infrastructure_and_utility_damage",
    "not_humanitarian",
    "other_relevant_information",
    "rescue_volunteering_or_donation_effort",
    "vehicle_damage",
    "affected_individuals",
    "injured_or_dead_people",
    "missing_or_found_people",
}

NON_ASCII_RE = re.compile(r"[^\x00-\x7F]+")
URL_RE = re.compile(r"https?://\S+")
MENTION_RE = re.compile(r"@\w+")
RT_PREFIX_RE = re.compile(r"^\s*RT\b", flags=re.IGNORECASE)
HASHTAG_SYMBOL_RE = re.compile(r"#")

FAILURES = []  # Lưu tên kiểm tra và các dòng lỗi để tổng hợp


def report(check_name: str, bad_df: pd.DataFrame, max_show: int = 5, cols=None):
    """Lưu và in kết quả kiểm tra."""
    n_bad = len(bad_df)
    if n_bad == 0:
        print(f"[PASS] {check_name}")
        return
    print(f"[FAIL] {check_name} -- {n_bad} offending row(s)")
    show_cols = cols if cols else bad_df.columns.tolist()
    with pd.option_context("display.max_colwidth", 60):
        print(bad_df[show_cols].head(max_show).to_string(index=False))
    FAILURES.append((check_name, bad_df))


# =============================================================================
# Bước trích xuất / copy ảnh sang thư mục processed
# =============================================================================

def copy_images_to_processed(df: pd.DataFrame,
                              image_root: str = IMAGE_ROOT,
                              out_dir: str = OUT_IMAGE_DIR):
    """
    Copy toàn bộ ảnh được tham chiếu bởi cột image_path (nằm trong image_root,
    ví dụ .../data_image/...) sang out_dir, giữ nguyên cấu trúc thư mục con
    tương đối của image_path.

    Ví dụ:
        image_root = "data/raw/CrisisMMD_v2.0"
        image_path = "data_image/hurricane_irma/917791.jpg"
        -> copy sang: "data/processed/pro_crisismmd/images/data_image/hurricane_irma/917791.jpg"

    Trả về (n_copied, n_skipped_existing, n_missing_source, missing_list).
    """
    os.makedirs(out_dir, exist_ok=True)

    # Lấy danh sách image_path duy nhất, bỏ NaN
    unique_paths = (
        df["image_path"]
        .dropna()
        .astype(str)
        .map(str.strip)
        .loc[lambda s: s != ""]
        .unique()
    )

    n_copied = 0
    n_skipped_existing = 0
    n_missing_source = 0
    missing_list = []

    for rel_path in unique_paths:
        src = os.path.join(image_root, rel_path)
        dst = os.path.join(out_dir, rel_path)

        if not os.path.isfile(src):
            n_missing_source += 1
            missing_list.append(rel_path)
            continue

        os.makedirs(os.path.dirname(dst), exist_ok=True)

        if os.path.isfile(dst):
            n_skipped_existing += 1
            continue

        shutil.copy2(src, dst)
        n_copied += 1

    return n_copied, n_skipped_existing, n_missing_source, missing_list


def main():
    if not os.path.isfile(IN_FILE):
        print(f"ERROR: cần TSV processed đã chuẩn bị -> {IN_FILE}. "
              "Script này kiểm tra/copy ảnh, không tạo TSV từ annotation thô.")
        sys.exit(1)

    df = pd.read_csv(IN_FILE, sep="\t", dtype=str)
    n = len(df)
    print(f"Loaded {n} rows from {IN_FILE}\n")

    # 1. Kiểm tra tên và thứ tự cột -------------------------------------------------
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    extra_cols = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    if missing_cols or extra_cols:
        print(f"[FAIL] expected columns -- missing={missing_cols} extra={extra_cols}")
        FAILURES.append(("expected columns", pd.DataFrame()))
    else:
        print("[PASS] expected columns")
    print()

    if missing_cols:
        print("Cannot continue further checks without required columns. Exiting.")
        sys.exit(1)

    # 2. Không thiếu giá trị ở cột bắt buộc ----------------------------------------
    for col in ["clean_text", "image_path", "label_informative", "label_humanitarian",
                "tweet_id", "image_id"]:
        bad = df[df[col].isna() | (df[col].astype(str).str.strip() == "")]
        report(f"no missing values in '{col}'", bad, cols=["tweet_id", "image_id", col])

    # 3. Nhãn informative thuộc bộ lớp cho phép ------------------------------
    bad = df[~df["label_informative"].isin(INFORMATIVE_CLASSES)]
    report("label_informative in expected 2 classes", bad,
           cols=["tweet_id", "label_informative"])

    # 4. Nhãn humanitarian thuộc bộ lớp cho phép -----------------------------
    bad = df[~df["label_humanitarian"].isin(HUMANITARIAN_CLASSES)]
    report("label_humanitarian in expected 8 classes", bad,
           cols=["tweet_id", "label_humanitarian"])

    # 5. clean_text chỉ chứa ký tự ASCII -------------------------------------------------
    bad_mask = df["clean_text"].apply(lambda t: bool(NON_ASCII_RE.search(str(t))))
    report("clean_text is fully ASCII", df[bad_mask], cols=["tweet_id", "clean_text"])

    # 6. Text không còn tiền tố RT, mention, URL hoặc ký hiệu hashtag -------
    bad_mask = df["clean_text"].apply(lambda t: bool(RT_PREFIX_RE.match(str(t))))
    report("no leftover 'RT' prefix in clean_text", df[bad_mask], cols=["tweet_id", "clean_text"])

    bad_mask = df["clean_text"].apply(lambda t: bool(MENTION_RE.search(str(t))))
    report("no leftover '@mentions' in clean_text", df[bad_mask], cols=["tweet_id", "clean_text"])

    bad_mask = df["clean_text"].apply(lambda t: bool(URL_RE.search(str(t))))
    report("no leftover URLs in clean_text", df[bad_mask], cols=["tweet_id", "clean_text"])

    bad_mask = df["clean_text"].apply(lambda t: bool(HASHTAG_SYMBOL_RE.search(str(t))))
    report("no leftover '#' symbol in clean_text", df[bad_mask], cols=["tweet_id", "clean_text"])

    # 7. Không trùng cặp tweet_id/image_id ------------------------------------
    dup_mask = df.duplicated(subset=["tweet_id", "image_id"], keep=False)
    report("no duplicate (tweet_id, image_id) pairs", df[dup_mask],
           cols=["tweet_id", "image_id"])

    # 8. Đường dẫn ảnh tồn tại ---------------------------------------------------
    if CHECK_IMAGE_EXISTS:
        def missing_image(rel_path):
            if pd.isna(rel_path):
                return True
            return not os.path.isfile(os.path.join(IMAGE_ROOT, rel_path))

        bad_mask = df["image_path"].apply(missing_image)
        report("image_path exists on disk", df[bad_mask],
               cols=["tweet_id", "image_path"])
    else:
        print("[SKIP] image_path exists on disk (CHECK_IMAGE_EXISTS=False)")

    # 9. Kiểm tra định dạng tweet_id/image_id -------------------------------
    id_re = re.compile(r"^\S+$")
    for col in ["tweet_id", "image_id"]:
        bad_mask = ~df[col].astype(str).apply(lambda v: bool(id_re.match(v)))
        report(f"'{col}' looks like a valid identifier", df[bad_mask], cols=["tweet_id", "image_id"])

    # --- Tổng kết ---------------------------------------------------------------------
    print("\n" + "=" * 70)
    if FAILURES:
        print(f"SUMMARY: {len(FAILURES)} check(s) FAILED out of the run above.")
        for name, bad_df in FAILURES:
            print(f"  - {name} ({len(bad_df)} bad rows)")
        checks_passed = False
    else:
        print(f"SUMMARY: all checks PASSED on {n} rows.")
        print("\nLabel distributions:")
        print("\nlabel_informative:")
        print(df["label_informative"].value_counts())
        print("\nlabel_humanitarian:")
        print(df["label_humanitarian"].value_counts())
        print("\nRows per event:")
        print(df["event"].value_counts())
        checks_passed = True

    # --- 10. NEW: copy ảnh sang data/processed/pro_crisismmd/images ------------------
    if COPY_IMAGES and checks_passed:
        print("\n" + "=" * 70)
        print(f"Copying images -> {OUT_IMAGE_DIR}")
        n_copied, n_skipped, n_missing, missing_list = copy_images_to_processed(df)
        print(f"  copied      : {n_copied}")
        print(f"  already had : {n_skipped}")
        print(f"  missing src : {n_missing}")
        if missing_list:
            print("  -- first missing paths --")
            for p in missing_list[:5]:
                print(f"     {p}")
    else:
        print("\n[SKIP] copy images (COPY_IMAGES=False)")

    sys.exit(0 if checks_passed else 1)


if __name__ == "__main__":
    main()
