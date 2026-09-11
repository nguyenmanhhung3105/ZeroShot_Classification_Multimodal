"""
Preprocess CrisisMMD v2.0 annotation files into a single, clean TSV ready
for zero-shot multimodal classification.

Input : data/raw/CrisisMMD_v2.0/annotations/*.tsv   (6 event files)
Output: data/processed/pro_CrisisMMD/pre_crisismmd.tsv

-----------------------------------------------------------------------------
LABELING STRATEGY (why there are TWO label columns)
-----------------------------------------------------------------------------
CrisisMMD annotates text and image SEPARATELY (text_human vs image_human,
text_info vs image_info), and they don't always agree. For a *multimodal*
task we need one ground truth per sample per task. Following the standard
practice used in the CrisisMMD paper for multimodal experiments, for BOTH
tasks we:

  1. Keep only rows where the text label and the image label AGREE
     (text_human == image_human for Task 2, or text_info == image_info for
     Task 1). This becomes that task's label -> a much more reliable ground
     truth than picking either side alone.
  2. Additionally filter by confidence score (both text and image
     confidence must be >= CONF_THRESHOLD) to drop noisy/uncertain
     annotations.

Both tasks are computed independently and kept as two separate columns:

  - label_informative  -> Task 1, 2 classes (informative / not_informative)
  - label_humanitarian -> Task 2, 8 classes (humanitarian categories)

A row is kept ONLY IF it has a usable (non-NaN) label for BOTH tasks --
i.e. label_informative AND label_humanitarian must both be present. Rows
that pass one task's filter but not the other's are dropped entirely, so
neither column will ever contain NaN in the final output.

This naturally produces a smaller, high-quality evaluation set -- which is
exactly what a zero-shot project needs (no training data required, just a
trustworthy test set).

-----------------------------------------------------------------------------
TEXT CLEANING
-----------------------------------------------------------------------------
  - Removes "RT @user:" retweet prefix
  - Removes @mentions
  - Removes URLs (t.co links / any http(s) link)
  - Removes the "#" symbol but keeps the hashtag word
  - Fixes mojibake (encoding corruption), including double-corrupted text,
    using a known-pattern map + the `ftfy` library (if installed)
  - Strips any leftover non-ASCII garbage characters that neither of the
    above could fix (general catch-all, since these tweets are English and
    should end up fully ASCII after cleaning)
  - Collapses whitespace

-----------------------------------------------------------------------------
OUTPUT COLUMNS
-----------------------------------------------------------------------------
tweet_id | image_id | clean_text | image_path | image_url | label_informative | label_humanitarian | event

Usage:
    pip install pandas ftfy   # ftfy is optional but recommended
    python preprocess_crisismmd.py
"""

import os
import re
import glob
import pandas as pd

try:
    import ftfy  # pip install ftfy
    HAS_FTFY = True
except ImportError:
    HAS_FTFY = False
    print(
        "Note: 'ftfy' not installed (pip install ftfy). "
        "Mojibake fixing will rely only on the manual map + ASCII stripping."
    )

# =============================================================================
# CONFIG
# =============================================================================

RAW_DIR = "data/raw/CrisisMMD_v2.0/annotations"
IMAGE_ROOT = "data/raw/CrisisMMD_v2.0"  # folder containing "data_image/..."
OUT_DIR = "data/processed/pro_CrisisMMD"
OUT_FILE = os.path.join(OUT_DIR, "pre_crisismmd.tsv")

FILES = [
    "california_wildfires_final_data.tsv",
    "hurricane_harvey_final_data.tsv",
    "hurricane_irma_final_data.tsv",
    "hurricane_maria_final_data.tsv",
    "iraq_iran_earthquake_final_data.tsv",
    "mexico_earthquake_final_data.tsv",
    "srilanka_floods_final_data.tsv",
]

# Minimum confidence required on BOTH text and image annotation to keep a row
# (checked separately per task -- see load_and_clean).
CONF_THRESHOLD = 0.6

# Verify image_path actually exists on disk. Set to None to skip the check.
CHECK_IMAGE_EXISTS = True

# Strip any remaining non-ASCII character after mojibake fixing.
# Safe for this dataset (English tweets); disable if you need to preserve
# genuine accented characters (e.g. names) at the cost of some leftover junk.
STRIP_NON_ASCII = True

# Which task columns to keep from the raw files (both are always computed).
TASK_COLUMNS = {
    "informative": {
        "text_label": "text_info",
        "image_label": "image_info",
        "text_conf": "text_info_conf",
        "image_conf": "image_info_conf",
    },
    "humanitarian": {
        "text_label": "text_human",
        "image_label": "image_human",
        "text_conf": "text_human_conf",
        "image_conf": "image_human_conf",
    },
}

# =============================================================================
# TEXT CLEANING
# =============================================================================

# Known exact mojibake sequences seen in this dataset (including double-
# corrupted ones, e.g. an apostrophe corrupted twice in a row). Always
# applied first, regardless of whether ftfy is installed, since these are
# confirmed-correct fixes.
MOJIBAKE_MAP = {
    "\u221a\u00a2\u201a\u00c7\u00a8\u201a\u00d1\u00a2": "'",  # "√¢‚Ç¨‚Ñ¢" -> '
    "\u221a\u00a2\u201a\u00c7\u00ac": '"',                     # right double quote variant
    "\u221a\u00a2\u201a\u00c7\u00f4": '"',                     # left double quote variant
    "\u221a\u00a2\u201a\u00c7\u201d": "-",                     # dash variant
    "\u221a\u00a2\u201a\u00c7\u00a6": "...",                   # ellipsis variant
}

URL_RE = re.compile(r"https?://\S+")
MENTION_RE = re.compile(r"@\w+")
RT_PREFIX_RE = re.compile(r"^\s*RT\s*(@\w+)?:?\s*", flags=re.IGNORECASE)
HASHTAG_SYMBOL_RE = re.compile(r"#")
MULTI_SPACE_RE = re.compile(r"\s+")
NON_ASCII_RE = re.compile(r"[^\x00-\x7F]+")


def fix_mojibake(text: str, max_passes: int = 4) -> str:
    """Iteratively fix mojibake. Some entries in this dataset are corrupted
    TWICE in a row, so a single fix pass is often not enough."""
    if not text:
        return text

    for _ in range(max_passes):
        new_text = text

        # 1. Known exact fixes (always applied)
        for bad, good in MOJIBAKE_MAP.items():
            new_text = new_text.replace(bad, good)

        # 2. General-purpose fixer, if available
        if HAS_FTFY:
            new_text = ftfy.fix_text(new_text)

        if new_text == text:
            break
        text = new_text

    # 3. Last resort: manual encode/decode round-trip for anything still
    #    containing non-ASCII junk (common when text was UTF-8 misread as
    #    Latin-1/CP1252/Mac-Roman more than once).
    if NON_ASCII_RE.search(text):
        for enc in ("cp1252", "latin1", "mac_roman"):
            try:
                candidate = text.encode(enc, errors="ignore").decode(
                    "utf-8", errors="ignore"
                )
            except (UnicodeEncodeError, UnicodeDecodeError, LookupError):
                continue
            if candidate and len(NON_ASCII_RE.findall(candidate)) < len(
                NON_ASCII_RE.findall(text)
            ):
                text = candidate
                break

    # 4. General catch-all: strip whatever non-ASCII junk is still left
    if STRIP_NON_ASCII:
        text = NON_ASCII_RE.sub("", text)

    return text


def clean_tweet_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = fix_mojibake(text)
    text = RT_PREFIX_RE.sub("", text)
    text = URL_RE.sub("", text)
    text = MENTION_RE.sub("", text)
    text = HASHTAG_SYMBOL_RE.sub("", text)
    text = MULTI_SPACE_RE.sub(" ", text).strip()
    return text


# =============================================================================
# LOAD + CLEAN + LABEL ONE EVENT FILE
# =============================================================================

def load_and_clean(filepath: str, event_name: str) -> pd.DataFrame:
    info_cols = TASK_COLUMNS["informative"]
    human_cols = TASK_COLUMNS["humanitarian"]

    needed_raw_cols = [
        "tweet_id",
        "image_id",
        "tweet_text",
        "image_path",
        "image_url",
        info_cols["text_label"], info_cols["image_label"],
        info_cols["text_conf"], info_cols["image_conf"],
        human_cols["text_label"], human_cols["image_label"],
        human_cols["text_conf"], human_cols["image_conf"],
    ]

    df = pd.read_csv(filepath, sep="\t", dtype=str)

    missing = [c for c in needed_raw_cols if c not in df.columns]
    if missing:
        print(f"[{event_name}] Warning: missing columns {missing}, skipping file")
        return pd.DataFrame()

    df = df[needed_raw_cols].copy()
    df["event"] = event_name

    # Normalize literal "nan"/empty strings to real NaN
    df.replace({"nan": pd.NA, "NaN": pd.NA, "": pd.NA}, inplace=True)

    n0 = len(df)

    # --- Compute label_informative (Task 1, 2 classes) independently:
    #     confidence threshold on both sides + text/image agreement.
    df[info_cols["text_conf"]] = pd.to_numeric(df[info_cols["text_conf"]], errors="coerce")
    df[info_cols["image_conf"]] = pd.to_numeric(df[info_cols["image_conf"]], errors="coerce")
    info_conf_ok = (
        (df[info_cols["text_conf"]] >= CONF_THRESHOLD)
        & (df[info_cols["image_conf"]] >= CONF_THRESHOLD)
    )
    info_agree = df[info_cols["text_label"]] == df[info_cols["image_label"]]
    df["label_informative"] = df[info_cols["text_label"]].where(info_conf_ok & info_agree)

    # --- Compute label_humanitarian (Task 2, 8 classes) independently, same way.
    df[human_cols["text_conf"]] = pd.to_numeric(df[human_cols["text_conf"]], errors="coerce")
    df[human_cols["image_conf"]] = pd.to_numeric(df[human_cols["image_conf"]], errors="coerce")
    human_conf_ok = (
        (df[human_cols["text_conf"]] >= CONF_THRESHOLD)
        & (df[human_cols["image_conf"]] >= CONF_THRESHOLD)
    )
    human_agree = df[human_cols["text_label"]] == df[human_cols["image_label"]]
    df["label_humanitarian"] = df[human_cols["text_label"]].where(human_conf_ok & human_agree)

    n1_info = df["label_informative"].notna().sum()
    n1_human = df["label_humanitarian"].notna().sum()

    # --- Keep a row ONLY if it has a usable label for BOTH tasks
    df = df[df["label_informative"].notna() & df["label_humanitarian"].notna()]
    n1 = len(df)

    # --- Clean text
    df["clean_text"] = df["tweet_text"].apply(clean_tweet_text)
    df = df[df["clean_text"].str.len() > 0]
    n2 = len(df)

    # --- Drop duplicates
    df = df.drop_duplicates(subset=["tweet_id", "image_id"])
    n3 = len(df)

    # --- Verify image exists on disk
    if CHECK_IMAGE_EXISTS and IMAGE_ROOT:
        def image_exists(rel_path):
            if pd.isna(rel_path):
                return False
            return os.path.isfile(os.path.join(IMAGE_ROOT, rel_path))

        df = df[df["image_path"].apply(image_exists)]
    n4 = len(df)

    print(
        f"[{event_name}] raw={n0} "
        f"| usable_informative_label={n1_info} usable_humanitarian_label={n1_human} "
        f"-> after_keep_both_labels_present={n1} -> after_text_clean={n2} "
        f"-> after_dedupe={n3} -> after_image_check={n4}"
    )

    return df[[
        "tweet_id", "image_id", "clean_text", "image_path", "image_url",
        "label_informative", "label_humanitarian", "event",
    ]]


# =============================================================================
# MAIN
# =============================================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    all_dfs = []
    for filename in FILES:
        filepath = os.path.join(RAW_DIR, filename)
        if not glob.glob(filepath):
            print(f"Warning: file not found, skipping -> {filepath}")
            continue

        event_name = filename.replace("_final_data.tsv", "")
        df = load_and_clean(filepath, event_name)
        if not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        print("No data processed. Check RAW_DIR / FILES / CONF_THRESHOLD config.")
        return

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(OUT_FILE, sep="\t", index=False)

    print(f"\nSaved -> {OUT_FILE}")
    print(f"Total rows: {len(combined)}")

    print(f"\nRows with usable label_informative: {combined['label_informative'].notna().sum()}")
    print(combined["label_informative"].value_counts(dropna=False))

    print(f"\nRows with usable label_humanitarian: {combined['label_humanitarian'].notna().sum()}")
    print(combined["label_humanitarian"].value_counts(dropna=False))

    print("\nRows per event:")
    print(combined["event"].value_counts())

    # --- Final guarantee check: make sure no garbage characters survived.
    # This should always print 0 rows given STRIP_NON_ASCII=True; if it
    # doesn't, something in the cleaning pipeline was skipped/broken.
    leftover_mask = combined["clean_text"].apply(lambda t: bool(NON_ASCII_RE.search(t)))
    n_leftover = leftover_mask.sum()
    if n_leftover > 0:
        print(f"\nWARNING: {n_leftover} rows still contain non-ASCII characters:")
        print(combined.loc[leftover_mask, ["tweet_id", "clean_text"]].head(10))
    else:
        print("\nCheck passed: clean_text is fully ASCII in all rows.")


if __name__ == "__main__":
    main()