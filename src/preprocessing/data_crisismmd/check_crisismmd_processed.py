"""
Sanity-check script for the output of preprocess_crisismmd.py.

Loads data/processed/pro_CrisisMMD/pre_crisismmd.tsv and re-verifies
every guarantee the preprocessing pipeline is supposed to have enforced:

  1. Expected columns are present, in a sane order.
  2. No missing values in either label column (label_informative,
     label_humanitarian) or in clean_text / image_path.
  3. label_informative only contains the 2 expected classes.
  4. label_humanitarian only contains the 8 expected classes.
  5. clean_text is fully ASCII (no leftover mojibake / non-ASCII junk).
  6. clean_text has no leftover "RT @user:", "@mentions", URLs, or "#"
     symbols (hashtag word should remain, symbol should not).
  7. No duplicate (tweet_id, image_id) pairs.
  8. Every image_path actually exists on disk (relative to IMAGE_ROOT).
  9. tweet_id / image_id look like non-empty identifiers.

Any failed check prints the offending rows (first few) and the script exits
with a non-zero status code, so it can also be used as a CI gate.

Usage:
    pip install pandas
    python check_crisismmd_processed.py
"""

import os
import re
import sys
import pandas as pd

# =============================================================================
# CONFIG - keep in sync with preprocess_crisismmd.py
# =============================================================================

IN_FILE = "data/processed/pro_CrisisMMD/pre_crisismmd.tsv"
IMAGE_ROOT = "data/raw/CrisisMMD_v2.0"  # folder containing "data_image/..."
CHECK_IMAGE_EXISTS = True

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

FAILURES = []  # collect (check_name, DataFrame_of_bad_rows) for the summary


def report(check_name: str, bad_df: pd.DataFrame, max_show: int = 5, cols=None):
    """Record + print a check's result."""
    n_bad = len(bad_df)
    if n_bad == 0:
        print(f"[PASS] {check_name}")
        return
    print(f"[FAIL] {check_name} -- {n_bad} offending row(s)")
    show_cols = cols if cols else bad_df.columns.tolist()
    with pd.option_context("display.max_colwidth", 60):
        print(bad_df[show_cols].head(max_show).to_string(index=False))
    FAILURES.append((check_name, bad_df))


def main():
    if not os.path.isfile(IN_FILE):
        print(f"ERROR: input file not found -> {IN_FILE}")
        sys.exit(1)

    df = pd.read_csv(IN_FILE, sep="\t", dtype=str)
    n = len(df)
    print(f"Loaded {n} rows from {IN_FILE}\n")

    # 1. Column presence / order -------------------------------------------------
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

    # 2. No missing values in key columns ----------------------------------------
    for col in ["clean_text", "image_path", "label_informative", "label_humanitarian",
                "tweet_id", "image_id"]:
        bad = df[df[col].isna() | (df[col].astype(str).str.strip() == "")]
        report(f"no missing values in '{col}'", bad, cols=["tweet_id", "image_id", col])

    # 3. label_informative only has expected classes ------------------------------
    bad = df[~df["label_informative"].isin(INFORMATIVE_CLASSES)]
    report("label_informative in expected 2 classes", bad,
           cols=["tweet_id", "label_informative"])

    # 4. label_humanitarian only has expected classes -----------------------------
    bad = df[~df["label_humanitarian"].isin(HUMANITARIAN_CLASSES)]
    report("label_humanitarian in expected 8 classes", bad,
           cols=["tweet_id", "label_humanitarian"])

    # 5. clean_text is fully ASCII -------------------------------------------------
    bad_mask = df["clean_text"].apply(lambda t: bool(NON_ASCII_RE.search(str(t))))
    report("clean_text is fully ASCII", df[bad_mask], cols=["tweet_id", "clean_text"])

    # 6. clean_text has no leftover RT-prefix / mentions / URLs / '#' symbol -------
    bad_mask = df["clean_text"].apply(lambda t: bool(RT_PREFIX_RE.match(str(t))))
    report("no leftover 'RT' prefix in clean_text", df[bad_mask], cols=["tweet_id", "clean_text"])

    bad_mask = df["clean_text"].apply(lambda t: bool(MENTION_RE.search(str(t))))
    report("no leftover '@mentions' in clean_text", df[bad_mask], cols=["tweet_id", "clean_text"])

    bad_mask = df["clean_text"].apply(lambda t: bool(URL_RE.search(str(t))))
    report("no leftover URLs in clean_text", df[bad_mask], cols=["tweet_id", "clean_text"])

    bad_mask = df["clean_text"].apply(lambda t: bool(HASHTAG_SYMBOL_RE.search(str(t))))
    report("no leftover '#' symbol in clean_text", df[bad_mask], cols=["tweet_id", "clean_text"])

    # 7. No duplicate (tweet_id, image_id) pairs ------------------------------------
    dup_mask = df.duplicated(subset=["tweet_id", "image_id"], keep=False)
    report("no duplicate (tweet_id, image_id) pairs", df[dup_mask],
           cols=["tweet_id", "image_id"])

    # 8. image_path exists on disk ---------------------------------------------------
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

    # 9. tweet_id / image_id look like real identifiers -------------------------------
    id_re = re.compile(r"^\S+$")
    for col in ["tweet_id", "image_id"]:
        bad_mask = ~df[col].astype(str).apply(lambda v: bool(id_re.match(v)))
        report(f"'{col}' looks like a valid identifier", df[bad_mask], cols=["tweet_id", "image_id"])

    # --- Summary ---------------------------------------------------------------------
    print("\n" + "=" * 70)
    if FAILURES:
        print(f"SUMMARY: {len(FAILURES)} check(s) FAILED out of the run above.")
        for name, bad_df in FAILURES:
            print(f"  - {name} ({len(bad_df)} bad rows)")
        sys.exit(1)
    else:
        print(f"SUMMARY: all checks PASSED on {n} rows.")
        print("\nLabel distributions:")
        print("\nlabel_informative:")
        print(df["label_informative"].value_counts())
        print("\nlabel_humanitarian:")
        print(df["label_humanitarian"].value_counts())
        print("\nRows per event:")
        print(df["event"].value_counts())
        sys.exit(0)


if __name__ == "__main__":
    main()
