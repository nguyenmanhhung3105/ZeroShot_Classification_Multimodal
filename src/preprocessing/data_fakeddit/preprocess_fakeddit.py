"""Entry point for three audited Fakeddit splits; preserves legacy output files.

Use --download to fetch missing images, --inspect for metadata only,
--verify to recheck the published three splits. No work occurs on import.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from preprocessing.data_fakeddit.prepare_splits import main

if __name__ == "__main__":
    raise SystemExit(main())
