"""Audit all three Fakeddit splits and every selected image; no sample display."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from preprocessing.data_fakeddit.prepare_splits import main

if __name__ == "__main__":
    raise SystemExit(main(["--verify", *sys.argv[1:]]))
