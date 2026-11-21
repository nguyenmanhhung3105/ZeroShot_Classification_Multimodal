"""Decode at most five images per new benchmark, without model inference."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vlm_project.shared_data import smoke

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", default="/kaggle/input")
    parser.add_argument("--output", default="/kaggle/working/shared_data_smoke")
    args = parser.parse_args()
    print(smoke(args.input_root, args.output))
