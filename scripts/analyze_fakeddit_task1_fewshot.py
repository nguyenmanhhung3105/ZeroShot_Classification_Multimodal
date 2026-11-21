"""Offline: phân tích lỗi và train-only CV từ run few-shot min5 đã lưu."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from vlm_project.fewshot_task1_analysis import analyze

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--output-root', default='results/final/fewshot_task1_analysis')
    args = parser.parse_args()
    print(analyze(args.run, args.output_root))
