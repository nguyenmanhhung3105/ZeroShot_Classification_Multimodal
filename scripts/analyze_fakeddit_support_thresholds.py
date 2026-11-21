"""CPU offline: so support policy tại mục tiêu recall Fake, không dùng test/GPU/Kaggle."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from vlm_project.fewshot_support_thresholds import analyze


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, help='Run fakeddit_fewshot_support full dev đã hoàn tất')
    parser.add_argument('--output-root', default='results/final/fewshot_support_thresholds')
    parser.add_argument('--target-recall', nargs='+', type=float, default=[.87], help='Tỷ lệ trong (0,1], mặc định 0.87')
    parser.add_argument('--modalities', nargs='+', choices=['image', 'title', 'image_title'], default=['image_title'])
    parser.add_argument('--selection-fraction', type=float, default=.5)
    parser.add_argument('--split-seed', type=int, default=42)
    args = parser.parse_args()
    print(analyze(args.run, args.output_root, args.target_recall, args.modalities,
                  args.selection_fraction, args.split_seed))
