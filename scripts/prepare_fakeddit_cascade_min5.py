"""Xuất snapshot dev-only cho cascade min5 từ artifact Task 1, không dataset/GPU/Kaggle."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from vlm_project.fakeddit_fewshot_cascade import export_specification


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', required=True)
    parser.add_argument('--threshold-run', required=True)
    parser.add_argument('--config', default='configs/project_min5.yaml')
    parser.add_argument('--output', default='configs/fakeddit_fewshot_cascade_min5.json')
    args = parser.parse_args()
    result = export_specification(args.source_run, args.threshold_run, args.output, args.config)
    print('Dev cascade snapshot:', args.output, result['sha256'])
