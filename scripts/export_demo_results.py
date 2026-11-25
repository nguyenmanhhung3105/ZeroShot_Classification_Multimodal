"""Build read-only chart snapshots from published, audited experiment artifacts."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from vlm_project.demo_results import export

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default='results/published/manifest.json')
    parser.add_argument('--output', default='src/vlm_project/demo_web')
    args = parser.parse_args()
    result = export(ROOT, ROOT / args.manifest, ROOT / args.output)
    print(f"Đã xuất {len(result['runs'])} bộ biểu đồ từ artifact full đã kiểm checksum; không nạp model/dataset.")
