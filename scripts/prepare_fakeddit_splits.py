"""Entry point cho builder Fakeddit tại project root hiện tại; không tự chạy khi import."""
import importlib.util
from functools import partial
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / 'archive/legacy_20261103'


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    # Chạy builder qua import để ROOT không trỏ nhầm vào archive.
    source = ROOT / 'src/preprocessing/data_fakeddit/prepare_splits.py'
    sys.path.insert(0, str(ROOT / 'src'))
    spec = importlib.util.spec_from_file_location('fakeddit_split_builder', source)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    builder.ROOT = ROOT
    # Default argument của prepare đã được bind lúc import, phải thay riêng.
    builder.prepare = partial(builder.prepare, root=ROOT)
    previous_exclusions = builder.old_development_exclusions

    def exclusions(root):
        result = previous_exclusions(root)
        for historical_root in [LEGACY, ROOT / 'archive/report_cleanup_20261116']:
            archived = previous_exclusions(historical_root)
            for key in result:
                result[key].update(archived[key])
        # Những mẫu dev đã xem trong pipeline mới cũng không được vào test mới.
        for receipt in (ROOT / 'results/final/kaggle').glob('*/run/run.json'):
            run = json.loads(receipt.read_text())
            is_dev = run.get('split') == 'dev' or run.get('cohort', {}).get('split') == 'dev'
            if not is_dev or run.get('status') != 'complete':
                continue
            cohort = receipt.parent / 'cohort.json'
            if cohort.is_file():
                for row in json.loads(cohort.read_text()):
                    result['ids'].add(row['id'])
                    if row.get('text_sha256'):
                        result['texts'].add(row['text_sha256'])
                    if row.get('image_sha256'):
                        result['images'].add(row['image_sha256'])
        return result

    builder.old_development_exclusions = exclusions
    defaults = {'--output': 'data/processed/fakeddit_splits_min5_v1',
                '--base-config': 'configs/preprocessing/fakeddit_legacy_base.yaml',
                '--min-text-length': '5'}
    for flag, value in defaults.items():
        if not any(a == flag or a.startswith(flag + '=') for a in args):
            args.extend([flag, value])
    return builder.main(args)


if __name__ == '__main__':
    raise SystemExit(main())
