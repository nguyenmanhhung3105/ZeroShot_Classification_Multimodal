"""Protocol Task 1 min5 offline: prepare draft, seal hoặc validate; không chạy test."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from vlm_project.min5_protocol import prepare, seal, validate


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    draft = commands.add_parser('prepare', help='Tạo draft, chưa cho phép test')
    draft.add_argument('--source-run', required=True)
    draft.add_argument('--threshold-run', required=True)
    draft.add_argument('--config', default='configs/project_min5.yaml')
    draft.add_argument('--output', required=True)
    final = commands.add_parser('seal', help='Niêm phong riêng Task 1 sau khi chốt phạm vi')
    final.add_argument('--draft', required=True)
    final.add_argument('--test-manifest-sha256', required=True)
    final.add_argument('--output', required=True)
    check = commands.add_parser('validate')
    check.add_argument('--protocol', required=True)
    check.add_argument('--allow-draft', action='store_true')
    check.add_argument('--config', default=None, help='Đường dẫn config thay thế, nội dung phải khớp hash')
    args = parser.parse_args()
    if args.command == 'prepare':
        result = prepare(args.source_run, args.threshold_run, args.output, args.config)
    elif args.command == 'seal':
        result = seal(args.draft, args.test_manifest_sha256, args.output)
    else:
        result = validate(args.protocol, require_sealed=not args.allow_draft, settings_file=args.config)
    # Không in IDs/hashes ảnh hoặc toàn bộ payload ra terminal.
    print(f"Protocol {result['payload']['state']}: {result['sha256']}")
