"""Runner few-shot Task 1 min5; --prepare-only không đọc dataset/nạp model/gọi Kaggle."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from vlm_project.min5_test_runner import plan, run


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--config', default=None, help='Config cùng nội dung đã khóa; mặc định lấy đường dẫn trong protocol')
    parser.add_argument('--train-root', help='Đường dẫn train min5 tường minh; không tự tìm slug legacy')
    parser.add_argument('--test-root', help='Đường dẫn test min5 tường minh')
    parser.add_argument('--output-root', default='results/final/fakeddit_fewshot_test_min5')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if args.prepare_only:
        result = plan(args.protocol, args.config)
    else:
        if not args.train_root or not args.test_root:
            parser.error('Cần --train-root và --test-root tường minh; dùng --prepare-only để chỉ kiểm kế hoạch')
        result = {'output': str(run(args.protocol, args.train_root, args.test_root, args.output_root, args.config))}
    print(json.dumps(result, ensure_ascii=False, indent=2))
