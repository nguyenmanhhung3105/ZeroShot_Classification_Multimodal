"""Cascade few-shot min5 dev-only. Prepare-only không đọc dataset/nạp model."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from vlm_project.io import config
from vlm_project.fakeddit_fewshot_cascade import SPEC_FILE, run, specification


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/project_min5.yaml')
    parser.add_argument('--spec', default=SPEC_FILE)
    parser.add_argument('--train-root')
    parser.add_argument('--dev-root')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    settings = config(args.config); spec = specification(args.spec)
    if args.prepare_only:
        result = {'status': 'plan_only', 'kind': spec['kind'], 'split': 'dev', 'stages': 2,
                  'samples': 5 if args.smoke else 3000, 'support_per_seed': 384,
                  'seeds': spec['seeds'], 'gate_candidate_seed_groups': len(spec['gate_candidates']),
                  'flat_candidate_seed_groups': 10, 'test_opened': False}
    else:
        if not args.train_root or not args.dev_root:
            parser.error('Cần --train-root và --dev-root tường minh')
        result = {'output': str(run(settings, args.train_root, args.dev_root, args.smoke, spec_file=args.spec))}
    print(json.dumps(result, ensure_ascii=False, indent=2))
