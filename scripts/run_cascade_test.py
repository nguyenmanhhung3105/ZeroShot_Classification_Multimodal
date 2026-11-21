"""Test hai tầng min5 đã khóa; --prepare-only không đọc dataset hoặc nạp model."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from vlm_project.cascade_test_runner import plan,run


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',required=True);parser.add_argument('--config');parser.add_argument('--spec')
    parser.add_argument('--train-root');parser.add_argument('--test-root')
    parser.add_argument('--output-root',default='results/final/cascade_test_min5')
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    if args.prepare_only:result=plan(args.protocol,args.config,args.spec)
    else:
        if not args.train_root or not args.test_root:parser.error('Cần --train-root và --test-root tường minh')
        result={'output':str(run(args.protocol,args.train_root,args.test_root,args.output_root,args.config,spec_file=args.spec))}
    print(json.dumps(result,ensure_ascii=False,indent=2))
