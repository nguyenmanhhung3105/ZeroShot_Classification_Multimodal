"""Protocol test min5 hai tầng: prepare/capture-runtime/seal/validate; không mở test."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from vlm_project import cascade_protocol as protocol


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    prepare=commands.add_parser('prepare')
    prepare.add_argument('--dev-run',required=True);prepare.add_argument('--train-source',required=True)
    prepare.add_argument('--config',default='configs/project_min5.yaml')
    prepare.add_argument('--spec',default='configs/fakeddit_fewshot_cascade_min5.json')
    prepare.add_argument('--output',required=True)
    capture=commands.add_parser('capture-runtime');capture.add_argument('--output',required=True)
    seal=commands.add_parser('seal')
    seal.add_argument('--draft',required=True);seal.add_argument('--test-manifest-sha256',required=True)
    seal.add_argument('--runtime-file',required=True);seal.add_argument('--output',required=True)
    validate=commands.add_parser('validate');validate.add_argument('--protocol',required=True)
    validate.add_argument('--allow-draft',action='store_true');validate.add_argument('--config');validate.add_argument('--spec')
    args=parser.parse_args()
    if args.command=='prepare':
        result=protocol.prepare(args.dev_run,args.train_source,args.output,args.config,spec_file=args.spec)
    elif args.command=='capture-runtime':result=protocol.capture_runtime(args.output)
    elif args.command=='seal':result=protocol.seal(args.draft,args.test_manifest_sha256,args.runtime_file,args.output)
    else:result=protocol.validate(args.protocol,require_sealed=not args.allow_draft,settings_file=args.config,spec_file=args.spec)
    print(f"Protocol {result['payload'].get('state',result['payload']['kind'])}: {result['sha256']}")
