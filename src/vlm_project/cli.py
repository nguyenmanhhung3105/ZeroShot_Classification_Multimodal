"""Điểm vào duy nhất của bản dự án cuối."""
import argparse
import json
from .io import BENCHMARKS, config, locate_data

def main(argv=None):
    parser = argparse.ArgumentParser(description='MetaCLIP2: public benchmark và Fakeddit v2')
    parser.add_argument('--config', default='configs/project.yaml')
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ['benchmark', 'fakeddit']:
        sub = commands.add_parser(name)
        if name == 'benchmark':
            sub.add_argument('--dataset', choices=BENCHMARKS, default='dtd')
        else:
            sub.add_argument('--split', choices=['train', 'dev', 'test'], default='dev')
        sub.add_argument('--data-root', default='auto', help='Thư mục chứa manifest và images; mặc định tìm local/Kaggle Input')
        sub.add_argument('--smoke', action='store_true', help='Chỉ 5 mẫu; không dùng để báo cáo hoặc khóa test')
        sub.add_argument('--prepare-only', action='store_true', help='Kiểm tra cấu hình; không đọc dữ liệu hay tải model')
    freeze = commands.add_parser('freeze')
    freeze.add_argument('--dev-run', required=True)
    report = commands.add_parser('report')
    report.add_argument('--run', required=True)
    diagnostic = commands.add_parser('diagnostic', help='Thử fusion 1 tầng trên subset benchmark hoặc Fakeddit dev 2way')
    diagnostic.add_argument('--dataset', choices=[*BENCHMARKS, 'fakeddit_2way'], default='imagenetv2')
    diagnostic.add_argument('--data-root', default='auto')
    diagnostic.add_argument('--smoke', action='store_true')
    diagnostic.add_argument('--prepare-only', action='store_true')
    task1 = commands.add_parser('fakeddit-task1', help='Dev 2way: ảnh, tiêu đề và StatA một tầng, 3 seed')
    task1.add_argument('--data-root', default='auto')
    task1.add_argument('--smoke', action='store_true')
    task1.add_argument('--prepare-only', action='store_true')
    commands.add_parser('sources', help='Cài/kiểm tra ba repo prompt tác giả đúng commit')
    package = commands.add_parser('package', help='Xuất ZIP mã nguồn và notebook Kaggle')
    package.add_argument('--output', default='bundles/final')
    args = parser.parse_args(argv)
    settings = config(args.config)
    if args.command in ['benchmark', 'fakeddit']:
        if args.command == 'fakeddit':
            from .fakeddit import profile
            profile(settings)
            if args.split == 'test' and args.smoke:
                parser.error('Không smoke trên test Fakeddit; dùng dev')
        if args.prepare_only:
            result = {'status': 'plan_only', 'kind': args.command, 'model': settings['model'], 'dataset': getattr(args, 'dataset', 'fakeddit'), 'split': getattr(args, 'split', None), 'mode': 'smoke' if args.smoke else 'full', 'settings': settings}
        else:
            name = args.dataset if args.command == 'benchmark' else args.split
            root = locate_data(args.command, name, settings, args.data_root)
            if args.command == 'benchmark':
                from .benchmark import run
                result = {'output': str(run(settings, name, root, args.smoke))}
            else:
                from .fakeddit import run
                result = {'output': str(run(settings, name, root, args.smoke))}
    elif args.command == 'fakeddit-task1':
        from .fakeddit_task1 import run, specification
        spec = specification()
        if args.prepare_only:
            result = {'status': 'plan_only', 'kind': 'fakeddit_task1', 'split': 'dev',
                      'task': '2way', 'samples': 5 if args.smoke else spec['samples'], 'specification': spec}
        else:
            root = locate_data('fakeddit', 'dev', settings, args.data_root)
            result = {'output': str(run(settings, 'dev', root, args.smoke))}
    elif args.command == 'diagnostic':
        from .diagnostic import run, specification
        if args.prepare_only:
            result = {'status': 'plan_only', 'dataset': args.dataset,
                      'split': 'dev' if args.dataset == 'fakeddit_2way' else BENCHMARKS[args.dataset][2],
                      'samples': 5 if args.smoke else specification()['samples'], 'stages': 1}
        else:
            binary = args.dataset == 'fakeddit_2way'
            root = locate_data('fakeddit' if binary else 'benchmark', 'dev' if binary else args.dataset,
                               settings, args.data_root)
            result = {'output': str(run(settings, args.dataset, root, limit=5 if args.smoke else None))}
    elif args.command == 'sources':
        from .paper import install_sources
        result = install_sources()
    elif args.command == 'freeze':
        from .protocol import freeze
        lock = freeze(settings, args.dev_run)
        result = {'status': 'locked', 'sha256': lock['sha256']}
    elif args.command == 'report':
        from .protocol import report
        result = report(args.run)
    else:
        from .packaging import package
        result = package(args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
