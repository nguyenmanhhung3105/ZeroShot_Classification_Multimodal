"""Khóa bản triển khai bằng dev; chặn đổi code/prompt và trùng mẫu khi test."""
import csv
from .io import path, read_json, sha256, digest, protocol_hashes, write_json

def completed_run(directory):
    directory = path(directory)
    receipt = read_json(directory / 'completed.json')
    if receipt.get('status') != 'complete' or receipt.get('run_sha256') != sha256(directory / 'run.json'):
        raise ValueError('Run chưa hoàn tất hoặc manifest đã đổi')
    run = read_json(directory / 'run.json')
    if run.get('status') != 'complete':
        raise ValueError('Run không hoàn tất')
    for name, expected in run['output_sha256'].items():
        artifact = path(directory / name)
        if not artifact.is_relative_to(directory) or sha256(artifact) != expected:
            raise ValueError(f'Artifact đã đổi: {name}')
    return run

def freeze(settings, dev_run):
    from .fakeddit import SELECTED
    directory = path(dev_run)
    run = completed_run(directory)
    if run.get('kind') != 'fakeddit' or run.get('split') != 'dev' or run.get('mode') != 'full' or (run.get('samples') != 3000) or (run.get('selected_methods') != SELECTED):
        raise ValueError('Cần run Fakeddit dev hoàn tất đủ 3000 mẫu bằng bản code cuối')
    if run['protocol'] != protocol_hashes(settings):
        raise ValueError('Code/cấu hình đã đổi sau dev; cần chạy lại dev')
    expected_rows = {(task, method): 3 if method == 'proposed' else 7 if method == 'waffle_image' else 1 for task in ['2way', '6way'] for method in settings['fakeddit']['methods']}
    actual = {(r['task'], r['method']): r['seeds'] for r in run['summary'] if r['samples'] == 3000}
    if actual != expected_rows or len(run['summary']) != len(expected_rows):
        raise ValueError('Dev thiếu đối chứng/seed hoặc số mẫu không đủ')
    expected = run['profile']['development_source']['dev_cohort_id']
    if run['cohort']['cohort_id'] != expected:
        raise ValueError('Dev khác cohort đã chọn phương pháp lịch sử')
    cohort = read_json(directory / 'cohort.json')
    if len(cohort) != 3000 or len({r['id'] for r in cohort}) != 3000:
        raise ValueError('Cohort dev thiếu hoặc trùng ID')
    payload = {'version': 1, 'model': settings['model'], 'selected_methods': SELECTED, 'protocol': run['protocol'], 'dev_cohort_id': expected, 'dev_run_sha256': sha256(directory / 'run.json'), 'dev_summary': run['summary'], 'excluded_dev': [{k: r[k] for k in ['id', 'image_sha256', 'text_sha256']} for r in cohort], 'selection': 'Fixed historical dev winners, revalidated on final implementation; no test selection.'}
    lock = {'payload': payload, 'sha256': digest(payload)}
    destination = path(settings['fakeddit']['lock_file'])
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x', encoding='utf-8') as stream:
        import json
        json.dump(lock, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print('Đã khóa:', destination, flush=True)
    return lock

def validate_lock(settings):
    from .fakeddit import SELECTED
    file = path(settings['fakeddit']['lock_file'])
    if not file.is_file():
        raise ValueError('Chạy Fakeddit dev đủ 3000 rồi run.py freeze trước khi chạy test')
    lock = read_json(file)
    payload = lock['payload']
    if lock.get('sha256') != digest(payload) or payload.get('version') != 1 or payload.get('model') != settings['model'] or (payload.get('selected_methods') != SELECTED) or (payload.get('protocol') != protocol_hashes(settings)):
        raise ValueError('Code/prompt/cấu hình khác dev đã khóa; không dùng lock cũ để test')
    return lock

def check_test_cohort(lock, rows):
    excluded = lock['payload']['excluded_dev']
    for key in ['id', 'image_sha256', 'text_sha256']:
        values = {r[key] for r in excluded}
        if any((r[key] in values for r in rows)):
            raise ValueError(f'Test trùng dev theo {key}; dừng để tránh leakage')

def report(directory):
    run = completed_run(directory)
    print(f"{run['model']} | {run.get('dataset', 'fakeddit')} | {run['mode']} | {run['samples']} mẫu")
    print('task\tmethod\taccuracy (%)\tmacro-F1 (%)\tseeds')
    for row in run['summary']:
        print(f"{row['task']}\t{row['method']}\t{100 * row['accuracy_mean']:.2f} ± {100 * row['accuracy_std']:.2f}\t{100 * row['macro_f1_mean']:.2f} ± {100 * row['macro_f1_std']:.2f}\t{row['seeds']}")
    return {'status': 'complete', 'run': str(path(directory)), 'mode': run['mode'], 'samples': run['samples']}
