"""Launcher CLI từ local: chỉ gửi code, dùng dữ liệu Kaggle đã có."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time
import uuid
import zipfile
import yaml

from .io import ROOT, BENCHMARKS, config, path, read_json, write_json, digest, sha256
from .packaging import package, notebook


def cli(arguments, check=True):
    executable = Path(sys.executable).with_name('kaggle')
    command = str(executable) if executable.is_file() else shutil.which('kaggle')
    if not command:
        raise RuntimeError('Cần cài CLI: venv/bin/python -m pip install "kaggle>=2.2.5"')
    result = subprocess.run([command, *arguments], cwd=ROOT, capture_output=True, text=True)
    if check and result.returncode:
        action = ' '.join(arguments[:2])
        raise RuntimeError(f'Kaggle {action} thất bại:\n{result.stderr or result.stdout}')
    return result


def print_run_limits(plan):
    """Hiển thị cấu hình thật của lượt chạy, không suy đoán quota tài khoản."""
    model = plan.get('reference_model', plan['settings']['model'])
    seconds = plan['launcher']['timeout_seconds']
    print(f'[RUN LIMITS]\nJob: {plan["job"]}\nModel: {model}\n'
          f'Dataset/split: {plan["name"]} | mode={plan["mode"]} | limit={plan["samples"]} mẫu\n'
          f'GPU: {plan["launcher"]["accelerator"]} | timeout={seconds}s ({seconds/3600:g} giờ)\n'
          f'Stage: {plan["root"]}', flush=True)
    if plan.get('diagnostic_specification'):
        spec = plan['diagnostic_specification']
        print(f'Nhánh: CLIP / StatA / CALIP; trọng số={spec["fusion_weights"]}\n'
              f'Chọn ảnh seed={spec["selection_seed"]}; chia nhóm seed={spec["grouping_seed"]}; '
              f'batch ảnh={spec["image_batch_size"]}; batch StatA={spec["adaptation_batch_size"]}\n'
              'Chỉ một lượt diagnostic; temperature chưa fit, không tự chạy full benchmark.', flush=True)
    if plan.get('task1_specification'):
        spec = plan['task1_specification']
        print(f'Fakeddit dev 2way; seed chia nhóm={spec["grouping_seeds"]}; '
              f'CLIP/StatA={spec["clip_weight"]}; trọng số ảnh={spec["image_weight"]}. '
              '6 phương pháp; chỉ một lượt, không tạo lock test.', flush=True)
    if plan.get('fewshot_task1_specification'):
        spec = plan['fewshot_task1_specification']
        if plan.get('task') == 'fakeddit_fewshot_cascade':
            print('Few-shot min5 hai tầng: Real/Fake → 5 loại Fake; đối chứng flat 6way cùng 384 support/seed. '
                  'Ảnh–tiêu đề, 2 policy, 5 seed, C=1 balanced; gate/ngưỡng giữ nguyên, chỉ dev. '
                  'Smoke vẫn encode toàn bộ hợp support train, không chỉ 5 ảnh.', flush=True)
        else:
            print(f'Few-shot binary min5: support={6*spec["shots_per_6way_class"]}/seed, seed={spec["seeds"]}; '
                  'ảnh / tiêu đề / nối ảnh–tiêu đề; frozen MetaCLIP2, LogisticRegression C=1 balanced; chỉ dev.', flush=True)
        if 'support_policies' in spec and plan.get('task') != 'fakeddit_fewshot_cascade':
            print('Đối chứng support: 64 Real + 320 Fake versus 192 Real + 192 Fake; '
                  'cùng 384/seed, ba modality, không đổi C/ngưỡng.', flush=True)
    if plan.get('waffle_stata_specification'):
        spec = plan['waffle_stata_specification']
        print(f'Waffle→StatA: prompt seed={spec["prompt_seeds"]}, grouping seed={spec["grouping_seeds"]}; '
              'có CLIP, CLIP–StatA, Waffle và Waffle unit làm đối chứng; không fusion posterior.', flush=True)
        if 'semantic_weight' in spec:
            print(f'Có Waffle+semantic và Waffle+semantic+StatA; semantic weight={spec["semantic_weight"]}, '
                  'mô tả lớp cố định, không tìm trọng số theo kết quả.', flush=True)
    if plan.get('dclip_stata_specification'):
        print('DCLIP→StatA: DTD đủ split, grouping seed 0/1/2; '
              'đối chứng CLIP, CLIP–StatA, DCLIP và DCLIP unit; không fusion posterior.', flush=True)
    if plan.get('cifar100_specification'):
        spec = plan['cifar100_specification']
        print(f'CIFAR seed={spec["seed"]}; batch ảnh={spec["batch_size"]}; '
              f'phương pháp={", ".join(spec["methods"])}; '
              f'theo dõi mỗi {plan["launcher"]["poll_seconds"]} giây.', flush=True)
    if plan.get('fakeddit_hierarchy_specification'):
        spec=plan['fakeddit_hierarchy_specification']
        print(f'Fakeddit dev 6way; nhóm={spec["group_size"]}; seed={spec["grouping_seeds"]}; '
              f'theo dõi mỗi {plan["launcher"]["poll_seconds"]} giây. '
              + ('StatA gate; tầng 2 CLIP / CALIP / fusion; có binary metric.' if plan.get('fakeddit_calip_specification')
               else '8 đối chứng image-only + 3 đối chứng MM legacy; oracle chỉ chẩn đoán.'),flush=True)


def missing(result):
    return bool(re.search(r'\b404\b|not found|does not exist', result.stderr + result.stdout, re.I))


def permission_denied(result):
    return bool(re.search(r'\b403\b|permission[^\n]*denied|cannot access (kernel|dataset)',
                          result.stderr + result.stdout, re.I))


def owned_reference_exists(resource, reference):
    """Kiểm tra tài nguyên của đúng tài khoản; không suy 403 thành chưa tồn tại."""
    owner = reference.split('/', 1)[0].casefold()
    page, token, seen = 1, None, set()
    while True:
        arguments = [resource, 'list', '--mine', '--page-size', '100', '--format', 'json']
        arguments += ['--page-token', token] if token else ['--page', str(page)]
        response = cli(arguments)
        raw = response.stdout.strip()
        marker = re.match(r'Next Page Token = (\S+)\s*\n', raw)
        next_token = marker.group(1) if marker else None
        if marker:
            raw = raw[marker.end():]
        rows = json.loads(raw)
        if not isinstance(rows, list) or any(not isinstance(row, dict) or not isinstance(row.get('ref'), str) for row in rows):
            raise ValueError('Không xác minh được danh sách tài nguyên Kaggle của tài khoản')
        references = [row['ref'] for row in rows]
        if any(ref.split('/', 1)[0].casefold() != owner for ref in references):
            raise RuntimeError('Tài khoản Kaggle CLI khác username trong configs/kaggle_launcher.yaml')
        if reference in references:
            return True
        if not next_token and len(rows) < 100:
            return False
        fingerprint = (next_token, tuple(references))
        if fingerprint in seen:
            raise RuntimeError('Kaggle lặp trang danh sách; dừng để tránh gửi trùng job')
        seen.add(fingerprint)
        token = next_token
        page += 1


def confirmed_missing(resource, reference, response):
    if permission_denied(response):
        # Kaggle có thể trả kernels.get denied cả khi slug mới chưa được tạo.
        # Chỉ cho tạo sau khi list --mine thành công và không chứa đúng ref.
        return not owned_reference_exists(resource, reference)
    return missing(response)


def code_dataset_ready(plan):
    """Status có thể trả 403 sau upload; xác minh file list trước khi push GPU."""
    state = cli(['datasets', 'status', plan['code_dataset'], '--format', 'json'], check=False)
    if state.returncode == 0:
        return str(json.loads(state.stdout)['status']).lower() == 'ready'
    if not (permission_denied(state) or missing(state)):
        raise RuntimeError(state.stderr or state.stdout)
    listing = cli(['datasets', 'files', plan['code_dataset'], '--page-size', '200', '--format', 'json'], check=False)
    if listing.returncode:
        if permission_denied(listing) or missing(listing):
            return False
        raise RuntimeError(listing.stderr or listing.stdout)
    raw = re.sub(r'^Next Page Token = \S+\s*\n', '', listing.stdout.strip())
    rows = json.loads(raw)
    if not isinstance(rows, list):
        raise ValueError('Kaggle file list không đúng schema')
    observed = {row['name']: int(row['size']) for row in rows}
    source = path(plan['root'])/'code/final_model_source.zip'
    if observed.get(source.name) == source.stat().st_size:
        return True
    with zipfile.ZipFile(source) as archive:
        expected = {entry.filename:entry.file_size for entry in archive.infolist() if not entry.is_dir()}
    return any(all(observed.get(prefix+name) == size for name,size in expected.items())
               for prefix in ['', 'final_model_source/'])


def prepare_cifar100(launcher, job_name, limit=None):
    """Standalone CIFAR notebook uses the normal submit/wait/collect workflow."""
    import importlib.util
    from .cifar100_two_stage import specification
    job = launcher['jobs'][job_name]
    rerank = job['kind'] == 'cifar100_rerank'
    adapted = job['kind'] == 'cifar100_em_stata'
    spec = specification(ROOT / 'configs/cifar100_em_stata.json') if adapted else specification(ROOT / 'configs/cifar100_rerank.json') if rerank else specification()
    if job.get('split') != 'dev' or job.get('mode', 'full') not in ['full', 'smoke']:
        raise ValueError('CIFAR launcher supports dev/full or dev/smoke; test uses locked dev workflow')
    if limit not in [None, 5, spec['dev_samples']]:
        raise ValueError('CIFAR supports smoke 5 or fixed full dev only')
    smoke = limit == 5 or job.get('mode') == 'smoke'
    if smoke and limit not in [None, 5]:
        raise ValueError('Smoke job cannot request full sample count')
    owner = launcher['kaggle']['username']
    if not re.fullmatch(r'[A-Za-z0-9_-]+', owner):
        raise ValueError('Invalid Kaggle username')
    root = path('results/_system/launches/final') / (job_name + '_' + uuid.uuid4().hex[:12])
    root.mkdir(parents=True)
    module_spec = importlib.util.spec_from_file_location('cifar_job_builder', ROOT / 'scripts/prepare_cifar100_two_stage_job.py')
    builder = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(builder)
    builder_args = {'variant': 'em_stata'} if adapted else {'variant': 'rerank'} if rerank else {}
    kernel = builder.prepare(mode='smoke' if smoke else 'dev', output=root / 'kernel', **builder_args)
    metadata = read_json(kernel / 'kernel-metadata.json')
    metadata['id'] = owner + '/' + metadata['id'].split('/', 1)[1]
    metadata['machine_shape'] = launcher['kaggle']['accelerator']
    write_json(kernel / 'kernel-metadata.json', metadata)
    options = dict(launcher['kaggle'])
    options['poll_seconds'] = job.get('poll_seconds', 120)
    if type(options['poll_seconds']) is not int or options['poll_seconds'] <= 0:
        raise ValueError('poll_seconds must be a positive integer')
    plan = {'status': 'prepared', 'job': job_name, 'task': job['kind'],
            'name': 'cifar100/dev', 'mode': 'smoke' if smoke else 'full',
            'samples': 5 if smoke else spec['dev_samples'], 'root': str(root),
            'kernel': metadata['id'], 'code_dataset': None, 'inline_code': True,
            'data_dataset': spec['dataset_id'], 'settings': {'model': spec['model']},
            'launcher': options, 'cifar100_specification': spec}
    write_json(root / 'launch.json', plan)
    return plan


def prepare(launcher, job_name, limit=None):
    if launcher['jobs'][job_name]['kind'] in ['fakeddit_hierarchy','fakeddit_stata_calip']:
        return prepare_fakeddit_hierarchy(launcher, job_name, limit)
    if launcher['jobs'][job_name]['kind'] in ['cifar100_two_stage', 'cifar100_rerank', 'cifar100_em_stata']:
        return prepare_cifar100(launcher, job_name, limit)
    settings = config(launcher['experiment_config'])
    job = launcher['jobs'][job_name]
    task = job['kind']
    reference = task == 'paper_reference'
    diagnostic = task == 'diagnostic'
    task1 = task == 'fakeddit_task1'
    support_ablation = task == 'fakeddit_fewshot_support'
    cascade = task == 'fakeddit_fewshot_cascade'
    fewshot = task in ['fakeddit_fewshot_task1', 'fakeddit_fewshot_support', 'fakeddit_fewshot_cascade']
    semantic = task == 'waffle_semantic_stata'
    waffle_full = task in ['waffle_stata_full', 'waffle_semantic_stata']
    waffle_stata = task in ['waffle_stata', 'waffle_stata_full', 'waffle_semantic_stata']
    dclip_stata = task == 'dclip_stata'
    diagnostic_fakeddit = diagnostic and job.get('dataset') == 'fakeddit_2way'
    benchmark_task = task in ['benchmark', 'paper_reference', 'diagnostic', 'waffle_stata', 'waffle_stata_full', 'waffle_semantic_stata', 'dclip_stata'] and not diagnostic_fakeddit
    name = job.get('dataset') if benchmark_task or diagnostic_fakeddit else job.get('split')
    if task not in ['benchmark', 'fakeddit', 'paper_reference', 'diagnostic', 'fakeddit_task1', 'fakeddit_fewshot_task1', 'fakeddit_fewshot_support', 'fakeddit_fewshot_cascade', 'waffle_stata', 'waffle_stata_full', 'waffle_semantic_stata', 'dclip_stata'] or (benchmark_task and name not in BENCHMARKS):
        raise ValueError('Job chỉ hỗ trợ benchmark đã khai báo hoặc Fakeddit')
    if reference:
        from .paper_reference import REFERENCE_DATASETS
        if name not in REFERENCE_DATASETS:
            raise ValueError('Lượt paper_reference chỉ chạy 8 benchmark đã khai báo bằng OpenAI ViT-L/14')
    if task == 'fakeddit' and name not in ['train', 'dev', 'test']:
        raise ValueError('Split Fakeddit không hợp lệ')
    if task1 and name != 'dev':
        raise ValueError('Job Task 1 chỉ hỗ trợ Fakeddit dev 2way')
    if fewshot:
        if cascade:
            from .fakeddit_fewshot_cascade import specification
        elif support_ablation:
            from .fakeddit_fewshot_support import specification
        else:
            from .fakeddit_fewshot_task1 import specification
        if name != 'dev':
            raise ValueError('Few-shot Task 1 chỉ đánh giá dev, không train/test')
        fewshot_spec = specification()
        settings['kaggle']['fakeddit_slugs'].update(fewshot_spec['dataset_slugs'])
    diagnostic_spec = None
    if diagnostic:
        from .diagnostic import specification
        diagnostic_spec = specification()
    if task1:
        from .fakeddit_task1 import specification
        task1_spec = specification()
    if waffle_stata:
        if semantic:
            from .waffle_semantic_stata import specification, DATASETS
        elif waffle_full:
            from .waffle_stata_full import specification, DATASETS
        else:
            from .waffle_stata import specification, DATASETS
        if name not in DATASETS:
            raise ValueError('Dataset không thuộc job Waffle–StatA đã chọn')
        waffle_spec = specification(name) if semantic else specification()
    if dclip_stata:
        from .dclip_stata import specification, DATASETS
        if name not in DATASETS:
            raise ValueError('DCLIP–StatA hiện chỉ hỗ trợ DTD')
        dclip_spec = specification()
    expected = dclip_spec['samples'] if dclip_stata else waffle_spec['samples'] if waffle_stata else diagnostic_spec['samples'] if diagnostic else BENCHMARKS[name][1] if benchmark_task else 3000
    if limit not in [None, 5, expected]:
        raise ValueError(f'Chỉ hỗ trợ smoke 5 mẫu hoặc {expected} mẫu đã cấu hình; không đổi cohort báo cáo')
    smoke = limit == 5
    if task == 'fakeddit' and name == 'test':
        if smoke:
            raise ValueError('Không smoke trên test Fakeddit')
        lock = read_json(settings['fakeddit']['lock_file'])
        if lock['sha256'] != digest(lock['payload']):
            raise ValueError('Lock Fakeddit sai checksum')
    owner = launcher['kaggle']['username']
    if not re.fullmatch(r'[A-Za-z0-9_-]+', owner):
        raise ValueError('Kaggle username không hợp lệ')
    root = path('results/_system/launches/final') / (job_name + '_' + uuid.uuid4().hex[:12])
    root.mkdir(parents=True)
    receipt = package(root / 'bundle')
    files = read_json(root / 'bundle/source_bundle_manifest.json')['files']
    tag = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()[:12]
    code_id = f'{owner}/{launcher["kaggle"]["code_dataset_prefix"]}-{tag}'
    data_slug = settings['kaggle']['benchmark_slug'] if benchmark_task else settings['kaggle']['fakeddit_slugs']['dev' if diagnostic_fakeddit else name]
    data_id = f'{owner}/{data_slug}'
    suffix = name if task == 'benchmark' else 'fakeddit-' + name
    kernel_slug = (f'{launcher["kaggle"]["kernel_prefix"]}-{name}-openai-vitl14-paper' if reference
                   else f'{launcher["kaggle"]["kernel_prefix"]}-fakeddit-fewshot-cascade-min5' if cascade
                   else f'{launcher["kaggle"]["kernel_prefix"]}-fakeddit-fewshot-support-min5' if support_ablation
                   else f'{launcher["kaggle"]["kernel_prefix"]}-fakeddit-fewshot-task1-min5' if fewshot
                   else f'{launcher["kaggle"]["kernel_prefix"]}-{name}-waffle-semantic-stata-full' if semantic
                   else f'{launcher["kaggle"]["kernel_prefix"]}-{name}-dclip-stata-full' if dclip_stata
                   else f'{launcher["kaggle"]["kernel_prefix"]}-{name}-waffle-stata-full' if waffle_full
                   else f'{launcher["kaggle"]["kernel_prefix"]}-{name}-waffle-stata-pilot' if waffle_stata
                   else f'{launcher["kaggle"]["kernel_prefix"]}-fakeddit-dev-task1-single-stage' if task1
                   else f'{launcher["kaggle"]["kernel_prefix"]}-{name}-fusion-diagnostic' if diagnostic
                   else f'{launcher["kaggle"]["kernel_prefix"]}-{suffix}-metaclip2') + ('-smoke' if smoke else '')
    kernel_slug = kernel_slug.replace('_', '-')
    kernel_id = owner + '/' + kernel_slug
    upload = root / 'code'; upload.mkdir()
    shutil.copy2(receipt['zip'], upload / 'final_model_source.zip')
    write_json(upload / 'dataset-metadata.json', {
        'id': code_id, 'title': code_id.split('/')[1].replace('-', ' ').title(),
        'licenses': [{'name': 'unknown'}],
    })
    kernel = root / 'kernel'; kernel.mkdir()
    document = notebook()
    # Notebook mẫu mặc định đọc project.yaml; launcher có thể chọn cấu hình khác.
    config_file = str(path(launcher['experiment_config']).relative_to(ROOT))
    document['cells'][2]['source'] = [
        line.replace('settings = config()', f'settings = config({config_file!r})')
        for line in document['cells'][2]['source']
    ]
    document['cells'][1]['source'] = [
        f'TASK = {task!r}\n', f'DATASET = {job.get("dataset", "dtd")!r}\n',
        f'SPLIT = {job.get("split", "dev")!r}\n', 'DATA_ROOT = "auto"\n', 'TEST_LOCK_INPUT = ""\n',
    ]
    document['cells'][2]['source'].append(
        f'\nif code_id != {tag!r}:\n    raise ValueError("Kaggle Input khác bundle đã gửi từ local")\n'
        'settings["device"] = "cuda"\n')
    if fewshot:
        document['cells'][0]['source'] = ['# Fakeddit min5 — cascade hai tầng và flat 6way, cùng 384 support/seed\n' if cascade
                                         else '# Fakeddit min5 — few-shot binary, ba modality, 5 seed\n']
        document['cells'][3]['source'] = [
            f'from vlm_project.{task} import run, specification\n',
            'settings["kaggle"]["fakeddit_slugs"].update(specification()["dataset_slugs"])\n',
            'train_root = locate_data("fakeddit", "train", settings, "auto")\n',
            'dev_root = locate_data("fakeddit", "dev", settings, DATA_ROOT)\n',
            f'output = run(settings, train_root, dev_root, smoke={smoke!r})\n',
            'from vlm_project.protocol import report\n', 'report(output)\n',
        ]
        del document['cells'][4]
    elif waffle_stata or dclip_stata:
        document['cells'][0]['source'] = [f'# {name} — {task} prototype → StatA, một tầng\n']
        document['cells'][3]['source'] = [
            f'from vlm_project.{task} import run\n', f'name = {name!r}\n',
            'root = locate_data("benchmark", name, settings, DATA_ROOT)\n',
            f'output = run(settings, name, root, limit={5 if smoke else expected})\n',
            'from vlm_project.protocol import report\n', 'report(output)\n',
        ]
        del document['cells'][4]
    elif task1:
        document['cells'][0]['source'] = ['# Fakeddit dev 2way — ảnh, tiêu đề, StatA một tầng, 3 seed\n']
        document['cells'][3]['source'] = [
            'from vlm_project.fakeddit_task1 import run\n',
            'root = locate_data("fakeddit", "dev", settings, DATA_ROOT)\n',
            f'output = run(settings, "dev", root, smoke={smoke!r})\n',
            'from vlm_project.protocol import report\n', 'report(output)\n',
        ]
        del document['cells'][4]
    elif diagnostic:
        document['cells'][0]['source'] = [f'# {name} — chẩn đoán CLIP / StatA / CALIP và gộp xác suất\n']
        document['cells'][3]['source'] = [
            'from vlm_project.diagnostic import run\n',
            f'name = {name!r}\n',
            ('root = locate_data("fakeddit", "dev", settings, DATA_ROOT)\n' if diagnostic_fakeddit
             else 'root = locate_data("benchmark", name, settings, DATA_ROOT)\n'),
            f'output = run(settings, name, root, limit={5 if smoke else expected})\n',
            'from vlm_project.protocol import report\n',
            'report(output)\n',
        ]
        del document['cells'][4]
    elif reference:
        document['cells'][0]['source'] = [f'# {name} — kiểm chứng OpenAI CLIP ViT-L/14 theo paper\n']
        document['cells'][3]['source'] = [
            'from vlm_project.paper_reference import run\n',
            f'name = {name!r}\n',
            'root = locate_data("benchmark", name, settings, DATA_ROOT)\n',
            'smoke_output = run(settings, name, root, smoke=True)\n',
            'print("Reference smoke OK:", smoke_output)\n',
        ]
        document['cells'][4]['source'] = [
            'output = run(settings, name, root, smoke=False)\n',
            'from vlm_project.protocol import report\n',
            'report(output)\n',
        ]
    if smoke and not diagnostic and not task1 and not waffle_stata and not dclip_stata and not fewshot:
        document['cells'][3]['source'].append('\noutput = smoke_output\n')
        if task == 'benchmark':
            document['cells'][3]['source'].append(
                'benchmark_backend.unload()\nbenchmark_backend = None\n')
        del document['cells'][4]
    write_json(kernel / 'run.ipynb', document)
    metadata = {
        'id': kernel_id, 'title': kernel_slug.replace('-', ' ').title(),
        'code_file': 'run.ipynb', 'language': 'python', 'kernel_type': 'notebook',
        'is_private': 'true', 'enable_gpu': 'true', 'enable_internet': 'true',
        'machine_shape': launcher['kaggle']['accelerator'],
        'dataset_sources': [code_id, data_id],
        'competition_sources': [], 'kernel_sources': [], 'model_sources': [],
    }
    extra_data = owner + '/' + fewshot_spec['dataset_slugs']['train'] if fewshot else None
    if extra_data:
        metadata['dataset_sources'].append(extra_data)
    write_json(kernel / 'kernel-metadata.json', metadata)
    run_launcher = dict(launcher['kaggle'])
    if diagnostic:
        run_launcher['timeout_seconds'] = diagnostic_spec['timeout_seconds']
    if task1:
        run_launcher['timeout_seconds'] = task1_spec['timeout_seconds']
    if waffle_stata:
        run_launcher['timeout_seconds'] = waffle_spec['timeout_seconds']
    if dclip_stata:
        run_launcher['timeout_seconds'] = dclip_spec['timeout_seconds']
    if fewshot:
        run_launcher['timeout_seconds'] = fewshot_spec['timeout_seconds']
    plan = {'status': 'prepared', 'job': job_name, 'task': task, 'name': name,
            'mode': 'smoke' if smoke else 'pilot' if waffle_stata and not waffle_full else 'diagnostic' if diagnostic else 'full', 'samples': 5 if smoke else expected,
            'root': str(root), 'kernel': kernel_id, 'code_dataset': code_id, 'data_dataset': data_id,
            'code_tree': tag, 'settings': settings, 'launcher': run_launcher}
    if reference:
        plan['reference_model'] = 'openai_clip_vitl14'
    if diagnostic:
        plan['diagnostic_specification'] = diagnostic_spec
    if task1:
        plan['task1_specification'] = task1_spec
    if waffle_stata:
        plan['waffle_stata_specification'] = waffle_spec
    if dclip_stata:
        plan['dclip_stata_specification'] = dclip_spec
    if fewshot:
        plan['fewshot_task1_specification'] = fewshot_spec
        plan['data_datasets'] = [data_id, extra_data]
    write_json(root / 'launch.json', plan)
    return plan


def prepare_fakeddit_hierarchy(launcher, job_name, limit=None):
    import importlib.util
    from .fakeddit_hierarchy import specification
    spec=specification()
    settings=config(launcher['experiment_config'])
    job=launcher['jobs'][job_name]
    calip=job['kind']=='fakeddit_stata_calip'
    if calip:
        from .fakeddit_calip_ablation import specification as calip_specification
        calip_spec=calip_specification()
    if job.get('split')!='dev' or job.get('mode','full') not in ['full','smoke']:
        raise ValueError('Fakeddit hierarchy is dev-only')
    if limit not in [None,5,3000]:
        raise ValueError('Only smoke 5 or full dev 3000 supported')
    smoke=limit==5 or job.get('mode')=='smoke'
    if smoke and limit not in [None,5]:
        raise ValueError('Smoke cannot request full dev')
    owner=launcher['kaggle']['username']
    if not re.fullmatch(r'[A-Za-z0-9_-]+',owner):
        raise ValueError('Invalid owner')
    root=path('results/_system/launches/final')/(job_name+'_'+uuid.uuid4().hex[:12]);root.mkdir(parents=True)
    module_spec=importlib.util.spec_from_file_location('fakeddit_hierarchy_builder',ROOT/'scripts/prepare_fakeddit_hierarchy_job.py')
    module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
    variant_args={'variant':'stata_calip'} if calip else {}
    kernel=module.prepare(mode='smoke' if smoke else 'dev',output=root/'kernel',**variant_args)
    metadata=read_json(kernel/'kernel-metadata.json')
    metadata['id']=owner+'/'+metadata['id'].split('/',1)[1]
    metadata['machine_shape']=launcher['kaggle']['accelerator']
    metadata['dataset_sources']=[owner+'/'+settings['kaggle']['fakeddit_slugs']['dev']]
    # Keep the selected project config content/path consistent inside the notebook.
    project_config=path(launcher['experiment_config'])
    if project_config != ROOT/'configs/project.yaml':
        raise ValueError('Standalone Fakeddit hierarchy currently uses configs/project.yaml only')
    write_json(kernel/'kernel-metadata.json',metadata)
    options=dict(launcher['kaggle']);options['poll_seconds']=job.get('poll_seconds',30)
    if type(options['poll_seconds']) is not int or options['poll_seconds']<=0:
        raise ValueError('poll_seconds must be positive')
    plan={'status':'prepared','job':job_name,'task':job['kind'],'name':'fakeddit/dev/6way',
          'mode':'smoke' if smoke else 'full','samples':5 if smoke else 3000,'root':str(root),
          'kernel':metadata['id'],'code_dataset':None,'inline_code':True,'data_dataset':metadata['dataset_sources'][0],
          'settings':settings,'launcher':options,'fakeddit_hierarchy_specification':spec}
    if calip:plan['fakeddit_calip_specification']=calip_spec
    write_json(root/'launch.json',plan)
    return plan


def check_input_dataset(plan, reference):
    """Few-shot min5: status 403 có thể khác quyền ListDatasetFiles thực tế."""
    state = cli(['datasets', 'status', reference], check=False)
    if state.returncode == 0:
        return
    spec = plan.get('fewshot_task1_specification')
    if not spec or not permission_denied(state):
        raise RuntimeError(state.stderr or state.stdout)
    split = next((s for s, slug in spec['dataset_slugs'].items()
                  if reference == plan['kernel'].split('/')[0] + '/' + slug), None)
    if split not in ['train', 'dev']:
        raise ValueError('Không có schema few-shot cho dataset đầu vào này')
    files, token, seen_tokens = {}, None, set()
    while True:
        arguments = ['datasets', 'files', reference, '--page-size', '200', '--format', 'json']
        if token:
            arguments += ['--page-token', token]
        response = cli(arguments, check=False)
        if response.returncode:
            raise RuntimeError(f'Không đọc được dataset {reference}; chưa gửi job.\n'
                               + (response.stderr or response.stdout))
        raw = response.stdout.strip()
        marker = re.match(r'Next Page Token = (\S+)\s*\n', raw)
        next_token = marker.group(1) if marker else None
        if marker:
            raw = raw[marker.end():]
        rows = json.loads(raw)
        if not isinstance(rows, list):
            raise ValueError('Dataset file list sai schema')
        for row in rows:
            if (not isinstance(row, dict) or not isinstance(row.get('name'), str)
                    or type(row.get('size')) is not int or row['size'] <= 0):
                raise ValueError('Dataset file list thiếu tên/kích thước hợp lệ')
            if row['name'] in files:
                raise ValueError('Dataset file list trùng tên giữa các trang')
            files[row['name']] = row['size']
        if not next_token:
            break
        if next_token in seen_tokens:
            raise ValueError('Kaggle lặp token trang file; dừng trước khi gửi job')
        seen_tokens.add(next_token)
        token = next_token
    images = sum(name.startswith('images/') and not name.endswith('/') for name in files)
    if not {'manifest.json', 'fakeddit_sample.tsv'} <= set(files) or images != spec[split + '_samples']:
        raise RuntimeError(f'Dataset {reference} đọc được nhưng chưa thấy đủ manifest/TSV/'
                           f'{spec[split + "_samples"]} ảnh (thấy {images}). '
                           'Chờ Kaggle xử lý rồi thử lại, không tạo lại dataset.')
    print(f'[DATASET CHECK] {reference}: status trả 403 nhưng file API đọc được đủ '
          f'manifest/TSV/{images} ảnh; run sẽ kiểm SHA256 cohort trước khi nạp model.', flush=True)


def submit(plan):
    """Không ghi đè dataset ảnh hoặc job đang chạy; code bundle có ID theo hash."""
    status = cli(['kernels', 'status', plan['kernel']], check=False)
    if status.returncode == 0 and re.search(r'\b(running|queued)\b', status.stdout, re.I):
        raise RuntimeError('Job đang chạy. Kiểm tra status hoặc tải output; không gửi trùng job.')
    if status.returncode and not confirmed_missing('kernels', plan['kernel'], status):
        raise RuntimeError(status.stderr or status.stdout)
    # Giống launcher cũ: lấy quota thật và in trước khi upload hoặc push GPU.
    quota = cli(['quota'])
    print(f'[KAGGLE QUOTA]\n{quota.stdout.strip()}', flush=True)
    for dataset in plan.get('data_datasets', [plan['data_dataset']]):
        check_input_dataset(plan, dataset)
    if plan.get('inline_code'):
        return push_prepared_kernel(plan)
    state = cli(['datasets', 'status', plan['code_dataset'], '--format', 'json'], check=False)
    if state.returncode:
        absent = confirmed_missing('datasets', plan['code_dataset'], state)
        if not absent and not permission_denied(state):
            raise RuntimeError(state.stderr or state.stdout)
        if absent:
            print('Upload bundle code:', plan['code_dataset'], flush=True)
            created = cli(['datasets', 'create', '-p', str(path(plan['root']) / 'code'), '-t'])
            print(created.stdout, flush=True)
    deadline = time.monotonic() + 300
    while True:
        if code_dataset_ready(plan):
            break
        if time.monotonic() >= deadline:
            raise TimeoutError('Kaggle đang xử lý bundle code; chạy lại launcher sau, không tạo lại dataset ảnh')
        print('Kaggle đang xử lý bundle code...', flush=True)
        time.sleep(10)
    return push_prepared_kernel(plan)


def push_prepared_kernel(plan):
    result = cli(['kernels', 'push', '-p', str(path(plan['root']) / 'kernel'),
                  '--accelerator', plan['launcher']['accelerator'],
                  '-t', str(plan['launcher']['timeout_seconds'])])
    version = re.search(r'Kernel version (\d+) successfully pushed', result.stdout, re.I)
    if version:
        plan['kernel_version'] = int(version.group(1))
        plan['kernel_reference'] = plan['kernel'] + '/' + version.group(1)
    elif plan['task'] == 'paper_reference' or plan.get('inline_code'):
        plan.update(status='submission_uncertain', push_output=result.stdout)
        write_json(path(plan['root']) / 'launch.json', plan)
        raise RuntimeError('Kaggle chưa trả số version; kiểm tra notebook trước khi gửi lại để tránh chạy trùng')
    plan['status'] = 'submitted'
    plan['push_output'] = result.stdout
    write_json(path(plan['root']) / 'launch.json', plan)
    print(result.stdout, flush=True)
    return plan


def resume_error(plan, message):
    """Lỗi đọc sau push không đồng nghĩa job GPU đã dừng."""
    plan['last_wait_error'] = message
    write_json(path(plan['root'])/'launch.json', plan)
    command = shlex.join([sys.executable, '-B', str(ROOT/'scripts/launch_runs.py'),
                          '--resume-stage', plan['root']])
    return RuntimeError(f'{message}\nJob đã được gửi: {plan.get("kernel_reference", plan["kernel"])}.'
                        f'\nGiữ nguyên job; theo dõi/tải lại bằng:\n{command}')


def retryable_read(result):
    message = result.stderr + result.stdout
    if re.search(r'\b401\b|unauthenticated|unauthorized|authentication (failed|required)|invalid[^\n]*(token|api key)', message, re.I):
        return False
    return (permission_denied(result) or missing(result) or bool(re.search(
        r'\b(408|429|500|502|503|504)\b|NameResolutionError|ConnectionError|ConnectionReset|'
        r'Connection aborted|Max retries exceeded|timed?\s*out|ReadTimeout|Temporary failure',
        message, re.I)))


def read_submitted_job(arguments, plan, timeout_seconds=300):
    """Retry hữu hạn trên đúng version; không push hoặc chuyển sang latest."""
    deadline = time.monotonic() + timeout_seconds
    auth_checked = False
    while True:
        result = cli(arguments, check=False)
        if result.returncode == 0:
            return result
        message = (result.stderr or result.stdout).strip()
        if permission_denied(result) and not auth_checked:
            auth = cli(['kernels','list','--mine','--page-size','1','--format','json'], check=False)
            auth_checked = True
            if auth.returncode and re.search(
                    r'\b401\b|unauthenticated|unauthorized|authentication (failed|required)',
                    auth.stderr + auth.stdout, re.I):
                login = shlex.join([str(Path(sys.executable).with_name('kaggle')), 'auth', 'login'])
                raise resume_error(plan, f'Phiên Kaggle CLI chưa được xác thực. Đăng nhập lại:\n{login}')
        if not retryable_read(result) or time.monotonic() >= deadline:
            raise resume_error(plan, f'Kaggle {arguments[1]} tạm không đọc được:\n{message}')
        delay = min(plan['launcher']['poll_seconds'], max(0, deadline-time.monotonic()))
        print(f'Kaggle {arguments[1]} chưa truy cập được; giữ version đã gửi, thử lại sau {delay:.0f}s.', flush=True)
        time.sleep(delay)


def wait_and_collect(plan):
    kernel_reference = plan.get('kernel_reference', plan['kernel'])
    deadline = time.monotonic() + plan['launcher']['timeout_seconds'] + 900
    while time.monotonic() < deadline:
        result = read_submitted_job(['kernels', 'status', kernel_reference], plan,
                                    timeout_seconds=min(300, max(0, deadline-time.monotonic())))
        print(result.stdout.strip(), flush=True)
        if re.search(r'\b(error|failed|canceled|cancelled)\b', result.stdout, re.I):
            raise RuntimeError('Kaggle job lỗi; xem log tại https://www.kaggle.com/code/' + plan['kernel'])
        if re.search(r'\bcomplete\b', result.stdout, re.I):
            break
        time.sleep(plan['launcher']['poll_seconds'])
    else:
        raise resume_error(plan, 'Hết thời gian theo dõi local; không hủy job Kaggle')
    output = path('results/final/kaggle') / path(plan['root']).name
    output.mkdir(parents=True, exist_ok=True)
    read_submitted_job(['kernels', 'output', kernel_reference, '-p', str(output), '-o',
                        '--file-pattern', r'^metrics_.*\.zip$'], plan)
    archives = list(output.glob('metrics_*.zip'))
    if len(archives) != 1:
        raise ValueError('Cần đúng một ZIP metrics của run vừa hoàn tất')
    with zipfile.ZipFile(archives[0]) as archive:
        for name in archive.namelist():
            if not (output / name).resolve().is_relative_to(output.resolve()):
                raise ValueError('Path ngoài ZIP metrics')
        archive.extractall(output)
    from .protocol import completed_run, report
    completed_run(output / 'run')
    report(output / 'run')
    lock_file = output / 'fakeddit_test_lock.json'
    if plan['task'] == 'fakeddit' and plan['name'] == 'dev' and plan['mode'] == 'full':
        lock = read_json(lock_file)
        if (lock['sha256'] != digest(lock['payload'])
            or lock['payload']['dev_run_sha256'] != sha256(output / 'run/run.json')):
            raise ValueError('Lock không khớp run dev vừa tải')
        target = path(plan['settings']['fakeddit']['lock_file'])
        if target.exists() and target.read_bytes() != lock_file.read_bytes():
            raise ValueError('Đã có lock khác tại local; không ghi đè')
        if not target.exists():
            shutil.copy2(lock_file, target)
    plan.update(status='collected', output=str(output))
    write_json(path(plan['root']) / 'launch.json', plan)
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launcher-config', default='configs/kaggle_launcher.yaml')
    parser.add_argument('--job', default='benchmark_dtd')
    parser.add_argument('--use-kaggle', choices=['true', 'false'])
    parser.add_argument('--max-samples', type=int)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--wait', action='store_true', help='Chờ Kaggle xong, tải và kiểm tra metrics về local')
    parser.add_argument('--resume-stage', help='Đường dẫn root staging đã in; chỉ theo dõi/tải job cũ, không gửi lại')
    args = parser.parse_args(argv)
    if args.resume_stage:
        if args.prepare_only:
            parser.error('Không kết hợp resume-stage với prepare-only')
        plan = read_json(path(args.resume_stage) / 'launch.json')
        if plan['status'] != 'submitted':
            parser.error('Chỉ resume staging có job đã submitted')
        print_run_limits(plan)
        plan = wait_and_collect(plan)
        print(json.dumps({'status': plan['status'], 'output': plan['output']}, indent=2))
        return
    launcher = yaml.safe_load(path(args.launcher_config).read_text())
    if args.job not in launcher['jobs']:
        parser.error('Job không có trong launcher config')
    enabled = args.use_kaggle == 'true' if args.use_kaggle else launcher['use_kaggle']
    if not enabled:
        parser.error('Launcher này gửi Kaggle; chạy local dùng run.py benchmark/fakeddit')
    plan = prepare(launcher, args.job, args.max_samples)
    print_run_limits(plan)
    if not args.prepare_only:
        plan = submit(plan)
        if args.wait:
            plan = wait_and_collect(plan)
    else:
        print('[KAGGLE QUOTA] Chưa truy vấn (--prepare-only).', flush=True)
    print(json.dumps({key: plan[key] for key in ['status', 'kernel', 'code_dataset', 'data_dataset', 'root']}, indent=2))
