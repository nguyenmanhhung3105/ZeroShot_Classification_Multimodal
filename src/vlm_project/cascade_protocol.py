"""Protocol test hai tầng min5: sáu nhánh cố định, runtime đích và cohort khóa riêng."""
import csv
import importlib.metadata
import platform
import re

from packaging.requirements import Requirement
import yaml

from . import fakeddit_fewshot_cascade as cascade, min5_protocol
from .io import ROOT, code_hashes, digest, path, read_json, sha256

LIBRARIES = ['numpy','pillow','pyyaml','scipy','scikit-learn','torch','torchvision','open_clip_torch',
             'transformers','sentencepiece','protobuf','ftfy','regex','packaging','tqdm']


def runtime_versions():
    versions = {}
    for name in LIBRARIES:
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = None
    return {'python': platform.python_version(), 'libraries': versions}


def _check_runtime(value):
    if (not isinstance(value, dict) or not isinstance(value.get('python'),str)
            or not re.fullmatch(r'3\.\d+\.\d+', value['python'])
            or not isinstance(value.get('libraries'),dict)
            or set(value.get('libraries', {})) != set(LIBRARIES)
            or any(not isinstance(v, str) or not v for v in value['libraries'].values())):
        raise ValueError('Runtime thiếu Python hoặc phiên bản thư viện đích')
    versions = {re.sub(r'[-_.]+','-',k).lower():v for k,v in value['libraries'].items()}
    for line in (ROOT / 'requirements.txt').read_text().splitlines():
        if line.strip() and not line.startswith('#'):
            requirement = Requirement(line)
            key = re.sub(r'[-_.]+','-',requirement.name).lower()
            if key not in versions or not requirement.specifier.contains(versions[key], prereleases=True):
                raise ValueError(f'Runtime không thỏa requirements: {requirement.name}')


def capture_runtime(destination):
    value = runtime_versions()
    _check_runtime(value)
    return min5_protocol.exclusive_write(destination, {'version':1, 'kind':'cascade_target_runtime', 'runtime':value})


def fingerprint(settings_file, spec_file):
    return {'code': code_hashes(), 'files': {
        'settings': sha256(settings_file), 'recipe': sha256(spec_file), 'requirements': sha256(ROOT/'requirements.txt'),
        'protocol_cli': sha256(ROOT/'scripts/prepare_cascade_protocol.py'),
        'runner_cli': sha256(ROOT/'scripts/run_cascade_test.py')}}


def methods(recipe):
    cascade._check_specification(recipe)
    expected = {(p,s,d) for p in cascade.POLICIES for s in range(5) for d in ['default','target_0.87']}
    found = set()
    for c in recipe['gate_candidates']:
        key = (c['policy'], c['seed'], c['decision'])
        if key in found or c['modality'] != 'image_title' or type(c['seed']) is not int:
            raise ValueError('Bộ gate không khớp sáu nhánh đã chạy dev')
        found.add(key)
        if c['decision'] != 'default' and c.get('target_recall_fake') != .87:
            raise ValueError('Không đổi mục tiêu ngưỡng sau dev')
    if found != expected:
        raise ValueError('Phải giữ đúng default và target 87% cho hai policy/năm seed')
    return [f'{p}_{suffix}' for p in cascade.POLICIES for suffix in ['flat','hierarchy_default','hierarchy_target_0.87']]


def _file_reference(file):
    file = path(file)
    return {'absolute':str(file), 'relative':str(file.relative_to(ROOT)) if file.is_relative_to(ROOT) else None}


def files(payload, settings_file=None, spec_file=None):
    def resolve(reference, override):
        return path(override if override is not None else reference['relative'] or reference['absolute'])
    return resolve(payload['settings_file'],settings_file), resolve(payload['spec_file'],spec_file)


def prepare(dev_run, train_source, destination, settings_file='configs/project_min5.yaml', *, spec_file=cascade.SPEC_FILE):
    dev_run, train_source, destination = path(dev_run), path(train_source), path(destination)
    settings_file, spec_file = path(settings_file), path(spec_file)
    if destination.is_relative_to(dev_run) or destination.is_relative_to(train_source):
        raise ValueError('Không ghi protocol vào source run')
    run, hashes = min5_protocol.checked_run(dev_run, ['support.json','cohort.json','summary.tsv','metrics_by_seed.tsv'])
    if (run.get('kind') != 'fakeddit_fewshot_cascade' or run.get('mode') != 'full' or run.get('split') != 'dev'
            or run.get('samples') != 3000 or run.get('stages') != 2 or run.get('test_opened') is not False):
        raise ValueError('Cần full dev cascade hai tầng đã hoàn tất, không smoke/test')
    recipe = cascade.specification(spec_file)
    suite = methods(recipe)
    if run['specification'] != recipe or run['specification_sha256'] != sha256(spec_file):
        raise ValueError('Recipe đã đổi sau dev')
    critical = ['fakeddit_fewshot_cascade.py','fakeddit_fewshot_task1.py','backbone.py','data.py','evaluate.py','io.py','fewshot_support_thresholds.py','min5_protocol.py']
    for name in critical:
        key = 'src/vlm_project/'+name
        if run.get('code_sha256',{}).get(key) != sha256(ROOT/key):
            raise ValueError(f'Code inference/dev đã đổi: {name}; không khóa kết quả bằng code khác')
    source, source_hashes = min5_protocol.checked_run(train_source, ['train_cohort.json','support.json'])
    if (source.get('kind') != 'fakeddit_fewshot_support' or source.get('split') != 'dev'
            or source.get('mode') != 'full' or source.get('test_opened') is not False
            or sha256(train_source/'run.json') != recipe['source_run_sha256']):
        raise ValueError('Nguồn train support không khớp recipe dev')
    train, dev = read_json(train_source/'train_cohort.json'), read_json(dev_run/'cohort.json')
    excluded_train, excluded_dev = min5_protocol.identities(train), min5_protocol.identities(dev)
    min5_protocol.no_overlap(excluded_train, excluded_dev)
    min5_protocol.validate_support(recipe['support_snapshot'], train)
    if (read_json(dev_run/'support.json') != recipe['support_snapshot']
            or read_json(train_source/'support.json') != recipe['support_snapshot']
            or digest(excluded_train) != recipe['train_identity_sha256']
            or digest(excluded_dev) != recipe['dev_identity_sha256']):
        raise ValueError('Support/cohort khác dev đã chạy')
    for split in ['train','dev']:
        if run[split+'_cohort']['manifest_sha256'] != recipe['manifest_sha256'][split]:
            raise ValueError('Cohort manifest không khớp recipe')
    expected = {(m,s) for m in suite for s in range(5)}
    with (dev_run/'metrics_by_seed.tsv').open() as stream:
        rows = list(csv.DictReader(stream,delimiter='\t'))
    if (len(rows) != 30 or {(r['method'],int(r['seed'])) for r in rows} != expected
            or any(r['task'] != '6way' or int(r['samples']) != 3000 for r in rows)
            or len(run.get('summary',[])) != 6 or {r['method'] for r in run['summary']} != set(suite)
            or any(r['seeds'] != 5 or r['samples'] != 3000 for r in run['summary'])):
        raise ValueError('Dev thiếu sáu nhánh/năm seed hoặc sai số mẫu')
    settings = yaml.safe_load(settings_file.read_text())
    if settings.get('model') != recipe['model'] or type(settings.get('batch_size')) is not int or not 1 <= settings['batch_size'] <= 64:
        raise ValueError('Model/batch_size không hợp lệ')
    payload = {'version':1, 'kind':'min5_cascade_test_protocol', 'state':'draft', 'stages':2,
        'split':'test', 'test_opened':False, 'test_samples':3000, 'test_manifest_sha256':None,
        'primary_metric':'macro_f1', 'report_metrics':['accuracy','macro_f1','balanced_accuracy','recall_per_class'],
        'methods':suite, 'recipe':recipe, 'recipe_sha256':digest(recipe),
        'settings_file':_file_reference(settings_file), 'spec_file':_file_reference(spec_file),
        'inference_settings':{'model':settings['model'],'batch_size':settings['batch_size']},
        'fingerprint':fingerprint(settings_file,spec_file), 'runtime':None,
        'dev_run_sha256':sha256(dev_run/'run.json'), 'dev_artifact_sha256':hashes,
        'train_source_sha256':sha256(train_source/'run.json'), 'train_artifact_sha256':source_hashes,
        'historical_dev_code_sha256':run['code_sha256'], 'dev_summary':run['summary'],
        'excluded_train':excluded_train, 'excluded_dev':excluded_dev,
        'scope':'Six predeclared branches, 5 seeds, same support/thresholds as full dev. Oracle diagnostic only, not ranked. No test threshold/seed selection.',
        'limitations':'New test wrappers verified with synthetic data; historical inference code retained. Target runtime must be captured and sealed explicitly; old dev did not record all dependency versions.'}
    return min5_protocol.exclusive_write(destination,payload)


def validate_lock(lock, *, require_sealed=True, settings_file=None, spec_file=None, check_runtime=True):
    if not isinstance(lock,dict) or not isinstance(lock.get('payload'),dict):
        raise ValueError('Protocol sai định dạng')
    p=lock['payload']
    if (lock.get('sha256') != digest(p) or p.get('version') != 1 or p.get('kind') != 'min5_cascade_test_protocol'
            or p.get('stages') != 2 or p.get('test_opened') is not False or p.get('split') != 'test'
            or p.get('test_samples') != 3000 or p.get('primary_metric') != 'macro_f1'):
        raise ValueError('Protocol checksum/schema/metric không hợp lệ')
    if p.get('state') not in ['draft','sealed_cascade'] or require_sealed and p['state'] != 'sealed_cascade':
        raise ValueError('Protocol còn draft hoặc sai trạng thái; không cho phép test')
    if p['methods'] != methods(p['recipe']) or digest(p['recipe']) != p['recipe_sha256']:
        raise ValueError('Không được đổi suite hoặc ngưỡng sau dev')
    config_file, recipe_file=files(p,settings_file,spec_file)
    if fingerprint(config_file,recipe_file) != p['fingerprint']:
        raise ValueError('Code/cấu hình/recipe đã đổi sau khi tạo protocol')
    if p['recipe'] != cascade.specification(recipe_file):
        raise ValueError('Recipe trong protocol khác snapshot đã khóa')
    for split in ['train','dev']:
        if digest(min5_protocol.identities(p['excluded_'+split])) != p['recipe'][split+'_identity_sha256']:
            raise ValueError('Danh sách chống trùng khác cohort nguồn')
    min5_protocol.no_overlap(p['excluded_train'],p['excluded_dev'])
    if p['state']=='sealed_cascade':
        h=p.get('test_manifest_sha256')
        if not isinstance(h,str) or not re.fullmatch(r'[0-9a-f]{64}',h) or h in p['recipe']['manifest_sha256'].values():
            raise ValueError('Hash test thiếu/sai hoặc trùng train/dev')
        _check_runtime(p['runtime'])
        if check_runtime and runtime_versions()!=p['runtime']:
            raise ValueError('Phiên bản runtime khác môi trường đích đã niêm phong')
    elif p.get('runtime') is not None or p.get('test_manifest_sha256') is not None:
        raise ValueError('Draft chưa được gắn runtime/test manifest')
    return lock


def validate(file, **kwargs):
    return validate_lock(read_json(file),**kwargs)


def seal(draft,test_manifest_sha256,runtime_file,destination):
    lock=validate(draft,require_sealed=False,check_runtime=False)
    p=lock['payload']
    if p['state']!='draft' or not isinstance(test_manifest_sha256,str) or not re.fullmatch(r'[0-9a-f]{64}',test_manifest_sha256):
        raise ValueError('Cần draft và SHA256 manifest test hợp lệ')
    if test_manifest_sha256 in p['recipe']['manifest_sha256'].values():
        raise ValueError('Không dùng train/dev làm test')
    rt=read_json(runtime_file);r=rt.get('payload',{})
    if rt.get('sha256')!=digest(r) or r.get('version')!=1 or r.get('kind')!='cascade_target_runtime':
        raise ValueError('File runtime sai schema/checksum')
    _check_runtime(r['runtime'])
    return min5_protocol.exclusive_write(destination,{**p,'state':'sealed_cascade','runtime':r['runtime'],
        'test_manifest_sha256':test_manifest_sha256,'draft_sha256':lock['sha256'],'runtime_file_sha256':sha256(runtime_file)})


def check_test_cohort(lock,rows,metadata,*,settings_file=None,spec_file=None):
    p=validate_lock(lock,settings_file=settings_file,spec_file=spec_file)['payload']
    if (metadata.get('split')!='test' or metadata.get('source_split')!='test_public'
            or metadata.get('manifest_sha256')!=p['test_manifest_sha256']):
        raise ValueError('Sai manifest/split nguồn test đã khóa')
    test=min5_protocol.identities(rows)
    if any(str(r.get('6way')) not in list('012345') for r in rows):
        raise ValueError('Taxonomy test sai')
    min5_protocol.no_overlap(test,p['excluded_train']);min5_protocol.no_overlap(test,p['excluded_dev'])
