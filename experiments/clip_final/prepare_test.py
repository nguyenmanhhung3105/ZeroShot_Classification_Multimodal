"""Người dùng chạy để chuẩn bị giao hai annotation TEST; không tải model.

Đọc đúng hai TSV test và ảnh tương ứng. Agent chỉ kiểm thử bằng fixture tạm;
không chạy module này trên dữ liệu thật khi AGENTS.md cấm truy cập raw.
"""
import csv
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
import yaml
from PIL import Image
from experiments.clip_final.protocol import PLAN, load_plan
from experiments.clip_two_stage.prepare_dev import read_dev as read_annotations
from experiments.waffle_lab.run import digest, file_hash


def prepare(root=ROOT):
    root = Path(root).resolve()
    plan = load_plan(root/PLAN, root); protocol = plan['protocol']; reference = protocol['exclusion']
    directory = root/'data/processed/pro_crisismmd_test_final'
    config_file = root/'configs/experiment_crisismmd_test_final.yaml'
    launcher_file = root/'experiments/clip_final/launcher_test.yaml'
    if any(p.exists() for p in [directory, config_file, launcher_file]):
        raise FileExistsError('Test đã chuẩn bị; không tạo lại hoặc ghi đè cohort đã khóa')
    source = root/'data/raw/CrisisMMD_v2.0'
    files = {task: source/f'crisismmd_datasplit_all/task_{name}_text_img_test.tsv'
             for task,name in [('informativeness','informative'), ('humanitarian','humanitarian')]}
    records = {task: read_annotations(path, task) for task,path in files.items()}
    common = records['informativeness'].keys() & records['humanitarian'].keys()
    # Thứ tự dựa trên ID, không dùng lớp/độ khó/confidence để chọn test.
    ordered = sorted(common, key=lambda key: (hashlib.sha256(('final_test_v1:'+key).encode()).hexdigest(),key))
    ids, tweets, hashes = [set(reference[key]) for key in ['image_ids','tweet_ids','image_sha256']]
    selected = []; excluded = []; seen_hashes = set()
    for key in ordered:
        a, b = [records[task][key] for task in ['informativeness','humanitarian']]
        if any(a[field] != b[field] for field in ['tweet_id','image','event_name','tweet_text']):
            raise ValueError('Hai annotation test không khớp metadata ảnh/tweet')
        image = (source/a['image']).resolve()
        if not image.is_relative_to((source/'data_image').resolve()): raise ValueError('Ảnh ngoài CrisisMMD data_image')
        reason = None; sha = None
        if key in ids or a['tweet_id'] in tweets: reason = 'overlap_train_dev_id_or_tweet'
        elif not image.is_file(): reason = 'missing_image'
        else:
            sha = file_hash(image)
            if sha in hashes: reason = 'overlap_train_dev_exact_image_hash'
            elif sha in seen_hashes: reason = 'duplicate_image_hash_within_test'
            else:
                try:
                    with Image.open(image) as im: im.verify()
                    with Image.open(image) as im: im.convert('RGB').load()
                except (OSError,ValueError,Image.DecompressionBombError): reason = 'invalid_image'
        if reason:
            excluded.append({'id':key,'reason':reason});continue
        seen_hashes.add(sha);selected.append((key,a,b,image,sha))
        if len(selected) == protocol['rules']['max_test_samples']: break
    if not selected: raise ValueError('Không có ảnh test chung hợp lệ sau loại trùng train/dev')
    directory.mkdir(parents=True);(directory/'images').mkdir()
    sample_file = directory/'pre_crisismmd.tsv'; entries = []
    fields = ['tweet_id','image_id','clean_text','event','label_informative','label_humanitarian','image_path',
              'source_split','label_source','source_file_informative','source_row_informative',
              'source_file_humanitarian','source_row_humanitarian']
    with sample_file.open('x',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields,delimiter='\t');writer.writeheader()
        for key,a,b,image,sha in selected:
            target=directory/'images'/(key+image.suffix.lower());shutil.copy2(image,target)
            writer.writerow({'tweet_id':a['tweet_id'],'image_id':key,'clean_text':a['tweet_text'],'event':a['event_name'],
                'label_informative':a['label_image'],'label_humanitarian':b['label_image'],
                'image_path':str(target.relative_to(root)),'source_split':'test','label_source':'image',
                'source_file_informative':str(files['informativeness'].relative_to(root)),'source_row_informative':a['source_row'],
                'source_file_humanitarian':str(files['humanitarian'].relative_to(root)),'source_row_humanitarian':b['source_row']})
            entries.append({'id':key,'image_sha256':sha,'y_true':{'informativeness':a['label_image'],'humanitarian':b['label_image']}})
    manifest = {'status':'ready','dataset':'crisismmd','task':'both','source_split':'test','label_source':'image',
        'cohort_id':digest(entries),'samples_sha256':file_hash(sample_file),'samples':entries,
        'selected':{'samples':len(entries)},'source_counts':{task:len(rows) for task,rows in records.items()},
        'intersection_samples':len(common),'selection_rule':protocol['rules']['selection'],
        'unscanned_common_after_cap':len(common)-len(selected)-len(excluded),
        'outside_common_split':{task:len(rows)-len(common) for task,rows in records.items()},
        'excluded_counts':dict(Counter(r['reason'] for r in excluded)),'excluded':excluded,
        'gate_labels_available':True,'final_protocol_sha256':plan['protocol_sha256'],
        'support':{task:dict(Counter(e['y_true'][task] for e in entries)) for task in records},
        'source_hashes':{str(p.relative_to(root)):file_hash(p) for p in files.values()},
        'scope':'locked_final_test_common_official_test_splits', 'raw_files_read':list(str(p.relative_to(root)) for p in files.values())}
    (directory/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    config=yaml.safe_load((root/'configs/experiment_config.yaml').read_text())
    config['models']=['metaclip2_vith14']
    for item in config['datasets'].values(): item['enabled']=False
    config['datasets']['crisismmd'].update(enabled=True,task='humanitarian',source_split='test',label_source='image',
        data_path=str(sample_file.relative_to(root)),image_dir=str((directory/'images').relative_to(root)),
        cohort_manifest=str((directory/'manifest.json').relative_to(root)),max_samples=len(entries),
        gate_labels_available=True,final_dn_unit_plan=PLAN)
    with config_file.open('x') as f: yaml.safe_dump(config,f,allow_unicode=True,sort_keys=False)
    launcher=yaml.safe_load((root/'experiments/clip_two_stage/launcher.yaml').read_text())
    launcher['experiment_config']=str(config_file.relative_to(root))
    launcher['jobs']={'final_dn_unit_test':{**launcher['jobs']['two_stage'],'max_samples':len(entries),
        'evaluation_scope':'heldout','export_features':True,'tda_evaluate':False,'stata_gate':False}}
    with launcher_file.open('x') as f: yaml.safe_dump(launcher,f,sort_keys=False)
    print(json.dumps({key:manifest[key] for key in ['selected','source_counts','intersection_samples','excluded_counts','support','final_protocol_sha256']},ensure_ascii=False,indent=2))
    return directory


if __name__ == '__main__': print(prepare())
