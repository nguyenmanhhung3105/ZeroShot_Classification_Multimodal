"""Chuẩn bị riêng DEV CrisisMMD, không mở train/test, không tải model.

Đọc streaming humanitarian dev; gate không cần nhãn informative thật.
"""
from collections import Counter
from copy import deepcopy
import csv
import json
from pathlib import Path
import re
import shutil
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import yaml
from PIL import Image
from experiments.waffle_lab.run import file_hash,digest
from prompts.crisismmd_prompts import get_label_names


def read_dev(path,task):
    allowed=set(get_label_names(task));records={}
    with path.open(encoding='utf-8-sig',newline='') as stream:
        reader=csv.DictReader(stream,delimiter='\t')
        required={'tweet_id','image_id','image','event_name','tweet_text','label_image'}
        if not required.issubset(reader.fieldnames or []):raise ValueError('Thiếu schema raw dev')
        for number,row in enumerate(reader,2):
            if number>20001:raise ValueError('Annotation dev vượt giới hạn 20.000 dòng')
            key=row['image_id']
            if not re.fullmatch(r'\d+_\d+',key) or row['tweet_id']!=key.rsplit('_',1)[0]:raise ValueError('ID không nguyên vẹn')
            if key in records or row['label_image'] not in allowed:raise ValueError('ID lặp hoặc nhãn ảnh sai')
            records[key]={**row,'source_row':number}
    return records


def prepare(root=ROOT):
    root=Path(root).resolve()
    source=root/'data/raw/CrisisMMD_v2.0'
    directory=root/'data/processed/pro_crisismmd_dev_humanitarian'
    config_file=root/'configs/experiment_crisismmd_dev_humanitarian.yaml'
    launcher_file=root/'experiments/clip_two_stage/launcher_dev_humanitarian.yaml'
    if any(p.exists() for p in [directory,config_file,launcher_file]):raise FileExistsError('Đầu ra dev đã tồn tại; không ghi đè')
    sources=[source/'crisismmd_datasplit_all/task_humanitarian_text_img_dev.tsv']
    human=read_dev(sources[0],'humanitarian')
    reference_path=root/'configs/prompts/crisismmd_development_exclusion.json'
    reference=json.loads(reference_path.read_text())
    ids=set(reference['image_ids']);hashes=set(reference['image_sha256']);tweets=set(reference['tweet_ids'])
    common=sorted(human);selected=[];excluded=[]
    for key in common:
        a=b=human[key]
        image=(source/a['image']).resolve()
        if not image.is_relative_to((source/'data_image').resolve()):raise ValueError('Ảnh ngoài thư mục raw CrisisMMD')
        reason=None
        if key in ids or a['tweet_id'] in tweets:reason='overlap_development_id_or_tweet'
        elif not image.is_file():reason='missing_image'
        else:
            sha=file_hash(image)
            if sha in hashes:reason='overlap_development_image_hash'
            else:
                try:
                    with Image.open(image) as im:im.verify()
                    with Image.open(image) as im:im.convert('RGB').load()
                except (OSError,ValueError,Image.DecompressionBombError):reason='invalid_image'
        if reason:
            excluded.append({'id':key,'reason':reason});continue
        selected.append((key,a,b,image,sha))
    if not selected:raise ValueError('Không có ảnh dev hợp lệ')
    directory.mkdir(parents=True);(directory/'images').mkdir()
    fields=['tweet_id','image_id','clean_text','event','label_informative','label_humanitarian','image_path',
            'source_split','label_source','source_file_informative','source_row_informative','source_file_humanitarian','source_row_humanitarian']
    sample_file=directory/'pre_crisismmd.tsv';entries=[]
    with sample_file.open('x',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,delimiter='\t');writer.writeheader()
        for key,a,b,image,sha in selected:
            target=directory/'images'/(key+image.suffix.lower());shutil.copy2(image,target)
            writer.writerow({'tweet_id':a['tweet_id'],'image_id':key,'clean_text':a['tweet_text'],'event':a['event_name'],
                             'label_informative':'','label_humanitarian':b['label_image'],
                             'image_path':str(target.relative_to(root)),'source_split':'dev','label_source':'image',
                             'source_file_informative':'','source_row_informative':'',
                             'source_file_humanitarian':str(sources[0].relative_to(root)),'source_row_humanitarian':b['source_row']})
            entries.append({'id':key,'image_sha256':sha,'y_true':{'informativeness':None,'humanitarian':b['label_image']}})
    manifest={'status':'ready','dataset':'crisismmd','task':'both','source_split':'dev','label_source':'image',
              'cohort_id':digest(entries),'samples_sha256':file_hash(sample_file),'samples':entries,
              'selected':{'samples':len(entries)},'source_counts':{'humanitarian':len(human)},
              'gate_labels_available':False,
              'excluded':excluded,'excluded_counts':dict(Counter(r['reason'] for r in excluded)),
              'source_hashes':{str(p.relative_to(root)):file_hash(p) for p in sources},
              'exclusion_reference_sha256':file_hash(reference_path),
              'support':{'humanitarian':dict(Counter(e['y_true']['humanitarian'] for e in entries))},
              'scope':'development_validation_not_final_test','test_accessed':False}
    (directory/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    config=yaml.safe_load((root/'configs/experiment_config.yaml').read_text())
    config['models']=['metaclip2_vith14']
    for item in config['datasets'].values():item['enabled']=False
    task=config['datasets']['crisismmd'];task.update(enabled=True,task='humanitarian',source_split='dev',
        data_path=str(sample_file.relative_to(root)),image_dir=str((directory/'images').relative_to(root)),
        cohort_manifest=str((directory/'manifest.json').relative_to(root)),max_samples=len(entries),gate_labels_available=False)
    config_file.write_text(yaml.safe_dump(config,allow_unicode=True,sort_keys=False))
    launcher=yaml.safe_load((root/'experiments/clip_two_stage/launcher.yaml').read_text())
    launcher['experiment_config']=str(config_file.relative_to(root))
    launcher['jobs']={'two_stage_dev':{**launcher['jobs']['two_stage'],'max_samples':len(entries),'evaluation_scope':'heldout'}}
    launcher_file.write_text(yaml.safe_dump(launcher,sort_keys=False))
    print(json.dumps({k:manifest[k] for k in ['source_counts','selected','excluded_counts','support','test_accessed']},ensure_ascii=False,indent=2))
    return directory


if __name__=='__main__':print(prepare())
