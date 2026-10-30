"""Kiểm tra protocol đánh giá độc lập; không nhận hoặc sửa nhãn dự đoán."""
import json
from pathlib import Path


def check_holdout(rows, reference, expected_split):
    if expected_split not in ('dev','val','validate','validation','test'):
        raise ValueError('Đánh giá độc lập cần source_split validation/dev/test, không phải train')
    if not rows or any(r.get('source_split') != expected_split or r.get('label_source') != 'image' for r in rows):
        raise ValueError('Nguồn split/nhãn ảnh không đúng protocol')
    ids=set(reference['image_ids']);hashes=set(reference['image_sha256']);tweets=set(reference['tweet_ids'])
    collisions={'image_id':0,'image_sha256':0,'tweet_id':0}
    for row in rows:
        key=row['id']
        # CrisisMMD image_id có dạng tweet_id_image_index; từ chối tên ngoài schema.
        pieces=key.rsplit('_',1)
        if len(pieces)!=2 or not all(piece.isdigit() for piece in pieces):
            raise ValueError('Không xác định được tweet_id từ image_id CrisisMMD')
        collisions['image_id']+=key in ids
        collisions['image_sha256']+=row['image_sha256'] in hashes
        collisions['tweet_id']+=pieces[0] in tweets
    if any(collisions.values()):
        raise ValueError(f'Tập đánh giá trùng dữ liệu phát triển: {collisions}; không bỏ mẫu âm thầm')
    return {'status':'disjoint_from_recorded_development','samples':len(rows),
            'expected_split':expected_split,'collisions':collisions,
            'limitations':'Kiểm tra trùng chính xác ID/tweet/hash ảnh; không phát hiện ảnh gần giống hoặc cùng sự kiện.'}


def lock_holdout(settings, rows, root):
    from experiments.waffle_lab.run import digest
    split=rows[0]['source_split']
    if split not in ('dev','val','validate','validation','test'):
        raise ValueError('Split không hợp lệ')
    fingerprint=digest(rows)
    directory=Path(root)/'results/_system/sample_locks/clip_two_stage'/split
    directory.mkdir(parents=True,exist_ok=True)
    path=directory/f"{settings['task']}_samples_{len(rows)}.json"
    cohort={'cohort_fingerprint':fingerprint,'samples':len(rows),'rows':rows}
    try:
        with path.open('x') as stream:json.dump(cohort,stream,indent=2)
    except FileExistsError:
        if json.loads(path.read_text())['cohort_fingerprint']!=fingerprint:
            raise ValueError('Tập đánh giá khác cohort đã khóa; không trộn hai tập')
    return cohort
