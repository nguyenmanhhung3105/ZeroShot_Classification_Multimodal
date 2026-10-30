"""Đối chứng phân cấp từ score đã lưu; không tải model hoặc đọc dataset."""
import argparse
import json
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.waffle_lab.analyze import load_run
from experiments.waffle_lab.run import evaluate
import numpy as np


def reconcile(info, human, info_labels, human_labels):
    """Thay tổng xác suất hai nhóm bằng phân phối informativeness.

    Giữ tỷ lệ tương đối giữa bảy lớp humanitarian; không nhân xác suất
    từng lớp với prior rồi vô tình tính prior hai lần.
    """
    info, human = np.asarray(info, float), np.asarray(human, float)
    if set(info_labels) != {'informative', 'not_informative'} or 'not_humanitarian' not in human_labels:
        raise ValueError('Sai taxonomy')
    if info.shape != (len(human), 2) or human.shape != (len(info), len(human_labels)):
        raise ValueError('Sai shape')
    for values in (info, human):
        if not np.isfinite(values).all() or (values < 0).any() or not np.allclose(values.sum(1), 1):
            raise ValueError('Cần phân phối xác suất hợp lệ')
    out = human.copy()
    negative = human_labels.index('not_humanitarian')
    active = [i for i in range(len(human_labels)) if i != negative]
    conditional = human[:, active]
    total = conditional.sum(1, keepdims=True)
    conditional = np.divide(conditional, total, out=np.full_like(conditional, 1/len(active)), where=total > 0)
    out[:, active] = conditional * info[:, [info_labels.index('informative')]]
    out[:, negative] = info[:, info_labels.index('not_informative')]
    return out


def analyze(info, human):
    for key in ('model', 'samples', 'device', 'compute_dtype', 'inference_backend'):
        if info['manifest'][key] != human['manifest'][key]:
            raise ValueError(f'Khác {key}')
    if info['manifest']['task'] != 'informativeness' or human['manifest']['task'] != 'humanitarian':
        raise ValueError('Sai task')
    def identity(run):
        # Hai task có nhãn/fingerprint khác nhau nhưng phải cùng ảnh và thứ tự.
        return [(r['id'], r['image_sha256'], r['source_split'], r['label_source'])
                for r in run['cohort']['rows']]
    if identity(info) != identity(human):
        raise ValueError('Hai task khác ảnh hoặc thứ tự')
    p, q = [np.asarray([json.loads(r['probabilities']) for r in run['rows']]) for run in (info, human)]
    labels = human['plan']['label_order']
    out = reconcile(p, q, info['plan']['label_order'], labels)
    predicted = [labels[i] for i in out.argmax(1)]
    truth = [r['y_true'] for r in human['rows']]
    metrics = evaluate(truth, predicted, out, labels)
    baseline = human['metrics']
    return {'scope': 'exploratory_train_cohort_not_independent_test',
            'method': 'hierarchy_group_mass_v1', 'inputs': [info['directory'], human['directory']],
            'baseline_metrics': baseline, 'candidate_metrics': metrics,
            'delta_macro_f1': metrics['macro_f1']-baseline['macro_f1'],
            'label_consistency_violations': sum((a['y_true']=='not_informative') != (b['y_true']=='not_humanitarian')
                                                for a,b in zip(info['rows'],human['rows'])),
            'note': 'Xác suất softmax chưa hiệu chuẩn; không chọn temperature/weight bằng nhãn.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--info', required=True)
    parser.add_argument('--human', required=True)
    args = parser.parse_args()
    runs = []
    for directory in (args.info, args.human):
        run = load_run(directory)
        with __import__('zipfile').ZipFile(Path(directory)/'run_artifacts.zip') as archive:
            run['cohort'] = json.loads(archive.read('cohort.json'))
        runs.append(run)
    result = analyze(*runs)
    output = ROOT/'results/analysis/crisis_hierarchy'/uuid.uuid4().hex[:12]
    output.mkdir(parents=True)
    (output/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    print(output/'analysis.json')


if __name__ == '__main__':
    main()
