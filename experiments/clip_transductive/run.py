"""Thử transductive trên artifact dev đã lưu; không mở dataset hoặc tải model."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from experiments.clip_transductive.core import predict_two_stage, probability_features
from experiments.clip_two_stage.analyze import analyze, read_artifact
from experiments.clip_two_stage.run import decide

REFERENCE = '9066ad371757c794e4b8255e852f837ac9a1d3b1'


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def sha256(value):
    return hashlib.sha256(value).hexdigest()


def load_source(directory):
    # Validator cũ đối chiếu ID, hash cohort/protocol, score và dự đoán baseline.
    analyze(directory, bootstraps=1)
    names = ['run_manifest.json', 'predictions.tsv'] + [
        f'{task}/{name}.json' for task in ['informativeness', 'humanitarian']
        for name in ['frozen_plan', 'cohort']]
    contents = {name: read_artifact(directory, name) for name in names}
    manifest = json.loads(contents['run_manifest.json'])
    plans = [json.loads(contents[f'{task}/frozen_plan.json']) for task in ['informativeness', 'humanitarian']]
    cohort = json.loads(contents['humanitarian/cohort.json'])['rows']
    if (manifest.get('evaluation_scope') != 'heldout' or manifest.get('input') != 'image_only' or
            manifest.get('full_frame_view', False) or manifest.get('label_fit') is not False):
        raise ValueError('Cần artifact hai tầng image-only heldout không fit nhãn')
    if any(p['dataset_config'].get('source_split') != 'dev' for p in plans):
        raise ValueError('Thử nghiệm này chỉ cho dev; không tự dùng test để chọn phương pháp')
    if not cohort or any(r.get('source_split') != 'dev' or r.get('label_source') != 'image' for r in cohort):
        raise ValueError('Cohort phải là dev và nhãn ảnh')
    labels = plans[1]['label_order']
    info_labels = plans[0]['label_order']
    if len(labels) != 8 or len(set(labels)) != 8 or 'not_humanitarian' not in labels:
        raise ValueError('Cần đúng taxonomy humanitarian tám lớp')
    if labels != manifest.get('label_order') or info_labels != manifest.get('info_label_order'):
        raise ValueError('Thứ tự lớp khác manifest')
    rows = list(csv.DictReader(io.StringIO(contents['predictions.tsv']), delimiter='\t'))
    scores = [np.asarray([json.loads(r[key]) for r in rows], dtype=np.float64)
              for key in ['info_cosine_scores', 'human_cosine_scores']]
    for matrix in scores:
        probability_features(matrix)
    if any(r['y_true'] not in labels for r in rows):
        raise ValueError('Ground truth ngoài taxonomy')
    return {'manifest': manifest, 'info_labels': info_labels, 'labels': labels, 'rows': rows,
            'cohort': cohort, 'scores': scores,
            'artifact_sha256': {k: sha256(v.encode('utf-8')) for k, v in contents.items()}}


def metric(truth, predicted, labels):
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, labels=labels, zero_division=0)
    return {'accuracy': float(accuracy_score(truth, predicted)), 'macro_f1': float(f1.mean()),
            'per_class': {label: {'precision': float(precision[i]), 'recall': float(recall[i]),
                                  'f1': float(f1[i]), 'support': int(support[i]),
                                  'predicted': int(np.sum(np.asarray(predicted) == label))}
                          for i, label in enumerate(labels)}}


def validate_config(settings):
    if (settings.get('algorithm') != 'EM_DIRICHLET' or settings.get('reference_commit') != REFERENCE or
            settings.get('protocol') != 'fixed_image_gate_then_seven_class_transductive' or
            settings.get('label_fit') is not False or settings.get('support_samples') != 0):
        raise ValueError('Sai protocol zero-shot cố định')
    seeds = settings.get('grouping_seeds')
    if (not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds) or
            any(type(s) is not int or not 0 <= s < 2**32 for s in seeds)):
        raise ValueError('Grouping seeds phải là số nguyên không lặp')
    for key, minimum in [('query_size', 2), ('iterations', 1), ('mm_iterations', 1)]:
        if type(settings.get(key)) is not int or settings[key] < minimum:
            raise ValueError(f'{key} không hợp lệ')
    if not isinstance(settings.get('temperature'), (float, int)) or isinstance(settings['temperature'], bool):
        raise ValueError('Temperature phải là số dương hữu hạn')
    probability_features(np.zeros((1, 2)), settings['temperature'])


def run(settings, output_dir=None):
    validate_config(settings)
    source_dir = Path(settings['source_run'])
    if not source_dir.is_absolute():
        source_dir = ROOT / source_dir
    source = load_source(source_dir)
    output = (Path(output_dir) if output_dir else ROOT / 'results/analysis/clip_transductive' / uuid.uuid4().hex[:12])
    output.mkdir(parents=True, exist_ok=False)
    manifest = {'status': 'running', 'evaluation_scope': 'dev_exploratory_transductive',
                'settings': settings, 'source_run': str(source_dir), 'model': source['manifest']['model'],
                'prompt_seed': source['manifest']['seed'], 'samples': len(source['rows']),
                'source_artifact_sha256': source['artifact_sha256'],
                'source_cohort_fingerprints': source['manifest']['cohort_fingerprints'],
                'source_protocol_fingerprints': source['manifest']['protocol_fingerprints'],
                'code_sha256': {name: sha256((ROOT / name).read_bytes()) for name in [
                    'experiments/clip_transductive/core.py', 'experiments/clip_transductive/run.py',
                    'experiments/clip_two_stage/analyze.py', 'experiments/clip_two_stage/run.py']},
                'gate': 'unchanged_image_argmax', 'label_fit': False, 'support_samples': 0,
                'labels_used_for': 'artifact_validation_and_metrics_only',
                'grouping': 'shuffle_gate_positive_indices_only_without_labels_or_events',
                'probabilities_calibrated': False,
                'notes': ['Group seeds thay cách chia nhóm; không sinh lại prompt Waffle.',
                          'Cùng tập dev đã dùng thăm dò; không phải kết quả test cuối.']}
    write_json(output / 'run_manifest.json', manifest)
    started = time.perf_counter()
    try:
        a, b = source['scores']
        info_labels, labels = source['info_labels'], source['labels']
        baseline, gate = decide(a, b, info_labels, labels)
        positive = [i for i, label in enumerate(labels) if label != 'not_humanitarian']
        # Softmax T=30 phải giữ argmax baseline; không có thay đổi do nhiệt độ đơn thuần.
        baseline_t30 = np.asarray(labels)[np.asarray(positive)[probability_features(b[:, positive], settings['temperature']).argmax(1)]]
        baseline_t30[~gate] = 'not_humanitarian'
        if baseline_t30.tolist() != baseline:
            raise ValueError('Softmax đã đổi baseline argmax ngoài dự kiến')
        truth = np.asarray([r['y_true'] for r in source['rows']])
        old_ok = np.asarray(baseline) == truth
        baseline_metric = metric(truth, baseline, labels)
        reports = []
        for seed in settings['grouping_seeds']:
            print(f'[TRANSDUCTIVE] grouping_seed={seed}; images={int(gate.sum())}; query_size={settings["query_size"]}', flush=True)
            predicted, batch_ids, diagnostics = predict_two_stage(
                a, b, info_labels, labels, grouping_seed=seed,
                **{key: settings[key] for key in ['query_size', 'temperature', 'iterations', 'mm_iterations']})
            if np.any(batch_ids[gate] < 0) or np.any(batch_ids[~gate] != -1):
                raise AssertionError('Thiếu ảnh trong nhóm hoặc nhóm chứa ảnh bị gate chặn')
            new_ok = np.asarray(predicted) == truth
            fixed = ~old_ok & new_ok
            broken = old_ok & ~new_ok
            report = {'grouping_seed': seed, **metric(truth, predicted, labels),
                      'fixed': int(fixed.sum()), 'broken': int(broken.sum()),
                      'changed': int((np.asarray(predicted) != baseline).sum()),
                      'delta_macro_f1': metric(truth, predicted, labels)['macro_f1'] - baseline_metric['macro_f1'],
                      'batches': len(set(batch_ids.tolist()) - {-1}),
                      'by_event': {event: {'samples': sum(r['event'] == event for r in source['cohort']),
                                          'fixed': int(sum(f and r['event'] == event for f, r in zip(fixed, source['cohort']))),
                                          'broken': int(sum(f and r['event'] == event for f, r in zip(broken, source['cohort'])))}
                                   for event in sorted({r['event'] for r in source['cohort']})}}
            reports.append(report)
            seed_dir = output / f'grouping_seed_{seed}'
            seed_dir.mkdir()
            with (seed_dir / 'predictions.tsv').open('x', newline='', encoding='utf-8') as handle:
                writer = csv.writer(handle, delimiter='\t')
                writer.writerow(['id', 'y_true', 'baseline_pred', 'y_pred', 'batch_id', 'gate_informative'])
                writer.writerows((r['id'], r['y_true'], old, new, int(batch), bool(g)) for r, old, new, batch, g in
                                 zip(source['rows'], baseline, predicted, batch_ids, gate))
            write_json(seed_dir / 'diagnostics.json', diagnostics)
            write_json(seed_dir / 'metrics.json', report)
            print(f'[RESULT] seed={seed}; accuracy={report["accuracy"]:.6f}; macro_f1={report["macro_f1"]:.6f}', flush=True)
        summary = {'samples': len(truth), 'gate_positive': int(gate.sum()), 'label_order': labels,
                   'baseline_two_stage': baseline_metric, 'softmax_preserves_baseline': True,
                   'transductive_by_seed': reports,
                   'transductive_mean': {key: float(np.mean([r[key] for r in reports])) for key in ['accuracy', 'macro_f1']},
                   'transductive_std_population': {key: float(np.std([r[key] for r in reports])) for key in ['accuracy', 'macro_f1']},
                   'limitations': ['Không chọn seed tốt nhất; trung bình metric, không ensemble dự đoán.',
                                   'Đây là thay đổi sang suy luận transductive, dùng thông tin từ nhiều query.',
                                   'Dev đã được dùng chọn phương pháp trước đó; chưa đánh giá test.',
                                   'Ba grouping seed chỉ đo độ nhạy chia nhóm, không chứng minh ý nghĩa thống kê.',
                                   'Không báo CI bootstrap từng ảnh vì dự đoán trong cùng nhóm phụ thuộc nhau.',
                                   'Lớp hiếm và cụm giả có thể làm Hungarian gán nhãn sai; không ép cân bằng lớp.',
                                   'Gate không đổi nên không thể sửa false negative của tầng 1.']}
        write_json(output / 'analysis.json', summary)
        with (output / 'summary.tsv').open('x', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle, delimiter='\t')
            writer.writerow(['method', 'grouping_seed', 'samples', 'accuracy', 'macro_f1'])
            writer.writerow(['two_stage', '', len(truth), baseline_metric['accuracy'], baseline_metric['macro_f1']])
            for report in reports:
                writer.writerow(['transductive', report['grouping_seed'], len(truth), report['accuracy'], report['macro_f1']])
            writer.writerow(['transductive_mean', '', len(truth), summary['transductive_mean']['accuracy'], summary['transductive_mean']['macro_f1']])
        manifest['status'] = 'complete'
        manifest['elapsed_seconds'] = time.perf_counter() - started
        manifest['output_sha256'] = {str(p.relative_to(output)): sha256(p.read_bytes())
                                     for p in sorted(output.rglob('*')) if p.is_file() and p.name != 'run_manifest.json'}
        write_json(output / 'run_manifest.json', manifest)
        print(f'[SAVED] {output}', flush=True)
        return output
    except Exception as error:
        manifest.update(status='failed', error=f'{type(error).__name__}: {error}')
        write_json(output / 'run_manifest.json', manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT / 'experiments/clip_transductive/config.json'))
    parser.add_argument('--source-run', help='Artifact dev hai tầng image-only đã chạy xong')
    parser.add_argument('--output-dir', help='Thư mục mới, từ chối ghi đè')
    args = parser.parse_args()
    settings = json.loads(Path(args.config).read_text(encoding='utf-8'))
    if args.source_run:
        settings['source_run'] = args.source_run
    run(settings, args.output_dir)


if __name__ == '__main__':
    main()
