"""Bảo vệ informative có margin lớn; ngưỡng cố định từ train không dùng nhãn."""
import argparse
import csv
import json
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import numpy as np
from experiments.clip_transductive.core import probability_features
from experiments.clip_transductive.gate import load_binary_source, summarize
from experiments.clip_transductive.run import load_source, metric, sha256, write_json
from experiments.clip_two_stage.run import decide


def margin(scores, labels):
    scores = np.asarray(scores, dtype=float)
    if len(labels) != 2 or set(labels) != {'informative', 'not_informative'} or scores.ndim != 2 or scores.shape[1] != 2:
        raise ValueError('Cần score và taxonomy informativeness hai lớp')
    probability_features(scores)
    return scores[:, labels.index('informative')] - scores[:, labels.index('not_informative')]


def fit_label_free_threshold(scores, labels):
    differences = margin(scores, labels)
    positive = differences[differences > 0]
    if not len(positive): raise ValueError('Không có margin dương để xác định ngưỡng')
    return {'threshold': float(np.median(positive)), 'samples': len(differences),
            'positive_margins': len(positive), 'rule': 'median_strictly_positive_train_cosine_margin',
            'uses_ground_truth': False}


def apply_guard(scores, labels, transductive_predictions, threshold):
    differences = margin(scores, labels)
    predictions = np.asarray(transductive_predictions, dtype=object)
    if predictions.shape != differences.shape or not set(predictions) <= set(labels):
        raise ValueError('Sai nhãn transductive hoặc số lượng ảnh')
    if not np.isfinite(threshold) or threshold <= 0: raise ValueError('Ngưỡng phải dương hữu hạn')
    protected = differences >= threshold
    restored = protected & (predictions == 'not_informative')
    result = predictions.copy(); result[protected] = 'informative'
    return result.tolist(), restored


def run(settings, output_dir=None):
    if (settings.get('method') != 'confidence_guarded_transductive_gate' or settings.get('label_fit') is not False or
            settings.get('threshold_rule') != 'median_strictly_positive_train_cosine_margin'):
        raise ValueError('Chỉ thử quy tắc median không nhãn đã khóa')
    gate_dir = Path(settings['gate_run'])
    if not gate_dir.is_absolute(): gate_dir = ROOT / gate_dir
    previous = json.loads((gate_dir / 'run_manifest.json').read_text())
    if previous.get('status') != 'complete' or previous['settings']['protocol'] != 'transductive_binary_gate_then_independent_humanitarian':
        raise ValueError('Cần lượt transductive gate hoàn tất')
    if settings.get('grouping_seeds') != previous['settings']['grouping_seeds']:
        raise ValueError('Giữ nguyên mọi grouping seed của đối chứng')
    def path(name):
        value = Path(previous['settings'][name]); return value if value.is_absolute() else ROOT / value
    train = load_binary_source(path('binary_source_run'))
    threshold_plan = fit_label_free_threshold(train['scores'], train['labels'])
    output = Path(output_dir) if output_dir else ROOT / 'results/analysis/clip_transductive_guard' / uuid.uuid4().hex[:12]
    output.mkdir(parents=True, exist_ok=False)
    # Khóa ngưỡng trước khi nạp cohort/ground truth dev hoặc tính metric.
    write_json(output / 'frozen_guard_plan.json', {'settings': settings, **threshold_plan,
                                                'train_source_sha256': train['artifact_sha256']})
    manifest = {'status': 'running', 'settings': settings, 'label_fit': False,
                'uses_unlabeled_train_statistics': True, 'threshold': threshold_plan['threshold'],
                'source_gate_manifest_sha256': sha256((gate_dir / 'run_manifest.json').read_bytes()),
                'code_sha256': {name: sha256((ROOT / name).read_bytes()) for name in [
                    'experiments/clip_transductive/confidence_guard.py', 'experiments/clip_transductive/gate.py',
                    'experiments/clip_transductive/run.py', 'experiments/clip_two_stage/run.py']}}
    write_json(output / 'run_manifest.json', manifest)
    try:
        for name, expected in previous['output_sha256'].items():
            if sha256((gate_dir / name).read_bytes()) != expected: raise ValueError('Output đối chứng bị thay đổi')
        human = load_source(path('humanitarian_source_run'))
        if (train['artifact_sha256'] != previous['source_artifact_sha256']['binary_train'] or
                human['artifact_sha256'] != previous['source_artifact_sha256']['humanitarian_dev']):
            raise ValueError('Artifact nguồn khác lượt gate gốc')
        results = {}
        for task, source, scores, binary_labels, final_labels in [
                ('binary_train', train, train['scores'], train['labels'], train['labels']),
                ('humanitarian_dev', human, human['scores'][0], human['info_labels'], human['labels'])]:
            truth = np.asarray([r['y_true'] for r in source['rows']])
            baseline = (np.asarray(binary_labels)[scores.argmax(1)].tolist() if task == 'binary_train'
                        else decide(*source['scores'], binary_labels, final_labels)[0])
            result = {'samples': len(truth), 'baseline': metric(truth, baseline, final_labels), 'by_seed': []}
            for seed in settings['grouping_seeds']:
                source_file = gate_dir / task / f'grouping_seed_{seed}/predictions.tsv'
                with source_file.open() as f: rows = list(csv.DictReader(f, delimiter='\t'))
                if [(r['id'], r['y_true']) for r in rows] != [(r['id'], r['y_true']) for r in source['rows']]:
                    raise ValueError('Khác ID/nhãn/thứ tự của đối chứng')
                if [r['baseline_pred'] for r in rows] != baseline: raise ValueError('Baseline không khớp score')
                trans_binary = [r['info_pred'] for r in rows]
                guarded_binary, restored = apply_guard(scores, binary_labels, trans_binary, threshold_plan['threshold'])
                trans_final = [r['y_pred'] for r in rows]
                if task == 'binary_train':
                    if trans_final != trans_binary: raise ValueError('Binary prediction khác info_pred')
                    guarded = guarded_binary
                else:
                    positive = [i for i, label in enumerate(final_labels) if label != 'not_humanitarian']
                    candidate = np.asarray(final_labels)[np.asarray(positive)[source['scores'][1][:, positive].argmax(1)]]
                    original = candidate.copy(); original[np.asarray(trans_binary) != 'informative'] = 'not_humanitarian'
                    if original.tolist() != trans_final: raise ValueError('Tầng 2 đối chứng không khớp')
                    candidate[np.asarray(guarded_binary) != 'informative'] = 'not_humanitarian'
                    guarded = candidate.tolist()
                report = {'grouping_seed': seed, 'transductive': metric(truth, trans_final, final_labels),
                          'guarded': summarize(truth, baseline, guarded, final_labels),
                          'restored_informative': int(restored.sum())}
                result['by_seed'].append(report)
                directory = output / task / f'grouping_seed_{seed}'; directory.mkdir(parents=True)
                with (directory / 'predictions.tsv').open('x', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f, delimiter='\t')
                    writer.writerow(['id', 'y_true', 'baseline_pred', 'transductive_pred', 'y_pred', 'info_pred', 'restored_informative'])
                    writer.writerows((r['id'], r['y_true'], old, trans, new, info, bool(flag)) for r, old, trans, new, info, flag in
                                     zip(source['rows'], baseline, trans_final, guarded, guarded_binary, restored))
                write_json(directory / 'metrics.json', report)
                print(f'[GUARD] {task}; seed={seed}; restored={int(restored.sum())}; macro_f1={report["guarded"]["macro_f1"]:.6f}', flush=True)
            for method in ['transductive', 'guarded']:
                result[method + '_mean'] = {k: float(np.mean([r[method][k] for r in result['by_seed']])) for k in ['accuracy', 'macro_f1']}
            results[task] = result
        results['limitations'] = ['Ngưỡng là thống kê train không nhãn; đây là thích nghi có dùng dữ liệu tham chiếu không nhãn.',
                                  'Guard là giả thuyết của dự án, không phải implementation StatA/TDA/DPE.',
                                  'Chỉ một quy tắc median đã khóa; không chọn ngưỡng bằng kết quả dev.',
                                  'Dev đã xem nhiều lần, chưa có test độc lập; nhãn informativeness dev không có.']
        write_json(output / 'analysis.json', results)
        manifest['status'] = 'complete'
        manifest['output_sha256'] = {str(p.relative_to(output)): sha256(p.read_bytes()) for p in output.rglob('*')
                                     if p.is_file() and p.name != 'run_manifest.json'}
        write_json(output / 'run_manifest.json', manifest)
        print(f'[SAVED] {output}', flush=True)
        return output
    except Exception as error:
        manifest.update(status='failed', error=f'{type(error).__name__}: {error}')
        write_json(output / 'run_manifest.json', manifest)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT / 'experiments/clip_transductive/config_guard.json'))
    parser.add_argument('--output-dir')
    args = parser.parse_args()
    run(json.loads(Path(args.config).read_text()), args.output_dir)
