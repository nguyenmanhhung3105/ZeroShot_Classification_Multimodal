"""Đối chứng bỏ một lớp: cùng ảnh, cùng score, tái suy luận cả hai phương pháp."""
import argparse
import csv
import json
from pathlib import Path
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import numpy as np
from experiments.clip_transductive.core import predict_two_stage
from experiments.clip_transductive.run import load_source, metric, sha256, validate_config, write_json
from experiments.clip_two_stage.run import decide


def restrict_task(source, excluded_label):
    """Loại mẫu theo nhãn thật để định nghĩa task mới, không theo dự đoán đúng/sai."""
    labels = source['labels']
    if excluded_label not in labels or excluded_label == 'not_humanitarian':
        raise ValueError('Chỉ bỏ một lớp humanitarian dương đang có')
    keep_rows = np.asarray([r['y_true'] != excluded_label for r in source['rows']])
    if not keep_rows.any():
        raise ValueError('Không còn mẫu để đánh giá')
    keep_columns = [i for i, label in enumerate(labels) if label != excluded_label]
    return {'labels': [labels[i] for i in keep_columns], 'info_labels': source['info_labels'],
            'scores': [source['scores'][0][keep_rows], source['scores'][1][np.ix_(keep_rows, keep_columns)]],
            'rows': [r for r, keep in zip(source['rows'], keep_rows) if keep],
            'cohort': [r for r, keep in zip(source['cohort'], keep_rows) if keep],
            'source_row_indices': np.flatnonzero(keep_rows).tolist(),
            'removed': [r for r, keep in zip(source['rows'], keep_rows) if not keep]}


def run(settings, excluded_label='missing_or_found_people', output_dir=None):
    validate_config(settings)
    source_dir = Path(settings['source_run'])
    if not source_dir.is_absolute(): source_dir = ROOT / source_dir
    source = load_source(source_dir)
    subset = restrict_task(source, excluded_label)
    output = Path(output_dir) if output_dir else ROOT / 'results/analysis/clip_transductive_drop_class' / uuid.uuid4().hex[:12]
    output.mkdir(parents=True, exist_ok=False)
    manifest = {'status': 'running', 'evaluation_scope': 'dev_exploratory_reduced_taxonomy',
                'settings': settings, 'excluded_label': excluded_label,
                'effective_protocol': 'fixed_image_gate_then_remaining_positive_classes',
                'source_run': str(source_dir), 'source_artifact_sha256': source['artifact_sha256'],
                'model': source['manifest']['model'], 'prompt_seed': source['manifest']['seed'],
                'original_label_order': source['labels'], 'label_order': subset['labels'],
                'original_samples': len(source['rows']), 'samples': len(subset['rows']),
                'removed_samples': len(subset['removed']),
                'subset_rule': 'exclude_true_label_rows_and_exclude_same_candidate_label_for_both_methods',
                'prompts': 'reuse_original_eight_class_bank_scores_then_remove_one_column',
                'gate': 'original_image_gate_unchanged_on_retained_rows',
                'label_fit': False, 'support_samples': 0,
                'ground_truth_used_for': 'task_subset_definition_artifact_validation_and_metrics_only',
                'code_sha256': {name: sha256((ROOT / name).read_bytes()) for name in [
                    'experiments/clip_transductive/drop_class.py', 'experiments/clip_transductive/core.py',
                    'experiments/clip_transductive/run.py', 'experiments/clip_two_stage/run.py',
                    'experiments/clip_two_stage/analyze.py']}}
    write_json(output / 'run_manifest.json', manifest)
    started = time.perf_counter()
    try:
        write_json(output / 'subset.json', {'source_row_indices': subset['source_row_indices'],
                                          'rows': subset['cohort'],
                                          'removed': [{'id': r['id'], 'y_true': r['y_true']} for r in subset['removed']]})
        a, b = subset['scores']
        labels, info_labels = subset['labels'], subset['info_labels']
        truth = np.asarray([r['y_true'] for r in subset['rows']])
        baseline, gate = decide(a, b, info_labels, labels)
        original_baseline, original_gate = decide(*source['scores'], source['info_labels'], source['labels'])
        np.testing.assert_array_equal(gate, original_gate[subset['source_row_indices']])
        baseline_metric = metric(truth, baseline, labels)
        reports = []
        for seed in settings['grouping_seeds']:
            print(f'[DROP CLASS] label={excluded_label}; samples={len(truth)}; grouping_seed={seed}', flush=True)
            predicted, batch_ids, diagnostics = predict_two_stage(
                a, b, info_labels, labels, grouping_seed=seed,
                **{k: settings[k] for k in ['query_size', 'temperature', 'iterations', 'mm_iterations']})
            if excluded_label in predicted or excluded_label in baseline:
                raise AssertionError('Dự đoán vẫn còn lớp đã bỏ')
            if not set(predicted) <= set(labels): raise AssertionError('Dự đoán ngoài taxonomy mới')
            if np.any(batch_ids[gate] < 0) or np.any(batch_ids[~gate] != -1):
                raise AssertionError('Sai thành viên nhóm')
            new_ok = np.asarray(predicted) == truth
            old_ok = np.asarray(baseline) == truth
            report = {'grouping_seed': seed, **metric(truth, predicted, labels),
                      'fixed': int((~old_ok & new_ok).sum()), 'broken': int((old_ok & ~new_ok).sum()),
                      'delta_macro_f1': metric(truth, predicted, labels)['macro_f1'] - baseline_metric['macro_f1']}
            reports.append(report)
            directory = output / f'grouping_seed_{seed}'; directory.mkdir()
            with (directory / 'predictions.tsv').open('x', encoding='utf-8', newline='') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerow(['id', 'y_true', 'baseline_pred', 'y_pred', 'batch_id', 'gate_informative'])
                writer.writerows((r['id'], r['y_true'], old, new, int(batch), bool(g))
                                 for r, old, new, batch, g in zip(subset['rows'], baseline, predicted, batch_ids, gate))
            write_json(directory / 'metrics.json', report)
            write_json(directory / 'diagnostics.json', diagnostics)
            print(f'[RESULT] seed={seed}; accuracy={report["accuracy"]:.6f}; macro_f1={report["macro_f1"]:.6f}', flush=True)
        summary = {'excluded_label': excluded_label, 'samples': len(truth), 'removed_samples': len(subset['removed']),
                   'label_order': labels, 'gate_positive': int(gate.sum()), 'baseline_two_stage': baseline_metric,
                   'transductive_by_seed': reports,
                   'transductive_mean': {k: float(np.mean([r[k] for r in reports])) for k in ['accuracy', 'macro_f1']},
                   'transductive_std_population': {k: float(np.std([r[k] for r in reports])) for k in ['accuracy', 'macro_f1']},
                   'original_eight_class_baseline_context_only': metric(
                       [r['y_true'] for r in source['rows']], original_baseline, source['labels']),
                   'limitations': ['Bài toán ít lớp hơn, không thay thế kết quả tám lớp gốc.',
                                   'Macro-F1 mới trung bình trên bảy lớp; không so như cùng thước đo tám lớp.',
                                   'Lọc theo ground truth chỉ để định nghĩa tập đánh giá; chưa giải quyết ảnh lớp bị bỏ khi triển khai.',
                                   'Tái chạy transductive từ score còn lại, không chỉ xóa một dòng metric cũ.',
                                   'Cùng backbone, prompt gốc, ảnh và gate giữa hai phương pháp; không sinh lại Waffle.',
                                   'Dev thăm dò đã được xem nhiều lần, chưa phải test độc lập.',
                                   'Trung bình cả ba grouping seed, không chọn seed tốt nhất.']}
        write_json(output / 'analysis.json', summary)
        with (output / 'summary.tsv').open('x', encoding='utf-8', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['method', 'grouping_seed', 'samples', 'classes', 'accuracy', 'macro_f1'])
            writer.writerow(['two_stage', '', len(truth), len(labels), baseline_metric['accuracy'], baseline_metric['macro_f1']])
            for report in reports:
                writer.writerow(['transductive', report['grouping_seed'], len(truth), len(labels), report['accuracy'], report['macro_f1']])
            writer.writerow(['transductive_mean', '', len(truth), len(labels), summary['transductive_mean']['accuracy'], summary['transductive_mean']['macro_f1']])
        manifest.update(status='complete', elapsed_seconds=time.perf_counter() - started)
        manifest['output_sha256'] = {str(p.relative_to(output)): sha256(p.read_bytes())
                                     for p in output.rglob('*') if p.is_file() and p.name != 'run_manifest.json'}
        write_json(output / 'run_manifest.json', manifest)
        print(f'[BASELINE] accuracy={baseline_metric["accuracy"]:.6f}; macro_f1={baseline_metric["macro_f1"]:.6f}', flush=True)
        print(f'[SAVED] {output}', flush=True)
        return output
    except Exception as error:
        manifest.update(status='failed', error=f'{type(error).__name__}: {error}')
        write_json(output / 'run_manifest.json', manifest)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT / 'experiments/clip_transductive/config.json'))
    parser.add_argument('--source-run')
    parser.add_argument('--drop-label', default='missing_or_found_people')
    parser.add_argument('--output-dir')
    args = parser.parse_args()
    settings = json.loads(Path(args.config).read_text())
    if args.source_run: settings['source_run'] = args.source_run
    run(settings, args.drop_label, args.output_dir)
