"""Transductive chỉ ở cổng 2 nhãn; tầng humanitarian giữ argmax độc lập."""
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
from experiments.clip_transductive.core import em_dirichlet, probability_features
from experiments.clip_transductive.run import load_source, metric, sha256, validate_config, write_json
from experiments.clip_two_stage.analyze import read_artifact
from experiments.clip_two_stage.run import decide
from experiments.waffle_lab.analyze import load_run


def predict_binary(scores, labels, grouping_seed=0, query_size=75, temperature=30.,
                   iterations=20, mm_iterations=1000):
    if len(labels) != 2 or set(labels) != {'informative', 'not_informative'}:
        raise ValueError('Cần đúng hai nhãn informativeness')
    if type(grouping_seed) is not int or not 0 <= grouping_seed < 2**32:
        raise ValueError('Grouping seed không hợp lệ')
    if type(query_size) is not int or query_size < 2:
        raise ValueError('query_size phải là số nguyên >= 2')
    features = probability_features(scores, temperature)
    if features.shape[1] != 2 or len(features) < 1:
        raise ValueError('Điểm phải có hai cột và ít nhất một ảnh')
    order = np.random.default_rng(grouping_seed).permutation(len(features))
    batches = [order[start:start + query_size] for start in range(0, len(order), query_size)]
    assigned = np.full(len(features), -1, dtype=int)
    batch_ids = np.full(len(features), -1, dtype=int)
    diagnostics = []
    for size in sorted({len(batch) for batch in batches}, reverse=True):
        ids = [i for i, batch in enumerate(batches) if len(batch) == size]
        indices = np.stack([batches[i] for i in ids])
        predicted, log, _ = em_dirichlet(features[indices], iterations, mm_iterations)
        assigned[indices] = predicted
        batch_ids[indices] = np.asarray(ids)[:, None]
        diagnostics.append({'batch_ids': ids, 'batch_size': size, **log})
    if np.any(assigned < 0) or np.any(batch_ids < 0):
        raise AssertionError('Bỏ sót ảnh khi suy luận cổng')
    return np.asarray(labels)[assigned].tolist(), batch_ids, diagnostics


def predict_two_stage_gate(info_scores, human_scores, info_labels, human_labels, **settings):
    # Tận dụng kiểm tra shape/taxonomy và quy tắc tie của baseline hiện tại.
    decide(info_scores, human_scores, info_labels, human_labels)
    binary, batches, diagnostics = predict_binary(info_scores, info_labels, **settings)
    gate = np.asarray(binary) == 'informative'
    positive = [i for i, label in enumerate(human_labels) if label != 'not_humanitarian']
    # Giữ y nguyên cách chọn lớp ở tầng 2; không transductive tầng này.
    predicted = np.asarray(human_labels)[np.asarray(positive)[np.asarray(human_scores)[:, positive].argmax(1)]]
    predicted[~gate] = 'not_humanitarian'
    return predicted.tolist(), binary, batches, diagnostics


def load_binary_source(directory):
    source = load_run(directory)
    manifest, plan = source['manifest'], source['plan']
    if manifest.get('task') != 'informativeness' or manifest.get('variant') != 'semantic' or manifest.get('label_fit') is not False:
        raise ValueError('Cần nguồn informativeness semantic đã hoàn tất, không fit nhãn')
    cohort = json.loads(read_artifact(directory, 'cohort.json'))['rows']
    if (plan['dataset_config'].get('source_split') != 'train' or
            any(r.get('source_split') != 'train' or r.get('label_source') != 'image' for r in cohort)):
        raise ValueError('Phần đo binary dùng train có nhãn ảnh, không tự dùng test')
    labels = plan['label_order']
    rows = source['rows']
    if len({r['id'] for r in rows}) != len(rows) or any(r['y_true'] not in labels for r in rows):
        raise ValueError('ID lặp hoặc nhãn không hợp lệ')
    scores = np.asarray([json.loads(r['cosine_scores']) for r in rows], dtype=float)
    probability_features(scores)
    if len(labels) != 2 or set(labels) != {'informative', 'not_informative'}:
        raise ValueError('Sai taxonomy binary')
    source.update(scores=scores, labels=labels, cohort=cohort,
                  artifact_sha256={name: sha256(read_artifact(directory, name).encode('utf-8'))
                                   for name in ['run_manifest.json', 'frozen_plan.json', 'cohort.json', 'predictions.tsv']})
    return source


def summarize(truth, baseline, predictions, labels):
    old_ok = np.asarray(baseline) == truth
    new_ok = np.asarray(predictions) == truth
    return {**metric(truth, predictions, labels), 'fixed': int((~old_ok & new_ok).sum()),
            'broken': int((old_ok & ~new_ok).sum()), 'changed': int((np.asarray(baseline) != predictions).sum())}


def run(settings, output_dir=None):
    if settings.get('protocol') != 'transductive_binary_gate_then_independent_humanitarian':
        raise ValueError('Sai protocol transductive gate')
    # Giữ cùng kiểm tra tham số EM/zero-shot với thử nghiệm đã đối chiếu tác giả.
    validate_config({**settings, 'protocol': 'fixed_image_gate_then_seven_class_transductive'})
    def path(name):
        result = Path(settings[name])
        return result if result.is_absolute() else ROOT / result
    binary = load_binary_source(path('binary_source_run'))
    human = load_source(path('humanitarian_source_run'))
    if binary['manifest']['model'] != human['manifest']['model']:
        raise ValueError('Hai nguồn phải cùng backbone')
    info_plan = json.loads(read_artifact(path('humanitarian_source_run'), 'informativeness/frozen_plan.json'))
    if (binary['labels'] != human['info_labels'] or
            binary['plan']['prompt_banks']['semantic'] != info_plan['prompt_banks']['semantic']):
        raise ValueError('Prompt hoặc thứ tự lớp informative khác nhau giữa hai nguồn')
    output = Path(output_dir) if output_dir else ROOT / 'results/analysis/clip_transductive_gate' / uuid.uuid4().hex[:12]
    output.mkdir(parents=True, exist_ok=False)
    manifest = {'status': 'running', 'evaluation_scope': 'train_binary_and_dev_humanitarian_exploratory',
                'settings': settings, 'model': human['manifest']['model'],
                'binary_samples': len(binary['rows']), 'humanitarian_samples': len(human['rows']),
                'source_artifact_sha256': {'binary_train': binary['artifact_sha256'], 'humanitarian_dev': human['artifact_sha256']},
                'label_fit': False, 'support_samples': 0, 'humanitarian_classes': len(human['labels']),
                'humanitarian_stage2': 'unchanged_independent_argmax_over_seven_positive_classes',
                'binary_dev_ground_truth_available': False, 'lambda_over_n': 0,
                'lambda_rule': 'floor(K/5), K=2; giữ công thức gốc, không tune lambda',
                'code_sha256': {name: sha256((ROOT / name).read_bytes()) for name in [
                    'experiments/clip_transductive/gate.py', 'experiments/clip_transductive/core.py',
                    'experiments/clip_transductive/run.py', 'experiments/waffle_lab/analyze.py',
                    'experiments/clip_two_stage/run.py', 'experiments/clip_two_stage/analyze.py']}}
    write_json(output / 'run_manifest.json', manifest)
    started = time.perf_counter()
    try:
        info_labels = binary['labels']; labels = human['labels']
        train_truth = np.asarray([r['y_true'] for r in binary['rows']])
        dev_truth = np.asarray([r['y_true'] for r in human['rows']])
        binary_baseline = np.asarray(info_labels)[binary['scores'].argmax(1)].tolist()
        human_baseline, old_gate = decide(*human['scores'], human['info_labels'], labels)
        summaries = {'binary_train': {'samples': len(train_truth), 'label_order': info_labels,
                                    'baseline': metric(train_truth, binary_baseline, info_labels), 'transductive_by_seed': []},
                     'humanitarian_dev': {'samples': len(dev_truth), 'label_order': labels,
                                          'baseline': metric(dev_truth, human_baseline, labels), 'transductive_by_seed': [],
                                          'informativeness_metric': None,
                                          'informativeness_metric_unavailable_reason': 'Không có nhãn informativeness thật trên cohort dev này.'}}
        for seed in settings['grouping_seeds']:
            kwargs = {k: settings[k] for k in ['query_size', 'temperature', 'iterations', 'mm_iterations']}
            print(f'[GATE] seed={seed}; binary_train={len(train_truth)}; humanitarian_dev={len(dev_truth)}', flush=True)
            binary_pred, train_batches, train_logs = predict_binary(binary['scores'], info_labels, grouping_seed=seed, **kwargs)
            human_pred, dev_info_pred, dev_batches, dev_logs = predict_two_stage_gate(
                *human['scores'], human['info_labels'], labels, grouping_seed=seed, **kwargs)
            reports = [summarize(train_truth, binary_baseline, binary_pred, info_labels),
                       summarize(dev_truth, human_baseline, human_pred, labels)]
            reports[1]['gate_informative'] = int(sum(p == 'informative' for p in dev_info_pred))
            reports[1]['baseline_gate_informative'] = int(old_gate.sum())
            for task, report, rows, truth, baseline, pred, info_pred, batches, logs in [
                    ('binary_train', reports[0], binary['rows'], train_truth, binary_baseline, binary_pred, binary_pred, train_batches, train_logs),
                    ('humanitarian_dev', reports[1], human['rows'], dev_truth, human_baseline, human_pred, dev_info_pred, dev_batches, dev_logs)]:
                report['grouping_seed'] = seed
                report['delta_macro_f1'] = report['macro_f1'] - summaries[task]['baseline']['macro_f1']
                summaries[task]['transductive_by_seed'].append(report)
                directory = output / task / f'grouping_seed_{seed}'; directory.mkdir(parents=True)
                with (directory / 'predictions.tsv').open('x', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f, delimiter='\t')
                    writer.writerow(['id', 'y_true', 'baseline_pred', 'y_pred', 'info_pred', 'batch_id'])
                    writer.writerows((r['id'], str(t), old, new, info, int(batch))
                                     for r, t, old, new, info, batch in zip(rows, truth, baseline, pred, info_pred, batches))
                write_json(directory / 'metrics.json', report)
                write_json(directory / 'diagnostics.json', logs)
                print(f'[RESULT] {task}; seed={seed}; accuracy={report["accuracy"]:.6f}; macro_f1={report["macro_f1"]:.6f}', flush=True)
        for summary in summaries.values():
            reports = summary['transductive_by_seed']
            summary['transductive_mean'] = {k: float(np.mean([r[k] for r in reports])) for k in ['accuracy', 'macro_f1']}
            summary['transductive_std_population'] = {k: float(np.std([r[k] for r in reports])) for k in ['accuracy', 'macro_f1']}
        summaries['limitations'] = ['Binary đo trên train đã dùng thăm dò; humanitarian đo trên dev đã xem nhiều lần.',
                                   'Không suy nhãn informative từ humanitarian để báo binary dev accuracy/F1.',
                                   'Tầng 2 giữ đủ tám nhãn đầu ra gốc; không kế thừa ablation bỏ lớp.',
                                   'Không chọn seed tốt nhất, không tune temperature/lambda/query size.',
                                   'Suy luận transductive phụ thuộc nhóm; chưa có kết quả test độc lập.']
        write_json(output / 'analysis.json', summaries)
        with (output / 'summary.tsv').open('x', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['task', 'method', 'grouping_seed', 'samples', 'accuracy', 'macro_f1'])
            for task in ['binary_train', 'humanitarian_dev']:
                s = summaries[task]
                writer.writerow([task, 'baseline', '', s['samples'], s['baseline']['accuracy'], s['baseline']['macro_f1']])
                for report in s['transductive_by_seed']:
                    writer.writerow([task, 'transductive_gate', report['grouping_seed'], s['samples'], report['accuracy'], report['macro_f1']])
                writer.writerow([task, 'transductive_gate_mean', '', s['samples'], s['transductive_mean']['accuracy'], s['transductive_mean']['macro_f1']])
        manifest.update(status='complete', elapsed_seconds=time.perf_counter() - started)
        manifest['output_sha256'] = {str(p.relative_to(output)): sha256(p.read_bytes())
                                     for p in output.rglob('*') if p.is_file() and p.name != 'run_manifest.json'}
        write_json(output / 'run_manifest.json', manifest)
        print(f'[SAVED] {output}', flush=True)
        return output
    except Exception as error:
        manifest.update(status='failed', error=f'{type(error).__name__}: {error}')
        write_json(output / 'run_manifest.json', manifest)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT / 'experiments/clip_transductive/config_gate.json'))
    parser.add_argument('--output-dir')
    args = parser.parse_args()
    run(json.loads(Path(args.config).read_text()), args.output_dir)
