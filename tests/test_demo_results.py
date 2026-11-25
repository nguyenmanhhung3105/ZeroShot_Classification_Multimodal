"""Verified experiment exports: numerical correctness, provenance and exclusion rules."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


def sha(file):
    return hashlib.sha256(file.read_bytes()).hexdigest()


def write_table(file, rows):
    with file.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter='\t')
        writer.writeheader(); writer.writerows(rows)


class DemoResultsTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('vlm_project.demo_results'),
                             'Missing verified experiment export for report charts')
        from vlm_project import demo_results
        self.module = demo_results

    def fixture(self, root):
        source = root / 'results/final/synthetic'; source.mkdir(parents=True)
        published = root / 'results/published'; published.mkdir(parents=True)
        predictions = []
        for seed in range(2):
            for i, (truth, pred) in enumerate([(0,0), (0,1), (1,1), (1,1)]):
                predictions.append({'task': '2way', 'method': 'clip_image', 'seed': seed,
                                    'id': str(i), 'truth': truth, 'predicted': pred})
        write_table(source / 'predictions.tsv', predictions)
        metrics = [{'task': '2way', 'method': 'clip_image', 'seed': seed, 'samples': 4,
                    'accuracy': .75, 'macro_f1': (2/3 + .8)/2, 'balanced_accuracy': .75} for seed in range(2)]
        write_table(source / 'metrics_by_seed.tsv', metrics)
        summary = [{'task': '2way', 'method': 'clip_image', 'samples': 4, 'seeds': 2, 'std_ddof': 1,
                    **{k + suffix: value for k, mean in [('accuracy', .75), ('macro_f1', (2/3+.8)/2), ('balanced_accuracy', .75)]
                       for suffix, value in [('_mean', mean), ('_std', 0.)]}}]
        write_table(source / 'summary.tsv', summary)
        run = {'kind': 'fakeddit_task1', 'mode': 'full', 'status': 'complete', 'model': 'metaclip2_vith14',
               'dataset': 'fakeddit_2way', 'split': 'dev', 'samples': 4, 'summary': summary,
               'cohort': {'cohort_sha256': 'synthetic-cohort'},
               'output_sha256': {p.name: sha(p) for p in source.iterdir()}}
        (source / 'run.json').write_text(json.dumps(run))
        (source / 'completed.json').write_text(json.dumps({'status': 'complete', 'run_sha256': sha(source / 'run.json')}))
        table = published / 'summary.tsv'; table.write_bytes((source / 'summary.tsv').read_bytes())
        audit = root / 'audit.json'; audit.write_text(json.dumps({'status': 'passed'}))
        entry = {'summary': str(table.relative_to(root)), 'summary_sha256': sha(table),
                 'source_run': str(source.relative_to(root)), 'source_run_sha256': sha(source / 'run.json'),
                 'kind': run['kind'], 'mode': 'full', 'samples': 4, 'audit_status': 'passed'}
        manifest = {'version': 1, 'exported_on': 'synthetic', 'audit': 'audit.json',
                    'audit_sha256': sha(audit), 'runs': [entry]}
        file = published / 'manifest.json'; file.write_text(json.dumps(manifest))
        return file, source, entry

    def test_export_recomputes_metrics_and_canonical_binary_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); manifest, _, _ = self.fixture(root)
            index = self.module.export(root, manifest, root / 'export')
            self.assertEqual(len(index['runs']), 1)
            detail = json.loads((root / 'export/experiments' / (index['runs'][0]['id'] + '.json')).read_text())
            row = detail['methods'][0]
            self.assertAlmostEqual(row['metrics']['macro_f1']['mean'], (2/3+.8)/2)
            self.assertEqual(row['labels'], ['Real', 'Fake'])
            self.assertEqual(row['confusion'], [[2., 0.], [1., 1.]])
            self.assertEqual(sum(map(sum, row['confusion'])), 4)
            self.assertEqual(row['seeds'], 2)
            self.assertNotIn('id', row['seed_metrics'][0])
            self.assertNotIn('predictions', detail)

    def test_tampered_artifact_and_summary_are_rejected(self):
        for name in ('predictions.tsv', 'summary.tsv'):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); manifest, source, _ = self.fixture(root)
                with (source / name).open('a') as stream: stream.write('tampered')
                with self.assertRaises(ValueError): self.module.export(root, manifest, root / 'export')

    def test_excludes_smoke_and_does_not_invent_loss_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); file, _, entry = self.fixture(root)
            manifest = json.loads(file.read_text()); manifest['runs'].append({**entry, 'mode': 'smoke'})
            file.write_text(json.dumps(manifest))
            index = self.module.export(root, file, root / 'export')
            self.assertEqual(len(index['runs']), 1)
            self.assertNotIn('loss', json.dumps(index))

    def test_composite_requires_same_cohort_model_and_samples(self):
        a = {'cohort': 'one', 'model': 'a', 'samples': 4, 'methods': []}
        for b in ({**a, 'cohort': 'two'}, {**a, 'model': 'b'}, {**a, 'samples': 5}):
            with self.assertRaises(ValueError): self.module.combine([a, b])

    def test_score_curves_handle_ties_without_inventing_training_history(self):
        from sklearn.metrics import average_precision_score, roc_auc_score
        values = [(0.9, True), (0.8, False), (0.8, True), (0.1, False)]
        result = self.module.probability_curve(values)
        self.assertAlmostEqual(result['auc'], roc_auc_score([p[1] for p in values], [p[0] for p in values]))
        self.assertAlmostEqual(result['average_precision'], average_precision_score([p[1] for p in values], [p[0] for p in values]))
        self.assertEqual(result['points'][-1]['tpr'], 1)
        self.assertEqual(result['points'][-1]['fpr'], 1)
        with self.assertRaises(ValueError): self.module.probability_curve([(0.1, True)])

    def test_committed_snapshots_have_valid_checksums_and_counts(self):
        root = Path(__file__).resolve().parents[1] / 'src/vlm_project/demo_web'
        index = json.loads((root / 'experiment-results.json').read_text())
        self.assertGreaterEqual(len(index['runs']), 3)
        for entry in index['runs']:
            file = root / entry['file']
            self.assertEqual(sha(file), entry['sha256'])
            run = json.loads(file.read_text())
            for row in run['methods']:
                for score in row['metrics'].values():
                    self.assertGreaterEqual(score['mean'], 0)
                    self.assertLessEqual(score['mean'], 1)
                if 'confusion' in row:
                    self.assertAlmostEqual(sum(map(sum, row['confusion'])), row['samples'])
        comparison = next((r for r in index['runs'] if r['kind'] == 'fakeddit_comparison'), None)
        self.assertIsNotNone(comparison)
        run = json.loads((root / comparison['file']).read_text())
        winner = next(r for r in run['methods'] if r['method'] == 'balanced_binary_image_title')
        self.assertAlmostEqual(winner['metrics']['accuracy']['mean'], .8240666666666666)
        self.assertEqual(len(run['sources']), 2)


if __name__ == '__main__': unittest.main()
