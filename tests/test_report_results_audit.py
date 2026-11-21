"""Audit phát hiện sai số metric ngay cả khi artifact được rehash hợp lệ."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from vlm_project.evaluate import metric, publish

spec = importlib.util.spec_from_file_location('report_audit', ROOT / 'scripts/audit_report_results.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class ReportAuditTests(unittest.TestCase):
    def fixture(self, directory, *, incorrect_metric=False):
        truth, predicted = [0, 1, 0, 1], [0, 1, 1, 1]
        values = metric(truth, predicted, [0, 1])
        details = values.pop('per_class')
        if incorrect_metric:
            values['macro_f1'] = 1.
        record = {'task': '2way', 'method': 'image', 'seed': None, 'samples': 4, **values}
        # Old deterministic CIFAR exports omit seed in predictions.
        predictions = [{'task': '2way', 'method': 'image', 'id': str(i), 'truth': y, 'predicted': p}
                       for i, (y, p) in enumerate(zip(truth, predicted))]
        publish(directory, [record], predictions, details,
                {'kind': 'synthetic', 'mode': 'smoke', 'samples': 4})

    def test_seedless_predictions_and_recomputed_metrics(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            directory = Path(folder)
            self.fixture(directory)
            result = audit.audit(directory)
            self.assertEqual(result['status'], 'passed')
            self.assertEqual(result['prediction_rows'], 4)

    def test_bad_metric_detected_with_valid_checksums(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            directory = Path(folder)
            self.fixture(directory, incorrect_metric=True)
            with self.assertRaisesRegex(ValueError, 'macro_f1'):
                audit.audit(directory)

    def test_changed_predictions_rejected_by_checksum(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            directory = Path(folder)
            self.fixture(directory)
            with (directory / 'predictions.tsv').open('a') as stream:
                stream.write('changed\n')
            with self.assertRaisesRegex(ValueError, 'Artifact'):
                audit.audit(directory)
