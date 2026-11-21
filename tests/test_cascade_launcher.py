"""Kaggle cascade chỉ chuẩn bị offline, không gửi API hoặc gắn test."""
import contextlib
import copy
import io as stdio
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from vlm_project import io, kaggle


class CascadeLauncherTests(unittest.TestCase):
    def test_prepare_smoke_and_full_one_run_two_min5_inputs_no_test(self):
        launcher = {'experiment_config': 'configs/project_min5.yaml', 'kaggle': {
            'username': 'nguyenhung3105', 'code_dataset_prefix': 'vlm-final-code', 'kernel_prefix': 'vlm-min5',
            'accelerator': 'NvidiaTeslaT4', 'timeout_seconds': 7200, 'poll_seconds': 120},
            'jobs': {'cascade': {'kind': 'fakeddit_fewshot_cascade', 'split': 'dev'}}}
        with tempfile.TemporaryDirectory() as temp:
            real_path = kaggle.path
            with patch.object(kaggle, 'path', side_effect=lambda value: Path(temp) / 'stage' if str(value) == 'results/_system/launches/final' else real_path(value)), \
                 patch.object(kaggle, 'cli', side_effect=AssertionError('no network')), \
                 contextlib.redirect_stdout(stdio.StringIO()):
                for limit in [5, None]:
                    plan = kaggle.prepare(launcher, 'cascade', limit)
                    display = stdio.StringIO()
                    with contextlib.redirect_stdout(display): kaggle.print_run_limits(plan)
                    self.assertIn('hai tầng', display.getvalue())
                    self.assertNotIn('ba modality', display.getvalue())
                    root = Path(plan['root'])
                    doc = io.read_json(root / 'kernel/run.ipynb')
                    code = '\n'.join(''.join(c['source']) for c in doc['cells'] if c['cell_type'] == 'code')
                    self.assertIn('fakeddit_fewshot_cascade import run', code)
                    self.assertEqual(code.count('output = run('), 1)
                    self.assertNotIn('freeze(settings', code)
                    self.assertIn('fakeddit-fewshot-cascade-min5', plan['kernel'])
                    self.assertEqual(plan['samples'], 5 if limit else 3000)
                    meta = io.read_json(root / 'kernel/kernel-metadata.json')
                    self.assertEqual(set(meta['dataset_sources']), {plan['code_dataset'],
                        'nguyenhung3105/vlm-fakeddit-train-min5-v1', 'nguyenhung3105/vlm-fakeddit-dev-min5-v1'})
                    self.assertFalse(any('test' in d for d in meta['dataset_sources']))
                    for c in doc['cells']:
                        if c['cell_type'] == 'code': compile(''.join(c['source']), '<cascade notebook>', 'exec')
                for split in ['train', 'test']:
                    invalid = copy.deepcopy(launcher); invalid['jobs']['cascade']['split'] = split
                    with self.assertRaisesRegex(ValueError, 'chỉ đánh giá dev'):
                        kaggle.prepare(invalid, 'cascade')


if __name__ == '__main__': unittest.main()
