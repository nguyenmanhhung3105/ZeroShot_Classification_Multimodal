import contextlib
import io
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from vlm_project import kaggle


class CifarLauncherTests(unittest.TestCase):
    def test_prepare_and_smoke_use_same_launcher_without_api(self):
        import yaml
        launcher = yaml.safe_load((ROOT / 'configs/kaggle_launcher.yaml').read_text())
        with tempfile.TemporaryDirectory() as tmp:
            real_path = kaggle.path
            def local_path(value):
                return Path(tmp) if str(value) == 'results/_system/launches/final' else real_path(value)
            with patch.object(kaggle, 'path', side_effect=local_path), \
                    patch.object(kaggle, 'cli', side_effect=AssertionError('No API during prepare')):
                plan = kaggle.prepare(launcher, 'cifar100_two_stage_dev')
                smoke = kaggle.prepare(launcher, 'cifar100_two_stage_dev', 5)
            self.assertEqual(plan['samples'], 10000)
            self.assertEqual(plan['launcher']['poll_seconds'], 120)
            self.assertTrue(plan['inline_code'])
            self.assertIsNone(plan['code_dataset'])
            self.assertEqual(smoke['samples'], 5)
            self.assertNotEqual(plan['kernel'], smoke['kernel'])
            notebook = json.loads((Path(plan['root']) / 'kernel/run.ipynb').read_text())
            self.assertIn('metrics_', ''.join(notebook['cells'][-1]['source']))
            with self.assertRaises(ValueError):
                kaggle.prepare(launcher, 'cifar100_two_stage_dev', 256)

    def test_quota_before_push_no_code_upload_version_saved(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = {'root': tmp, 'status': 'prepared', 'kernel': 'owner/cifar',
                    'task': 'cifar100_two_stage', 'inline_code': True,
                    'data_dataset': 'owner/source', 'code_dataset': None,
                    'launcher': {'accelerator': 'NvidiaTeslaT4', 'timeout_seconds': 14400}}
            calls = []
            def cli(args, check=True):
                calls.append(args)
                if args[:2] == ['kernels', 'status']:
                    return subprocess.CompletedProcess(args, 1, '', '404 not found')
                if args == ['quota']:
                    return subprocess.CompletedProcess(args, 0, 'quota fixture', '')
                if args[:2] == ['datasets', 'status']:
                    return subprocess.CompletedProcess(args, 0, 'ready', '')
                if args[:2] == ['kernels', 'push']:
                    return subprocess.CompletedProcess(args, 0, 'Kernel version 3 successfully pushed.', '')
                raise AssertionError(args)
            with patch.object(kaggle, 'cli', side_effect=cli), contextlib.redirect_stdout(io.StringIO()):
                result = kaggle.submit(plan)
            self.assertEqual([c[:2] for c in calls], [['kernels', 'status'], ['quota'], ['datasets', 'status'], ['kernels', 'push']])
            self.assertEqual(result['kernel_reference'], 'owner/cifar/3')
            self.assertEqual(kaggle.read_json(Path(tmp) / 'launch.json')['status'], 'submitted')

    def test_running_cifar_does_not_query_quota_or_push(self):
        plan = {'kernel': 'owner/cifar'}
        with patch.object(kaggle, 'cli', return_value=subprocess.CompletedProcess([], 0, 'running', '')) as cli:
            with self.assertRaisesRegex(RuntimeError, 'đang chạy'):
                kaggle.submit(plan)
        self.assertEqual(cli.call_count, 1)

    def test_wait_collects_zip_checksums_and_prints_cifar_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            stage = folder / 'stage'
            stage.mkdir()
            plan = {'root': str(stage), 'status': 'submitted', 'task': 'cifar100_two_stage',
                    'name': 'cifar100/dev', 'mode': 'full', 'kernel': 'owner/cifar',
                    'kernel_reference': 'owner/cifar/3',
                    'launcher': {'poll_seconds': 120, 'timeout_seconds': 14400}}
            summary = b'method\nflat\n'
            run = {'status': 'complete', 'model': 'metaclip2_vith14', 'dataset': 'cifar100',
                   'mode': 'full', 'samples': 10000,
                   'output_sha256': {'summary.tsv': hashlib.sha256(summary).hexdigest()},
                   'summary': [{'task': 'cifar100', 'method': 'flat', 'accuracy_mean': .5,
                                'accuracy_std': 0, 'macro_f1_mean': .4, 'macro_f1_std': 0, 'seeds': 1}]}
            run_bytes = json.dumps(run).encode()
            completion = {'status': 'complete', 'run_sha256': hashlib.sha256(run_bytes).hexdigest()}
            calls = []
            def read(args, *_args, **_kwargs):
                calls.append(args)
                if args[1] == 'status':
                    return subprocess.CompletedProcess(args, 0, 'complete', '')
                destination = Path(args[args.index('-p')+1])
                with zipfile.ZipFile(destination / 'metrics_fixture.zip', 'w') as archive:
                    archive.writestr('run/summary.tsv', summary)
                    archive.writestr('run/run.json', run_bytes)
                    archive.writestr('run/completed.json', json.dumps(completion))
                return subprocess.CompletedProcess(args, 0, 'downloaded', '')
            real_path = kaggle.path
            def local_path(value):
                return folder / 'collected' if str(value) == 'results/final/kaggle' else real_path(value)
            with patch.object(kaggle, 'path', side_effect=local_path), \
                    patch.object(kaggle, 'read_submitted_job', side_effect=read), \
                    contextlib.redirect_stdout(io.StringIO()) as shown:
                result = kaggle.wait_and_collect(plan)
            self.assertEqual(result['status'], 'collected')
            self.assertTrue((Path(result['output']) / 'run/completed.json').exists())
            self.assertTrue(all(args[2] == 'owner/cifar/3' for args in calls))
            self.assertIn('cifar100 | full | 10000', shown.getvalue())


if __name__ == '__main__':
    unittest.main()
