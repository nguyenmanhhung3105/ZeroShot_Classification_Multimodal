"""Kiểm tra quota được in trước upload/push; không gọi Kaggle thật."""
import contextlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from vlm_project import kaggle


class LauncherLimitsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.plan = {
            'status':'prepared', 'root':self.temp.name, 'job':'benchmark_imagenetv2',
            'task':'benchmark', 'name':'imagenetv2', 'mode':'full', 'samples':10000,
            'kernel':'nguyenhung3105/limits-example',
            'code_dataset':'nguyenhung3105/limits-code',
            'data_dataset':'nguyenhung3105/vlm-public-benchmarks-eval-v1',
            'settings':{'model':'metaclip2_vith14'},
            'launcher':{'accelerator':'NvidiaTeslaT4','timeout_seconds':14400},
        }

    def test_quota_prints_before_any_dataset_operation_or_gpu_push(self):
        output = io.StringIO()
        calls = []
        quota = 'GPU quota from test API: remaining=12.5h'
        def fake_cli(arguments, check=True):
            calls.append(arguments)
            if arguments[:2] == ['kernels','status']:
                return subprocess.CompletedProcess(arguments,1,'','404 not found')
            if arguments == ['quota']:
                return subprocess.CompletedProcess(arguments,0,quota,'')
            self.assertIn('[KAGGLE QUOTA]\n'+quota, output.getvalue())
            if arguments[:2] == ['kernels','push']:
                return subprocess.CompletedProcess(arguments,0,'Kernel version 2 successfully pushed.','')
            return subprocess.CompletedProcess(arguments,0,'{"status":"ready"}','')
        with patch.object(kaggle,'cli',side_effect=fake_cli), contextlib.redirect_stdout(output):
            plan = kaggle.submit(self.plan)
        self.assertEqual(calls[1],['quota'])
        self.assertEqual(plan['kernel_reference'],self.plan['kernel']+'/2')

    def test_quota_failure_stops_before_upload_or_push(self):
        calls = []
        def fake_cli(arguments, check=True):
            calls.append(arguments)
            if arguments == ['quota']:
                raise RuntimeError('Kaggle quota thất bại: Authentication required')
            return subprocess.CompletedProcess(arguments,1,'','404 not found')
        with patch.object(kaggle,'cli',side_effect=fake_cli):
            with self.assertRaisesRegex(RuntimeError,'quota.*Authentication required'):
                kaggle.submit(self.plan)
        self.assertEqual(calls,[['kernels','status',self.plan['kernel']],['quota']])
        self.assertEqual(self.plan['status'],'prepared')

    def test_single_word_cli_failure_preserves_actual_error(self):
        with patch.object(kaggle.subprocess,'run',return_value=subprocess.CompletedProcess(
                [],1,'','Authentication required to call the Kaggle API.')):
            with self.assertRaisesRegex(RuntimeError,'Kaggle quota thất bại:.*\nAuthentication required'):
                kaggle.cli(['quota'])

    def test_prepare_only_prints_limits_without_querying_quota(self):
        output = io.StringIO()
        with patch.object(kaggle,'prepare',return_value=self.plan),\
             patch.object(kaggle,'cli',side_effect=AssertionError('no API in prepare-only')),\
             contextlib.redirect_stdout(output):
            kaggle.main(['--job','benchmark_imagenetv2','--prepare-only'])
        shown = output.getvalue()
        self.assertIn('limit=10000 mẫu',shown)
        self.assertIn('metaclip2_vith14',shown)
        self.assertIn('timeout=14400s (4 giờ)',shown)
        self.assertIn('[KAGGLE QUOTA] Chưa truy vấn',shown)

    def test_smoke_reference_limits_use_reference_model_and_actual_sample_limit(self):
        self.plan.update(reference_model='openai_clip_vitl14',mode='smoke',samples=5)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):kaggle.print_run_limits(self.plan)
        self.assertIn('Model: openai_clip_vitl14',output.getvalue())
        self.assertIn('mode=smoke | limit=5 mẫu',output.getvalue())


if __name__ == '__main__':unittest.main()
