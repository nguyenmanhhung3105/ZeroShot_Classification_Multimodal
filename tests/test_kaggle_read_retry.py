"""Mô phỏng lỗi đọc Kaggle sau push; tuyệt đối không gửi job thật."""
import contextlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from vlm_project import kaggle


class SubmittedJobReadTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.plan={'status':'submitted','root':str(self.root),
                   'kernel':'nguyenhung3105/example',
                   'kernel_reference':'nguyenhung3105/example/1',
                   'launcher':{'poll_seconds':20,'timeout_seconds':600}}

    def test_status_403_retries_same_version_without_resubmission(self):
        denied=subprocess.CompletedProcess([],1,'',"Permission 'kernels.get' was denied")
        authenticated=subprocess.CompletedProcess([],0,'[]','')
        running=subprocess.CompletedProcess([],0,'running','')
        arguments=['kernels','status',self.plan['kernel_reference']]
        with patch.object(kaggle,'cli',side_effect=[denied,authenticated,running]) as cli,\
             patch.object(kaggle.time,'sleep') as sleep,contextlib.redirect_stdout(io.StringIO()):
            result=kaggle.read_submitted_job(arguments,self.plan)
        self.assertEqual(result.stdout,'running')
        self.assertEqual(sleep.call_count,1)
        status_calls=[call for call in cli.call_args_list if call.args[0][:2]==['kernels','status']]
        self.assertEqual(len(status_calls),2)
        self.assertTrue(all(call.args[0]==arguments for call in status_calls))
        self.assertTrue(all(call.kwargs=={'check':False} for call in cli.call_args_list))

    def test_hidden_auth_failure_is_identified_from_owned_list_without_new_job(self):
        denied=subprocess.CompletedProcess([],1,'',"Permission 'kernels.get' was denied")
        auth=subprocess.CompletedProcess([],1,'','Authentication required to call the Kaggle API.')
        with patch.object(kaggle,'cli',side_effect=[denied,auth]) as cli,patch.object(kaggle.time,'sleep') as sleep:
            with self.assertRaisesRegex(RuntimeError,'auth login'):
                kaggle.read_submitted_job(['kernels','status',self.plan['kernel_reference']],self.plan)
        self.assertEqual(cli.call_count,2);sleep.assert_not_called()
        self.assertTrue(all(call.args[0][1] in ['status','list'] for call in cli.call_args_list))

    def test_persistent_permission_failure_keeps_receipt_and_resume_command(self):
        denied=subprocess.CompletedProcess([],1,'','403 Forbidden')
        with patch.object(kaggle,'cli',return_value=denied),\
             patch.object(kaggle.time,'monotonic',side_effect=[0,301]):
            with self.assertRaisesRegex(RuntimeError,'--resume-stage'):
                kaggle.read_submitted_job(['kernels','status',self.plan['kernel_reference']],self.plan)
        saved=kaggle.read_json(self.root/'launch.json')
        self.assertEqual(saved['status'],'submitted')
        self.assertEqual(saved['kernel_reference'],self.plan['kernel_reference'])
        self.assertIn('403',saved['last_wait_error'])

    def test_authentication_failure_returns_resume_instructions_immediately(self):
        denied=subprocess.CompletedProcess([],1,'','401 Unauthorized')
        with patch.object(kaggle,'cli',return_value=denied) as cli,patch.object(kaggle.time,'sleep') as sleep:
            with self.assertRaisesRegex(RuntimeError,'--resume-stage'):
                kaggle.read_submitted_job(['kernels','status',self.plan['kernel_reference']],self.plan)
        self.assertEqual(cli.call_count,1);sleep.assert_not_called()

    def test_output_index_delay_is_retried_on_pinned_version(self):
        responses=[subprocess.CompletedProcess([],1,'','404 not found'),
                   subprocess.CompletedProcess([],0,'downloaded','')]
        arguments=['kernels','output',self.plan['kernel_reference'],'-p',str(self.root),'-o']
        with patch.object(kaggle,'cli',side_effect=responses) as cli,\
             patch.object(kaggle.time,'sleep'),contextlib.redirect_stdout(io.StringIO()):
            result=kaggle.read_submitted_job(arguments,self.plan)
        self.assertEqual(result.stdout,'downloaded')
        self.assertTrue(all(call.args[0]==arguments for call in cli.call_args_list))


if __name__=='__main__':unittest.main()
