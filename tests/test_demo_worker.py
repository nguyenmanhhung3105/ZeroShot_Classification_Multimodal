"""Credential transport and CUDA-only guard, without any remote request/model."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch, MagicMock
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from vlm_project.demo_worker import Client, run


class WorkerTests(unittest.TestCase):
    def test_requires_https_origin_and_sufficient_token(self):
        for url in ('http://example.com', 'https://user:pass@example.com',
                    'https://example.com/path', 'https://example.com?secret=x'):
            with self.subTest(url=url), self.assertRaises(ValueError):
                Client(url, 'synthetic-test-token-long-enough')
        with self.assertRaises(ValueError):
            Client('https://example.com', 'short')

    def test_bearer_not_query_string_and_json_roundtrip(self):
        client = Client('https://example.com', 'synthetic-test-token-long-enough')
        response = MagicMock()
        response.__enter__.return_value.read.return_value = b'{"job": null}'
        with patch.object(client.opener, 'open', return_value=response) as opened:
            self.assertEqual(client.post('/worker/claim', {'worker': 'test'}), {'job': None})
        request = opened.call_args.args[0]
        self.assertEqual(request.full_url, 'https://example.com/worker/claim')
        self.assertTrue(request.get_header('Authorization').startswith('Bearer '))
        self.assertEqual(json.loads(request.data), {'worker': 'test'})

    def test_cpu_fallback_is_forbidden(self):
        with patch('torch.cuda.is_available', return_value=False), self.assertRaises(RuntimeError):
            run()
