"""Demo contract: no model required locally; worker leases cannot overwrite new runs."""
import base64
import importlib.util
import io
from pathlib import Path
import sys
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import json
from unittest.mock import patch, MagicMock
from http.server import ThreadingHTTPServer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


class DemoServiceTests(unittest.TestCase):
    def test_container_binds_web_only_and_keeps_gateway_private(self):
        from vlm_project.demo_service import serve
        web, gateway = MagicMock(), MagicMock()
        web.serve_forever.side_effect = KeyboardInterrupt
        with patch('vlm_project.demo_service.ThreadingHTTPServer', side_effect=[web, gateway]) as factory, \
                patch('vlm_project.demo_service.threading.Thread'):
            serve(web_host='0.0.0.0')
        self.assertEqual(factory.call_args_list[0].args[0], ('0.0.0.0', 8765))
        self.assertEqual(factory.call_args_list[1].args[0], ('127.0.0.1', 8766))
        gateway.shutdown.assert_called_once()

    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('vlm_project.demo_service'),
                             'Missing localhost queue and validation implementation')
        from vlm_project.demo_service import Store, validate_request
        self.Store, self.validate = Store, validate_request

    def request(self):
        from PIL import Image
        stream = io.BytesIO()
        Image.new('RGB', (3, 3)).save(stream, format='PNG')
        return {'image': base64.b64encode(stream.getvalue()).decode(), 'title': 'A street',
                'task': '2way', 'pipeline': 'all', 'backbones': ['metaclip2'],
                'methods': ['clip_image', 'clip_image_title']}

    def test_validate_image_and_reject_invalid_options(self):
        value = self.validate(self.request())
        self.assertEqual(value['title'], 'A street')
        for change in ({'image': 'bad'}, {'task': 'test'}, {'backbones': ['shell']},
                       {'methods': []}, {'title': ' '}, {'pipeline': 'invalid'}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.validate({**self.request(), **change})

    def test_claim_partial_finish_and_private_history(self):
        store = self.Store()
        job = store.submit(self.validate(self.request()))
        claimed = store.claim('worker')
        self.assertEqual(job['id'], claimed['id'])
        self.assertIsNone(store.claim('other'))
        store.update(job['id'], claimed['lease'], {'rows': [{'method': 'clip_image'}]})
        self.assertEqual(len(store.get(job['id'])['rows']), 1)
        store.update(job['id'], claimed['lease'], {'status': 'complete'})
        self.assertEqual(store.get(job['id'])['status'], 'complete')
        self.assertNotIn('image', store.history()[0])
        self.assertNotIn('payload', store.history()[0])
        self.assertNotIn('lease', store.get(job['id']))

    def test_expired_lease_rejects_old_worker_and_cancel(self):
        clock = [100.]
        store = self.Store(clock=lambda: clock[0], lease_seconds=10)
        job = store.submit(self.validate(self.request()))
        first = store.claim('first')
        clock[0] += 11
        second = store.claim('second')
        self.assertNotEqual(first['lease'], second['lease'])
        with self.assertRaises(ValueError):
            store.update(job['id'], first['lease'], {'status': 'complete'})
        store.cancel(job['id'])
        with self.assertRaises(ValueError):
            store.update(job['id'], second['lease'], {'status': 'complete'})

    def test_offline_health_and_queue_capacity(self):
        store = self.Store(max_jobs=1)
        self.assertFalse(store.health()['online'])
        store.heartbeat({'worker': 'gpu', 'state': 'ready', 'backbones': ['metaclip2']})
        self.assertTrue(store.health()['online'])
        store.submit(self.validate(self.request()))
        with self.assertRaises(ValueError):
            store.submit(self.validate(self.request()))

    def test_http_end_to_end_and_auth_boundaries(self):
        from vlm_project.demo_service import make_handler
        store, token = self.Store(), 'synthetic-test-token'
        web = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(store, token))
        port = web.server_port
        web.RequestHandlerClass = make_handler(store, token, web_port=port)
        gateway = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(store, token, worker_only=True))
        for server in (web, gateway):
            threading.Thread(target=server.serve_forever, daemon=True).start()

        def call(path, body=None, *, worker=False, headers=None):
            request_headers = {'Content-Type': 'application/json', 'X-Demo-Client': 'localhost'}
            if worker:
                request_headers['Authorization'] = 'Bearer ' + token
            request_headers.update(headers or {})
            request = Request(f'http://127.0.0.1:{gateway.server_port if worker else port}' + path,
                              data=None if body is None else json.dumps(body).encode(), headers=request_headers)
            with urlopen(request) as response:
                data = response.read()
                return json.loads(data) if 'application/json' in response.headers['Content-Type'] else data
        try:
            self.assertIn(b'VLM Classifier', call('/'))
            self.assertIn(b'renderMethods', call('/app.js'))
            self.assertIn(b'--blue', call('/style.css'))
            self.assertIn(b'<svg', call('/favicon.svg'))
            self.assertIn(b'<svg', call('/favicon.ico'))
            import gzip
            for encoding, compressed in [('gzip', True), ('gzip;q=0', False)]:
                request = Request(f'http://127.0.0.1:{port}/api/experiments',
                                  headers={'Accept-Encoding': encoding})
                with urlopen(request) as response:
                    payload = response.read()
                    self.assertEqual(response.headers.get('Content-Encoding') == 'gzip', compressed)
                    self.assertEqual(response.headers['Cache-Control'], 'no-store')
                    self.assertIn("frame-ancestors 'none'", response.headers['Content-Security-Policy'])
                    if compressed:
                        payload = gzip.decompress(payload)
                    self.assertIn('runs', json.loads(payload))
            request = Request(f'http://127.0.0.1:{port}/api/connection', data=b'{}',
                              headers={'Accept-Encoding': 'gzip', 'Content-Type': 'application/json',
                                       'X-Demo-Client': 'localhost'})
            with urlopen(request) as response:
                self.assertIsNone(response.headers.get('Content-Encoding'))
                self.assertEqual(json.loads(response.read())['token'], token)

            self.assertIn(b'ExperimentCharts', call('/analytics.js'))
            self.assertIn(b'MethodSimulations', call('/simulations.js'))
            self.assertIn(b'MethodStage', call('/simulations-stage.js'))
            self.assertIn(b'SimulationRecipes', call('/simulations-recipes.js'))
            self.assertTrue(call('/simulations-sample.jpg').startswith(b'\xff\xd8'))
            self.assertIn(b'.simulation-theater', call('/simulations.css'))
            experiments = call('/api/experiments')
            self.assertTrue(experiments['runs'])
            data = call('/api/experiments/' + experiments['default'])
            self.assertTrue(data['methods'])
            with self.assertRaises(HTTPError): call('/api/experiments/not-a-run')
            self.assertIn('metaclip2', call('/api/catalog')['backbones'])
            self.assertFalse(call('/api/health')['online'])
            self.assertEqual(call('/api/connection', {})['token'], token)
            for path, body, headers, worker in [
                ('/api/jobs', self.request(), {'Origin': 'https://evil.example'}, False),
                ('/api/jobs', self.request(), {'X-Demo-Client': ''}, False),
                ('/api/jobs', None, {'Host': 'evil.example'}, False),
                ('/worker/claim', {}, {'Authorization': 'bad'}, True),
                ('/', None, {}, True), ('/worker/claim', {}, {}, False),
                ('/missing', None, {}, False), ('/api/jobs', [], {}, False),
            ]:
                with self.subTest(path=path, headers=headers), self.assertRaises(HTTPError):
                    call(path, body, headers=headers, worker=worker)
            call('/worker/heartbeat', {'worker': 'synthetic', 'state': 'ready'}, worker=True)
            self.assertTrue(call('/api/health')['online'])
            job = call('/api/jobs', self.request())
            claimed = call('/worker/claim', {'worker': 'synthetic'}, worker=True)['job']
            self.assertEqual(claimed['id'], job['id'])
            self.assertIsNone(call('/worker/claim', {}, worker=True)['job'])
            call('/worker/update', {'id': job['id'], 'lease': claimed['lease'],
                                   'update': {'rows': [{'method': 'clip_image', 'predicted': 'Real'}],
                                              'status': 'complete', 'elapsed_seconds': .25}}, worker=True)
            self.assertEqual(call('/api/jobs/' + job['id'])['rows'][0]['predicted'], 'Real')
            self.assertEqual(len(call('/api/jobs')), 1)
            second = call('/api/jobs', self.request())
            self.assertEqual(call('/api/jobs/' + second['id'] + '/cancel', {})['status'], 'cancelled')
        finally:
            for server in (web, gateway):
                server.shutdown()
                server.server_close()


if __name__ == '__main__':
    unittest.main()
