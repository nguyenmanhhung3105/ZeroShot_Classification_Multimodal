"""Local UI and authenticated worker queue. This module never imports a model runtime."""
import base64
import binascii
import copy
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import secrets
import threading
import time
from urllib.parse import urlsplit
import uuid
import warnings

MAX_IMAGE = 8 * 1024 * 1024
MAX_BODY = 12 * 1024 * 1024
BACKBONES = {'metaclip2': 'MetaCLIP2 ViT-H/14', 'openai_clip': 'OpenAI CLIP ViT-L/14'}
METHODS = {
    'clip_image': ('CLIP · ảnh', 'image', 'single'),
    'clip_title': ('CLIP · tiêu đề', 'title', 'single'),
    'clip_image_title': ('CLIP · fusion', 'image_title', 'single'),
    'legacy_image': ('Baseline legacy · ảnh', 'image', 'single'),
    'legacy_title': ('Baseline legacy · tiêu đề', 'title', 'single'),
    'legacy_fusion': ('Baseline legacy · fusion', 'image_title', 'single'),
    'majority': ('Majority · ảnh–tiêu đề', 'image_title', 'single'),
    'dclip': ('DCLIP', 'image', 'single'),
    'waffle': ('WaffleCLIP', 'image', 'single'),
    'waffle_unit': ('WaffleCLIP · unit', 'image', 'single'),
    'calip': ('CALIP', 'image', 'single'),
    'em_image': ('EM-Dirichlet', 'image', 'single'),
    'flat_stata': ('StatA · phẳng 6 lớp', 'image', 'single'),
    'flat_majority': ('Majority · phẳng 6 lớp', 'image', 'single'),
    'stata_image': ('StatA · ảnh', 'image', 'single'),
    'clip_stata_image': ('CLIP–StatA · ảnh', 'image', 'single'),
    'clip_stata_image_title': ('CLIP–StatA · fusion', 'image_title', 'single'),
    'fewshot_image': ('Few-shot · ảnh', 'image', 'single'),
    'fewshot_title': ('Few-shot · tiêu đề', 'title', 'single'),
    'fewshot_image_title': ('Few-shot · ảnh–tiêu đề', 'image_title', 'single'),
    'fewshot_flat': ('Few-shot · phẳng 6 lớp', 'image_title', 'single'),
    'fewshot_cascade': ('Few-shot · cascade', 'image_title', 'cascade'),
    'fewshot_cascade_r87': ('Few-shot · cascade R87', 'image_title', 'cascade'),
    'hierarchy': ('Zero-shot · hierarchy StatA', 'image', 'cascade'),
    'hierarchy_clip': ('Zero-shot · hierarchy CLIP', 'image', 'cascade'),
    'hierarchy_em': ('Zero-shot · hierarchy EM', 'image', 'cascade'),
    'hierarchy_majority': ('Zero-shot · hierarchy Majority', 'image', 'cascade'),
}


def catalog():
    return {'backbones': BACKBONES, 'methods': [
        {'id': key, 'name': v[0], 'modality': v[1], 'pipeline': v[2]}
        for key, v in METHODS.items()], 'max_image_bytes': MAX_IMAGE}


def validate_request(value):
    from PIL import Image, UnidentifiedImageError
    if not isinstance(value, dict):
        raise ValueError('Yêu cầu phải là JSON object')
    title = value.get('title', '')
    if not isinstance(title, str) or not title.strip() or len(title) > 2000:
        raise ValueError('Nhập tiêu đề từ 1 đến 2.000 ký tự')
    for key, allowed in [('task', ['2way', '6way']), ('pipeline', ['single', 'cascade', 'all'])]:
        if value.get(key) not in allowed:
            raise ValueError(f'{key} không hợp lệ')
    for key, allowed in [('backbones', BACKBONES), ('methods', METHODS)]:
        values = value.get(key)
        if (not isinstance(values, list) or not values or len(values) > len(allowed)
                or any(not isinstance(v, str) or v not in allowed for v in values)
                or len(set(values)) != len(values)):
            raise ValueError(f'Chọn {key} hợp lệ, không trùng lặp')
    encoded = value.get('image')
    if not isinstance(encoded, str) or len(encoded) > (MAX_IMAGE * 4 // 3 + 4):
        raise ValueError('Ảnh vượt giới hạn 8 MB')
    try:
        raw = base64.b64decode(encoded, validate=True)
        if not raw or len(raw) > MAX_IMAGE:
            raise ValueError('Ảnh rỗng hoặc vượt 8 MB')
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as image:
                if image.format not in ('JPEG', 'PNG', 'WEBP') or image.width * image.height > 20_000_000:
                    raise ValueError('Chỉ nhận JPEG/PNG/WebP, tối đa 20 megapixel')
                image.verify()
    except (binascii.Error, OSError, UnidentifiedImageError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as exc:
        raise ValueError('File ảnh không hợp lệ') from exc
    return {key: (title.strip() if key == 'title' else value[key])
            for key in ['image', 'title', 'task', 'pipeline', 'backbones', 'methods']}


class Store:
    """Bounded in-memory session; one lease per job, renewed by worker progress."""
    def __init__(self, clock=time.time, lease_seconds=120, max_jobs=20):
        self.clock, self.lease_seconds, self.max_jobs = clock, lease_seconds, max_jobs
        self.jobs, self.worker = {}, {}
        self.lock = threading.RLock()

    @staticmethod
    def public(job):
        return copy.deepcopy({k: v for k, v in job.items() if k not in ('payload', 'lease', 'expires')})

    def submit(self, payload):
        with self.lock:
            if len(self.jobs) >= self.max_jobs:
                terminal = next((k for k, v in self.jobs.items()
                                 if v['status'] in ('complete', 'failed', 'cancelled')), None)
                if terminal is None:
                    raise ValueError('Hàng đợi đầy; chờ hoặc hủy yêu cầu cũ')
                del self.jobs[terminal]
            job = {'id': uuid.uuid4().hex, 'created': self.clock(), 'status': 'queued',
                   'title': payload['title'], 'task': payload['task'], 'rows': [],
                   'expected': len(payload['methods']) * len(payload['backbones']),
                   'message': 'Đang chờ GPU Kaggle', 'payload': copy.deepcopy(payload)}
            self.jobs[job['id']] = job
            return self.public(job)

    def get(self, identity):
        with self.lock:
            return self.public(self.jobs[identity])

    def history(self):
        with self.lock:
            return [self.public(v) for v in reversed(list(self.jobs.values()))]

    def claim(self, worker):
        with self.lock:
            for job in self.jobs.values():
                if job['status'] == 'queued' or (job['status'] == 'running' and job['expires'] < self.clock()):
                    job.update(status='running', lease=secrets.token_urlsafe(24),
                               expires=self.clock() + self.lease_seconds, rows=[], worker=worker,
                               message='GPU đã nhận yêu cầu')
                    return copy.deepcopy({k: job[k] for k in ('id', 'payload', 'lease')})
        return None

    def update(self, identity, lease, update):
        with self.lock:
            job = self.jobs[identity]
            if (job['status'] != 'running' or not secrets.compare_digest(job.get('lease', ''), lease)
                    or job['expires'] < self.clock()):
                raise ValueError('Lease đã hết hạn hoặc yêu cầu đã hủy')
            status = update.get('status', 'running')
            if status not in ('running', 'complete', 'failed'):
                raise ValueError('Trạng thái không hợp lệ')
            job['expires'] = self.clock() + self.lease_seconds
            for key in ('rows', 'message', 'elapsed_seconds', 'encoding_seconds'):
                if key in update:
                    job[key] = copy.deepcopy(update[key])
            job['status'] = status
            if status in ('complete', 'failed'):
                job.pop('payload', None)
            return self.public(job)

    def cancel(self, identity):
        with self.lock:
            job = self.jobs[identity]
            if job['status'] in ('queued', 'running'):
                job.update(status='cancelled', message='Đã hủy; bỏ kết quả đến muộn')
                job.pop('payload', None)
            return self.public(job)

    def heartbeat(self, info):
        with self.lock:
            self.worker = {k: copy.deepcopy(info[k]) for k in
                           ('worker', 'state', 'backbones', 'message', 'capabilities') if k in info}
            self.worker['seen'] = self.clock()

    def health(self):
        with self.lock:
            return {**copy.deepcopy(self.worker),
                    'online': bool(self.worker and self.clock() - self.worker['seen'] < 35)}


def make_handler(store, token, *, worker_only=False, web_port=8765):
    assets = Path(__file__).parent / 'demo_web'

    def experiment_asset(identity=None):
        file = assets / 'experiment-results.json'
        if identity is None:
            return file.read_bytes()
        index = json.loads(file.read_text())
        entry = next((r for r in index['runs'] if r['id'] == identity), None)
        if entry is None:
            raise KeyError(identity)
        payload = (assets / 'experiments' / (identity + '.json')).read_bytes()
        if hashlib.sha256(payload).hexdigest() != entry['sha256']:
            raise ValueError('Snapshot kết quả không khớp checksum')
        return payload

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # Avoid logging titles, tokens, and uploaded content.

        def respond(self, value, status=200, content_type='application/json; charset=utf-8'):
            data = json.dumps(value, ensure_ascii=False, allow_nan=False).encode() if isinstance(value, (dict, list)) else value
            self.send_response(status)
            for k, v in [('Content-Type', content_type), ('Content-Length', str(len(data))),
                         ('Cache-Control', 'no-store'), ('X-Content-Type-Options', 'nosniff'),
                         ('Referrer-Policy', 'no-referrer'),
                         ('Content-Security-Policy', "default-src 'self'; img-src 'self' blob: data:; frame-ancestors 'none'")]:
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(data)

        def handle_request(self):
            self.connection.settimeout(20)
            path = urlsplit(self.path).path
            if worker_only:
                auth = self.headers.get('Authorization', '')
                if not secrets.compare_digest(auth, 'Bearer ' + token):
                    self.respond({'error': 'Unauthorized'}, 401)
                    return
                if not path.startswith('/worker/'):
                    self.respond({'error': 'Not found'}, 404)
                    return
            else:
                allowed_hosts = {f'localhost:{web_port}', f'127.0.0.1:{web_port}'}
                origin = self.headers.get('Origin')
                if (self.headers.get('Host') not in allowed_hosts
                        or (origin and origin not in {f'http://{h}' for h in allowed_hosts})
                        or self.headers.get('Sec-Fetch-Site') == 'cross-site'):
                    self.respond({'error': 'Local origin required'}, 403)
                    return
                if path.startswith('/worker/'):
                    self.respond({'error': 'Use worker port'}, 404)
                    return
            try:
                body = {}
                if self.command == 'POST':
                    if not worker_only and self.headers.get('X-Demo-Client') != 'localhost':
                        self.respond({'error': 'Missing local client header'}, 403)
                        return
                    length = int(self.headers.get('Content-Length', 0))
                    if not 0 < length <= MAX_BODY:
                        raise ValueError('Kích thước yêu cầu không hợp lệ')
                    if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                        raise ValueError('Cần Content-Type application/json')
                    body = json.loads(self.rfile.read(length))
                    if not isinstance(body, dict):
                        raise ValueError('Cần JSON object')
                if self.command == 'GET' and not worker_only:
                    if path == '/api/catalog': self.respond(catalog())
                    elif path == '/api/health': self.respond(store.health())
                    elif path == '/api/jobs': self.respond(store.history())
                    elif path.startswith('/api/jobs/'): self.respond(store.get(path.split('/')[-1]))
                    elif path == '/api/experiments':
                        self.respond(experiment_asset(), content_type='application/json; charset=utf-8')
                    elif path.startswith('/api/experiments/'):
                        self.respond(experiment_asset(path.split('/')[-1]), content_type='application/json; charset=utf-8')
                    elif path in ('/', '/app.js', '/analytics.js', '/style.css'):
                        name = 'index.html' if path == '/' else path[1:]
                        mime = {'index.html': 'text/html', 'app.js': 'application/javascript',
                                'analytics.js': 'application/javascript', 'style.css': 'text/css'}[name]
                        self.respond((assets / name).read_bytes(), content_type=mime + '; charset=utf-8')
                    else: self.respond({'error': 'Not found'}, 404)
                elif self.command == 'POST' and not worker_only:
                    if path == '/api/jobs': self.respond(store.submit(validate_request(body)), 201)
                    elif path.endswith('/cancel') and path.startswith('/api/jobs/'):
                        self.respond(store.cancel(path.split('/')[-2]))
                    elif path == '/api/connection': self.respond({'token': token, 'worker_port': web_port + 1})
                    else: self.respond({'error': 'Not found'}, 404)
                elif self.command == 'POST' and worker_only:
                    if path == '/worker/heartbeat':
                        store.heartbeat(body); self.respond({'ok': True})
                    elif path == '/worker/claim': self.respond({'job': store.claim(str(body.get('worker', 'kaggle')))})
                    elif path == '/worker/update':
                        self.respond(store.update(body['id'], body['lease'], body['update']))
                    else: self.respond({'error': 'Not found'}, 404)
                else: self.respond({'error': 'Method not allowed'}, 405)
            except KeyError:
                self.respond({'error': 'Không tìm thấy yêu cầu hoặc thiếu trường'}, 404)
            except FileNotFoundError:
                self.respond({'error': 'Chưa có snapshot kết quả thực nghiệm'}, 503)
            except (ValueError, TypeError) as exc:
                self.respond({'error': str(exc)}, 400)

        do_GET = handle_request
        do_POST = handle_request

    return Handler


def serve(port=8765, *, web_host='127.0.0.1'):
    store, token = Store(), secrets.token_urlsafe(32)
    web = ThreadingHTTPServer((web_host, port), make_handler(store, token, web_port=port))
    worker = ThreadingHTTPServer(('127.0.0.1', port + 1),
                                make_handler(store, token, worker_only=True, web_port=port))
    threading.Thread(target=worker.serve_forever, daemon=True).start()
    print(f'Web demo: http://127.0.0.1:{port}', flush=True)
    print(f'Worker gateway: http://127.0.0.1:{port + 1} (chỉ tunnel cổng này)', flush=True)
    try:
        web.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        worker.shutdown()
        worker.server_close()
        web.server_close()
