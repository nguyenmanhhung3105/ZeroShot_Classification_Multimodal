"""Outbound-only Kaggle worker. Credentials come from Kaggle Secrets or environment."""
import json
import os
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
import uuid


def secret(name):
    value = os.environ.get(name)
    if value:
        return value
    from kaggle_secrets import UserSecretsClient
    return UserSecretsClient().get_secret(name)


class Client:
    def __init__(self, url, token):
        parsed = urlsplit(url)
        if (parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password
                or parsed.query or parsed.fragment or parsed.path not in ('', '/')):
            raise ValueError('DEMO_BROKER_URL phải là HTTPS origin, không có user/password/path')
        if len(token) < 24:
            raise ValueError('DEMO_WORKER_TOKEN quá ngắn')
        self.url, self.token = url.rstrip('/'), token
        class NoRedirect(HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                raise HTTPError(req.full_url, code, 'Gateway redirects are disabled', headers, fp)
        self.opener = build_opener(NoRedirect)

    def post(self, path, body):
        request = Request(self.url + path, data=json.dumps(body, allow_nan=False).encode(),
                          headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + self.token},
                          method='POST')
        with self.opener.open(request, timeout=20) as response:
            return json.load(response)


def run(backbones=('metaclip2',), assets=None, max_seconds=14400, prepare_prompts=True):
    from .demo_inference import Engine
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Bật GPU trong Kaggle trước khi chạy worker')
    client = Client(secret('DEMO_BROKER_URL'), secret('DEMO_WORKER_TOKEN'))
    worker = 'kaggle-' + uuid.uuid4().hex[:8]
    state = {'worker': worker, 'state': 'loading', 'backbones': list(backbones),
             'message': 'Đang nạp backbone và prototype'}
    current, lock, stop = {}, threading.RLock(), threading.Event()

    def heartbeat():
        while not stop.is_set():
            try:
                with lock:
                    snapshot, job = dict(state), dict(current)
                client.post('/worker/heartbeat', snapshot)
                if job:
                    client.post('/worker/update', {**job, 'update': {}})
            except (OSError, ValueError):
                pass  # Main loop reports connectivity failures, never credentials.
            stop.wait(10)

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        engine = Engine(backbones, assets)
        if prepare_prompts:
            engine.prepare_prompts()
        engine.load(backbones[0])
        # Warm prototype caches for the first backbone before accepting requests.
        for task, banks in engine.profile['image_prompts'].items():
            for bank in banks.values():
                engine.backend.prototypes(bank)
            engine.backend.prototypes(engine.profile['text_prompts'][task])
        state.update(state='ready', message='GPU sẵn sàng', capabilities={
            'prompt_banks': bool(engine.prompt_banks), 'assets': list(engine.assets),
            'resident_backbone': backbones[0], 'model_policy': 'one_resident_backbone'})
        deadline = time.monotonic() + max_seconds
        print('Worker sẵn sàng; model chỉ được nạp trên Kaggle.', flush=True)
        while time.monotonic() < deadline:
            try:
                response = client.post('/worker/claim', {'worker': worker})
                job = response['job']
                if not job:
                    stop.wait(2)
                    continue
                identity = {k: job[k] for k in ('id', 'lease')}
                with lock:
                    current.update(identity)
                    state.update(state='busy', message='Đang phân loại')

                def emit(update):
                    # Repeat updates are idempotent; transient disconnect must not discard results.
                    for attempt in range(3):
                        try:
                            return client.post('/worker/update', {**identity, 'update': update})
                        except HTTPError:
                            raise
                        except (URLError, TimeoutError):
                            if attempt == 2:
                                raise
                            stop.wait(2)

                try:
                    emit(engine.run(job['payload'], emit))
                except HTTPError as exc:
                    print(f'Yêu cầu đã hủy/hết lease hoặc gateway từ chối: HTTP {exc.code}', flush=True)
                except Exception as exc:
                    try:
                        emit({'status': 'failed', 'message': f'Lỗi worker: {type(exc).__name__}'})
                    except OSError:
                        pass
                finally:
                    with lock:
                        current.clear()
                        state.update(state='ready', message='GPU sẵn sàng')
            except (OSError, ValueError, KeyError) as exc:
                print(f'Chưa kết nối gateway ({type(exc).__name__}); thử lại sau 5 giây.', flush=True)
                stop.wait(5)
    finally:
        stop.set()
        thread.join(timeout=25)
        if 'engine' in locals() and engine.backend is not None:
            engine.backend.unload()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbones', nargs='+', choices=['metaclip2', 'openai_clip'], default=['metaclip2'])
    parser.add_argument('--assets')
    parser.add_argument('--max-seconds', type=int, default=14400)
    parser.add_argument('--skip-author-prompts', action='store_true')
    args = parser.parse_args()
    run(args.backbones, args.assets, args.max_seconds, not args.skip_author_prompts)
