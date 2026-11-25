"""Real-Chrome UI regression, using only synthetic uploads and worker results.

Run: PYTHONPATH=/path/to/qa-packages venv/bin/python tests/demo_browser_smoke.py
"""
import io
from http.server import ThreadingHTTPServer
from pathlib import Path
import sys
import threading

from PIL import Image
from playwright.sync_api import sync_playwright, expect

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from vlm_project.demo_service import Store, make_handler


def main():
    store = Store()
    server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(store, 'synthetic-browser-only'))
    server.RequestHandlerClass = make_handler(store, 'synthetic-browser-only', web_port=server.server_port)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    errors = []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(executable_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', headless=True)
            page = browser.new_page(viewport={'width': 1440, 'height': 1050})
            page.on('pageerror', lambda exc: errors.append(str(exc)))
            page.goto(f'http://127.0.0.1:{server.server_port}')
            navigation = page.get_by_role('navigation', name='Điều hướng báo cáo')
            expect(navigation).to_be_visible()
            expect(page.locator('#view-overview')).to_be_visible()
            expect(page.locator('#view-setup')).to_be_hidden()
            expect(page.locator('#local-state')).to_have_text('Worker chưa kết nối')
            page.screenshot(path='/private/tmp/vlm-redesign-overview.png', full_page=True)
            navigation.get_by_role('link', name='Cấu hình suy luận', exact=True).click()
            expect(page.locator('#view-overview')).to_be_hidden()
            expect(page.locator('#view-setup')).to_be_visible()
            expect(page.get_by_role('heading', name='Bài đăng đầu vào')).to_be_visible()
            for name in ('Zero-shot', 'Thích nghi không nhãn', 'Few-shot'):
                expect(page.locator('#methods').get_by_role('heading', name=name, exact=True)).to_be_visible()
            expect(page.locator('#methods details')).not_to_have_attribute('open')
            stream = io.BytesIO()
            Image.new('RGB', (240, 160), '#547baa').save(stream, format='PNG')
            page.locator('#image').set_input_files({'name': 'synthetic.png', 'mimeType': 'image/png', 'buffer': stream.getvalue()})
            title = 'Synthetic UI test <script>no execution</script>'
            page.locator('#title').fill(title)
            page.screenshot(path='/private/tmp/vlm-redesign-setup.png', full_page=True)
            navigation.get_by_role('link', name='Bài toán', exact=True).click()
            navigation.get_by_role('link', name='Cấu hình suy luận', exact=True).click()
            expect(page.locator('#title')).to_have_value(title)
            expect(page.locator('#preview')).to_be_visible()
            page.get_by_role('button', name='Chạy phân loại', exact=True).click()
            expect(page.locator('#view-results')).to_be_visible()
            expect(page.locator('#view-setup')).to_be_hidden()
            expect(page.locator('#job-message')).to_contain_text('Chờ worker')
            job = store.claim('synthetic-browser-test')
            assert job is not None
            store.update(job['id'], job['lease'], {'status': 'complete', 'elapsed_seconds': .5,
                         'encoding_seconds': {'metaclip2': .2}, 'rows': [
                {'method': 'clip_image', 'name': 'CLIP · ảnh', 'backbone': 'metaclip2', 'pipeline': 'single',
                 'modality': 'image', 'status': 'complete', 'predicted': 'Real', 'score': .75,
                 'scores': {'Real': .75, 'Fake': .25}, 'seconds': .001, 'note': 'Synthetic UI fixture'},
                {'method': 'stata_image', 'name': 'StatA · ảnh', 'backbone': 'metaclip2', 'pipeline': 'single',
                 'modality': 'image', 'status': 'unavailable', 'message': 'Thiếu nhóm tham chiếu', 'seconds': .01},
                {'method': 'fewshot_image_title', 'name': 'Few-shot · ảnh–tiêu đề', 'backbone': 'metaclip2',
                 'pipeline': 'single', 'modality': 'image_title', 'status': 'complete', 'predicted': 'Real',
                 'score': .8, 'scores': {'Real': .8, 'Fake': .2}, 'seconds': .01,
                 'variants': [{'policy': 'balanced_binary', 'seed': 0, 'predicted': 'Real', 'score': .8}]}]})
            expect(page.locator('#results')).to_contain_text('Thiếu nhóm tham chiếu', timeout=8000)
            expect(page.locator('#results .family-row')).to_have_count(3)
            expect(page.locator('#results')).to_contain_text('< 0,01 s')
            expect(page.locator('#result-subtitle')).to_contain_text(title)
            expect(page.locator('#encoding-time')).to_have_text('0.20 s')
            page.get_by_role('button', name='Chi tiết CLIP · ảnh metaclip2', exact=True).click()
            expect(page.locator('#details')).to_be_visible()
            expect(page.locator('#details-body')).to_contain_text('75.0%')
            page.locator('#details-close').click()
            page.locator('[data-family="fewshot"]').click()
            expect(page.locator('#results .family-row')).to_have_count(1)
            expect(page.locator('#results')).not_to_contain_text('CLIP · ảnh')
            page.get_by_role('button', name='Chi tiết Few-shot · ảnh–tiêu đề metaclip2', exact=True).click()
            expect(page.locator('#details-body')).to_contain_text('balanced_binary')
            page.locator('#details-close').click()
            page.locator('[data-family="control"]').click()
            expect(page.locator('#filtered-empty')).to_be_visible()
            page.locator('[data-family="all"]').click()
            expect(page.locator('#results .family-row')).to_have_count(3)
            with page.expect_download() as download:
                page.get_by_role('button', name='Tải JSON', exact=True).click()
            assert download.value.suggested_filename.endswith('.json')
            page.screenshot(path='/private/tmp/vlm-redesign-results.png', full_page=True)
            navigation.get_by_role('link', name='Lịch sử', exact=True).click()
            expect(page.locator('#view-results')).to_be_hidden()
            expect(page.locator('#history')).to_contain_text(title)
            page.locator('#history .history-item').first.click()
            expect(page.locator('#view-results')).to_be_visible()
            navigation.get_by_role('link', name='Cấu hình suy luận', exact=True).click()
            expect(page.locator('#title')).to_have_value(title)
            page.locator('#pipeline').select_option('cascade')
            expect(page.locator('#task')).to_have_value('6way')
            expect(page.locator('#methods')).not_to_contain_text('Few-shot · ảnh–tiêu đề')
            page.locator('#title').fill('Synthetic cancellation test')
            page.get_by_role('button', name='Chạy phân loại', exact=True).click()
            expect(page.locator('#job-message')).to_contain_text('Chờ worker')
            page.get_by_role('button', name='Hủy yêu cầu', exact=True).click()
            expect(page.locator('#job-message')).to_contain_text('Đã hủy')
            page.get_by_role('button', name='Thiết lập kết nối', exact=True).click()
            expect(page.locator('#connection')).to_be_visible()
            page.keyboard.press('Escape')
            expect(page.locator('#connection')).to_be_hidden()
            # A history endpoint failure must not be mislabeled as loss of local API.
            store.heartbeat({'worker': 'synthetic', 'state': 'ready'})
            page.route('**/api/jobs', lambda route: route.fulfill(status=500, content_type='application/json', body='{"error":"Synthetic history failure"}'))
            expect(page.locator('#local-state')).to_have_text('Worker sẵn sàng', timeout=8000)
            expect(page.locator('#app-error')).to_contain_text('Synthetic history failure')
            page.unroute('**/api/jobs')
            expect(page.locator('#app-error')).to_be_hidden(timeout=8000)
            navigation.get_by_role('link', name='Lịch sử', exact=True).click()
            page.locator('#history .history-item').filter(has_text=title).click()
            expect(page.locator('#results .family-row')).to_have_count(3)
            for width, height in [(1440, 1000), (768, 1024), (390, 844)]:
                page.set_viewport_size({'width': width, 'height': height})
                for section, link in [('overview', 'Bài toán'), ('setup', 'Cấu hình suy luận'), ('results', 'Kết quả suy luận')]:
                    navigation.get_by_role('link', name=link, exact=True).click()
                    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), f'Page overflows at {width}: {section}'
                    assert page.locator('.page-view:visible').count() == 1
                    page.screenshot(path=f'/private/tmp/vlm-redesign-{section}-{width}.png', full_page=True)
            assert not errors, errors
            browser.close()
            print('PASS: independent views, state retention, grouped methods/results, filters, upload, details, download, history, cancel, connection status; 3 widths; no JS errors.')
    finally:
        server.shutdown()
        server.server_close()


if __name__ == '__main__':
    main()
