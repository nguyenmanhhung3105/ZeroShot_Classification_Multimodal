"""Chrome QA for read-only experiment charts, using exported evaluation artifacts only."""
import argparse
from http.server import ThreadingHTTPServer
from pathlib import Path
import sys
import threading


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--qa-packages', help='Optional directory containing Playwright')
    args = parser.parse_args()
    if args.qa_packages: sys.path.insert(0, args.qa_packages)
    from playwright.sync_api import sync_playwright, expect
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / 'src'))
    from vlm_project.demo_service import Store, make_handler
    server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(Store(), 'synthetic-chart-qa'))
    server.RequestHandlerClass = make_handler(Store(), 'synthetic-chart-qa', web_port=server.server_port)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    errors, checked = [], 0
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(executable_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', headless=True)
            page = browser.new_page(viewport={'width': 1440, 'height': 1100})
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.goto(f'http://127.0.0.1:{server.server_port}/#view-experiments')
            expect(page.locator('#exp-run')).to_be_enabled(timeout=15000)
            expect(page.locator('#view-experiments')).to_have_attribute('data-run-id', 'fakeddit-binary-comparison')
            expect(page.locator('#exp-chart svg')).to_have_count(1)
            page.screenshot(path='/private/tmp/vlm-experiments-comparison.png', full_page=True)
            index = page.request.get(f'http://127.0.0.1:{server.server_port}/api/experiments').json()
            # Every exported run and every available chart must render without invalid SVG data.
            for entry in index['runs']:
                page.locator('#exp-run').select_option(entry['id'])
                expect(page.locator('#view-experiments')).to_have_attribute('data-run-id', entry['id'], timeout=15000)
                buttons = page.locator('#exp-tabs button').all_text_contents()
                for title in buttons:
                    page.locator('#exp-tabs').get_by_role('button', name=title, exact=True).click()
                    expect(page.locator('#exp-chart svg').first).to_be_visible()
                    assert page.locator('#exp-data-table th').count() > 1
                    invalid = page.locator('#exp-chart svg').evaluate_all('nodes => nodes.some(node => /NaN|Infinity|undefined/.test(node.outerHTML))')
                    assert not invalid, (entry['kind'], title)
                    checked += 1
                    if entry['kind'] == 'fakeddit_fewshot_cascade' and title == 'Gate và head con':
                        page.screenshot(path='/private/tmp/vlm-experiments-cascade.png', full_page=True)
                    if entry['kind'] == 'fewshot_support_thresholds' and title == 'Precision–Recall và ngưỡng':
                        page.screenshot(path='/private/tmp/vlm-experiments-threshold.png', full_page=True)
                assert not errors, errors
            page.locator('#exp-run').select_option('fakeddit-binary-comparison')
            expect(page.locator('#view-experiments')).to_have_attribute('data-run-id', 'fakeddit-binary-comparison')
            page.locator('#exp-tabs').get_by_role('button', name='So sánh phương pháp', exact=True).click()
            for metric in ['accuracy', 'macro_f1', 'balanced_accuracy']:
                page.locator('#exp-metric').select_option(metric)
                expect(page.locator('#exp-data-table')).to_contain_text('CLIP')
            for button, extension in [('Tải bảng CSV', '.csv'), ('Tải SVG chính', '.svg'), ('Tải dữ liệu run', '.json')]:
                with page.expect_download() as download:
                    page.get_by_role('button', name=button, exact=True).click()
                assert download.value.suggested_filename.endswith(extension)
            page.locator('#exp-tabs').get_by_role('button', name='Ma trận và cặp lớp nhầm lẫn', exact=True).click()
            page.locator('#exp-control-1').select_option('count')
            page.locator('#exp-control-0').select_option(index=1)
            page.screenshot(path='/private/tmp/vlm-experiments-confusion.png', full_page=True)
            for width, height in [(1440, 1000), (768, 1024), (390, 844)]:
                page.set_viewport_size({'width': width, 'height': height})
                assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), f'Overflow at {width}'
                page.screenshot(path=f'/private/tmp/vlm-experiments-{width}.png', full_page=True)
            assert not errors, errors
            browser.close()
            print(f'PASS: {len(index["runs"])} experiment views, {checked} charts, metric/seed controls, CSV/SVG/JSON, 3 widths, no JS errors.')
    finally:
        server.shutdown(); server.server_close()


if __name__ == '__main__': main()
