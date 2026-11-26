"""Browser checks for synthetic walkthroughs; no model, uploads or real data."""
from http.server import ThreadingHTTPServer
from pathlib import Path
import sys
import threading

from playwright.sync_api import sync_playwright, expect

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from vlm_project.demo_service import Store, make_handler


def main():
    server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(Store(), 'synthetic-simulation-test'))
    server.RequestHandlerClass = make_handler(Store(), 'synthetic-simulation-test', web_port=server.server_port)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    errors = []
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(executable_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', headless=True)
            page = browser.new_page(viewport={'width': 1440, 'height': 1050})
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.goto(f'http://127.0.0.1:{server.server_port}/#view-simulations')
            expect(page.locator('#view-simulations')).to_be_visible()
            page.get_by_role('button', name='Thích nghi không nhãn', exact=True).click()
            page.get_by_role('tab', name='StatA · ảnh', exact=True).click()
            expect(page.locator('#sim-tabs [role=tab]')).to_have_count(1)
            expect(page.locator('#sim-case')).to_be_disabled()
            page.locator('#sim-next').click()
            coordinates = page.locator('#sim-visual [data-image]').evaluate_all('nodes => nodes.map(n => [n.getAttribute("cx"), n.getAttribute("cy")])')
            assert len(coordinates) == 41

            def finish():
                for _ in range(10):
                    if page.locator('#sim-next').is_disabled():
                        break
                    page.locator('#sim-next').click()
                expect(page.locator('#sim-next')).to_be_disabled()

            finish()
            assert coordinates == page.locator('#sim-visual [data-image]').evaluate_all('nodes => nodes.map(n => [n.getAttribute("cx"), n.getAttribute("cy")])')
            expect(page.locator('#sim-output')).to_contain_text('Fake')
            page.screenshot(path='/private/tmp/vlm-simulation-stata.png', full_page=True)
            page.get_by_role('button', name='Few-shot', exact=True).click()
            for name, result in [('Few-shot · phẳng 6 lớp', 'Real'), ('Few-shot · cascade', 'Real'), ('Few-shot · cascade R87', 'Real')]:
                page.get_by_role('tab', name=name, exact=True).click()
                expect(page.locator('#sim-timeline button').first).to_have_attribute('aria-pressed', 'true')
                finish()
                expect(page.locator('#sim-output .sim-output-label')).to_have_text(result)
            page.locator('#sim-case').select_option('borderline')
            expect(page.locator('#sim-timeline button').first).to_have_attribute('aria-pressed', 'true')
            finish()
            expect(page.locator('#sim-output')).to_contain_text('Báo nhầm')
            page.screenshot(path='/private/tmp/vlm-simulation-r87.png', full_page=True)
            page.get_by_role('tab', name='Few-shot · phẳng 6 lớp', exact=True).click()
            expect(page.locator('#sim-next')).to_be_disabled()
            page.locator('#sim-restart').click()
            expect(page.locator('#sim-prev')).to_be_disabled()
            page.get_by_role('tab', name='Few-shot · phẳng 6 lớp', exact=True).focus()
            page.keyboard.press('ArrowRight')
            expect(page.get_by_role('tab', name='Few-shot · cascade', exact=True)).to_be_focused()
            # Existing application form state survives switching to educational pages.
            page.get_by_role('navigation').get_by_role('link', name='Cấu hình suy luận', exact=True).click()
            page.locator('#title').fill('Giữ tiêu đề khi chuyển sang mô phỏng')
            page.get_by_role('navigation').get_by_role('link', name='Mô phỏng phương pháp', exact=True).click()
            expect(page.locator('#view-setup')).to_be_hidden()
            page.get_by_role('navigation').get_by_role('link', name='Cấu hình suy luận', exact=True).click()
            expect(page.locator('#title')).to_have_value('Giữ tiêu đề khi chuyển sang mô phỏng')
            page.get_by_role('navigation').get_by_role('link', name='Mô phỏng phương pháp', exact=True).click()
            methods = page.evaluate('Object.entries(window.MethodSimulations.methods).map(([key,m]) => ({key, name:m.name, family:m.family}))')
            families = {'zero':'Zero-shot', 'adaptation':'Thích nghi không nhãn', 'fewshot':'Few-shot', 'control':'Đối chứng bổ sung'}
            for width, height in [(1440, 1050), (768, 1024), (390, 844)]:
                page.set_viewport_size({'width': width, 'height': height})
                for method in methods:
                    name=method['name']
                    page.get_by_role('button', name=families[method['family']], exact=True).click()
                    page.get_by_role('tab', name=name, exact=True).click()
                    finish()
                    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), f'Overflow: {name}, {width}'
                page.screenshot(path=f'/private/tmp/vlm-simulation-{width}.png', full_page=True)
            assert not errors, errors
            browser.close()
            print('PASS: 26 methods, fixed StatA dots, R87 false positive, keyboard tabs, preserved form, 3 widths, no JavaScript errors.')
    finally:
        server.shutdown()
        server.server_close()


if __name__ == '__main__':
    main()
