"""Report UI contract: separate views, technical copy and grouped comparisons."""
from html.parser import HTMLParser
from pathlib import Path
import unittest

WEB = Path(__file__).resolve().parents[1] / 'src/vlm_project/demo_web'


class Document(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.ids, self.navigation, self.content = [], [], []
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if 'id' in attrs:
            self.ids.append(attrs['id'])
        if tag == 'a' and attrs.get('href', '').startswith('#view-'):
            self.navigation.append(attrs['href'][1:])

    def handle_data(self, data):
        self.content.append(data)


class PresentationTests(unittest.TestCase):
    def test_simulation_keeps_stage_visible_without_preface_banners(self):
        html = (WEB / 'index.html').read_text()
        self.assertNotIn('Theo dõi riêng từng phương pháp, từ bài đăng đầu vào đến nhãn dự đoán.', html)
        self.assertNotIn('Mọi ảnh vẽ, điểm và phân bố ở đây là dữ liệu giả lập', html)
        for identity in ('sim-play', 'sim-timeline', 'sim-details'):
            self.assertIn('id="' + identity + '"', html)

    def test_independent_report_views_and_unique_controls(self):
        document = Document((WEB / 'index.html').read_text())
        for view in ('overview', 'setup', 'results', 'experiments', 'simulations', 'history'):
            self.assertIn('view-' + view, document.ids)
            self.assertIn('view-' + view, document.navigation)
        self.assertEqual(len(document.ids), len(set(document.ids)))
        html = (WEB / 'index.html').read_text()
        navigation = Document(html.split('<nav aria-label="Điều hướng báo cáo">')[1].split('</nav>')[0]).navigation
        self.assertEqual(navigation[navigation.index('view-overview') + 1], 'view-simulations')

    def test_professional_project_identity_and_no_unverified_metrics(self):
        text = ' '.join(Document((WEB / 'index.html').read_text()).content)
        self.assertIn('Phân loại nội dung đa phương thức', text)
        self.assertNotIn('Một bài đăng.', text)
        self.assertNotIn('Nhiều góc nhìn.', text)
        self.assertNotIn('82,41%', text)


if __name__ == '__main__':
    unittest.main()
