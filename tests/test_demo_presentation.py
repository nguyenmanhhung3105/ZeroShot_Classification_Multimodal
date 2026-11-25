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
    def test_independent_report_views_and_unique_controls(self):
        document = Document((WEB / 'index.html').read_text())
        for view in ('overview', 'setup', 'results', 'experiments', 'history'):
            self.assertIn('view-' + view, document.ids)
            self.assertIn('view-' + view, document.navigation)
        self.assertEqual(len(document.ids), len(set(document.ids)))

    def test_professional_project_identity_and_no_unverified_metrics(self):
        text = ' '.join(Document((WEB / 'index.html').read_text()).content)
        self.assertIn('Phân loại nội dung đa phương thức', text)
        self.assertNotIn('Một bài đăng.', text)
        self.assertNotIn('Nhiều góc nhìn.', text)
        self.assertNotIn('82,41%', text)


if __name__ == '__main__':
    unittest.main()
