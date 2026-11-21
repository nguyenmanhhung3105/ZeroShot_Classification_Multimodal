import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import subprocess

from PIL import Image
import yaml
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from preprocessing.data_fakeddit import prepare_splits as splits


class FakedditSplitTests(unittest.TestCase):
    def fixture(self, root):
        (root/'configs').mkdir()
        (root/'configs/experiment_config.yaml').write_text(yaml.safe_dump({
            'datasets': {'fakeddit': {'enabled': True}}, 'output': {}}))
        raw = root/'data/raw/fakeddit'; raw.mkdir(parents=True)
        images = root/'data/processed/pro_fakeddit/images'; images.mkdir(parents=True)
        for split_index, (logical, name) in enumerate(splits.SOURCES.items()):
            rows = []
            for label in range(6):
                for i in range(2):
                    key = f'{logical}_{label}_{i}'
                    rows.append({'id':key, 'clean_title':'A sufficiently long title for '+key,
                                 'image_url':'https://example.invalid/'+key+'.jpg', 'hasImage':'True',
                                 '2_way_label':'1' if label == 0 else '0', '6_way_label':str(label)})
                    Image.new('RGB',(64,64),(20+split_index*60, 10+label*30, 20+i*100)).save(images/(key+'.jpg'))
            with (raw/name).open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=splits.COLUMNS,delimiter='\t');w.writeheader();w.writerows(rows)
        return SimpleNamespace(output='data/processed/fakeddit_splits_v2', samples=6, seed=42,
                               base_config='configs/experiment_config.yaml', min_text_length=20,
                               min_side=32, near_distance=2, workers=2, timeout=1,
                               max_download_attempts=100, download=False, inspect=False)

    def test_three_splits_exact_quota_and_repeat_verification(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);args=self.fixture(root)
            report=splits.prepare(args,root)
            self.assertEqual(report['total_samples'],18)
            for part in report['splits'].values():self.assertEqual(part['support'],{str(i):1 for i in range(6)})
            self.assertEqual(splits.prepare(args,root)['status'],'passed')
            output=root/args.output
            manifest=json.loads((output/'train/manifest.json').read_text())
            self.assertEqual(manifest['source_split'],'train')
            config=yaml.safe_load((output/'dev/experiment.yaml').read_text())
            self.assertEqual(config['datasets']['fakeddit']['source_split'],'validate')
            self.assertEqual(config['datasets']['fakeddit']['label_col_2way'],'2_way_label')
            (output/'train/fakeddit_sample.tsv').write_text('corrupted')
            with self.assertRaisesRegex(ValueError,'checksum'):splits.verify(output,6,32,2)

    def test_missing_source_does_not_silently_split_train_or_publish(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);args=self.fixture(root)
            (root/'data/raw/fakeddit/multimodal_test_public.tsv').unlink()
            with self.assertRaises(FileNotFoundError):splits.prepare(args,root)
            self.assertFalse((root/args.output).exists())

    def test_dedup_tracks_pixels_urls_titles_and_near_hash(self):
        index=splits.DisjointIndex(2)
        row={'id':'one','image_url':'https://example.invalid/a','text_sha256':'t',
             'image_sha256':'b','pixel_sha256':'p','dhash':'123456789abcdef0'}
        index.add(row)
        self.assertEqual(index.metadata_reason({**row,'id':'two'}),'duplicate_image_url')
        self.assertEqual(index.image_reason({**row,'image_sha256':'x'}),'duplicate_image_pixels')
        self.assertEqual(index.image_reason({**row,'image_sha256':'x','pixel_sha256':'z','dhash':'123456789abcdef1'}),'near_duplicate_image')

    def test_network_failure_saves_state_and_does_not_publish(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);args=self.fixture(root);args.download=True
            for path in (root/'data/processed/pro_fakeddit/images').iterdir():path.unlink()
            with (patch.object(splits,'fetch_image',return_value=(None,'dns_unavailable')),
                  patch.object(splits,'network_available',return_value=False)):
                with self.assertRaises(ConnectionError):splits.prepare(args,root)
            self.assertFalse((root/args.output).exists())
            self.assertTrue((root/'data/processed/.fakeddit_splits_v2_build/state.json').is_file())

    def test_dead_source_host_is_replaced_within_same_class(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);args=self.fixture(root);args.download=True
            images=root/'data/processed/pro_fakeddit/images'
            stored=(images/'test_0_0.jpg').read_bytes()
            (images/'test_0_0.jpg').unlink();(images/'test_0_1.jpg').unlink()
            attempts=[]
            def fetch(row,cache,timeout):
                attempts.append(row['id'])
                if len(attempts)==1:return None,'dns_unavailable'
                target=cache/(row['id']+'.jpg');target.write_bytes(stored);return target,None
            with (patch.object(splits,'fetch_image',side_effect=fetch),
                  patch.object(splits,'network_available',return_value=True)):
                report=splits.prepare(args,root)
            self.assertEqual(len(attempts),2)
            self.assertEqual(report['splits']['test']['support']['0'],1)

    def test_downloader_enforces_total_timeout_and_cleans_partial_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache=Path(tmp);row={'id':'sample','image_url':'https://example.invalid/image'}
            def response(command,**kwargs):
                self.assertEqual(command[command.index('--max-time')+1],'2')
                self.assertEqual(kwargs['timeout'],4)
                target=Path(command[command.index('--output')+1])
                Image.new('RGB',(64,64),'blue').save(target,format='PNG')
                return SimpleNamespace(returncode=0,stdout='200\nhttps://example.invalid/image.png')
            with patch.object(splits.subprocess,'run',side_effect=response):
                path,error=splits.fetch_image(row,cache,2)
            self.assertIsNone(error);self.assertTrue(path.is_file());self.assertEqual(list(cache.glob('*.part')),[])
            with patch.object(splits.subprocess,'run',side_effect=subprocess.TimeoutExpired('curl',4)):
                self.assertEqual(splits.fetch_image({**row,'id':'timeout'},cache,2),(None,'download_timeout'))
            self.assertEqual(list(cache.glob('*.part')),[])


if __name__=='__main__':unittest.main()
