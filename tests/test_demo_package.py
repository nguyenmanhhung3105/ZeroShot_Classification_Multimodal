"""Offline notebook packaging must exclude local data, secrets, and model weights."""
import base64
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
spec = importlib.util.spec_from_file_location('demo_cli', ROOT / 'scripts/demo.py')
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


class DemoPackageTests(unittest.TestCase):
    def test_notebook_is_private_gpu_and_contains_only_allowlisted_source(self):
        with tempfile.TemporaryDirectory() as temp:
            destination = demo.prepare(Path(temp) / 'bundle', 'test-owner', 'vlm-demo', ['metaclip2'])
            metadata = json.loads((destination / 'kernel-metadata.json').read_text())
            self.assertTrue(metadata['is_private'])
            self.assertTrue(metadata['enable_gpu'])
            self.assertTrue(metadata['enable_internet'])
            notebook = json.loads((destination / 'demo.ipynb').read_text())
            code = ''.join(notebook['cells'][1]['source'])
            compile(code, 'demo.ipynb', 'exec')
            # Find the literal embedded archive without evaluating the notebook.
            import ast
            tree = ast.parse(code)
            calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                     and isinstance(n.func, ast.Attribute) and n.func.attr == 'b64decode']
            content = base64.b64decode(ast.literal_eval(calls[0].args[0]))
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                names = archive.namelist()
            self.assertIn('src/vlm_project/demo_worker.py', names)
            self.assertTrue(all(n.startswith(('src/vlm_project/', 'configs/')) or n == 'requirements.txt' for n in names))
            self.assertFalse(any(n.endswith(('.npz', '.pth', '.key', '.env')) for n in names))
            with self.assertRaises(FileExistsError):
                demo.prepare(destination, 'test-owner', 'vlm-demo', ['metaclip2'])

    def test_export_only_completed_dev_and_keep_head_arrays(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source = root / 'run'; source.mkdir()
            run = {'model': 'metaclip2_vith14', 'mode': 'full', 'split': 'dev'}
            raw = json.dumps(run).encode(); (source / 'run.json').write_bytes(raw)
            (source / 'completed.json').write_text(json.dumps({'status': 'complete', 'run_sha256': hashlib.sha256(raw).hexdigest()}))
            np.savez(source / 'linear_models.npz', image_s0_coef=np.ones((1, 2)), image_s0_intercept=np.zeros(1))
            result = demo.export_assets(root / 'assets', binary=source)
            self.assertIn('binary', result['metaclip2'])
            np.savez(source / 'features.npz', train_image=np.eye(4), train_title=np.eye(4),
                     dev_image=np.full((4, 4), 99), dev_labels=np.arange(4))
            demo.export_assets(root / 'references', reference=source)
            with np.load(root / 'references/reference.npz', allow_pickle=False) as ref:
                self.assertEqual(set(ref.files), {'image', 'title'})
                np.testing.assert_array_equal(ref['image'], np.eye(4))
            (source / 'run.json').write_text(json.dumps({**run, 'split': 'test'}))
            with self.assertRaises(ValueError):
                demo.export_assets(root / 'rejected', binary=source)
