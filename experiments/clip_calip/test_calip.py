import csv
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import yaml
from experiments.clip_calip.core import score
from experiments.clip_calip.extract import encode
from experiments.clip_calip import run as runner
from experiments.clip_calip.analyze import analyze
from experiments.clip_calip.prepare import ASSET, load_reference
from experiments.waffle_lab import run as lab


class TinyModel(torch.nn.Module):
    def __init__(self, after=False):
        super().__init__()
        from open_clip.transformer import VisionTransformer
        self.visual = VisionTransformer(image_size=16, patch_size=4, width=16, layers=1,
                                        heads=2, mlp_ratio=2, output_dim=8, final_ln_after_pool=after)
        self.eval()

    def encode_image(self, x): return self.visual(x)


class TinyBackend:
    def __init__(self, after=False):
        self.model = TinyModel(after); self.device = 'cpu'; self.logit_scale = 100.; self.closed = False

    def preprocess(self, image):
        return torch.tensor(np.array(image.resize((16, 16))).transpose(2, 0, 1), dtype=torch.float32)/255.

    def encode_texts(self, prompts): return torch.tensor(np.stack([self.texts[p] for p in prompts]))
    def unload(self): self.closed = True


class CalipTests(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads(); torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, self.threads)

    def test_local_evidence_changes_ambiguous_global_prediction(self):
        g = F.normalize(torch.ones(1, 2), dim=-1); t = torch.eye(2)
        p1 = t[0].reshape(1, 1, 2).repeat(1, 4, 1)
        p2 = t[1].reshape(1, 1, 2).repeat(1, 4, 1)
        a, _, _ = score(g, p1, t); b, _, _ = score(g, p2, t)
        self.assertEqual(int(a.argmax()), 0); self.assertEqual(int(b.argmax()), 1)

    def test_permutations_and_invalid_features(self):
        rng = np.random.default_rng(2)
        norm = lambda shape: F.normalize(torch.tensor(rng.normal(size=shape), dtype=torch.float32), dim=-1)
        g, p, t = norm((3, 8)), norm((3, 9, 8)), norm((4, 8))
        a, _, att = score(g, p, t); b, _, _ = score(g, p.flip(1), t.flip(0))
        torch.testing.assert_close(a, b.flip(1)); torch.testing.assert_close(att.sum(1), torch.ones(3, 4))
        with self.assertRaises(ValueError): score(g*2, p, t)
        with self.assertRaises(ValueError): score(g, p[:, :, :4], t)

    def test_openclip_extractor_preserves_global_and_projects_both_ln_orders(self):
        torch.manual_seed(3)
        image = Image.new('RGB', (16, 16), 'red')
        try:
            for after in [False, True]:
                backend = TinyBackend(after)
                g, p, d = encode(backend, [image], verify_global=True)
                self.assertEqual(p.shape, (1, 16, 8)); self.assertEqual(g.shape, (1, 8))
                self.assertEqual(d['global_reference_max_abs_error'], 0.)
                self.assertFalse(backend.model.visual.output_tokens)
                torch.testing.assert_close(p.norm(dim=-1), torch.ones(1, 16))
                with patch.object(backend.model.visual, 'forward', side_effect=RuntimeError('fixture')):
                    with self.assertRaises(RuntimeError): encode(backend, [image])
                self.assertFalse(backend.model.visual.output_tokens)
        finally: image.close()

    def test_launcher_routes_and_packages_calip_without_changing_final_plan(self):
        sys.path.insert(0, str(runner.ROOT/'scripts')); import launch_runs as launcher
        from experiments.clip_final.protocol import load_plan, PLAN
        load_plan(runner.ROOT/PLAN)
        for filename, name, folder in [('launcher.yaml', 'calip_train', 'clip_calip'),
                                      ('launcher_dev.yaml', 'calip_dev', 'clip_calip_heldout')]:
            config = yaml.safe_load((runner.ROOT/'experiments/clip_calip'/filename).read_text())
            launcher.validate(config, [name])
            with tempfile.TemporaryDirectory() as tmp:
                _, _, output, command, _, _, research = launcher.job_plan(config, name, Path(tmp))
                self.assertEqual(command[0], 'experiments/clip_calip/run.py')
                self.assertEqual(output.parts[1], folder)
                self.assertTrue(yaml.safe_load(research.read_text())['calip'])
            bad = deepcopy(config); bad['jobs'][name]['multimodal'] = True
            with self.assertRaises(ValueError): launcher.validate(bad, [name])
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp)/'code.zip'; launcher.archive_code(archive)
            with zipfile.ZipFile(archive) as z:
                self.assertIn(ASSET, z.namelist()); self.assertIn('experiments/clip_calip/run.py', z.namelist())

    def fixture(self, root, split):
        torch.manual_seed(9); backend = TinyBackend()
        original = load_reference(runner.ROOT/ASSET)['payload']
        rng = np.random.default_rng(15)
        norm = lambda n: F.normalize(torch.tensor(rng.normal(size=(n, 8)), dtype=torch.float32), dim=-1).numpy()
        info, human = norm(2), norm(8)
        backend.texts = {}
        for task, vectors in [('informativeness', info), ('humanitarian', human)]:
            for i, prompts in enumerate(original['banks'][task].values()):
                for prompt in prompts: backend.texts[prompt] = vectors[i]
        folder = root/'data/processed/fixture'; folder.mkdir(parents=True)
        images = []; entries = []
        il, hl = original['info_labels'], original['human_labels']
        for i in range(5):
            image = Image.fromarray(rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8))
            path = folder/f'{100+i}_0.png'; image.save(path); images.append(image)
            entries.append({'image_id': f'{100+i}_0', 'image_path': str(path.relative_to(root)),
                            'label_informative': il[i%2], 'label_humanitarian': hl[i%8],
                            'source_split': split, 'label_source': 'image'})
        with torch.no_grad():
            x = backend.model.encode_image(torch.stack([backend.preprocess(im) for im in images]))
            x = F.normalize(x.float(), dim=-1).numpy(); x /= np.linalg.norm(x, axis=1, keepdims=True)
        for image in images: image.close()
        tsv = folder/'samples.tsv'
        with tsv.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(entries[0]), delimiter='\t'); w.writeheader(); w.writerows(entries)
        scope = {'samples': 5, 'ids': [r['image_id'] for r in entries],
                 'image_sha256': [lab.file_hash(root/r['image_path']) for r in entries],
                 'human_truth': [r['label_humanitarian'] for r in entries],
                 'binary_truth': [r['label_informative'] for r in entries] if split == 'train' else None,
                 'info_scores': (x@info.T).tolist(), 'human_scores': (x@human.T).tolist(),
                 'dn_leaf': (x@human.T).tolist(),
                 'majority_gates': {str(seed): [bool(i%2 == seed%2) for i in range(5)] for seed in [0, 1, 2]}}
        payload = {'model': 'metaclip2_vith14', 'prompt_seed': 0, 'banks': original['banks'],
                   'info_labels': il, 'human_labels': hl, 'info_mean_prototypes': info.tolist(),
                   'human_mean_prototypes': human.tolist(), 'cohorts': {'train': scope, 'dev': scope}}
        ref = root/ASSET; ref.parent.mkdir(parents=True)
        ref.write_text(json.dumps({'payload': payload, 'sha256': lab.digest(payload)}))
        data = {'data_path': str(tsv.relative_to(root)), 'image_dir': str(folder.relative_to(root)),
                'id_col': 'image_id', 'label_col_info': 'label_informative', 'label_col_human': 'label_humanitarian',
                'source_split': split, 'gate_labels_available': split == 'train'}
        config = root/'configs/active.yaml'; config.write_text(yaml.safe_dump({'datasets': {'crisismmd': data}}))
        settings = yaml.safe_load((runner.ROOT/'experiments/clip_two_stage/config.yaml').read_text())
        settings.update(calip=True, base_config=str(config), max_samples=5, batch_size=2,
                        output_dir=str(root/'result'), evaluation_scope='exploratory' if split == 'train' else 'heldout')
        return backend, settings

    def test_end_to_end_train_dev_zip_audit_and_tampering(self):
        for split in ['train', 'dev']:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); backend, settings = self.fixture(root, split)
                with patch.object(lab, 'ROOT', root), patch.object(runner, 'load_backend', return_value=backend):
                    output = runner.run(settings)
                self.assertTrue(backend.closed)
                with self.assertRaisesRegex(ValueError, 'synthetic'): analyze(output)
                report = analyze(output, allow_synthetic=True)
                self.assertEqual(report['status'], 'passed')
                self.assertEqual(report['mean']['majority_calip_leaf']['binary'] is None, split == 'dev')
                cloud = root/'cloud'; cloud.mkdir()
                with zipfile.ZipFile(cloud/'run_artifacts.zip', 'w') as z:
                    for p in output.rglob('*'):
                        if p.is_file(): z.write(p, str(p.relative_to(output)))
                self.assertEqual(analyze(cloud, allow_synthetic=True)['status'], 'passed')
                with (output/'predictions_seed_0.tsv').open('a') as f: f.write('tampered\n')
                with self.assertRaisesRegex(ValueError, 'hash'): analyze(output, allow_synthetic=True)

    def test_prepare_only_and_reference_mismatch_do_not_load_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); backend, settings = self.fixture(root, 'train')
            with patch.object(lab, 'ROOT', root), patch.object(runner, 'load_backend') as loader, \
                 patch.object(lab, 'read_cohort') as reader:
                runner.run(settings, prepare_only=True); loader.assert_not_called(); reader.assert_not_called()
            settings['output_dir'] = str(root/'bad')
            ref = json.loads((root/ASSET).read_text()); ref['payload']['cohorts']['train']['image_sha256'][0] = 'changed'
            ref['sha256'] = lab.digest(ref['payload']); (root/ASSET).write_text(json.dumps(ref))
            with patch.object(lab, 'ROOT', root), patch.object(runner, 'load_backend') as loader:
                with self.assertRaisesRegex(ValueError, 'Cohort'): runner.run(settings)
                loader.assert_not_called()


if __name__ == '__main__': unittest.main()
