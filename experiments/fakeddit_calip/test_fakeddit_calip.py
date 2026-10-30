import csv
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from scipy.special import softmax
import torch
import yaml
from experiments.fakeddit_calip import run as runner
from experiments.fakeddit_calip.core import hierarchy, majority
from experiments.clip_calip.test_calip import TinyBackend
from experiments.clip_calip.extract import encode
from experiments.waffle_lab import run as lab


class FakedditCalipTests(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads(); torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, self.threads)

    def test_hierarchy_uses_opposite_true_ids_and_excludes_true_for_fake(self):
        scores = np.array([[100, 1, 2, 3, 4, 5], [0, 10, 0, 0, 0, 0]])
        pred = hierarchy(['0', '1'], scores, [str(k) for k in range(6)])
        self.assertEqual(pred.tolist(), ['5', '0'])

    def test_majority_is_reproducible_label_free_and_uses_three_votes(self):
        rng = np.random.default_rng(8)
        x = rng.normal(size=(8, 6)).astype('float32'); x /= np.linalg.norm(x, axis=1, keepdims=True)
        means = x[:2]*.8
        a, meta = majority(x, means, 0); b, _ = majority(x, means, 0)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(a, (meta['votes'].sum(1) >= 2).astype(int))
        self.assertEqual(meta['batch_ids'].tolist(), [0]*8)

    def test_launcher_routes_both_tasks_and_rejects_wrong_backbone(self):
        sys.path.insert(0, str(runner.ROOT/'scripts'))
        import launch_runs as launcher
        config = yaml.safe_load((runner.ROOT/'experiments/fakeddit_calip/launcher.yaml').read_text())
        launcher.validate(config, ['fakeddit_calip'])
        with tempfile.TemporaryDirectory() as tmp:
            _, _, output, command, _, _, research = launcher.job_plan(config, 'fakeddit_calip', Path(tmp))
            self.assertEqual(command[0], 'experiments/fakeddit_calip/run.py')
            self.assertEqual(output.parts[1], 'fakeddit_calip')
            self.assertIsNone(research)
        bad = deepcopy(config); bad['jobs']['fakeddit_calip']['models'] = ['openai_clip_vit_b_32']
        with self.assertRaises(ValueError): launcher.validate(bad, ['fakeddit_calip'])

    def test_end_to_end_and_reject_modified_baseline(self):
        torch.manual_seed(2); backend = TinyBackend()
        rng = np.random.default_rng(4)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); folder = root/'data/processed/fixture'; folder.mkdir(parents=True)
            (root/'configs/prompts').mkdir(parents=True)
            semantic = root/runner.SEMANTIC; semantic.write_text('{}')
            images = [Image.fromarray(rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)) for _ in range(5)]
            rows = []
            for i, image in enumerate(images):
                path = folder/f'{i}.png'; image.save(path)
                rows.append({'id': str(i), 'image_path': str(path.relative_to(root)),
                             'image_stat': {'size': path.stat().st_size}})
            g, _, _ = encode(backend, images, verify_global=True); x = g.numpy()
            for image in images: image.close()
            payload = {'model': 'metaclip2_vith14', 'semantic_sha256': lab.file_hash(semantic), 'tasks': {}}
            backend.texts = {}
            for task, classes in [('2way', 2), ('6way', 6)]:
                labels = [str(k) for k in range(classes)]; bank = {}; means = []
                for label in labels:
                    text = task+'_'+label; vector = rng.normal(size=8).astype('float32'); vector /= np.linalg.norm(vector)
                    backend.texts[text] = vector; bank[label] = [text]; means.append(vector)
                probs = softmax(100*x@np.asarray(means).T, axis=1)
                truth = ['1', '0', '0', '1', '0'] if classes == 2 else ['0', '1', '2', '0', '5']
                payload['tasks'][task] = {'labels': labels, 'banks': {k: bank for k in ['legacy', 'semantic', 'semantic_waffle']},
                                         'rows': [{**row, 'label': truth[i], 'baseline_pred': truth[i],
                                                   'baseline_image_prob': probs[i].tolist()} for i, row in enumerate(rows)],
                                         'logit_scale': 100., 'source_split': 'validate'}
            ref = {'payload': payload, 'sha256': lab.digest(payload)}
            (root/runner.ASSET).write_text(json.dumps(ref))
            job = yaml.safe_load((runner.ROOT/'experiments/fakeddit_calip/launcher.yaml').read_text())['jobs']['fakeddit_calip']
            job['max_samples'] = 5
            base = {'datasets': {'fakeddit': {'image_dir': 'data/processed/fixture', 'source_split': 'validate'}}}
            with patch.object(runner, 'ROOT', root), patch.object(lab, 'ROOT', root), patch.object(runner, 'load_backend', return_value=backend):
                result = runner.run(base, job, root/'result')
                self.assertEqual(json.loads((result/'run_manifest.json').read_text())['status'], 'complete')
                with (result/'summary.tsv').open() as stream:
                    summary = list(csv.DictReader(stream, delimiter='\t'))
                self.assertEqual({r['task'] for r in summary}, {'2way', '6way'})
                self.assertTrue(any(r['method'] == 'majority_calip_mean' for r in summary))
                payload['tasks']['2way']['rows'][0]['baseline_image_prob'] = [.5, .5]
                (root/runner.ASSET).write_text(json.dumps({'payload': payload, 'sha256': lab.digest(payload)}))
                with self.assertRaises(AssertionError): runner.run(base, job, root/'bad')
                self.assertEqual(json.loads((root/'bad/run_manifest.json').read_text())['status'], 'failed')


if __name__ == '__main__': unittest.main()
