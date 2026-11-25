"""Synthetic features exercise demo decisions without loading models or datasets."""
import importlib.util
from pathlib import Path
import sys
import unittest
import tempfile
import json
import hashlib
import base64
import io
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


class DemoInferenceTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('vlm_project.demo_inference'),
                             'Missing single-query inference adapters')
        from vlm_project.demo_inference import linear_probability, cascade_decision, label_names
        self.linear = linear_probability
        self.route = cascade_decision
        self.labels = label_names

    def test_binary_and_multiclass_head_probabilities(self):
        p = self.linear(np.array([[1., 0.]]), np.array([[2., 0.]]), np.array([0.]))
        self.assertAlmostEqual(p[0, 1], 1 / (1 + np.exp(-2)))
        p = self.linear(np.eye(3), np.eye(3), np.zeros(3))
        np.testing.assert_allclose(p.sum(1), 1)
        np.testing.assert_array_equal(p.argmax(1), [0, 1, 2])
        with self.assertRaises(ValueError):
            self.linear(np.eye(3), np.eye(2), np.zeros(2))

    def test_cascade_uses_hard_gate_not_joint_argmax(self):
        predicted, joint = self.route(np.array([[.4, .6]]), np.ones((1, 5)) / 5)
        self.assertEqual(predicted, 1)
        self.assertEqual(int(joint.argmax()), 0)
        predicted, _ = self.route(np.array([[.5, .5]]), np.ones((1, 5)) / 5)
        self.assertEqual(predicted, 0)
        predicted, _ = self.route(np.array([[.5, .5]]), np.ones((1, 5)) / 5, .5)
        self.assertEqual(predicted, 1)

    def test_binary_prompt_and_fewshot_have_different_class_orders(self):
        self.assertEqual(self.labels('2way', fewshot=False), ['Fake', 'Real'])
        self.assertEqual(self.labels('2way', fewshot=True), ['Real', 'Fake'])
        self.assertEqual(len(self.labels('6way')), 6)

    def test_engine_shared_encoding_partial_results_and_unavailable_assets(self):
        from vlm_project.demo_inference import Engine
        from PIL import Image
        from vlm_project.demo_service import METHODS
        class Backend:
            def __init__(self):
                self.encodes = 0
                self.encodes_text = 0
            def encode_images(self, images):
                self.encodes += 1
                return np.array([[1., 0.]])
            def encode_texts(self, texts):
                self.encodes_text += 1
                return np.array([[0., 1.]])
            def prototypes(self, bank):
                x = np.zeros((len(bank), 2)); x[:, 0] = 1; x[1] = [0, 1]
                return x
            def unload(self): pass
        backend = Backend()
        engine = Engine(['metaclip2'], backend_factory=lambda _: backend)
        stream = io.BytesIO(); Image.new('RGB', (3, 3)).save(stream, format='PNG')
        payload = {'backbones': ['metaclip2'], 'image': base64.b64encode(stream.getvalue()).decode(),
                   'title': 'A title', 'task': '2way', 'pipeline': 'all',
                   'methods': [k for k in METHODS if k != 'calip']}
        updates = []
        result = engine.run(payload, lambda x: updates.append(json.loads(json.dumps(x))))
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['rows'][0]['predicted'], 'Fake')
        self.assertEqual(result['rows'][1]['predicted'], 'Real')
        self.assertEqual(backend.encodes, 1)
        self.assertTrue(any(r['status'] == 'unavailable' for r in result['rows']))
        self.assertGreater(len(updates), 3)
        engine.run({**payload, 'title': 'Changed title'}, lambda _: None)
        self.assertEqual(backend.encodes, 1)
        self.assertEqual(backend.encodes_text, 2)
        result = engine.run({**payload, 'backbones': ['openai_clip']}, lambda _: None)
        self.assertTrue(all(r['status'] == 'unavailable' for r in result['rows']))

    def test_saved_heads_policy_seed_and_asset_checksum(self):
        from vlm_project.demo_inference import Engine
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            arrays = {}
            for policy in ('legacy_6way', 'balanced_binary'):
                for seed in range(5):
                    for head, classes in [('flat', list(range(6))), ('gate', [0, 1]), ('child', list(range(1, 6)))]:
                        key = f'{head}_{policy}_s{seed}'
                        n = 1 if head == 'gate' else len(classes)
                        arrays[key + '_coef'] = np.zeros((n, 4))
                        arrays[key + '_intercept'] = np.zeros(n)
                        arrays[key + '_classes'] = np.array(classes)
            file = root / 'cascade.npz'; np.savez(file, **arrays)
            manifest = {'metaclip2': {'cascade': {'file': file.name, 'sha256': hashlib.sha256(file.read_bytes()).hexdigest()},
                                     'thresholds': {f'{p}_s{s}': .5 for p in ('legacy_6way', 'balanced_binary') for s in range(5)}}}
            (root / 'manifest.json').write_text(json.dumps(manifest))
            engine = Engine(['metaclip2'], root)
            x = np.array([[1., 0.]])
            result = engine.fewshot('fewshot_cascade', 'metaclip2', x, x, '6way')
            self.assertEqual(result['predicted'], 'Real')
            self.assertEqual(result['policy'], 'balanced_binary')
            self.assertEqual(len(result['variants']), 10)
            result = engine.fewshot('fewshot_cascade_r87', 'metaclip2', x, x, '6way')
            self.assertEqual(result['predicted'], 'Satire/Parody')
            engine.fewshot('fewshot_flat', 'metaclip2', x, x, '6way')
            with self.assertRaises(ValueError):
                engine.fewshot('fewshot_flat', 'metaclip2', x, x, '2way')
            file.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                engine.asset('metaclip2', 'cascade')

    def test_reference_group_runs_real_adaptation_and_preserves_method_identity(self):
        from vlm_project.demo_inference import Engine
        class Backend:
            def prototypes(self, bank):
                return np.eye(8)[:len(bank)]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            file = root / 'reference.npz'
            np.savez(file, image=np.eye(8)[2:6], title=np.eye(8)[2:6])
            (root / 'manifest.json').write_text(json.dumps({'metaclip2': {'reference': {
                'file': file.name, 'sha256': hashlib.sha256(file.read_bytes()).hexdigest()}}}))
            engine = Engine(['metaclip2'], root)
            backend = Backend(); image, title = np.eye(8)[:1], np.eye(8)[1:2]
            means = np.eye(8)[:2]
            # Real StatA is run on five unlabeled synthetic queries, never singleton fallback.
            for method in ('stata_image', 'clip_stata_image', 'clip_stata_image_title', 'em_image', 'majority'):
                result = engine.transductive(method, 'metaclip2', backend, image, title, '2way', means, None)
                self.assertIn(result['predicted'], ['Real', 'Fake'])
            engine.prediction_cache.clear()
            for method in ('hierarchy', 'hierarchy_em', 'hierarchy_majority', 'flat_stata', 'flat_majority'):
                result = engine.transductive(method, 'metaclip2', backend, image, title, '6way', np.eye(8)[:6], None)
                self.assertIn(result['predicted'], self.labels('6way'))
                self.assertIsNone(result['score'])

    def test_prompt_variants_and_binary_heads(self):
        from vlm_project.demo_inference import Engine
        class Backend:
            def prototypes(self, bank): return np.eye(6)[:len(bank)]
        engine = Engine(['metaclip2'])
        backend = Backend(); x = np.eye(6)[:1]
        for task in ('2way', '6way'):
            for key in ('dclip', 'waffle'):
                engine.prompt_banks[task, key] = engine.profile['image_prompts'][task]['legacy']
            for method in ('dclip', 'waffle', 'waffle_unit', 'legacy_image', 'legacy_title', 'legacy_fusion'):
                result = engine.predict_method(method, 'metaclip2', backend, x, x, {'task': task})
                self.assertIn(result['predicted'], self.labels(task))
        engine.predict_method('hierarchy_clip', 'metaclip2', backend, x, x, {'task': '6way'})
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); file = root / 'binary.npz'
            arrays = {}
            for seed in range(5):
                for modality, width in [('image', 6), ('title', 6), ('image_title', 12)]:
                    arrays[f'{modality}_s{seed}_coef'] = np.ones((1, width))
                    arrays[f'{modality}_s{seed}_intercept'] = np.zeros(1)
            np.savez(file, **arrays)
            (root / 'manifest.json').write_text(json.dumps({'metaclip2': {'binary': {
                'file': file.name, 'sha256': hashlib.sha256(file.read_bytes()).hexdigest()}}}))
            engine = Engine(['metaclip2'], root)
            for method in ('fewshot_image', 'fewshot_title', 'fewshot_image_title'):
                result = engine.fewshot(method, 'metaclip2', x, x, '2way')
                self.assertEqual(result['predicted'], 'Fake')
                self.assertEqual(result['policy'], 'legacy_6way')
                self.assertEqual(len(result['variants']), 5)


if __name__ == '__main__':
    unittest.main()
