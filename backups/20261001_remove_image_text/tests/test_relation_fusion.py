import unittest
import numpy as np
from test_runtime_fixes import FakeVLM
from inference import predict_single_label
from pair_relation import fuse_false_connection, relation_fusion_settings


class RelationFusionTests(unittest.TestCase):
    def test_formula_can_change_label_and_preserves_missing(self):
        p = np.array([[.3, .1, .29, .1, .1, .11]] * 3)
        out = fuse_false_connection(p, [-1, 1, None], list(range(6)), .1)
        self.assertEqual(p[0].argmax(), 0)
        self.assertEqual(out[0].argmax(), 2)
        np.testing.assert_allclose(out.sum(axis=1), 1)
        np.testing.assert_array_equal(out[1:], p[1:])
        np.testing.assert_array_equal(fuse_false_connection(p, [-1]*3, list(range(6)), 0), p)

    def test_inference_output_and_diagnostic_are_consistent(self):
        classes = np.array([[1., 0.], [0., 1.], [.7, .7], [-1., 0.], [0., -1.], [-.7, -.7]])
        args = ([None, None], ['text', ''], classes, list(range(6)))
        base = predict_single_label(FakeVLM(), *args, show_progress=False)
        output = {}
        new = predict_single_label(FakeVLM(), *args, show_progress=False,
            relation_fusion_enabled=True, relation_fusion_weight=.1, relation_scores=output)
        np.testing.assert_allclose(new[1], fuse_false_connection(base[1], output['image_text_similarity'], list(range(6)), .1))
        np.testing.assert_array_equal(new[2], base[2])
        np.testing.assert_array_equal(new[3], base[3])
        self.assertEqual(new[0], new[1].argmax(axis=1).tolist())
        np.testing.assert_array_equal(new[1][1], base[1][1])

    def test_invalid_scope_and_weight(self):
        for dataset, task in [('fakeddit','2way'), ('crisismmd','humanitarian'), ('mmimdb','genres')]:
            with self.assertRaises(ValueError):
                relation_fusion_settings({'task':task,'relation_fusion_enabled':True}, dataset)
        for weight in [-1, 2, float('nan'), True]:
            with self.assertRaises(ValueError):
                relation_fusion_settings({'relation_fusion_weight':weight}, 'fakeddit')
