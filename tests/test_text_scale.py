"""Text scale tests with synthetic embeddings, no datasets or model weights."""
import unittest
import numpy as np
from test_runtime_fixes import FakeVLM
from inference import predict_single_label, predict_multi_label
from text_scale import text_scale_settings, validate_text_scale


class TextScaleTests(unittest.TestCase):
    def test_config_override_and_relative_status(self):
        config = {'inference': {'text_logit_scale': 50,
                  'text_logit_scale_overrides': {'fakeddit': 25, 'crisismmd': None, 'mmimdb': 25}}}
        model = FakeVLM()
        value, meta = text_scale_settings(config, 'fakeddit', model, {}, .75)
        self.assertEqual(value, 25)
        self.assertEqual(meta['text_logit_scale_effective'], 25)
        value, meta = text_scale_settings(config, 'crisismmd', model, {}, .5)
        self.assertIsNone(value)
        self.assertEqual(meta['text_logit_scale_effective'], model.logit_scale)
        _, meta = text_scale_settings(config, 'mmimdb', model, {}, .5)
        self.assertEqual(meta['text_logit_scale_status'], 'ignored_relative_z')
        self.assertIsNone(meta['text_logit_scale_effective'])
        _, meta = text_scale_settings(config, 'mmimdb', model, {'decision_mode': 'fixed_threshold'}, .5)
        self.assertEqual(meta['text_logit_scale_effective'], 25)
        _, meta = text_scale_settings(config, 'fakeddit', model, {}, 1)
        self.assertEqual(meta['text_logit_scale_status'], 'unused_image_only')

    def test_invalid_scales(self):
        for value in [True, False, 0, -1, float('nan'), float('inf'), '25']:
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_text_scale(value)

    def test_scale_changes_only_text_and_default_is_identical(self):
        args = ([None, None], ['some text', ''], np.eye(2), ['a', 'b'])
        baseline = predict_single_label(FakeVLM(), *args, show_progress=False)
        explicit = predict_single_label(FakeVLM(), *args, text_logit_scale=2, show_progress=False)
        smaller = predict_single_label(FakeVLM(), *args, text_logit_scale=.5, show_progress=False)
        for a, b in zip(baseline[1:], explicit[1:]):
            np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(baseline[2], smaller[2])
        np.testing.assert_array_equal(baseline[1][1], smaller[1][1])
        self.assertLess(smaller[3][0].max(), baseline[3][0].max())
        np.testing.assert_allclose(smaller[3].sum(axis=1), 1)
        for mode in ['relative_z', 'fixed_threshold']:
            old = predict_multi_label(FakeVLM(), *args, decision_mode=mode, show_progress=False)
            new = predict_multi_label(FakeVLM(), *args, decision_mode=mode, text_logit_scale=.5, show_progress=False)
            np.testing.assert_array_equal(old[2], new[2])
            if mode == 'relative_z':
                for a, b in zip(old[1:], new[1:]):
                    np.testing.assert_array_equal(a, b)
            else:
                self.assertFalse(np.array_equal(old[3], new[3]))


if __name__ == '__main__':
    unittest.main()
