"""Synthetic relation diagnostics: no datasets or model weights."""
import unittest
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np
from test_runtime_fixes import FakeVLM
from inference import predict_single_label
from pair_relation import relation_settings, attach_relation
from logging_utils import save_single_label_log


class RelationTests(unittest.TestCase):
    def test_switches(self):
        self.assertEqual(relation_settings({}, .75)[0], .75)
        self.assertEqual(relation_settings({'image_label_enabled': False}, .75)[0], 0)
        self.assertEqual(relation_settings({'text_label_enabled': False}, .75)[0], 1)
        for cfg in [{'image_label_enabled': False, 'text_label_enabled': False},
                    {'image_text_relation_enabled': 'false'}]:
            with self.assertRaises(ValueError):
                relation_settings(cfg, .5)

    def test_relation_does_not_change_scores_or_predictions(self):
        args = ([None, None], ['text', ''], np.eye(2), ['a', 'b'])
        for weight in [0, .5, 1]:
            baseline = predict_single_label(FakeVLM(), *args, image_weight=weight, show_progress=False)
            model, output = FakeVLM(), {}
            result = predict_single_label(model, *args, image_weight=weight,
                image_text_relation_enabled=True, relation_scores=output, show_progress=False)
            self.assertEqual(result[0], baseline[0])
            for a, b in zip(result[1:], baseline[1:]):
                np.testing.assert_array_equal(a, b)
            self.assertEqual(output['image_text_similarity'], [0., None])
            self.assertEqual(model.image_batches, [2])
            self.assertEqual(model.text_batches, [1])

    def test_relation_saved_without_evaluation(self):
        frame = pd.DataFrame({'id': ['one', 'two'], 'text': ['text', '']})
        attach_relation(frame, {'image_text_similarity': [.2, None]})
        with tempfile.TemporaryDirectory() as directory:
            root = save_single_label_log(frame, ['a', 'a'], ['a', 'b'],
                np.array([[.8, .2], [.3, .7]]), ['a', 'b'], 'id', 'text',
                'fake', 'fakeddit', '6way', directory)
            saved = pd.read_csv(Path(root) / 'raw_predictions.tsv', sep='\t')
            self.assertEqual(saved.iloc[0]['image_text_similarity'], .2)
            self.assertTrue(pd.isna(saved.iloc[1]['image_text_similarity']))


if __name__ == '__main__':
    unittest.main()
