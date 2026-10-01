"""Regression against the dataset author's published IDs; no dataset access."""
import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from prompts import fakeddit_prompts as prompts
from inference import build_class_embeddings, predict_single_label
from test_runtime_fixes import FakeVLM


class FakedditTaxonomyTests(unittest.TestCase):
    def test_author_mapping_and_distinct_binary_ids(self):
        # https://github.com/entitize/Fakeddit/issues/14#issuecomment-815355653
        expected = {0: 'True', 1: 'Satire/Parody', 2: 'False Connection',
                    3: 'Imposter Content', 4: 'Manipulated Content', 5: 'Misleading Content'}
        self.assertEqual(prompts.get_label_names('6way'), expected)
        self.assertEqual(list(prompts.get_prompt_set('6way')), list(expected))
        self.assertEqual(prompts.get_label_names('2way'), {0: 'fake', 1: 'real'})
        self.assertEqual(prompts.normalize_labels(['0', '1', '2', '3', '4', '5'], '6way'), list(expected))

    def test_prompt_prototype_order_and_unit_norm(self):
        vectors, labels = build_class_embeddings(FakeVLM(), prompts.get_prompt_set('6way'))
        self.assertEqual(labels, list(range(6)))
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1)
        # Distinct synthetic class vectors ensure argmax maps to label IDs,
        # rather than assuming predicted index is the public label.
        model = FakeVLM()
        classes = np.array([[0., 1.], [1., 0.]])
        pred, *_ = predict_single_label(model, [None], ['text'], classes, [5, 2],
                                         image_weight=1, show_progress=False)
        self.assertEqual(pred, [2])


if __name__ == '__main__':
    unittest.main()
