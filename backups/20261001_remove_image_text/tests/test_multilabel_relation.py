"""Relation diagnostics with synthetic plots; no dataset/model access."""
import unittest
import numpy as np
from test_text_chunking import ChunkVLM
from inference import predict_multi_label
from text_chunking import encode_chunked_texts


class MultiRelationTests(unittest.TestCase):
    def test_all_modes_preserve_predictions_and_batch_counts(self):
        texts = ['Long plot. Another part. The final scene.', '']
        args = ([None, None], texts, np.eye(2), ['a', 'b'])
        for chunking in [False, True]:
            for aggregation in ['mean_embed', 'max_sim']:
                for mode in ['relative_z', 'fixed_threshold']:
                    for weight in [0., .5, 1.]:
                        with self.subTest(chunking=chunking, aggregation=aggregation, mode=mode, weight=weight):
                            options = dict(chunk_long_text=chunking, chunk_aggregation=aggregation,
                                decision_mode=mode, image_weight=weight, show_progress=False)
                            baseline_model, model = ChunkVLM(), ChunkVLM()
                            baseline = predict_multi_label(baseline_model, *args, **options)
                            output = {}
                            result = predict_multi_label(model, *args, **options,
                                image_text_relation_enabled=True, relation_scores=output)
                            self.assertEqual(result[0], baseline[0])
                            for a, b in zip(result[1:], baseline[1:]):
                                np.testing.assert_allclose(a, b)
                            self.assertIsNone(output['image_text_similarity'][1])
                            self.assertTrue(np.isfinite(output['image_text_similarity'][0]))
                            if weight < 1:
                                self.assertEqual(model.text_batches, baseline_model.text_batches)
                            if chunking:
                                vector, _ = encode_chunked_texts(ChunkVLM(), texts[:1], 16)
                                self.assertAlmostEqual(output['image_text_similarity'][0], vector[0, 0], places=9)


if __name__ == '__main__':
    unittest.main()
