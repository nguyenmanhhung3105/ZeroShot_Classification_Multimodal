"""Synthetic max-sim regressions; no project data or model loading."""
import unittest
import numpy as np
from test_text_chunking import ChunkVLM
from text_chunking import encode_chunked_max_sim, split_text, token_counter
from inference import predict_multi_label, multilabel_settings, relative_multilabel_scores


class MaxSimTests(unittest.TestCase):
    def test_max_is_per_label_and_batch_invariant(self):
        classes = np.array([[1., 0.], [0., 1.], [-1., 0.]])
        text = 'Short. A long sentence with several chunks. End.'
        model = ChunkVLM()
        count, context = token_counter(model)
        chunks = split_text(text, count, context)
        values = np.array([[1., 1. + len(t)] for t in chunks])
        values /= np.linalg.norm(values, axis=1, keepdims=True) + 1e-12
        expected = (values @ classes.T).max(axis=0)
        for batch in [1, 2, 20]:
            model = ChunkVLM()
            scores, counts = encode_chunked_max_sim(model, [text], batch, classes)
            np.testing.assert_allclose(scores[0], expected)
            self.assertEqual(counts, [len(chunks)])
            self.assertTrue(all(n <= batch for n in model.text_batches))

    def test_relative_pipeline_empty_and_unused_branches(self):
        classes = np.array([[1., 0.], [0., 1.], [-1., 0.]])
        texts = ['Long plot. Some other sentence. End.', '']
        args = ([None, None], texts, classes, ['a', 'b', 'c'])
        for weight in [0., .5, 1.]:
            model, audit = ChunkVLM(), {}
            result = predict_multi_label(model, *args, image_weight=weight,
                decision_mode='relative_z', chunk_long_text=True, chunk_aggregation='max_sim',
                diagnostics=audit, show_progress=False)
            self.assertEqual(audit['chunking']['aggregation'], 'max_sim')
            self.assertEqual(audit['chunking']['counts'][1], 0)
            np.testing.assert_array_equal(result[1][1], result[2][1])
            if weight == 1:
                self.assertEqual(model.seen, [])
            else:
                scores, _ = encode_chunked_max_sim(ChunkVLM(), texts[:1], 16, classes)
                expected = relative_multilabel_scores(np.zeros_like(scores), scores, 0)[2]
                np.testing.assert_allclose(result[3][:1], expected)
            self.assertEqual(sum(model.image_batches), 1 if weight == 0 else 2)

    def test_one_chunk_baseline_and_disabled(self):
        for text, chunking in [('short', True), ('A longer plot. Another part.', False)]:
            args = ([None], [text], np.eye(2), ['a', 'b'])
            for mode in ['relative_z', 'fixed_threshold']:
                base = predict_multi_label(ChunkVLM(), *args, chunk_long_text=chunking,
                                           decision_mode=mode, show_progress=False)
                new = predict_multi_label(ChunkVLM(), *args, chunk_long_text=chunking,
                    chunk_aggregation='max_sim', decision_mode=mode, show_progress=False)
                self.assertEqual(base[0], new[0])
                for a, b in zip(base[1:], new[1:]):
                    np.testing.assert_allclose(a, b)
        options, meta = multilabel_settings({'chunk_long_text': True, 'chunk_aggregation': 'max_sim'}, .22)
        self.assertEqual(options['chunk_aggregation'], 'max_sim')
        self.assertEqual(meta['chunk_aggregation'], 'max_sim')
        with self.assertRaises(ValueError):
            multilabel_settings({'chunk_aggregation': 'unknown'}, .22)


if __name__ == '__main__':
    unittest.main()
