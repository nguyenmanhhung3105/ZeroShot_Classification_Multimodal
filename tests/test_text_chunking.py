"""Synthetic only: no dataset reads, weights or network."""
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
import tempfile
import json
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from text_chunking import split_text, token_counter, encode_chunked_texts
from inference import predict_multi_label, multilabel_settings
from test_runtime_fixes import FakeVLM, Tensor
from evaluation_audit import prepare_audit, finish_audit, save_audit


class SimpleTokenizer:
    def encode(self, text):
        return list(text)


class ChunkVLM(FakeVLM):
    tokenizer = SimpleTokenizer()
    model = SimpleNamespace(context_length=12)

    def __init__(self):
        super().__init__()
        self.seen = []

    def encode_texts(self, texts):
        self.seen.extend(texts)
        self.text_batches.append(len(texts))
        return Tensor([[1., 1. + len(t)] for t in texts])


class ChunkTests(unittest.TestCase):
    def test_sentences_words_unicode_and_no_loss(self):
        count, limit = token_counter(ChunkVLM())
        self.assertEqual(split_text('Hello. Bye.', count, limit), ['Hello.', 'Bye.'])
        for text in ['Short', 'One long sentence with many words', 'x' * 57,
                     '你好世界' * 12, 'First.\nSecond! Third?']:
            chunks = split_text(text, count, limit)
            self.assertEqual(''.join(''.join(chunks).split()), ''.join(text.split()))
            self.assertTrue(all(count(c) <= limit for c in chunks))
        self.assertEqual(split_text(' ', count, limit), [])

    def test_mean_normalized_bounded_batches(self):
        model = ChunkVLM()
        embeddings, counts = encode_chunked_texts(model, ['a ' * 55, 'short'], 2)
        self.assertGreater(counts[0], 1)
        self.assertEqual(counts[1], 1)
        self.assertTrue(all(n <= 2 for n in model.text_batches))
        self.assertAlmostEqual(np.linalg.norm(embeddings[0]), 1.)
        first = model.seen[:counts[0]]
        vectors = np.array([[1., 1. + len(t)] for t in first])
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        expected = vectors.mean(axis=0)
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(embeddings[0], expected)

    def test_baseline_short_texts_and_empty_fallback(self):
        args = ([None, None], ['short', ''], np.eye(2), ['a', 'b'])
        old = predict_multi_label(ChunkVLM(), *args, show_progress=False)
        audit = {}
        new = predict_multi_label(ChunkVLM(), *args, show_progress=False,
                                  chunk_long_text=True, diagnostics=audit)
        self.assertEqual(old[0], new[0])
        for a, b in zip(old[1:], new[1:]):
            np.testing.assert_array_equal(a, b)
        self.assertEqual(audit['chunking']['counts'], [1, 0])
        image_only = ChunkVLM()
        predict_multi_label(image_only, *args, image_weight=1, chunk_long_text=True,
                            show_progress=False)
        self.assertEqual(image_only.seen, [])

    def test_disabled_and_unsupported_tokenizer(self):
        model = FakeVLM()
        predict_multi_label(model, [None], ['long ' * 30], np.eye(2), ['a', 'b'],
                            show_progress=False, chunk_long_text=False)
        with self.assertRaises(ValueError):
            token_counter(model)
        options, meta = multilabel_settings({'chunk_long_text': True}, .22)
        self.assertTrue(options['chunk_long_text'])
        self.assertEqual(meta['chunk_aggregation'], 'mean_embed')
        with self.assertRaises(ValueError):
            multilabel_settings({'chunk_long_text': 'false'}, .22)

    def test_hf_special_tokens(self):
        class HFTokenizer:
            tokenizer_mode = ''
            clean_fn = staticmethod(str.lower)
            tokenizer = SimpleNamespace(encode=lambda text, **kwargs: [0] + list(text) + [1])
        model = SimpleNamespace(tokenizer=HFTokenizer(), model=SimpleNamespace(context_length=8))
        count, context = token_counter(model)
        self.assertEqual(count('HELLO'), 7)
        self.assertTrue(all(count(c) <= context for c in split_text('ABCDEF GHIJKL', count, context)))

    def test_chunk_audit_files_and_unchanged_image_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            texts = ['A long plot. Another sentence. Ending.', 'short']
            frame = pd.DataFrame({'id': ['one', 'two'], 'text': texts, 'label': ['a', 'b']})
            config = {'evaluation': {'enabled': True, 'sample_lock_mode': 'off'}}
            audit = prepare_audit(frame, {}, config, 'mmimdb', 'genres',
                                  'id', 'text', 'label', {'a': ['a'], 'b': ['b']})
            model = ChunkVLM()
            options, meta = multilabel_settings({'chunk_long_text': True}, .22)
            args = ([None, None], texts, np.eye(2), ['a', 'b'])
            predictions, fused, image, text, fallback, empty = predict_multi_label(
                model, *args, **options, diagnostics=audit, show_progress=False)
            baseline = predict_multi_label(ChunkVLM(), *args, decision_mode='relative_z',
                                            show_progress=False)
            np.testing.assert_array_equal(image, baseline[2])
            finish_audit(audit, model, frame, texts, [['a'], ['b']], ['a', 'b'],
                         fused, image, text, empty, .5, multilabel=True, decision_meta=meta)
            save_audit(audit, directory)
            lengths = pd.read_csv(Path(directory) / 'text_lengths.tsv', sep='\t')
            self.assertGreater(lengths.iloc[0]['n_chunks'], 1)
            self.assertTrue(lengths.iloc[0]['truncated'])
            self.assertFalse(lengths['encoded_chunk_truncated'].any())
            snapshot = json.loads((Path(directory) / 'evaluation_snapshot.json').read_text())
            self.assertTrue(snapshot['effective_inference']['chunk_long_text'])
            self.assertEqual(snapshot['chunking']['n_chunked_texts'], 1)


if __name__ == '__main__':
    unittest.main()
