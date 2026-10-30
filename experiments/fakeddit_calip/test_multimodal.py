import unittest
import numpy as np
from experiments.fakeddit_calip.multimodal import fuse, replace_clip_vote, decide


class MultimodalTests(unittest.TestCase):
    def test_fusion_uses_probabilities_and_rejects_logits(self):
        result = fuse([[.8, .2]], [[.1, .9]], .75)
        np.testing.assert_allclose(result, [[.625, .375]])
        with self.assertRaises(ValueError): fuse([[8, 2]], [[.1, .9]], .75)
        with self.assertRaises(ValueError): fuse([[.8, .2]], [[.1, .3, .6]], .75)

    def test_text_changes_gate_only_when_other_two_voters_disagree(self):
        votes = np.array([[0, 0, 1], [0, 0, 0], [1, 1, 1]])
        before = votes.copy()
        pred, after = replace_clip_vote(votes, [[.1, .9], [.1, .9], [.9, .1]])
        np.testing.assert_array_equal(pred, [1, 0, 1])
        np.testing.assert_array_equal(after[:, 1:], before[:, 1:])
        np.testing.assert_array_equal(votes, before)

    def test_hierarchy_remains_consistent_with_multimodal_gate(self):
        rng = np.random.default_rng(3)
        norm = lambda x: x/np.linalg.norm(x, axis=-1, keepdims=True)
        x = norm(rng.normal(size=(5, 8)))
        arrays = {'image_features': x}
        text = {}; labels = {}; weights = {}
        for task, classes in [('2way', 2), ('6way', 6)]:
            means = norm(rng.normal(size=(classes, 8)))
            arrays[task+'_legacy_means'] = means*.8
            arrays[task+'_semantic_means'] = means*.7
            arrays[task+'_calip'] = 100*x@means.T
            text[task] = np.full((5, classes), 1/classes)
            labels[task] = [str(k) for k in range(classes)]; weights[task] = .75
        for seed in [0, 1, 2]: arrays[f'votes_seed_{seed}'] = rng.integers(0, 2, (5, 3))
        predictions, _, _, _ = decide(arrays, text, weights, labels)
        for seed in [0, 1, 2]:
            gate = predictions['2way'][f'majority_clip_text_seed_{seed}']
            for leaf in ['legacy', 'calip']:
                result = predictions['6way'][f'clip_text_gate_{leaf}_multimodal_leaf_seed_{seed}']
                np.testing.assert_array_equal(result == '0', gate == '1')
                self.assertTrue(set(result) <= set(labels['6way']))


if __name__ == '__main__': unittest.main()
