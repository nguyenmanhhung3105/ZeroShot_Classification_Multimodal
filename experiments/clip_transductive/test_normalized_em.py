import unittest
import numpy as np
from experiments.clip_transductive.normalized_em import normalize_mean_scores


class NormalizedEmTests(unittest.TestCase):
    def test_score_division_equals_unit_prototype_cosine_without_labels(self):
        rng=np.random.default_rng(7)
        images=rng.normal(size=(12,8));images/=np.linalg.norm(images,axis=1,keepdims=True)
        prompts=rng.normal(size=(2,3,8));prompts/=np.linalg.norm(prompts,axis=2,keepdims=True)
        prototypes=prompts.mean(1);norms=np.linalg.norm(prototypes,axis=1)
        raw=(images@prompts.reshape(6,8).T).reshape(12,2,3).mean(2)
        np.testing.assert_allclose(normalize_mean_scores(raw,norms),images@(prototypes/norms[:,None]).T,atol=1e-12)
        with self.assertRaises(ValueError):normalize_mean_scores(raw,[0,1])
        with self.assertRaises(ValueError):normalize_mean_scores(np.ones((2,2)),[.1,.1])


if __name__=='__main__':unittest.main()
