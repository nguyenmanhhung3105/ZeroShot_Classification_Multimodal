import unittest
import numpy as np
from scipy.special import softmax
from experiments.clip_lame.core import adapt,batches


class LameTests(unittest.TestCase):
    def test_no_pairwise_retains_prior_and_finite_probabilities(self):
        rng=np.random.default_rng(11);x=rng.normal(size=(13,5));scores=rng.uniform(-.02,.08,(13,3))
        p,_=adapt(x,scores,'knn',bound_lambda=0.)
        np.testing.assert_allclose(p,softmax(100*scores,axis=1),atol=2e-7,rtol=1e-5)
        np.testing.assert_allclose(p.sum(1),1,atol=1e-6)

    def test_neighbor_can_correct_ambiguous_sample_without_truth(self):
        x=np.array([[1.,0.],[.99,.01],[-1.,0.]])
        scores=np.array([[.08,0.],[-.001,.001],[0.,.08]])
        p,_=adapt(x,scores,'knn',knn=1)
        self.assertEqual(p.argmax(1).tolist(),[0,0,1])

    def test_complete_batch_assignment_and_degenerate_fallbacks(self):
        rng=np.random.default_rng(4);x=rng.normal(size=(13,5));scores=rng.uniform(0,.1,(13,2))
        p,ids,log=batches(x,scores,0,'knn',query_size=5)
        self.assertEqual(p.shape,(13,2));self.assertEqual(set(ids),{0,1,2});self.assertEqual(len(log),3)
        one,d=adapt([[1.,0.]],[[.2,.1]])
        self.assertTrue(d['singleton_prior_fallback']);self.assertEqual(one.argmax(1).tolist(),[0])
        _,d=adapt([[1.,0.]]*4,[[.2,.1]]*4)
        self.assertTrue(d['degenerate_prior_fallback'])

    def test_bad_inputs_rejected(self):
        for x,s in [([[0.,0.]],[[.1,.2]]),([[1.,0.]],[[float('nan'),.2]])]:
            with self.assertRaises(ValueError):adapt(x,s)


if __name__=='__main__':unittest.main()
