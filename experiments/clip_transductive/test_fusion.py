import unittest
import numpy as np
from experiments.clip_transductive.fusion import name_binary_assignments,binary_em_probabilities,soft_mean,protect
from experiments.clip_transductive.gate import predict_binary
from experiments.clip_transductive.ensemble import humanitarian


class FusionTests(unittest.TestCase):
    def test_cluster_mapping_swaps_columns_and_completes_empty_cluster(self):
        p=np.array([[[.8,.2],[.6,.4]]])
        expected=np.array([[[.2,.8],[.4,.6]]])
        np.testing.assert_allclose(name_binary_assignments(p,[{0:1,1:0}]),expected)
        np.testing.assert_allclose(name_binary_assignments(p,[{0:1}]),expected)
        with self.assertRaises(ValueError):name_binary_assignments(p,[{0:0,1:0}])

    def test_soft_named_em_preserves_hard_control_and_all_rows(self):
        scores=np.random.default_rng(19).uniform(-.1,.2,(13,2));labels=['informative','not_informative']
        p,hard,groups,ties=binary_em_probabilities(scores,labels,0,query_size=4,iterations=2,mm_iterations=50)
        reference,ids,_=predict_binary(scores,labels,grouping_seed=0,query_size=4,iterations=2,mm_iterations=50)
        self.assertEqual(hard,reference);np.testing.assert_array_equal(groups,ids)
        np.testing.assert_allclose(p.sum(1),1.);self.assertEqual(p.shape,(13,2))

    def test_equal_soft_vote_uses_confidence_and_rejects_bad_probabilities(self):
        vote,p=soft_mean([[[.51,.49]],[[.51,.49]],[[.01,.99]]],0)
        self.assertEqual(vote.tolist(),[False]);np.testing.assert_allclose(p.sum(1),1.)
        with self.assertRaises(ValueError):soft_mean([[[.8,.8]],[[.5,.5]],[[.5,.5]]],0)

    def test_protection_preserves_raw_positive_predictions_without_truth(self):
        raw=np.array([True,False,True]);new=np.array([False,True,True])
        guard=protect(raw,new);self.assertEqual(guard.tolist(),[True,True,True])
        scores=np.array([[.9,.4,.2],[.8,.1,.5],[.9,.4,.5]])
        labels=['not_humanitarian','a','b']
        baseline=np.array(humanitarian(raw,scores,labels));candidate=np.array(humanitarian(guard,scores,labels))
        np.testing.assert_array_equal(baseline[raw],candidate[raw])


if __name__=='__main__':unittest.main()
