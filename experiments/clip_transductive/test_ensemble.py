import unittest
import numpy as np
from experiments.clip_transductive.ensemble import combine,humanitarian,align


class EnsembleTests(unittest.TestCase):
    def test_complete_truth_table_and_priority_equivalence(self):
        votes=np.array([[bool(i&4),bool(i&2),bool(i&1)] for i in range(8)])
        result=combine(votes)
        self.assertEqual(result['majority'].tolist(),[False,False,False,True,False,True,True,True])
        self.assertEqual(result['recall_or'].tolist(),[False,True,True,True,True,True,True,True])
        preferred=votes[:,1].copy();agree=votes[:,0]==votes[:,2];preferred[agree]=votes[agree,0]
        np.testing.assert_array_equal(preferred,result['majority'])
        self.assertTrue(np.all(result['recall_or'][:,None]>=votes))

    def test_gate_keeps_detail_argmax_and_rejects_invalid_scores(self):
        scores=np.array([[.9,.3,.2],[.1,.2,.8],[.9,.8,.1]])
        labels=['not_humanitarian','a','b']
        self.assertEqual(humanitarian(np.array([True,True,False]),scores,labels),['a','b','not_humanitarian'])
        with self.assertRaises(ValueError):humanitarian(np.array([True]),scores,labels)
        with self.assertRaises(ValueError):combine(np.ones((2,3)))
        with self.assertRaises(ValueError):combine(np.zeros((0,3),dtype=bool))

    def test_alignment_refuses_permutation_missing_or_changed_ground_truth(self):
        rows=[{'id':'a','y_true':'x'},{'id':'b','y_true':'y'}]
        align(rows,rows)
        for candidate in [rows[::-1],rows[:1],[rows[0],{'id':'b','y_true':'x'}]]:
            with self.assertRaises(ValueError):align(rows,candidate)


if __name__=='__main__':unittest.main()
