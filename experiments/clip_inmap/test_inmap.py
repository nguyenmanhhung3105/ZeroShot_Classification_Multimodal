import unittest
import numpy as np
import torch
from experiments.clip_zlap.core import unit_vectors
from experiments.clip_inmap.core import optimize,refine


class InmapTests(unittest.TestCase):
    def test_sinkhorn_balances_soft_mass_and_preserves_rows(self):
        logits=torch.tensor([[.2,.1],[.3,.1],[.25,.13],[.21,.12]])
        p=refine(logits,iterations=100)
        np.testing.assert_allclose(p.sum(1).numpy(),1,atol=1e-6)
        np.testing.assert_allclose(p.mean(0).numpy(),[.5,.5],atol=1e-6)

    def test_proxy_update_leaves_inputs_intact_unit_and_deterministic(self):
        rng=np.random.default_rng(3)
        x=unit_vectors(rng.uniform(.1,1.,(21,12)));t=unit_vectors(rng.uniform(.1,1.,(3,12)))
        x_before=x.copy();t_before=t.copy()
        scores,proxy,diag=optimize(x,t,iterations=60)
        again,_,_=optimize(x,t,iterations=60)
        np.testing.assert_array_equal(x,x_before);np.testing.assert_array_equal(t,t_before)
        np.testing.assert_array_equal(scores,again);np.testing.assert_allclose(np.linalg.norm(proxy,axis=1),1,atol=1e-6)
        self.assertTrue(diag['pseudo_label_optimization']);self.assertFalse(diag['ground_truth_fit'])

    def test_rejects_nonunit_and_bad_optimization_parameters(self):
        with self.assertRaises(ValueError):optimize([[2.,0.]],[[1.,0.],[0.,1.]])
        for settings in [{'iterations':0},{'tau_i':0},{'alpha':2},{'lr':float('nan')}]:
            with self.assertRaises(ValueError):optimize([[1.,0.]],[[1.,0.],[0.,1.]],**settings)


if __name__=='__main__':unittest.main()
