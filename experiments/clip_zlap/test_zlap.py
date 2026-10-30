import unittest
import numpy as np
from experiments.clip_zlap.core import predict, build_graph, unit_vectors
from experiments.clip_zlap.dn_unit import shifted_scores


class ZlapTests(unittest.TestCase):
    def test_sparse_solution_matches_dense_inverse(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(17, 6)); t = rng.normal(size=(3, 6))
        scores, diagnostics = predict(x, t, tol=1e-10)
        matrix, _ = build_graph(x, t)
        dense = np.linalg.solve(matrix.toarray(), np.eye(20)[:, :3])[3:]
        np.testing.assert_allclose(scores, dense, atol=2e-10, rtol=1e-7)
        self.assertTrue(diagnostics['all_cg_converged'])

    def test_permutation_equivariance_without_neighbor_ties(self):
        rng = np.random.default_rng(9)
        x = rng.uniform(.1, 1., (21, 9)); t = rng.uniform(.1, 1., (4, 9))
        order = rng.permutation(len(x))
        a, _ = predict(x, t); b, _ = predict(x[order], t)
        np.testing.assert_allclose(a[order], b, atol=2e-7, rtol=1e-4)
        np.testing.assert_array_equal(a[order].argmax(1), b.argmax(1))

    def test_separate_knn_has_cross_modal_edges_and_no_self_loop(self):
        x = np.array([[1., 0.], [.9, .1], [.8, .2]])
        t = np.array([[.2, 1.], [.1, 1.]])
        matrix, diagnostic = build_graph(x, t)
        np.testing.assert_allclose(matrix.diagonal(), 1.)
        self.assertEqual(diagnostic['image_k'], 3)
        self.assertEqual(diagnostic['text_k'], 2)
        self.assertGreater(np.count_nonzero(matrix.toarray()[:2, 2:]), 0)

    def test_single_image_and_disconnected_nodes_are_finite(self):
        scores, d = predict([[1., 0.]], [[1., 0.], [0., 1.]])
        self.assertEqual(scores.argmax(1).tolist(), [0])
        self.assertEqual(d['zero_degree_nodes'], 1)
        scores, d = predict([[-1., 0.]], [[1., 0.], [1., .1]])
        np.testing.assert_array_equal(scores, [[0., 0.]])
        self.assertEqual(d['zero_score_rows'], 1)

    def test_bad_inputs_rejected(self):
        for values in [[[0., 0.]], [[np.nan, 1.]]]:
            with self.assertRaises(ValueError): unit_vectors(values)
        for options in [{'k':0}, {'gamma':-1.}, {'alpha':1.}, {'tol':0.}, {'maxiter':0}]:
            with self.assertRaises(ValueError): predict([[1., 0.]], [[1., 0.], [0., 1.]], **options)

    def test_dn_formula_and_frozen_mean_independent_of_query_batch(self):
        rng = np.random.default_rng(42)
        x = unit_vectors(rng.normal(size=(13, 7)))
        t = unit_vectors(rng.normal(size=(3, 7)))
        mean = unit_vectors(rng.normal(size=(21, 7))).mean(0)
        expected = unit_vectors(x-.25*mean) @ unit_vectors(t-.25*t.mean(0)).T
        np.testing.assert_allclose(shifted_scores(x, t, mean), expected, atol=2e-7)
        np.testing.assert_allclose(shifted_scores(x[:1], t, mean)[0], shifted_scores(x, t, mean)[0], atol=2e-7)
        with self.assertRaises(ValueError): shifted_scores(x, t, mean, coefficient=.5)


if __name__ == '__main__': unittest.main()
