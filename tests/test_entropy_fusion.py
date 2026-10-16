"""Kiểm tra fusion entropy bằng điểm lớp giả; không đọc dữ liệu hoặc tải model."""

import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from inference import _entropy_weighted_fuse, _weighted_fuse


class EntropyFusionTests(unittest.TestCase):
    def test_concentrated_branch_gets_more_weight_and_empty_text_uses_image(self):
        image = np.array([[0.9, 0.1], [0.5, 0.5], [0.2, 0.8]])
        text = np.array([[0.5, 0.5], [0.5, 0.5], [0.9, 0.1]])
        actual = _entropy_weighted_fuse(image, text, np.array([False, False, True]))
        np.testing.assert_allclose(actual[0], image[0])
        np.testing.assert_allclose(actual[1], _weighted_fuse(image[1:2], text[1:2], .5)[0])
        np.testing.assert_allclose(actual[2], image[2])

    def test_equal_concentration_matches_fixed_fusion(self):
        image = np.array([[0.8, 0.2], [0.2, 0.8]])
        text = 1 - image
        np.testing.assert_allclose(_entropy_weighted_fuse(image, text),
                                   _weighted_fuse(image, text, .5))


if __name__ == "__main__":
    unittest.main()
