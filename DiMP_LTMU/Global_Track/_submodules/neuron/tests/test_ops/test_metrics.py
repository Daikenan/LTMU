from __future__ import absolute_import

import unittest
import numpy as np

import neuron.ops as ops


class TestMetrics(unittest.TestCase):

    def test_iou(self):
        r1 = np.random.rand(1000, 4) * 100
        r2 = np.random.rand(1000, 4) * 100
        r1[:, 2:] += r1[:, :2] - 1
        r2[:, 2:] += r2[:, :2] - 1

        for bound in [None, (50, 100), (100, 200)]:
            o1 = ops.rect_iou(r1, r2, bound=bound)
            o2 = ops.poly_iou(r1, r2, bound=bound)
            self.assertTrue((o1 - o2).max() < 1e-12)

            p1 = self._to_corner(r1)
            p2 = self._to_corner(r2)
            o3 = ops.poly_iou(p1, p2, bound=bound)
            self.assertTrue((o1 - o3).max() < 1e-12)
    
    def _to_corner(self, rects):
        rects = rects.copy()
        rects[:, 2:] += 1
        x1, y1, x2, y2 = rects.T
        return np.array([x1, y1, x1, y2, x2, y2, x2, y1]).T


if __name__ == '__main__':
    unittest.main()
