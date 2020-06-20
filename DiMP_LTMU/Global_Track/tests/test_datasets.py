import unittest
import numpy as np

import _init_paths
import neuron.ops as ops
from datasets import *
from mmcv.parallel import DataContainer as DC


class TestDatasets(unittest.TestCase):

    def setUp(self):
        self.visualize = False

    def test_pair_wrapper(self):
        dataset = PairWrapper(base_transforms='extra_partial')
        indices = np.random.choice(len(dataset), 10)
        for i in indices:
            item = dataset[i]

            # check keys
            keys = [
                'img_z',
                'img_x',
                'img_meta_z',
                'img_meta_x',
                'gt_bboxes_z',
                'gt_bboxes_x']
            self.assertTrue(all([k in item for k in keys]))

            # check data types
            for _, v in item.items():
                self.assertTrue(isinstance(v, DC))
            
            # check sizes
            self.assertEqual(
                len(item['gt_bboxes_z'].data),
                len(item['gt_bboxes_x'].data))
            if 'gt_labels' in item:
                self.assertEqual(
                    len(item['gt_bboxes_x'].data),
                    len(item['gt_labels'].data))
            
            # visualize pair
            if self.visualize:
                ops.sys_print('Item index:', i)
                self._show_image(
                    item['img_z'].data, item['gt_bboxes_z'].data,
                    fig=0, delay=1)
                self._show_image(
                    item['img_x'].data, item['gt_bboxes_x'].data,
                    fig=1, delay=0)
    
    def _show_image(self, img, bboxes, fig, delay):
        img = 255. * (img - img.min()) / (img.max() - img.min())
        img = img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        bboxes = bboxes.cpu().numpy()
        ops.show_image(img, bboxes, fig=fig, delay=delay)


if __name__ == '__main__':
    unittest.main()
