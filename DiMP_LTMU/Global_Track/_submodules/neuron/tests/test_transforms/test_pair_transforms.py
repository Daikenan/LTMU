import unittest
import numpy as np

import neuron.data as data
import neuron.ops as ops


class TestPairTransforms(unittest.TestCase):
    
    def setUp(self):
        self.seqs = data.OTB(version=2015)
        self.visualize = False
    
    def test_siamfc_transforms(self):
        transforms = data.SiamFC_Transforms()
        dataset = data.Seq2Pair(self.seqs, transforms=transforms)
        indices = np.random.choice(len(dataset), 10)
        for i in indices:
            img_z, img_x, target = dataset[i]
            img_z = img_z.permute(1, 2, 0).numpy().astype(np.uint8)
            img_x = img_x.permute(1, 2, 0).numpy().astype(np.uint8)
            target = target.squeeze(0).numpy().astype(np.float32)
            if self.visualize:
                ops.show_image(img_z, fig=1, delay=1)
                ops.show_image(img_x, fig=2, delay=1)
                ops.show_image(target, fig=3, delay=0)
    
    def test_mmdet_transforms(self):
        for transforms in [
            data.BasicPairTransforms(),
            data.ExtraPairTransforms()]:
            dataset = data.Seq2Pair(self.seqs, transforms=transforms)
            indices = np.random.choice(len(dataset), 10)
            for i in indices:
                item = dataset[i]
                img_z = ops.stretch_color(item['img_z'].permute(1, 2, 0).numpy())
                img_x = ops.stretch_color(item['img_x'].permute(1, 2, 0).numpy())
                bboxes_z = item['gt_bboxes_z'][0].numpy()
                bboxes_x = item['gt_bboxes_x'][0].numpy()
                if self.visualize:
                    ops.show_image(img_z, bboxes_z, fig=1, delay=1)
                    ops.show_image(img_x, bboxes_x, fig=2, delay=0)


if __name__ == '__main__':
    unittest.main()
