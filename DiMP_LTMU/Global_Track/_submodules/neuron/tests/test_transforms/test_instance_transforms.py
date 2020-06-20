import unittest
import numpy as np

import neuron.data as data
import neuron.ops as ops


class TestInstanceTransforms(unittest.TestCase):
    
    def setUp(self):
        self.seqs = data.OTB(version=2015)
        self.visualize = False
    
    def test_siamfc_transforms(self):
        transforms = data.ReID_Transforms()
        dataset = data.Seq2Instance(self.seqs, transforms=transforms)

        inds = np.random.choice(len(dataset), 10)
        for i in inds:
            img, target = dataset[i]
            img = 255. * (img - img.min()) / (img.max() - img.min())
            img = img.permute(1, 2, 0).numpy().astype(np.uint8)
            if self.visualize:
                ops.sys_print('Label: %d' % target['label'].item())
                ops.show_image(img, target['bbox'], delay=0)


if __name__ == '__main__':
    unittest.main()
