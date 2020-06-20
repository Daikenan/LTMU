import unittest
import random

import neuron.ops as ops
from neuron.data.datasets import *


class TestSeqDatasets(unittest.TestCase):

    def setUp(self):
        self.visualize = False
    
    def test_seq_datasets(self):
        datasets = [
            OTB(version=2015),
            VOT(version=2019, anno_type='rect'),
            DTB70(),
            TColor128(),
            NfS(fps=30),
            NfS(fps=240),
            UAV123(version='UAV123'),
            UAV123(version='UAV20L'),
            OxUvA(subset='dev'),
            OxUvA(subset='test'),
            GOT10k(subset='train'),
            GOT10k(subset='test'),
            LaSOT(subset='train'),
            LaSOT(subset='test'),
            TrackingNet(subset='train'),
            TrackingNet(subset='test'),
            VisDroneSOT(subset='train'),
            VisDroneSOT(subset='test'),
            POT(),
            MOT(version=2016, subset='train'),
            ImageNetVID(subset=['train', 'val']),
            VisDroneVID(subset=['train', 'val'])]
        for dataset in datasets:
            self._check_dataset(dataset)
    
    def _check_dataset(self, dataset):
        seq_num = len(dataset)
        if seq_num == 0:
            ops.sys_print('Warning: {} dataset is empty, '
                          'skip test...'.format(dataset.name))
            return
        self.assertGreater(seq_num, 0)
        ops.sys_print('{} contains {} sequences'.format(
            dataset.name, seq_num))

        # sanity check
        inds = random.sample(range(seq_num), min(seq_num, 10))
        for i in inds:
            img_files, target = dataset[i]
            anno, meta = target['anno'], target['meta']
            if anno.shape[0] == 1:
                continue  # test sets
            if anno.shape[1] in [4, 8]:
                self.assertEqual(len(img_files), len(anno))
            else:
                self.assertGreaterEqual(
                    len(img_files) - 1, anno[:, 0].max())
        
        # visualization
        if self.visualize:
            img_files, target = random.choice(dataset)
            anno = target['anno']
            
            for f, img_file in enumerate(img_files):
                if f >= anno.shape[0]:
                    break  # test sets
                if anno.shape[1] == 9:
                    bboxes = anno[anno[:, 0] == f, 2:6]
                elif anno.shape[1] == 8:
                    break  # TODO: points are not supported yet
                else:
                    bboxes = anno[f, :]
                img = ops.read_image(img_file)
                ops.show_image(img, bboxes, delay=1)


if __name__ == '__main__':
    unittest.main()
