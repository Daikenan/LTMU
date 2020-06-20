import unittest

import _init_paths
import neuron.data as data
from trackers import *


class TestTrackers(unittest.TestCase):

    def setUp(self):
        self.evaluator = data.EvaluatorOTB(version=2015)
        self.visualize = False

    def test_global_track(self):
        # settings
        cfg_files = [
            'configs/qg_rpn_r50_fpn.py',
            'configs/qg_rcnn_r50_fpn.py',
            'configs/qg_rpn_r18_fpn.py']
        ckp_files = [
            'checkpoints/qg_rpn_r50_fpn_coco_got10k_lasot.pth',
            'checkpoints/qg_rcnn_r50_fpn_coco_got10k_lasot.pth',
            'work_dirs/qg_rpn_r18_fpn/epoch_12.pth']
        transforms = data.BasicPairTransforms(train=False)

        # run evaluation over different settings
        for cfg_file, ckp_file in zip(cfg_files, ckp_files):
            tracker = GlobalTrack(cfg_file, ckp_file, transforms)
            self.evaluator.run(tracker, visualize=self.visualize)
            self.evaluator.report(tracker.name)


if __name__ == '__main__':
    unittest.main()
