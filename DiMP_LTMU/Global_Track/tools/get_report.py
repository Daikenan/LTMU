import _init_paths
import neuron.data as data
import os
from trackers import *


if __name__ == '__main__':
    cfg_file = 'configs/qg_rcnn_r50_fpn.py'
    ckp_file = 'checkpoints/qg_rcnn_r50_fpn_coco_got10k_lasot.pth'
    transforms = data.BasicPairTransforms(train=False)
    tracker = GlobalTrack(
        cfg_file, ckp_file, transforms,
        name_suffix='qg_rcnn_r50_fpn')
    # evaluators = [
    #     data.EvaluatorLaSOT(frame_stride=1),
    #     data.EvaluatorTLP()]

    evaluators = [data.EvaluatorTLP()]
    trackers = os.listdir('/home/dkn/Tracking/GlobalTrack/results/TLP')
    # trackers = ['BACF']
    for e in evaluators:
        e.report(trackers, plot_curves=True)
