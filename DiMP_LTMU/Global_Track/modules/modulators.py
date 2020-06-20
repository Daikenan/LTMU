import torch.nn as nn

from mmdet.models.roi_extractors import SingleRoIExtractor
from mmdet.core import bbox2roi
from mmcv.cnn import normal_init


__all__ = ['RPN_Modulator', 'RCNN_Modulator']


class RPN_Modulator(nn.Module):

    def __init__(self,
                 roi_out_size=7,
                 roi_sample_num=2,
                 channels=256,
                 strides=[4, 8, 16, 32],
                 featmap_num=5):
        super(RPN_Modulator, self).__init__()
        self.roi_extractor = SingleRoIExtractor(
            roi_layer={
                'type': 'RoIAlign',
                'out_size': roi_out_size,
                'sample_num': roi_sample_num},
            out_channels=channels,
            featmap_strides=strides)
        self.proj_modulator = nn.ModuleList([
            nn.Conv2d(channels, channels, roi_out_size, padding=0)
            for _ in range(featmap_num)])
        self.proj_out = nn.ModuleList([
            nn.Conv2d(channels, channels, 1, padding=0)
            for _ in range(featmap_num)])
    
    def forward(self, feats_z, feats_x, gt_bboxes_z):
        return self.inference(
            feats_x,
            modulator=self.learn(feats_z, gt_bboxes_z))
    
    def inference(self, feats_x, modulator):
        n_imgs = len(feats_x[0])
        for i in range(n_imgs):
            n_instances = len(modulator[i])
            for j in range(n_instances):
                query = modulator[i][j:j + 1]
                gallary = [f[i:i + 1] for f in feats_x]
                out_ij = [self.proj_modulator[k](query) * gallary[k]
                          for k in range(len(gallary))]
                out_ij = [p(o) for p, o in zip(self.proj_out, out_ij)]
                yield out_ij, i, j

    def learn(self, feats_z, gt_bboxes_z):
        rois = bbox2roi(gt_bboxes_z)
        bbox_feats = self.roi_extractor(
            feats_z[:self.roi_extractor.num_inputs], rois)
        modulator = [bbox_feats[rois[:, 0] == j]
                     for j in range(len(gt_bboxes_z))]
        return modulator
    
    def init_weights(self):
        for m in self.proj_modulator:
            normal_init(m, std=0.01)
        for m in self.proj_out:
            normal_init(m, std=0.01)


class RCNN_Modulator(nn.Module):

    def __init__(self, channels=256):
        super(RCNN_Modulator, self).__init__()
        self.proj_z = nn.Conv2d(channels, channels, 3, padding=1)
        self.proj_x = nn.Conv2d(channels, channels, 3, padding=1)
        self.proj_out = nn.Conv2d(channels, channels, 1)
    
    def forward(self, z, x):
        return self.inference(x, self.learn(z))
    
    def inference(self, x, modulator):
        # assume one image and one instance only
        assert len(modulator) == 1
        return self.proj_out(self.proj_x(x) * modulator)
    
    def learn(self, z):
        # assume one image and one instance only
        assert len(z) == 1
        return self.proj_z(z)
    
    def init_weights(self):
        normal_init(self.proj_z, std=0.01)
        normal_init(self.proj_x, std=0.01)
        normal_init(self.proj_out, std=0.01)
