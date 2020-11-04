import torch
import torch.nn as nn
import torch.nn.functional as F

import neuron.ops as ops
from neuron.config import registry


@registry.register_module
class Precision(nn.Module):

    def forward(self, scores, labels):
        return ops.topk_precision(scores, labels)


@registry.register_module
class R1_mAP(nn.Module):

    def __init__(self, normalize_feats=False):
        super(R1_mAP, self).__init__()
        self.normalize_feats = normalize_feats
    
    def forward(self, feats, labels):
        if self.normalize_feats:
            feats = F.normalize(feats, dim=1, p=2)
        dist_mat = ops.euclidean(feats, feats, sqrt=False)
        return ops.r1_map(dist_mat, labels)
