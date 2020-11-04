import torch
import torch.nn as nn

import neuron.ops as ops
from neuron.config import registry


@registry.register_module
class ReID_Metric(nn.Module):
    
    def __init__(self, metric_cls, metric_rank):
        super(ReID_Metric, self).__init__()
        self.metric_cls = metric_cls
        self.metric_rank = metric_rank
    
    def forward(self, *args):
        if len(args) == 2:
            scores = None
            feats, labels = args
        elif len(args) == 3:
            scores, feats, labels = args
        else:
            raise ValueError('Expected to have 2 or 3 inputs,'
                             'but got {}'.format(len(args)))
        
        metrics = self.metric_rank(feats, labels)
        if scores is not None:
            metrics.update(self.metric_cls(scores, labels))
        
        return metrics
