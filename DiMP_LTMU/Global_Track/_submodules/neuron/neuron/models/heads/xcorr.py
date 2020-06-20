import numbers
import torch
import torch.nn as nn
from collections import OrderedDict

from neuron.config import registry
from neuron import ops


__all__ = ['XCorr']


@registry.register_module
class XCorr(nn.Module):

    def __init__(self, scale=0.001, learnable=False):
        assert isinstance(scale, (numbers.Number, list))
        if isinstance(scale, numbers.Number):
            scale = [scale]
        super(XCorr, self).__init__()
        
        # store logarithm of scale to avoid explosion during training
        self.scale = nn.Parameter(torch.Tensor(scale).log())
        if not learnable:
            self.scale.requires_grad = False
    
    def forward(self, z, x):
        assert isinstance(z, (torch.Tensor, OrderedDict))
        scale = self.scale.exp()

        if isinstance(z, torch.Tensor):
            assert len(scale) == 1
            out = scale * ops.fast_xcorr(z, x)
        elif isinstance(z, OrderedDict):
            if len(scale) == 1:
                scale = scale.expand(len(z))
            assert len(scale) == len(z)
            out = sum([scale[i] * ops.fast_xcorr(z[k], x[k])
                       for i, k in enumerate(z)])
        return out
