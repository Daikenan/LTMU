import torch.nn as nn
import abc


class Backbone(nn.Module):
    __metaclass__  = abc.ABCMeta

    @abc.abstractproperty
    def out_channels(self):
        raise NotImplementedError
    
    @abc.abstractproperty
    def out_stride(self):
        raise NotImplementedError
