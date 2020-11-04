import torch.nn as nn
from collections import OrderedDict

from .backbone import Backbone
from neuron.config import registry


__all__ = ['AlexNetV1', 'AlexNetV2', 'AlexNetV3']


class _BatchNorm2d(nn.BatchNorm2d):

    def __init__(self, num_features, *args, **kwargs):
        super(_BatchNorm2d, self).__init__(
            num_features, *args, eps=1e-6, momentum=0.05, **kwargs)


class _AlexNet(Backbone):

    def __init__(self, out_layers='conv5'):
        super(_AlexNet, self).__init__()
        self._check_out_layers(out_layers)
        self.out_layers = out_layers
    
    def forward(self, x, out_layers=None):
        if out_layers is None:
            out_layers = self.out_layers
        self._check_out_layers(out_layers)
        outputs = OrderedDict()

        # conv1
        x = self.conv1(x)
        if self._add_and_check('conv1', out_layers, outputs, x):
            if isinstance(out_layers, str):
                return outputs[out_layers]
            else:
                return outputs
        
        # conv2
        x = self.conv2(x)
        if self._add_and_check('conv2', out_layers, outputs, x):
            if isinstance(out_layers, str):
                return outputs[out_layers]
            else:
                return outputs
        
        # conv3
        x = self.conv3(x)
        if self._add_and_check('conv3', out_layers, outputs, x):
            if isinstance(out_layers, str):
                return outputs[out_layers]
            else:
                return outputs
        
        # conv4
        x = self.conv4(x)
        if self._add_and_check('conv4', out_layers, outputs, x):
            if isinstance(out_layers, str):
                return outputs[out_layers]
            else:
                return outputs
        
        # conv5
        x = self.conv5(x)
        if self._add_and_check('conv5', out_layers, outputs, x):
            if isinstance(out_layers, str):
                return outputs[out_layers]
            else:
                return outputs
    
    def _check_out_layers(self, out_layers):
        valid_layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']
        if isinstance(out_layers, str):
            assert out_layers in valid_layers
        elif isinstance(out_layers, (list, tuple)):
            assert all([l in valid_layers for l in out_layers])
        else:
            raise ValueError('Invalid type of "out_layers".')
    
    def _add_and_check(self, name, out_layers, outputs, x):
        if isinstance(out_layers, str):
            out_layers = [out_layers]
        assert isinstance(out_layers, (list, tuple))
        if name in out_layers:
            outputs[name] = x
        return len(out_layers) == len(outputs)


@registry.register_module
class AlexNetV1(_AlexNet):
    _out_stride = 8

    def __init__(self, out_layers='conv5'):
        super(AlexNetV1, self).__init__(out_layers=out_layers)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, 11, 2),
            _BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, groups=2),
            _BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1),
            _BatchNorm2d(384),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1, groups=2),
            _BatchNorm2d(384),
            nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, 3, 1, groups=2))
    
    @property
    def out_channels(self):
        return self.conv5[-1].out_channels
    
    @property
    def out_stride(self):
        return self._out_stride


@registry.register_module
class AlexNetV2(_AlexNet):
    _out_stride = 4

    def __init__(self, out_layers='conv5'):
        super(AlexNetV2, self).__init__(out_layers=out_layers)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, 11, 2),
            _BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, groups=2),
            _BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 1))
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1),
            _BatchNorm2d(384),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1, groups=2),
            _BatchNorm2d(384),
            nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 32, 3, 1, groups=2))
    
    @property
    def out_channels(self):
        return self.conv5[-1].out_channels
    
    @property
    def out_stride(self):
        return self._out_stride


@registry.register_module
class AlexNetV3(_AlexNet):
    _out_stride = 8

    def __init__(self, out_layers='conv5'):
        super(AlexNetV3, self).__init__(out_layers=out_layers)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 192, 11, 2),
            _BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(192, 512, 5, 1),
            _BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(512, 768, 3, 1),
            _BatchNorm2d(768),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(768, 768, 3, 1),
            _BatchNorm2d(768),
            nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(768, 512, 3, 1),
            _BatchNorm2d(512))
    
    @property
    def out_channels(self):
        return self.conv5[-1].num_features

    @property
    def out_stride(self):
        return self._out_stride
