import unittest
import torch
import time

import neuron.ops as ops
from neuron.models.backbones import *


class TestBackbones(unittest.TestCase):

    def setUp(self):
        cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if cuda else 'cpu')
        self.input = torch.rand((2, 3, 640, 384)).to(self.device)
        # initialize GPUs
        _ = AlexNetV1(out_layers='conv5').to(self.device)(self.input)
    
    def test_alexnet(self):
        alexnets = [
            AlexNetV1(out_layers='conv5'),
            AlexNetV2(out_layers=['conv4', 'conv5']),
            AlexNetV3(out_layers=['conv5'])]
        for net in alexnets:
            self._check_net(net.to(self.device))
    
    def test_resnet(self):
        resnets = [
            resnet18(pretrained=True, last_stride=1),
            resnet34(pretrained=True, last_stride=1),
            resnet50(pretrained=True, last_stride=2),
            resnet101(pretrained=False, last_stride=2),
            resnet152(pretrained=False, last_stride=2)]
        for net in resnets:
            self._check_net(net.to(self.device))
    
    def _check_net(self, net):
        begin = time.time()
        out = net(self.input)
        end = time.time()

        # preserve the last-layer output
        if isinstance(out, dict):
            key = list(out.keys())[-1]
            out = out[key]
        elif isinstance(out, list):
            out = out[-1]
        
        # print inference information
        ops.sys_print('[{}] input: {} output: {} stride: {} '
                      'speed: {:.1f} fps'.format(
            net.__class__.__name__,
            tuple(self.input.shape),
            tuple(out.shape),
            net.out_stride,
            1. / (end - begin)))


if __name__ == '__main__':
    unittest.main()
