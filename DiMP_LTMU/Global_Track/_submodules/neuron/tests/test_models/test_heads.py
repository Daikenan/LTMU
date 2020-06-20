import unittest
import torch
import time

import neuron.ops as ops
import neuron.models as models


class TestHeads(unittest.TestCase):

    def setUp(self):
        cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if cuda else 'cpu')
        self.z = torch.rand(2, 3, 127, 127).to(self.device)
        self.x = torch.rand(2, 3, 255, 255).to(self.device)
    
    def test_xcorr(self):
        backbone = models.AlexNetV1()
        heads = [
            models.XCorr(scale=0.001, learnable=False),
            models.XCorr(scale=0.001, learnable=True),
            models.XCorr(scale=[0.001], learnable=False),
            models.XCorr(scale=[0.001], learnable=True)]
        for head in heads:
            ops.sys_print(head.scale.exp())
            self._check_xcorr_head(
                backbone.to(self.device),
                head.to(self.device))
    
    def _check_xcorr_head(self, backbone, head):
        begin = time.time()
        out = head(backbone(self.z), backbone(self.x))
        end = time.time()

        # print inference information
        ops.sys_print('[{}] z: {} x: {} output: {} time: {:.5f}s'.format(
            head.__class__.__name__,
            tuple(self.z.shape),
            tuple(self.x.shape),
            tuple(out.shape),
            end - begin))


if __name__ == '__main__':
    unittest.main()
