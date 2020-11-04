import torch.nn as nn

import neuron.ops as ops
from neuron.config import registry
from neuron.models.model import Model


__all__ = ['ReID_Baseline']


@registry.register_module
class ReID_Baseline(Model):

    def __init__(self, backbone, num_classes):
        super(ReID_Baseline, self).__init__()

        # build network
        self.backbone = backbone
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.neck = nn.BatchNorm1d(backbone.out_channels)
        self.neck.bias.requires_grad_(False)  # no shift
        self.classifier = nn.Linear(
            backbone.out_channels, num_classes, bias=False)
        self.num_classes = num_classes

        # initialize weights
        self.neck.apply(ops.kaiming_init)
        self.classifier.apply(ops.classifier_init)

    def forward(self, x, training=None):
        x = self.pool(self.backbone(x)).view(len(x), -1)
        x_neck = self.neck(x)

        if training is None:
            training = self.training
        if training:
            scores = self.classifier(x_neck)
            return scores, x
        else:
            return x_neck
    
    def forward_train(self, img, target):
        scores, feats = self.forward(img, training=True)
        return scores, feats, target['label']
    
    def forward_val(self, img, target):
        feats = self.forward(img, training=False)
        return feats, target['label']
    
    def forward_test(self, img, target):
        feats = self.forward(img, training=False)
        return feats, target['label']
