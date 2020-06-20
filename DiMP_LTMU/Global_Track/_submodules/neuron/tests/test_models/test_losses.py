import unittest
import torch
import torch.nn as nn

import neuron.ops as ops
from neuron.models.losses import *


class TestLosses(unittest.TestCase):

    def setUp(self):
        cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if cuda else 'cpu')
    
    def test_classification_losses(self):
        x = torch.rand(64)
        y = (x > 0.5).float()
        criterions = [
            nn.BCEWithLogitsLoss(reduction='mean'),
            BalancedBCELoss(neg_weight=1.),
            FocalLoss(gamma=2.),
            GHMC_Loss(bins=30, momentum=0.5),
            OHEM_BCELoss(neg_ratio=3.)]
        
        # check losses
        for criterion in criterions:
            loss = criterion(x, y)
            ops.sys_print('Loss[{}]: {:.3f}'.format(
                criterion.__class__.__name__, loss.item()))
            self.assertGreaterEqual(loss.item(), 0)
        
        # check losses on correct predictions
        for criterion in criterions:
            loss = criterion(y * 1e3, y)
            ops.sys_print('GT Loss[{}]: {:.3f}'.format(
                criterion.__class__.__name__, loss.item()))
            self.assertGreaterEqual(loss.item(), 0)
    
    def test_loc_losses(self):
        x = torch.rand(64, 4)
        x[:, 2:] += x[:, :2]
        y = x + 0.3 * (torch.rand(x.size()) - 0.5)
        y = y.clamp_(0)
        criterions = [
            nn.SmoothL1Loss(),
            SmoothL1Loss(beta=1. / 9),
            IoULoss(),
            GHMR_Loss(mu=0.02, bins=10, momentum=0.1)]
        
        # check losses
        for criterion in criterions:
            loss = criterion(x, y)
            ops.sys_print('Loss[{}]: {:.3f}'.format(
                criterion.__class__.__name__, loss.item()))
            self.assertGreaterEqual(loss.item(), 0)
        
        # check losses on correct predictions
        for criterion in criterions:
            loss = criterion(y, y)
            ops.sys_print('GT Loss[{}]: {:.3f}'.format(
                criterion.__class__.__name__, loss.item()))
            self.assertGreaterEqual(loss.item(), 0)
    
    def test_metric_losses(self):
        x = torch.rand(16, 2048)
        y = torch.LongTensor([
            0, 1, 2, 3, 2, 3, 1, 0, 0, 3, 2, 1, 0, 2, 3, 1])
        criterions = [
            TripletLoss(margin=None, normalize_feats=False),
            TripletLoss(margin=0.3, normalize_feats=False),
            TripletLoss(margin=None, normalize_feats=True),
            TripletLoss(margin=0.3, normalize_feats=True),
            CenterLoss(731, 2048)]
        
        # check losses
        for criterion in criterions:
            loss = criterion(x, y)
            ops.sys_print('Loss[{}]: {:.3f}'.format(
                criterion.__class__.__name__, loss.item()))
            self.assertGreaterEqual(loss.item(), 0)


if __name__ == '__main__':
    unittest.main()
