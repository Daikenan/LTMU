import torch
import torch.nn as nn

import neuron.ops as ops
from neuron.config import registry


__all__ = ['BalancedBCELoss', 'FocalLoss', 'GHMC_Loss', 'OHEM_BCELoss',
           'LabelSmoothLoss', 'SmoothL1Loss', 'IoULoss', 'GHMR_Loss',
           'TripletLoss', 'CenterLoss']


@registry.register_module
class BalancedBCELoss(nn.Module):

    def __init__(self, neg_weight=1.):
        super(BalancedBCELoss, self).__init__()
        self.neg_weight = neg_weight
    
    def forward(self, input, target):
        return ops.balanced_bce_loss(input, target, self.neg_weight)


@registry.register_module
class FocalLoss(nn.Module):

    def __init__(self, alpha=0.25, gamma=2.):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, input, target):
        return ops.focal_loss(input, target, self.alpha, self.gamma)


@registry.register_module
class GHMC_Loss(nn.Module):

    def __init__(self, bins=30, momentum=0.5):
        super(GHMC_Loss, self).__init__()
        self.bins = bins
        self.momentum = momentum
    
    def forward(self, input, target):
        return ops.ghmc_loss(input, target, self.bins, self.momentum)


@registry.register_module
class OHEM_BCELoss(nn.Module):

    def __init__(self, neg_ratio=3.):
        super(OHEM_BCELoss, self).__init__()
        self.neg_ratio = neg_ratio
    
    def forward(self, input, target):
        return ops.ohem_bce_loss(input, target, self.neg_ratio)


@registry.register_module
class LabelSmoothLoss(nn.Module):

    def __init__(self, num_classes, eps=0.1, calc_metrics=False):
        super(LabelSmoothLoss, self).__init__()
        self.num_classes = num_classes
        self.eps = eps
        self.calc_metrics = calc_metrics
    
    def forward(self, input, target):
        loss = ops.label_smooth_loss(
            input, target, self.num_classes, self.eps)
        
        if self.calc_metrics and not self.training:
            metrics = ops.topk_precision(input, target)
            loss = {'loss': loss}
            loss.update(metrics)
        
        return loss


@registry.register_module
class SmoothL1Loss(nn.Module):

    def __init__(self, beta=1. / 9):
        super(SmoothL1Loss, self).__init__()
        self.beta = beta
    
    def forward(self, input, target):
        return ops.smooth_l1_loss(input, target, self.beta)


@registry.register_module
class IoULoss(nn.Module):

    def forward(self, input, target, weight=None):
        return ops.iou_loss(input, target, weight)


@registry.register_module
class GHMR_Loss(nn.Module):

    def __init__(self, mu=0.02, bins=10, momentum=0):
        super(GHMR_Loss, self).__init__()
        self.mu = mu
        self.bins = bins
        self.momentum = momentum
    
    def forward(self, input, target):
        return ops.ghmr_loss(input, target)


@registry.register_module
class TripletLoss(nn.Module):

    def __init__(self, margin=None, normalize_feats=False, calc_metrics=False):
        super(TripletLoss, self).__init__()
        self.margin = margin
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()
        self.normalize_feats = normalize_feats
        self.calc_metrics = calc_metrics
    
    def forward(self, input, target):
        if self.normalize_feats:
            input = self._normalize(input, dim=-1)
        dist_mat = ops.euclidean(input, input, sqrt=True)
        dist_ap, dist_an = self._ohem(dist_mat, target)
        y = dist_an.new_ones(dist_an.size())

        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        
        if self.calc_metrics and not self.training:
            metrics = ops.r1_map(dist_mat, target)
            loss = {'loss': loss}
            loss.update(metrics)
        
        return loss
    
    def _normalize(self, x, dim=-1):
        norm = torch.norm(x, 2, dim=dim, keepdim=True)
        x = x / (norm.expand_as(x) + 1e-12)
        return x
    
    def _ohem(self, dist_mat, target, return_indices=False):
        n = dist_mat.size(0)

        label_mat = target.expand(n, n)
        pos_mask = label_mat.eq(label_mat.t())
        neg_mask = label_mat.ne(label_mat.t())
        
        dist_ap, indices_p = torch.max(
            dist_mat[pos_mask].contiguous().view(n, -1),
            dim=1, keepdim=True)
        dist_an, indices_n = torch.min(
            dist_mat[neg_mask].contiguous().view(n, -1),
            dim=1, keepdim=True)
        dist_ap = dist_ap.squeeze(1)
        dist_an = dist_an.squeeze(1)

        if return_indices:
            indices = target.new_zeros(
                target.size()).copy_(torch.arange(n).long())
            indices = indices.unsqueeze(0).expand(n, n)
            
            indices_p = torch.gather(
                indices[pos_mask].contiguous().view(n, -1),
                1, indices_p).squeeze(1)
            indices_n = torch.gather(
                indices[neg_mask].contiguous().view(n, -1),
                1, indices_n).squeeze(1)
            
            return dist_ap, dist_an, indices_p, indices_n
        else:
            return dist_ap, dist_an


@registry.register_module
class CenterLoss(nn.Module):

    def __init__(self, num_classes, num_channels):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.centers = nn.Parameter(
            torch.randn(num_classes, num_channels))
    
    def forward(self, input, target):
        assert len(input) == len(target)

        self.centers = self.centers.to(input.device)
        dist_mat = ops.euclidean(input, self.centers, sqrt=False)

        classes = torch.arange(self.num_classes).long()
        classes = classes.to(input.device)

        n = len(input)
        target = target.unsqueeze(1).expand(n, self.num_classes)
        mask = target.eq(classes.expand(n, self.num_classes))

        dist_mat = dist_mat * mask.float()
        loss = dist_mat.clamp_(min=1e-12, max=1e+12).sum() / n

        return loss
