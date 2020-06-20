import mmcv
import numpy as np
import pycocotools.mask as maskUtils
import torch
import torch.nn as nn

from mmdet.models.registry import DETECTORS
from mmdet.models.detectors.two_stage import TwoStageDetector
from mmdet.core import auto_fp16, get_classes, tensor2imgs, \
    bbox2result, bbox2roi, build_assigner, build_sampler

from .modulators import RPN_Modulator, RCNN_Modulator


__all__ = ['QG_RCNN']


@DETECTORS.register_module
class QG_RCNN(TwoStageDetector):

    def __init__(self,
                 backbone,
                 rpn_head,
                 bbox_roi_extractor,
                 bbox_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 shared_head=None,
                 pretrained=None):
        super(QG_RCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            shared_head=shared_head,
            rpn_head=rpn_head,
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)
        # build modulators
        self.rpn_modulator = RPN_Modulator()
        self.rcnn_modulator = RCNN_Modulator()
        # initialize weights
        self.rpn_modulator.init_weights()
        self.rcnn_modulator.init_weights()
    
    @auto_fp16(apply_to=('img_z', 'img_x'))
    def forward(self,
                img_z,
                img_x,
                img_meta_z,
                img_meta_x,
                return_loss=True,
                **kwargs):
        if return_loss:
            return self.forward_train(
                img_z, img_x, img_meta_z, img_meta_x, **kwargs)
        else:
            return self.forward_test(
                img_z, img_x, img_meta_z, img_meta_x, **kwargs)
    
    def forward_dummy(self, *args, **kwargs):
        raise NotImplementedError(
            'forward_dummy is not implemented for QG_RCNN')

    def forward_train(self,
                      img_z,
                      img_x,
                      img_meta_z,
                      img_meta_x,
                      gt_bboxes_z,
                      gt_bboxes_x,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None):
        z = self.extract_feat(img_z)
        x = self.extract_feat(img_x)

        # common parameters
        proposal_cfg = self.train_cfg.get(
            'rpn_proposal', self.test_cfg.rpn)
        bbox_assigner = build_assigner(
            self.train_cfg.rcnn.assigner)
        bbox_sampler = build_sampler(
            self.train_cfg.rcnn.sampler, context=self)
        
        losses = {}
        total = 0.
        for x_ij, i, j in self.rpn_modulator(z, x, gt_bboxes_z):
            losses_ij = {}

            # select the j-th bbox/meta/label of the i-th image
            gt_bboxes_ij = gt_bboxes_x[i:i + 1]
            gt_bboxes_ij[0] = gt_bboxes_ij[0][j:j + 1]
            gt_labels_ij = gt_labels[i:i + 1]
            gt_labels_ij[0] = gt_labels_ij[0][j:j + 1]
            img_meta_xi = img_meta_x[i:i + 1]

            # RPN forward and loss
            rpn_outs = self.rpn_head(x_ij)
            rpn_loss_inputs = rpn_outs + (
                gt_bboxes_ij, img_meta_xi, self.train_cfg.rpn)
            rpn_losses_ij = self.rpn_head.loss(
                *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses_ij.update(rpn_losses_ij)
            
            # parse proposal list
            proposal_inputs = rpn_outs + (img_meta_xi, proposal_cfg)
            proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)

            # assign gts and sample proposals
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None]
            assign_result = bbox_assigner.assign(
                proposal_list[0],
                gt_bboxes_ij[0],
                gt_bboxes_ignore[0],
                gt_labels_ij[0])
            sampling_result = bbox_sampler.sample(
                assign_result,
                proposal_list[0],
                gt_bboxes_ij[0],
                gt_labels_ij[0],
                feats=[lvl_feat[0][None] for lvl_feat in x_ij])
            sampling_results = [sampling_result]

            # bbox head forward of query
            z_ij = [u[i:i + 1] for u in z]
            gt_bboxes_z_ij = gt_bboxes_z[i:i + 1]
            gt_bboxes_z_ij[0] = gt_bboxes_z_ij[0][j:j + 1]
            rois_z = bbox2roi(gt_bboxes_z_ij)
            bbox_feats_z = self.bbox_roi_extractor(
                z_ij[:self.bbox_roi_extractor.num_inputs], rois_z)
            
            # bbox head forward of gallary
            x_ij = [u[i:i + 1] for u in x]
            rois_x = bbox2roi([res.bboxes for res in sampling_results])
            bbox_feats_x = self.bbox_roi_extractor(
                x_ij[:self.bbox_roi_extractor.num_inputs], rois_x)
            
            # do modulation
            bbox_feats = self.rcnn_modulator(
                bbox_feats_z, bbox_feats_x)
            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)
            cls_score, bbox_pred = self.bbox_head(bbox_feats)

            # calculate bbox losses
            bbox_targets = self.bbox_head.get_target(
                sampling_results,
                gt_bboxes_ij,
                gt_labels_ij,
                self.train_cfg.rcnn)
            loss_bbox = self.bbox_head.loss(
                cls_score, bbox_pred, *bbox_targets)
            losses_ij.update(loss_bbox)

            # update losses
            for k, v in losses_ij.items():
                if k in losses:
                    if isinstance(v, (tuple, list)):
                        for u in range(len(v)):
                            losses[k][u] += v[u]
                    else:
                        losses[k] += v
                else:
                    losses[k] = v
            total += 1.

        # average the losses over instances
        for k, v in losses.items():
            if isinstance(v, (tuple, list)):
                for u in range(len(v)):
                    losses[k][u] /= total
            else:
                losses[k] /= total

        return losses
    
    def forward_test(self,
                     img_z,
                     img_x,
                     img_meta_z,
                     img_meta_x,
                     gt_bboxes_z,
                     **kwargs):
        # assume one image and one instance only
        return self.simple_test(
            img_z, img_x, img_meta_z, img_meta_x,
            gt_bboxes_z, **kwargs)

    def simple_test(self,
                    img_z,
                    img_x,
                    img_meta_z,
                    img_meta_x,
                    gt_bboxes_z,
                    **kwargs):
        # assume one image and one instance only
        assert len(img_z) == 1
        z = self.extract_feat(img_z)
        x = self.extract_feat(img_x)
        
        # RPN forward
        rpn_feats = next(self.rpn_modulator(z, x, gt_bboxes_z))[0]
        proposal_list = self.simple_test_rpn(
            rpn_feats, img_meta_x, self.test_cfg.rpn)
        
        # RCNN forward
        det_bboxes, det_labels = self.simple_test_bboxes(
            z, x, img_meta_x, gt_bboxes_z,
            proposal_list, self.test_cfg.rcnn, **kwargs)
        bbox_results = bbox2result(
            det_bboxes, det_labels, self.bbox_head.num_classes)
        
        # format results
        proposals = proposal_list[0].cpu().numpy()
        bboxes = bbox_results[0]

        return proposals, bboxes
    
    def simple_test_bboxes(self,
                           z,
                           x,
                           img_meta_x,
                           gt_bboxes_z,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False,
                           keep_order=False,
                           **kwargs):
        # bbox head forward of query
        rois_z = bbox2roi(gt_bboxes_z)
        bbox_feats_z = self.bbox_roi_extractor(
            z[:self.bbox_roi_extractor.num_inputs], rois_z)
        
        # bbox head forward of gallary
        rois_x = bbox2roi(proposals)
        bbox_feats_x = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois_x)
        
        # do modulation
        roi_feats = self.rcnn_modulator(bbox_feats_z, bbox_feats_x)
        if self.with_shared_head:
            roi_feats = self.shared_head(roi_feats)
        cls_score, bbox_pred = self.bbox_head(roi_feats)

        # get predictions
        img_shape = img_meta_x[0]['img_shape']
        scale_factor = img_meta_x[0]['scale_factor']

        if keep_order:
            det_bboxes, det_labels = self.bbox_head.get_det_bboxes(
                rois_x,
                cls_score,
                bbox_pred,
                img_shape,
                scale_factor,
                rescale=rescale,
                cfg=None)
            det_bboxes = det_bboxes[:, 4:]
            det_labels = det_labels[:, 1]
        else:
            det_bboxes, det_labels = self.bbox_head.get_det_bboxes(
                rois_x,
                cls_score,
                bbox_pred,
                img_shape,
                scale_factor,
                rescale=rescale,
                cfg=rcnn_test_cfg)
        
        return det_bboxes, det_labels

    def aug_test(self, *args, **kwargs):
        raise NotImplementedError(
            'aug_test is not implemented for QG_RCNN')

    def show_result(self, *args, **kwargs):
        raise NotImplementedError(
            'show_result is not implemented for QG_RCNN')
    
    def _process_query(self, img_z, gt_bboxes_z):
        self._query = self.extract_feat(img_z)
        self._gt_bboxes_z = gt_bboxes_z
    
    def _process_gallary(self, img_x, img_meta_x, **kwargs):
        x = self.extract_feat(img_x)

        # RPN forward
        rpn_feats = next(self.rpn_modulator(
            self._query, x, self._gt_bboxes_z))[0]
        proposal_list = self.simple_test_rpn(
            rpn_feats, img_meta_x, self.test_cfg.rpn)
        
        # RCNN forward
        det_bboxes, det_labels = self.simple_test_bboxes(
            self._query, x, img_meta_x, self._gt_bboxes_z,
            proposal_list, self.test_cfg.rcnn, **kwargs)
        if not kwargs.get('keep_order', False):
            bbox_results = bbox2result(
                det_bboxes, det_labels, self.bbox_head.num_classes)
        else:
            bbox_results = [np.concatenate([
                det_bboxes.cpu().numpy(),
                det_labels.cpu().numpy()[:, None]], axis=1)]
        
        return bbox_results[0]
