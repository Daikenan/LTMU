import mmcv
from mmdet.core import bbox_mapping, tensor2imgs, auto_fp16
from mmdet.models import builder
from mmdet.models.registry import DETECTORS
from mmdet.models.detectors.base import BaseDetector
from mmdet.models.detectors.test_mixins import RPNTestMixin

from .modulators import RPN_Modulator


__all__ = ['QG_RPN']


@DETECTORS.register_module
class QG_RPN(BaseDetector, RPNTestMixin):

    def __init__(self,
                 backbone,
                 neck,
                 rpn_head,
                 train_cfg,
                 test_cfg,
                 pretrained=None):
        super(QG_RPN, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        self.neck = builder.build_neck(neck) if neck is not None else None
        self.rpn_head = builder.build_head(rpn_head)
        self.rpn_modulator = RPN_Modulator()
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        super(QG_RPN, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            self.neck.init_weights()
        self.rpn_head.init_weights()
        self.rpn_modulator.init_weights()

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, *args, **kwargs):
        raise NotImplementedError(
            'forward_dummy is not implemented for QG_RPN')

    def forward_train(self,
                      img_z,
                      img_x,
                      img_meta_z,
                      img_meta_x,
                      gt_bboxes_z=None,
                      gt_bboxes_x=None,
                      gt_bboxes_ignore=None):
        if self.train_cfg.rpn.get('debug', False):
            self.rpn_head.debug_imgs = tensor2imgs(img_x)
        
        z = self.extract_feat(img_z)
        x = self.extract_feat(img_x)

        losses = {}
        num = 0.  # total number of instances
        for x_ij, i, j in self.rpn_modulator(z, x, gt_bboxes_z):
            # select the j-th bbox of the i-th image
            gt_bboxes_ij = gt_bboxes_x[i:i + 1]
            gt_bboxes_ij[0] = gt_bboxes_ij[0][j:j + 1]
            # RPN forward and losses
            rpn_outs = self.rpn_head(x_ij)
            rpn_loss_inputs = rpn_outs + (
                gt_bboxes_ij, img_meta_x[i:i + 1], self.train_cfg.rpn)
            losses_ij = self.rpn_head.loss(
                *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            # update with RPN losses
            for k, v in losses_ij.items():
                if k in losses:
                    if isinstance(v, (tuple, list)):
                        for o in range(len(v)):
                            losses[k][o] += v[o]
                    else:
                        losses[k] += v
                else:
                    losses[k] = v
            # update total instance number
            num += 1.
        
        # average the losses over instances
        for k, v in losses.items():
            for o in range(len(v)):
                losses[k][o] /= num
        
        return losses
    
    def forward_test(self,
                     img_z,
                     img_x,
                     img_meta_z,
                     img_meta_x,
                     gt_bboxes_z,
                     **kwargs):
        # assume one image and one query instance only
        return self.simple_test(img_z, img_x, img_meta_z, img_meta_x,
                                gt_bboxes_z, **kwargs)
    
    @auto_fp16(apply_to=('img_z', 'img_x'))
    def forward(self, img_z, img_x, img_meta_z, img_meta_x,
                return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(
                img_z, img_x, img_meta_z, img_meta_x, **kwargs)
        else:
            return self.forward_test(
                img_z, img_x, img_meta_z, img_meta_x, **kwargs)
    
    def simple_test(self,
                    img_z,
                    img_x,
                    img_meta_z,
                    img_meta_x,
                    gt_bboxes_z,
                    **kwargs):
        # assume one image and one query instance only
        assert len(img_z) == 1
        z = self.extract_feat(img_z)
        x = self.extract_feat(img_x)
        x = next(self.rpn_modulator(z, x, gt_bboxes_z))[0]
        proposal_list = self.simple_test_rpn(x, img_meta_x, self.test_cfg.rpn)
        if rescale:
            for proposals, meta in zip(proposal_list, img_meta_x):
                proposals[:, :4] /= meta['scale_factor']
        # TODO: remove this restriction
        return proposal_list[0].cpu().numpy()

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError(
            'aug_test is not implemented for QG-RPN')

    def show_result(self, data, result, dataset=None, top_k=20):
        """Show QG-RPN proposals on the image.

        Although we assume batch size is 1, this method supports arbitrary
        batch size.
        """
        img_tensor = data['img_x'][0]
        img_metas = data['img_meta_x'][0].data[0]
        imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
        assert len(imgs) == len(img_metas)
        for img, img_meta in zip(imgs, img_metas):
            h, w, _ = img_meta['img_shape']
            img_show = img[:h, :w, :]
            mmcv.imshow_bboxes(img_show, result, top_k=top_k)
    
    def _process_query(self, img_z, gt_bboxes_z):
        self._query = self.extract_feat(img_z)
        self._gt_bboxes_z = gt_bboxes_z
    
    def _process_gallary(self, img_x, img_meta_x, rescale=False):
        x = self.extract_feat(img_x)
        x = next(self.rpn_modulator(self._query, x, self._gt_bboxes_z))[0]
        proposal_list = self.simple_test_rpn(x, img_meta_x, self.test_cfg.rpn)
        if rescale:
            for proposals, meta in zip(proposal_list, img_meta_x):
                proposals[:, :4] /= meta['scale_factor']
        return proposal_list[0].cpu().numpy()
