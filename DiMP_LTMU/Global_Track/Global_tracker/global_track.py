import torch
import numpy as np

from neuron.models import Tracker
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.core import wrap_fp16_model


__all__ = ['GlobalTrack']


class GlobalTrack(Tracker):

    def __init__(self, cfg_file, ckp_file, transforms, name_suffix=''):
        name = 'GlobalTrack'
        if name_suffix:
            name += '_' + name_suffix
        super(GlobalTrack, self).__init__(
            name=name, is_deterministic=True)
        self.transforms = transforms
        
        # build config
        cfg = Config.fromfile(cfg_file)
        if cfg.get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = True
        cfg.model.pretrained = None
        self.cfg = cfg

        # build model
        model = build_detector(
            cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
        fp16_cfg = cfg.get('fp16', None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)
        checkpoint = load_checkpoint(
            model, ckp_file, map_location='cpu')
        model.CLASSES = ('object', )

        # GPU usage
        cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if cuda else 'cpu')
        self.model = model.to(self.device)
    
    @torch.no_grad()
    def init(self, img, bbox):
        self.model.eval()

        # prepare query data
        img_meta = {'ori_shape': img.shape}
        bboxes = np.expand_dims(bbox, axis=0)
        img, img_meta, bboxes = \
            self.transforms._process_query(img, img_meta, bboxes)
        img = img.unsqueeze(0).contiguous().to(
            self.device, non_blocking=True)
        bboxes = bboxes.to(self.device, non_blocking=True)

        # initialize the modulator
        self.model._process_query(img, [bboxes])
    
    @torch.no_grad()
    def update(self, img, **kwargs):
        self.model.eval()

        # prepare gallary data
        img_meta = {'ori_shape': img.shape}
        img, img_meta, _ = \
            self.transforms._process_gallary(img, img_meta, None)
        img = img.unsqueeze(0).contiguous().to(
            self.device, non_blocking=True)

        # get detections
        results = self.model._process_gallary(
            img, [img_meta], rescale=True, **kwargs)
        return results
        # if not kwargs.get('return_all', False):
        #     # return the top-1 detection
        #     max_ind = results[:, -1].argmax()
        #     return results[max_ind, :4]
        # else:
        #     # return all detections
        #     return results
