import torch
import numpy as np
import cv2

import neuron.ops as ops
from neuron.config import registry
from .tracker import Tracker


__all__ = ['TrackerSiamFC']


@registry.register_module
class TrackerSiamFC(Tracker):

    def __init__(self, backbone, head, cfg):
        super(TrackerSiamFC, self).__init__(
            name=cfg.get('name', 'SiamFC'),
            is_deterministic=True)
        self.backbone = backbone
        self.head = head
        self.cfg = cfg
    
    def forward(self, z, x):
        z = self.backbone(z)
        x = self.backbone(x)
        return self.head(z, x)
    
    def forward_train(self, img_z, img_x, target):
        pred = self.forward(img_z, img_x)
        return pred, target
    
    def forward_val(self, img_z, img_x, target):
        pred = self.forward(img_z, img_x)
        return pred, target
    
    @torch.no_grad()
    def init(self, img, init_bbox):
        self.eval()
        
        bbox = init_bbox.copy()
        self.center = (bbox[:2] + bbox[2:]) / 2.
        self.target_sz = bbox[2:] - bbox[:2] + 1

        # create hanning window
        self.upscale_sz = self.cfg.response_up * self.cfg.response_sz
        self.hann_window = np.outer(
            np.hanning(self.upscale_sz),
            np.hanning(self.upscale_sz))
        self.hann_window /= self.hann_window.sum()

        # search scale factors
        self.scale_factors = self.cfg.scale_step ** np.linspace(
            -(self.cfg.scale_num // 2),
            self.cfg.scale_num // 2, self.cfg.scale_num)
        
        # exemplar and search sizes
        context = self.cfg.context * np.sum(self.target_sz)
        self.z_sz = np.sqrt(np.prod(self.target_sz + context))
        self.x_sz = self.z_sz * \
            self.cfg.instance_sz / self.cfg.exemplar_sz
        
        # exemplar image
        self.avg_color = np.mean(img, axis=(0, 1))
        z = ops.crop_and_resize(
            img, self.center, self.z_sz,
            out_size=self.cfg.exemplar_sz,
            border_value=self.avg_color)
        
        # exemplar features
        device = next(self.backbone.parameters()).device
        z = torch.from_numpy(z).to(
            device).permute(2, 0, 1).unsqueeze(0).float()
        self.template = self.backbone(z)
    
    @torch.no_grad()
    def update(self, img):
        self.eval()

        # search images
        x = [ops.crop_and_resize(
            img, self.center, self.x_sz * f,
            out_size=self.cfg.instance_sz,
            border_value=self.avg_color) for f in self.scale_factors]
        device = next(self.backbone.parameters()).device
        x = torch.from_numpy(np.stack(x, axis=0)).to(
            device).permute(0, 3, 1, 2).float()
        
        # responses
        x = self.backbone(x)
        responses = self.head(self.template, x)
        responses = responses.squeeze(1).cpu().numpy()

        # upsample responses and penalize scale changes
        responses = np.stack([cv2.resize(
            u, (self.upscale_sz, self.upscale_sz),
            interpolation=cv2.INTER_CUBIC)
            for u in responses])
        responses[:self.cfg.scale_num // 2] *= self.cfg.scale_penalty
        responses[self.cfg.scale_num // 2 + 1:] *= self.cfg.scale_penalty

        # peak scale
        scale_id = np.argmax(np.amax(responses, axis=(1, 2)))

        # peak location
        response = responses[scale_id]
        response -= response.min()
        response /= response.sum() + 1e-16
        response = (1 - self.cfg.window_influence) * response + \
            self.cfg.window_influence * self.hann_window
        loc = np.unravel_index(response.argmax(), response.shape)

        # locate target center
        disp_in_response = np.array(loc) - (self.upscale_sz - 1) / 2
        disp_in_instance = disp_in_response * \
            self.cfg.out_stride / self.cfg.response_up
        disp_in_image = disp_in_instance * self.x_sz * \
            self.scale_factors[scale_id] / self.cfg.instance_sz
        self.center += disp_in_image

        # update target size
        scale =  (1 - self.cfg.scale_lr) * 1.0 + \
            self.cfg.scale_lr * self.scale_factors[scale_id]
        self.target_sz *= scale
        self.z_sz *= scale
        self.x_sz *= scale

        # tracking result
        bbox = np.concatenate([
            self.center - (self.target_sz - 1) / 2.,
            self.center + (self.target_sz - 1) / 2.])
        
        return bbox
