import unittest
import torch

import _init_paths
from mmcv import Config
from mmdet.models import build_detector
from mmdet.datasets import build_dataset
from mmdet.datasets import build_dataloader
from mmcv.parallel import MMDataParallel


class TestModules(unittest.TestCase):

    def test_qg_rpn(self):
        # build configurations
        cfg_file = 'configs/qg_rpn_r50_fpn.py'
        cfg = Config.fromfile(cfg_file)
        cfg.gpus = 1

        # load state_dict
        ckp_file = 'checkpoints/qg_rpn_r50_fpn_2x_20181010-88a4a471.pth'
        checkpoint = torch.load(ckp_file, map_location='cpu')
        state_dict = checkpoint['state_dict']

        # build model
        model = build_detector(
            cfg.model,
            train_cfg=cfg.train_cfg,
            test_cfg=cfg.test_cfg)
        model.load_state_dict(state_dict)
        model = MMDataParallel(model, device_ids=range(cfg.gpus)).cuda()
        
        # build dataloader
        dataset = build_dataset(cfg.data.train)
        dataloader = build_dataloader(
            dataset,
            cfg.data.imgs_per_gpu,
            cfg.data.workers_per_gpu,
            cfg.gpus,
            shuffle=False,
            dist=False)
        
        # run a forward pass
        batch = next(iter(dataloader))
        losses = model(**batch)
    
    def test_qg_rcnn(self):
        # build configuration
        cfg_file = 'configs/qg_rcnn_r50_fpn.py'
        cfg = Config.fromfile(cfg_file)
        cfg.gpus = 1

        # load state_dict
        ckp_file = 'checkpoints/qg_rcnn_r50_fpn_2x_20181010-443129e1.pth'
        checkpoint = torch.load(ckp_file, map_location='cpu')
        state_dict = checkpoint['state_dict']

        # build model
        model = build_detector(
            cfg.model,
            train_cfg=cfg.train_cfg,
            test_cfg=cfg.test_cfg)
        model.load_state_dict(state_dict)
        model = MMDataParallel(model, device_ids=range(cfg.gpus))

        # build dataloader
        dataset = build_dataset(cfg.data.train)
        dataloader = build_dataloader(
            dataset,
            cfg.data.imgs_per_gpu,
            cfg.data.workers_per_gpu,
            cfg.gpus,
            shuffle=False,
            dist=False)
        
        # run a forward pass
        batch = next(iter(dataloader))
        losses = model(**batch)


if __name__ == '__main__':
    unittest.main()
