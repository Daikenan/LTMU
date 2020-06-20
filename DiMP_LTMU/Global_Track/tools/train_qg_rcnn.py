import _init_paths
import argparse
import os
import torch

import mmdet
import neuron.ops as ops
from mmcv import Config
from mmdet.apis import init_dist, get_root_logger, set_random_seed, \
    train_detector
from mmdet.models import build_detector
from mmdet.datasets import build_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a query-guided detector')
    parser.add_argument(
        '--config',
        default='configs/qg_rcnn_r50_fpn.py')
    parser.add_argument(
        '--work_dir',
        default='work_dirs/qg_rcnn_r50_fpn')
    parser.add_argument('--load_from')
    parser.add_argument('--resume_from')
    parser.add_argument(
        '--base_dataset',  # names of training datasets, splitted by comma, see `datasets/wrappers` for options
        type=str,
        default='coco_train,got10k_train,lasot_train')
    parser.add_argument(
        '--base_transforms',  # names of transforms, see `datasets/wrappers` for options
        type=str,
        default='extra_partial')
    parser.add_argument(
        '--sampling_prob',  # probabilities for sampling training datasets, splitted by comma, sum should be 1
        type=str,
        default='0.4,0.4,0.2')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--workers', type=int)
    parser.add_argument('--gpus', type=int, default=4)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--autoscale_lr', default=True)
    parser.add_argument('--validate', action='store_true')
    args = parser.parse_args()
    if not 'LOCAL_RANK' in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    # parse arguments
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    if args.load_from is not None:
        cfg.load_from = args.load_from
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.base_dataset is not None:
        cfg.data.train.base_dataset = args.base_dataset
    if args.base_transforms is not None:
        cfg.data.train.base_transforms = args.base_transforms
    if args.sampling_prob is not None:
        probs = [float(p) for p in args.sampling_prob.split(',')]
        cfg.data.train.sampling_prob = probs
    if args.fp16:
        cfg.fp16 = {'loss_scale': 512.}
    if args.workers is not None:
        cfg.data.workers_per_gpu = args.workers
    cfg.gpus = args.gpus
    if args.autoscale_lr:
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * cfg.gpus / 8.
    ops.sys_print('Args:\n--', args)
    ops.sys_print('Configs:\n--', cfg)

    # init distributed env, logger and random seeds
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
    logger = get_root_logger(cfg.log_level)
    logger.info('Distributed training: {}'.format(distributed))
    if args.seed is not None:
        logger.info('Set random seed to {}'.format(args.seed))
        set_random_seed(args.seed)
    
    # build model
    model = build_detector(
        cfg.model,
        train_cfg=cfg.train_cfg,
        test_cfg=cfg.test_cfg)
    
    # build dataset
    train_dataset = build_dataset(cfg.data.train)
    if cfg.checkpoint_config is not None:
        cfg.checkpoint_config.meta = {
            'mmdet_version': mmdet.__version__,
            'config': cfg.text,
            'CLASSES': train_dataset.CLASSES}
    model.CLASSES = train_dataset.CLASSES

    # run training
    train_detector(
        model,
        train_dataset,
        cfg,
        distributed=distributed,
        validate=args.validate,
        logger=logger)


if __name__ == '__main__':
    main()
