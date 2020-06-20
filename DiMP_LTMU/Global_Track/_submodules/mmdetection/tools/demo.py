import mmdet.apis as apis
import mmcv


def faster_rcnn():
    cfg_file = 'configs/faster_rcnn_r50_fpn_1x.py'
    ckp_file = '../../checkpoints/' \
        'faster_rcnn_r50_fpn_2x_20181010-443129e1.pth'
    model = apis.init_detector(cfg_file, ckp_file, device='cuda:0')
    
    img_file = '/home/lhhuang/data/VOCdevkit/VOC2007/JPEGImages/007663.jpg'
    img = mmcv.imread(img_file)
    result = apis.inference_detector(model, img)

    out_file = 'out.jpg'
    apis.show_result(
        img, result, model.CLASSES,
        show=False, out_file=out_file)


def mask_rcnn():
    cfg_file = 'configs/mask_rcnn_r50_fpn_1x.py'
    ckp_file = '../../checkpoints/' \
        'mask_rcnn_r50_fpn_1x_20181010-069fa190.pth'
    model = apis.init_detector(cfg_file, ckp_file, device='cuda:0')
    
    img_file = '/home/lhhuang/data/coco/val2017/000000397133.jpg'
    img = mmcv.imread(img_file)
    result = apis.inference_detector(model, img)

    out_file = 'out.jpg'
    apis.show_result(
        img, result, model.CLASSES,
        show=False, out_file=out_file)


if __name__ == '__main__':
    mask_rcnn()
