# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import glob
import pdb
import numpy as np
import os
import math
import sys
sys.path.append('/home/daikenan/Tracking/MBMD_vot_code/MBMD_vot_code/SiamMask')
sys.path.append('/home/daikenan/Tracking/MBMD_vot_code/MBMD_vot_code/SiamMask/experiments/siammask')
from custom import Custom

os.environ["CUDA_VISIBLE_DEVICE"]="0"
from tools.test import *

parser = argparse.ArgumentParser(description='PyTorch Tracking Demo')

parser.add_argument('--resume', default='SiamMask/experiments/siammask/SiamMask_DAVIS.pth', type=str,
                    metavar='PATH',help='path to latest checkpoint (default: none)')
parser.add_argument('--config', dest='config', default='SiamMask/experiments/siammask/config_davis.json',
                    help='hyper-parameter of SiamMask in json format')
parser.add_argument('--base_path', default='../../data/tennis', help='datasets')
args = parser.parse_args()
def overlap_ratio(rect1, rect2):
    '''
    Compute overlap ratio between two rects
    - rect: 1d array of [x,y,w,h] or
            2d array of N x [x,y,w,h]
    '''

    if rect1.ndim==1:
        rect1 = rect1[None,:]
    if rect2.ndim==1:
        rect2 = rect2[None,:]

    left = np.maximum(rect1[:,0], rect2[:,0])
    right = np.minimum(rect1[:,0]+rect1[:,2], rect2[:,0]+rect2[:,2])
    top = np.maximum(rect1[:,1], rect2[:,1])
    bottom = np.minimum(rect1[:,1]+rect1[:,3], rect2[:,1]+rect2[:,3])

    intersect = np.maximum(0,right - left) * np.maximum(0,bottom - top)
    union = rect1[:,2]*rect1[:,3] + rect2[:,2]*rect2[:,3] - intersect
    iou = np.clip(intersect / union, 0, 1)
    return iou
if __name__ == '__main__':
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    # Setup Model
    cfg = load_config(args)

    siammask = Custom(anchors=cfg['anchors'])
    if args.resume:
        assert isfile(args.resume), '{} is not a valid file'.format(args.resume)
        siammask = load_pretrain(siammask, args.resume)

    siammask.eval().to(device)

    # Parse Image file
    seq = 'car16'
    seq_path = '/home/daikenan/dataset/VOT2019/lt2019/' + seq
    img_files = sorted([seq_path + '/color/' + p for p in os.listdir(seq_path + '/color') if os.path.splitext(p)[1] == '.jpg'])
    #img_files = [file for file in img_files ]
    #img_files = sorted(glob.glob(join(args.base_path, '*.jp*')))
    #ims = [cv2.imread(imf) for imf in img_files]
    gt = np.loadtxt(seq_path+'/groundtruth.txt',delimiter = ',')
    x,y,w,h = gt[0]

    # Select ROI
    #cv2.namedWindow("SiamMask", cv2.WND_PROP_FULLSCREEN)
    #cv2.setWindowProperty("SiamMask", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    #try:
    #    init_rect = cv2.selectROI('SiamMask', ims[0], False, False)
    #    x, y, w, h = init_rect
    #except:
    #    exit()

    toc = 0
    for f, im in enumerate(img_files):
        im = cv2.imread(im)
        tic = cv2.getTickCount()
        if f == 0:  # init
            target_pos = np.array([x + w / 2, y + h / 2])
            target_sz = np.array([w, h])
            state = siamese_init(im, target_pos, target_sz, siammask, cfg['hp'])  # init tracker
        elif f > 0:  # tracking
            state = siamese_track(state, im, mask_enable=True, refine_enable=True)  # track
            #pdb.set_trace()
            score = np.max(state['score'])
            location = state['ploygon'].flatten()
            mask = state['mask'] > state['p'].seg_thr

            im[:, :, 2] = (mask > 0) * 255 + (mask == 0) * im[:, :, 2]
            # cv2.polylines(im, [np.int0(location).reshape((-1, 1, 2))], True, (0, 255, 0), 3)
            # cv2.putText(im, str(score), (20,40), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3)
            # cv2.imshow('SiamMask', im)
            # cv2.waitKey(1)
            #if key > 0:
            #    break

            cv2.namedWindow("SiamMask",cv2.WINDOW_NORMAL)
            cv2.rectangle(im, (int(state['target_pos'][0]-state['target_sz'][0]/2.0), int(state['target_pos'][1]-state['target_sz'][1]/2.0)),
                          (int(state['target_pos'][0]+state['target_sz'][0]/2.0), int(state['target_pos'][1]+state['target_sz'][1]/2.0)), [0, 255, 0], 2)
            #cv2.imwrite("/home/xiaobai/Desktop/MBMD_vot_code/figure/%05d.jpg"%frame_id, im[:, :, -1::-1])
            cv2.imshow("SiamMask", im)
            cv2.waitKey(1)

        toc += cv2.getTickCount() - tic
    toc /= cv2.getTickFrequency()
    fps = f / toc
    print('SiamMask Time: {:02.1f}s Speed: {:3.1f}fps (with visulization!)'.format(toc, fps))


    # dataset_path = '/home/xiaobai/dataset/VOT2019_RGBT/'
    # seq_list = os.listdir(dataset_path)
    # seq_list = [seq for seq in seq_list if not seq.endswith("txt")]
    # seq_list.sort()

    # iou_list=[]
    # fps_list=dict()
    # bb_result = dict()
    # result = dict()

    # iou_list_nobb=[]
    # bb_result_nobb = dict()
    # for num,seq in enumerate(seq_list):
    #     img_list = sorted([dataset_path+seq + '/color/' + p for p in os.listdir(dataset_path+seq+'/color/') if os.path.splitext(p)[1] == '.jpg'])

    #     # img_list = os.listdir(dataset_path+seq+'/ir/')
    #     # img_list.sort()
    #     ir_gt = np.loadtxt(dataset_path+seq+'/groundtruth.txt', delimiter=',')

    #     gt = np.zeros((ir_gt.shape[0], 4))
    #     ir_gt1 = np.concatenate((ir_gt[:,0:1],ir_gt[:,2:3], ir_gt[:,4:5], ir_gt[:,6:7]), axis=1)
    #     ir_gt2 = np.concatenate((ir_gt[:,1:2],ir_gt[:,3:4], ir_gt[:,5:6], ir_gt[:,7:8]), axis=1)

    #     gt[:,0] = np.min(ir_gt1,axis=1)
    #     gt[:,1] = np.min(ir_gt2,axis=1)
    #     gt[:,2] = np.max(ir_gt1,axis=1)
    #     gt[:,3] = np.max(ir_gt2,axis=1)
    #     gt[:,2] = gt[:,2] - gt[:,0]
    #     gt[:,3] = gt[:,3] - gt[:,1]
    #     cv2.namedWindow("SiamMask", cv2.WND_PROP_FULLSCREEN)
    #     iou_result = np.zeros((len(img_list),1))
    #     result_bb = np.zeros((len(img_list),4))
    #     toc = 0
    #     #iou_result, result_bb, fps, result_nobb = run_mdnet(img_list, gt[0], gt, display=True)
    #     for f, im_name in enumerate(img_list):
    #         im = cv2.imread(im_name)
    #         tic = cv2.getTickCount()
    #         if f == 0:  # init
    #             target_pos = np.array([gt[0,0] + gt[0,2] / 2, gt[0,1] + gt[0,3] / 2])
    #             result_bb[0,:] = gt[0,:]
    #             target_sz = np.array([gt[0,2], gt[0,3]])
    #             state = siamese_init(im, target_pos, target_sz, siammask, cfg['hp'])  # init tracker
    #         elif f > 0:  # tracking
    #             state = siamese_track(state, im, mask_enable=True, refine_enable=True)  # track
    #             #pdb.set_trace()
    #             score = np.max(state['score'])
    #             location = state['ploygon'].flatten()
    #             mask = state['mask'] > state['p'].seg_thr

    #             im[:, :, 2] = (mask > 0) * 255 + (mask == 0) * im[:, :, 2]
    #             # cv2.polylines(im, [np.int0(location).reshape((-1, 1, 2))], True, (0, 255, 0), 3)
    #             # cv2.putText(im, str(score), (20,40), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3)
    #             # cv2.imshow('SiamMask', im)
    #             # cv2.waitKey(1)
    #             #if key > 0:
    #             #    break
    #             result_bb[f,0] = state['target_pos'][0]-state['target_sz'][0]/2.0
    #             result_bb[f,1] = state['target_pos'][1]-state['target_sz'][1]/2.0
    #             result_bb[f,2] = state['target_sz'][0]
    #             result_bb[f,3] = state['target_sz'][1]
    #             #cv2.namedWindow("SiamMask",cv2.WINDOW_NORMAL)
    #             cv2.rectangle(im, (int(state['target_pos'][0]-state['target_sz'][0]/2.0), int(state['target_pos'][1]-state['target_sz'][1]/2.0)),
    #                           (int(state['target_pos'][0]+state['target_sz'][0]/2.0), int(state['target_pos'][1]+state['target_sz'][1]/2.0)), [0, 255, 0], 2)
    #             #cv2.imwrite("/home/xiaobai/Desktop/MBMD_vot_code/figure/%05d.jpg"%frame_id, im[:, :, -1::-1])
    #             cv2.imshow("SiamMask", im)
    #             cv2.waitKey(1)
    #         toc += cv2.getTickCount() - tic
    #         iou_result[f]= overlap_ratio(gt[f],result_bb[f])[0]

    #     toc /= cv2.getTickFrequency()
    #     fps = f / toc
    #     enable_frameNum = 0.
    #     for iidx in range(len(iou_result)):
    #         if (math.isnan(iou_result[iidx])==False): 
    #             enable_frameNum += 1.
    #         else:
    #             ## gt is not alowed
    #             iou_result[iidx] = 0.

    #     iou_list.append(iou_result.sum()/enable_frameNum)
    #     bb_result[seq] = result_bb
    #     fps_list[seq]=fps

    #     print '{} {} : {} , total mIoU:{}, fps:{}'.format(num,seq,iou_result.mean(), sum(iou_list)/len(iou_list),sum(fps_list.values())/len(fps_list))

