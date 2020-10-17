import os
from os.path import join, isdir
from tracker import *
import numpy as np

import argparse

import pickle

import math
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def genConfig(seq_path, set_type):

    path, seqname = os.path.split(seq_path)


    if set_type == 'OTB':
        ############################################  have to refine #############################################

        img_list = sorted([seq_path + '/img/' + p for p in os.listdir(seq_path + '/img') if os.path.splitext(p)[1] == '.jpg'])

        if (seqname == 'Jogging_1') or (seqname == 'Skating2_1'):
            gt = np.loadtxt(seq_path + '/groundtruth_rect.txt')
        elif (seqname == 'Jogging_2') or (seqname == 'Skating2_2'):
            gt = np.loadtxt(seq_path + '/groundtruth_rect.txt')
        elif seqname =='Human4':
            gt = np.loadtxt(seq_path + '/groundtruth_rect.2.txt', delimiter=',')
        elif (seqname == 'BlurBody')  or (seqname == 'BlurCar1') or (seqname == 'BlurCar2') or (seqname == 'BlurCar3') \
                or (seqname == 'BlurCar4') or (seqname == 'BlurFace') or (seqname == 'BlurOwl') or (seqname == 'Board') \
                or (seqname == 'Box')   or (seqname == 'Car4')  or (seqname == 'CarScale') or (seqname == 'ClifBar') \
                or (seqname == 'Couple')  or (seqname == 'Crossing')  or (seqname == 'Dog') or (seqname == 'FaceOcc1') \
                or (seqname == 'Girl') or (seqname == 'Rubik') or (seqname == 'Singer1') or (seqname == 'Subway') \
                or (seqname == 'Surfer') or (seqname == 'Sylvester') or (seqname == 'Toy') or (seqname == 'Twinnings') \
                or (seqname == 'Vase') or (seqname == 'Walking') or (seqname == 'Walking2') or (seqname == 'Woman')   :
            gt = np.loadtxt(seq_path + '/groundtruth_rect.txt')
        else:
            gt = np.loadtxt(seq_path + '/groundtruth_rect.txt', delimiter=',')

        if seqname == 'David':
            img_list = img_list[299:]
          
        if seqname == 'Football1':
            img_list = img_list[0:74]
        if seqname == 'Freeman3':
            img_list = img_list[0:460]
        if seqname == 'Freeman4':
            img_list = img_list[0:283]
        if seqname == 'Diving':
            img_list = img_list[0:215]
        if seqname == 'Tiger1':
            img_list = img_list[5:]

        ##polygon to rect
    if gt.shape[1] == 8:
        x_min = np.min(gt[:, [0, 2, 4, 6]], axis=1)[:, None]
        y_min = np.min(gt[:, [1, 3, 5, 7]], axis=1)[:, None]
        x_max = np.max(gt[:, [0, 2, 4, 6]], axis=1)[:, None]
        y_max = np.max(gt[:, [1, 3, 5, 7]], axis=1)[:, None]
        gt = np.concatenate((x_min, y_min, x_max - x_min, y_max - y_min), axis=1)

    return img_list, gt


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-set_type", default = 'OTB' )
    parser.add_argument("-model_path", default = './models/rt-mdnet.pth')
    parser.add_argument("-result_path", default = './result.npy')
    parser.add_argument("-visual_log",default=False, action= 'store_true')
    parser.add_argument("-visualize",default=False, action='store_true')
    parser.add_argument("-adaptive_align",default=True, action='store_false')
    parser.add_argument("-padding",default=1.2, type = float)
    parser.add_argument("-jitter",default=True, action='store_false')

    args = parser.parse_args()

    ##################################################################################
    #########################Just modify opts in this script.#########################
    ######################Becuase of synchronization of options#######################
    ##################################################################################
    ## option setting
    opts['model_path']=args.model_path
    opts['result_path']=args.result_path
    opts['visual_log']=args.visual_log
    opts['set_type']=args.set_type
    opts['visualize'] = False
    opts['adaptive_align'] = args.adaptive_align
    opts['padding'] = args.padding
    opts['jitter'] = args.jitter
    ##################################################################################
    ############################Do not modify opts anymore.###########################
    ######################Becuase of synchronization of options#######################
    ##################################################################################
    print opts


    ## path initialization
    dataset_path = '/home/daikenan/dataset/'


    seq_home = dataset_path + opts['set_type']
    seq_list = [f for f in os.listdir(seq_home) if isdir(join(seq_home,f))]

    iou_list=[]
    fps_list=dict()
    bb_result = dict()
    result = dict()

    iou_list_nobb=[]
    bb_result_nobb = dict()
    for num,seq in enumerate(seq_list):
        if num<-1:
            continue
        seq_path = seq_home + '/' + seq
        img_list,gt=genConfig(seq_path,opts['set_type'])

        iou_result, result_bb, fps, result_nobb = run_mdnet(img_list, gt[0], gt, seq = seq, display=opts['visualize'])

        enable_frameNum = 0.
        for iidx in range(len(iou_result)):
            if (math.isnan(iou_result[iidx])==False): 
                enable_frameNum += 1.
            else:
                ## gt is not alowed
                iou_result[iidx] = 0.

        iou_list.append(iou_result.sum()/enable_frameNum)
        bb_result[seq] = result_bb
        fps_list[seq]=fps

        bb_result_nobb[seq] = result_nobb
        print '{} {} : {} , total mIoU:{}, fps:{}'.format(num,seq,iou_result.mean(), sum(iou_list)/len(iou_list),sum(fps_list.values())/len(fps_list))

    result['bb_result']=bb_result
    result['fps']=fps_list
    result['bb_result_nobb']=bb_result_nobb
    np.save(opts['result_path'],result)

