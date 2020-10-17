import os
from os.path import join, isdir
from tracker import *
import numpy as np
import cv2
import argparse
from tracking_utils import _compile_results, compute_overall_results
import pickle

import math
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class Region:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

def eval_tracking(Dataset, g, video_spe=None):
    if Dataset == "votlt":
        data_dir = '/home/daikenan/dataset/VOT18_long'
    elif Dataset == 'otb':
        data_dir = '/home/daikenan/dataset/OTB'
    sequence_list = os.listdir(data_dir)
    sequence_list.sort()
    sequence_list = [title for title in sequence_list if not title.endswith("txt")]
    if video_spe is not None:
        sequence_list = [video_spe]
    precisions = np.zeros(len(sequence_list))
    precisions_auc = np.zeros(len(sequence_list))
    ious = np.zeros(len(sequence_list))
    lengths = np.zeros(len(sequence_list))
    speed = np.zeros(len(sequence_list))
    base_save_path = '/home/daikenan/Desktop/RTMDNet/'
    if not os.path.exists(base_save_path):
        os.mkdir(base_save_path)

    for seq_id, video in enumerate(sequence_list):
        if Dataset == "votlt":
            sequence_dir = data_dir + '/' + video + '/color/'
            gt_dir = data_dir + '/' + video + '/groundtruth.txt'
        elif Dataset == "otb":
            sequence_dir = data_dir + '/' + video + '/img/'
            gt_dir = data_dir + '/' + video + '/groundtruth_rect.txt'
        result_save_path = os.path.join(base_save_path, video+'.txt')
        if os.path.exists(result_save_path):
            continue
        image_list = os.listdir(sequence_dir)
        image_list.sort()
        image_list = [im for im in image_list if im.endswith("jpg") or im.endswith("jpeg")]
        try:
            groundtruth = np.loadtxt(gt_dir, delimiter=',')
        except:
            groundtruth = np.loadtxt(gt_dir)
        region = Region(groundtruth[0, 0], groundtruth[0,1],groundtruth[0,2],groundtruth[0,3])
        image_dir = sequence_dir + image_list[0]
        image = cv2.imread(image_dir)
        # tracker = MDNetTracker(image, region)
        num_frames = len(image_list)
        image_lists = [sequence_dir + tmp for tmp in image_list]
        bBoxes = np.zeros((num_frames, 4))
        bBoxes[0, :] = groundtruth[0,:]
        # tracker results
        iou_result, result_bb, fps, result_nobb = run_mdnet(image_lists, groundtruth[0], groundtruth, seq=video, display=False)
        np.savetxt(result_save_path, result_bb, fmt="%.6f,%.6f,%.6f,%.6f")
        np.savetxt(os.path.join('/home/daikenan/Desktop/RTMDNet2/', video+'.txt'), result_nobb, fmt="%.6f,%.6f,%.6f,%.6f")
        lengths[seq_id], precisions[seq_id], precisions_auc[seq_id], ious[seq_id] = _compile_results(groundtruth.astype(np.float32), result_bb.astype(np.float32), 20)
        print(
            str(seq_id) + ' -- ' + sequence_list[seq_id] +
            ' -- Precision: ' + "%.2f" % precisions[seq_id] +
            ' -- Precisions AUC: ' + "%.2f" % precisions_auc[seq_id] +
            ' -- IOU: ' + "%.2f" % ious[seq_id] +
            ' -- Speed: ' + "%.2f" % speed[seq_id] + ' --')
        g.writelines([str(seq_id) + ' -- ' + sequence_list[seq_id] +
                      ' -- Precision: ' + "%.2f" % precisions[seq_id] +
                      ' -- Precisions AUC: ' + "%.2f" % precisions_auc[seq_id] +
                      ' -- IOU: ' + "%.2f" % ious[seq_id] +
                      ' -- Speed: ' + "%.2f" % speed[seq_id] + ' --\n'])
    tot_frames = np.sum(lengths)
    mean_precision = np.mean(precisions)
    mean_precision_auc = np.mean(precisions_auc)
    mean_iou = np.mean(ious)
    mean_speed = np.mean(speed)
    print('-- Overall stats (averaged per frame) on ' + str(len(sequence_list)))
    print(' -- Precision ' + "(20 px)" + ': ' + "%.2f" % mean_precision +
          ' -- Precisions AUC: ' + "%.2f" % mean_precision_auc +
          ' -- IOU: ' + "%.2f" % mean_iou +
          ' -- Speed: ' + "%.2f" % mean_speed + ' --')
    g.writelines(['-- Overall stats (averaged per frame) on ' + str(
        len(sequence_list)) + ' -- Precision ' + "(20 px)" + ': ' + "%.2f" % mean_precision +
                  ' -- Precisions AUC: ' + "%.2f" % mean_precision_auc +
                  ' -- IOU: ' + "%.2f" % mean_iou +
                  ' -- Speed: ' + "%.2f" % mean_speed + ' --\n'])
if __name__ == "__main__":
    # g = open('/home/daikenan/Desktop/pymdnet.txt', 'rb+')
    # g.read()
    # eval_tracking('otb', g)
    # g.close()
    compute_overall_results(dataset='otb', result_path='/home/daikenan/Desktop/RTMDNet/',
                            save_path='/home/daikenan/Desktop/RTMDNet/all.txt')