import _init_paths
import neuron.data as data
from Global_trackers import *
import numpy as np
import os
import cv2



def get_seq_list(Dataset, mode=None, classes=None):
    if Dataset == "votlt18":
        data_dir = votlt18_dir
    elif Dataset == 'otb':
        data_dir = otb_dir
    elif Dataset == "votlt19":
        data_dir = '/home/dkn/data/lt2019'
    elif Dataset == "tlp":
        data_dir = tlp_dir
    elif Dataset == "lasot":
        data_dir = os.path.join(lasot_dir, classes)

    sequence_list = os.listdir(data_dir)
    sequence_list.sort()
    sequence_list = [title for title in sequence_list if not title.endswith("txt")]

    return sequence_list, data_dir


def get_groundtruth(Dataset, data_dir, video):
    if Dataset == "votlt" or Dataset == "votlt19":
        sequence_dir = data_dir + '/' + video + '/color/'
        gt_dir = data_dir + '/' + video + '/groundtruth.txt'
    elif Dataset == "otb":
        sequence_dir = data_dir + '/' + video + '/img/'
        gt_dir = data_dir + '/' + video + '/groundtruth_rect.txt'
    elif Dataset == "lasot":
        sequence_dir = data_dir + '/' + video + '/img/'
        gt_dir = data_dir + '/' + video + '/groundtruth.txt'
    elif Dataset == "tlp":
        sequence_dir = data_dir + '/' + video + '/img/'
        gt_dir = data_dir + '/' + video + '/groundtruth_rect.txt'
    try:
        groundtruth = np.loadtxt(gt_dir, delimiter=',')
    except:
        groundtruth = np.loadtxt(gt_dir)
    if Dataset == 'tlp':
        groundtruth = groundtruth[:, 1:5]

    return sequence_dir, groundtruth


def show_res(im, boxes, win_name):
    cv2.namedWindow(win_name,cv2.WINDOW_NORMAL)
    for i in range(len(boxes)):
        box = boxes[i]
        if i == 0:
            color = [0, 0, 255]
        else:
            color = [0, 255, 255]
        cv2.rectangle(im, (box[0], box[1]),
                      (box[2], box[3]), color, 2)
    cv2.imshow(win_name, im)
    cv2.waitKey(1)


def run_seq_list(tracker, Dataset, mode=None, classes=None, video=None):
    sequence_list, data_dir = get_seq_list(Dataset, mode=mode, classes=classes)
    if video is not None:
        sequence_list = [video]
    for seq_id, video in enumerate(sequence_list):
        sequence_dir, groundtruth = get_groundtruth(Dataset, data_dir, video)

        image_list = os.listdir(sequence_dir)
        image_list.sort()
        image_list = [im for im in image_list if im.endswith("jpg") or im.endswith("jpeg")]
        image_dir = sequence_dir + image_list[0]
        image = cv2.cvtColor(cv2.imread(image_dir), cv2.COLOR_BGR2RGB)
        init_box = groundtruth[0]
        tracker.init(image, init_box)


        for im_id in range(1, len(image_list), 10):
            imagefile = sequence_dir + image_list[im_id]
            image = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)
            results = tracker.update(image)
            index = np.argsort(results[:, -1])[::-1]
            max_index = index[:10]
            can_boxes = results[max_index][:, :4]
            show_res(image, can_boxes, 'a')


if __name__ == '__main__':
    cfg_file = 'configs/qg_rcnn_r50_fpn.py'
    ckp_file = 'checkpoints/qg_rcnn_r50_fpn_coco_got10k_lasot.pth'
    transforms = data.BasicPairTransforms(train=False)
    tracker = GlobalTrack(
        cfg_file, ckp_file, transforms,
        name_suffix='qg_rcnn_r50_fpn')
    run_seq_list(tracker, 'votlt19', video='deer')
    # tracker.init(img, init_box)
    # tracker.update(img, {'return_all': True})


    # evaluators = [
    #     data.EvaluatorLaSOT(frame_stride=1),
    #     data.EvaluatorTLP()]
    # evaluators = [data.EvaluatorLaSOT(frame_stride=1)]
    # for e in evaluators:
    #     e.run(tracker, visualize=True)
    #     e.report(tracker.name, plot_curves=True)
