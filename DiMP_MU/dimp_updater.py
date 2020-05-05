#coding=utf-8
import cv2
import os
from region_to_bbox import region_to_bbox
import time
# import tensorflow as tf
import torch
import numpy as np
import tensorflow as tf
from local_path import base_path
save_path = '/media/dkn/Data2/dimp/'
from local_path import toolkit_path
import sys
# sys.path.append(toolkit_path + 'native/trax/support/python')
sys.path.append(os.path.join(base_path, 'lib'))
sys.path.append(os.path.join(base_path, 'lib/slim'))
# sys.path.append(os.path.join(base_path, 'mdcheckmask/py-MDNet'))
sys.path.append(os.path.join('./tracker_controller'))
sys.path.append(os.path.join('./tracker_controller/metric_net'))
from tcNet import tclstm
from tcopt import tcopts
from metric_model import ft_net
from torch.autograd import Variable
from me_sample_generator import *
# from google.protobuf import text_format
# from object_detection.protos import pipeline_pb2
# from core.model_builder import build_man_model
# from object_detection.core import box_list
# from object_detection.core import box_list_ops
# pymdnet
# from pymdnet_model import *
# from data_prov import *
# import torch
# import torch.utils.data as data
# import torch.optim as optim
# from torch.autograd import Variable
# from run_pymdnet import forward_samples, set_optimizer, train
# from options import *

# rtmdnet
# from rtmdnet_utils import *
# sys.path.insert(0,os.path.join(base_path, 'RT_MDNet/modules'))
# from rt_sample_generator import *
# from data_prov import *
#
# from rtmdnet_model import *
# from rtmdnet_options import *
# from img_cropper import *
#
# from roi_align.modules.roi_align import RoIAlignAvg,RoIAlignMax,RoIAlignAdaMax,RoIAlignDenseAdaMax
#
# from bbreg import *
#
# from RT_MDNet.tracker import set_optimizer, rt_train

# atom
import argparse
from pytracking.libs.tensorlist import TensorList
from pytracking.utils.plotting import show_tensor
from pytracking.features.preprocessing import numpy_to_torch
env_path = os.path.join(os.path.dirname(__file__))
if env_path not in sys.path:
    sys.path.append(env_path)
from pytracking.evaluation import Tracker

from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000
# import scipy.io as sio
# import vot
import random
#from vggm import vggM#, ConfNet
# from sample_generator import *
# from tracking_utils import *
# from tracking_utils import _init_video, _compile_results
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from numpy.random import seed
# from tensorflow import set_random_seed




class p_config(object):
    name = 'a'
    model_dir = 'test'
    lose_count = 5
    R_loss_thr = 0.3
    R_center_redet = 0.3
    R_global_redet = 0.5
    Verification = "rtmdnet"
    Regressor = "mrpn"
    visualization = True

def show_res(im, box, win_name,confidence=None,score=None,save_path=None,frame_id=None,mask=None,score_max=None, groundtruth=None, can=None, local_state=None, rtbox=None):
    cv2.namedWindow(win_name,cv2.WINDOW_NORMAL)
    cv2.rectangle(im, (box[1], box[0]),
                  (box[3], box[2]), [0, 255, 255], 2)
    if score_max > 0:
        rtcolor = [90, 90, 255]
    else:
        rtcolor = [90, 90, 0]
    if local_state is not None:
        local_state = local_state.reshape((4, 1))
        cv2.rectangle(im, (local_state[0], local_state[1]),
                      (local_state[2]+local_state[0], local_state[3]+local_state[1]), rtcolor, 2)
    if rtbox is not None:
        for i in range(len(rtbox)):
            # rtbox = rtbox.reshape((4, 1))
            cv2.rectangle(im, (rtbox[i][0], rtbox[i][1]),
                          (rtbox[i][2] + rtbox[i][0], rtbox[i][3] + rtbox[i][1]), [90, 90, 0], 2)
    if can is not None:
        for i in range(len(can)):
            box = can[i][:4]
            box = [int(s) for s in box]
            cv2.rectangle(im, (int(box[0]), int(box[1])),
                          (int(box[2]), int(box[3])), [255, 0, 0], 2)
    if mask is not None:
        im[:, :, 1] = (mask > 0) * 255 + (mask == 0) * im[:, :, 1]
    if groundtruth is not None and not groundtruth[frame_id][0]==np.nan:
        groundtruth = groundtruth.astype("int16")
        cv2.rectangle(im, (groundtruth[frame_id][0], groundtruth[frame_id][1]),
                      (groundtruth[frame_id][0]+groundtruth[frame_id][2], groundtruth[frame_id][1]+groundtruth[frame_id][3]), [0, 0, 255], 2)
    if confidence is not None:
        cv2.putText(im,str(confidence),(20,40), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1)
    if score_max is not None:
        cv2.putText(im,str(score_max),(20,80), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1)
    if score is not None:
        cv2.putText(im, str(score), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
    if frame_id is not None:
        cv2.putText(im,str(frame_id),(20,20), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1)
    #cv2.imwrite("/home/xiaobai/Desktop/MBMD_vot_code/figure/%05d.jpg"%frame_id, im[:, :, -1::-1])
    cv2.imshow(win_name, im)
    cv2.waitKey(1)

def _compute_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    if xA < xB and yA < yB:
        # compute the area of intersection rectangle
        interArea = (xB - xA) * (yB - yA)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the intersection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
    else:
        iou = 0

    assert iou >= 0
    assert iou <= 1.01

    return iou

def process_regions(regions):
    # regions = np.squeeze(regions, axis=0)
    regions = regions / 255.0
    regions[:, :, :, 0] = (regions[:, :, :, 0] - 0.485) / 0.229
    regions[:, :, :, 1] = (regions[:, :, :, 1] - 0.456) / 0.224
    regions[:, :, :, 2] = (regions[:, :, :, 2] - 0.406) / 0.225
    regions = np.transpose(regions, (0, 3, 1, 2))
    # regions = np.expand_dims(regions, axis=0)
    # regions = np.tile(regions, (2,1,1,1))

    return regions

class Region:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

class MobileTracker(object):
    def __init__(self, image, region, imagefile=None, video=None, p=None, groundtruth=None):
        # seed(1)
        # set_random_seed(2)
        # torch.manual_seed(456)
        # torch.cuda.manual_seed(789)
        self.p = p
        self.i = 0
        if groundtruth is not None:
            self.groundtruth = groundtruth

        result_path = os.path.join(save_path, self.p.name, 'longterm')
        RV_path = os.path.join(save_path, self.p.name, 'RV')
        if not os.path.exists(os.path.join(result_path, video)):
            os.makedirs(os.path.join(result_path, video))
        if not os.path.exists(RV_path):
            os.makedirs(RV_path)
        self.g_region = open(os.path.join(result_path, video, video + '_001.txt'), 'w')
        self.g_region.writelines('1\n')
        self.g_conf = open(os.path.join(result_path, video, video + '_001_confidence.value'), 'w')
        self.g_conf.writelines('\n')
        self.g_time = open(os.path.join(result_path, video, video + '_time.txt'), 'w')
        self.g_time.writelines(['0.021305\n'])

        tfconfig = tf.ConfigProto()
        tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.3
        self.sess = tf.Session(config=tfconfig)
        init_img = Image.fromarray(image)
        init_gt1 = [region.x,region.y,region.width,region.height]
        # init_gt1 = [region[0], region[1], region[2], region[3]]
        init_gt = [init_gt1[1], init_gt1[0], init_gt1[1]+init_gt1[3], init_gt1[0]+init_gt1[2]] # ymin xmin ymax xmax

        self.last_gt = init_gt


        # if self.p.Verification == "rtmdnet":
        #     self.init_rtmdnet(image, init_gt1)
        # else:
        #     ValueError()
        self.local_init(image, init_gt1)

        self.tc_init(self.p.model_dir)
        self.metric_init(image, np.array(init_gt1))
        self.dis_record = []
        self.state_record = []
        self.rv_record = []
        self.all_map = []


        local_state1, self.score_map, update, self.score_max, dis = self.local_track(image)
        self.local_Tracker.pos = torch.FloatTensor(
            [(self.last_gt[0] + self.last_gt[2] - 1) / 2, (self.last_gt[1] + self.last_gt[3] - 1) / 2])
        self.local_Tracker.target_sz = torch.FloatTensor(
            [(self.last_gt[2] - self.last_gt[0]), (self.last_gt[3] - self.last_gt[1])])


    def get_first_state(self):
        return self.score_map, self.score_max


    def metric_init(self, im, init_box):
        self.metric_model = ft_net(class_num=1120)
        path = 'tracker_controller/metric_net/metric_model/metric_model_zj_57470.pt'
        self.metric_model.eval()
        self.metric_model = self.metric_model.cuda()
        self.metric_model.load_state_dict(torch.load(path))
        tmp = np.random.rand(1, 3, 107, 107)
        tmp = (Variable(torch.Tensor(tmp))).type(torch.FloatTensor).cuda()
        # get target feature
        self.metric_model(tmp)
        init_box = init_box.reshape((1, 4))
        anchor_region = me_extract_regions(im, init_box)
        anchor_region = process_regions(anchor_region)
        anchor_region = torch.Tensor(anchor_region)
        anchor_region = (Variable(anchor_region)).type(torch.FloatTensor).cuda()
        self.anchor_feature, _ = self.metric_model(anchor_region)

    def metric_eval(self, im, boxes, anchor_feature):
        box_regions = me_extract_regions(np.array(im), boxes)
        box_regions = process_regions(box_regions)
        box_regions = torch.Tensor(box_regions)
        box_regions = (Variable(box_regions)).type(torch.FloatTensor).cuda()
        box_features, class_result = self.metric_model(box_regions)

        class_result = torch.softmax(class_result, dim=1)
        ap_dist = torch.norm(anchor_feature - box_features, 2, dim=1).view(-1)
        return ap_dist

    def tc_init(self, model_dir):
        self.tc_model = tclstm()
        self.X_input = tf.placeholder("float", [None, tcopts['time_steps'], tcopts['lstm_num_input']])
        self.maps = tf.placeholder("float", [None, 19, 19, 1])
        self.map_logits = self.tc_model.map_net(self.maps)
        self.Inputs = tf.concat((self.X_input, self.map_logits), axis=2)
        self.logits, _ = self.tc_model.net(self.Inputs)

        variables_to_restore = [var for var in tf.global_variables() if
                                (var.name.startswith('tclstm') or var.name.startswith('mapnet'))]
        saver = tf.train.Saver(var_list=variables_to_restore)
        checkpoint = tf.train.latest_checkpoint(os.path.join('./tracker_controller', model_dir))
        # if checkpoint is not None:
        saver.restore(self.sess, checkpoint)

    def local_init(self, image, init_bbox):
        local_tracker = Tracker('dimp', 'dimp50')
        params = local_tracker.get_parameters()

        debug_ = getattr(params, 'debug', 0)
        params.debug = debug_

        params.tracker_name = local_tracker.name
        params.param_name = local_tracker.parameter_name

        self.local_Tracker = local_tracker.tracker_class(params)
        init_box = dict()
        init_box['init_bbox']=init_bbox
        self.local_Tracker.initialize(image, init_box)
        # if self.p.visualization:
        #     show_res(cv.cvtColor(image, cv.COLOR_RGB2BGR), np.array(self.last_gt, dtype=np.int32), '2', groundtruth=self.groundtruth,frame_id=self.i)
    def local_track(self, image):
        state, score_map, test_x, scale_ind, sample_pos, sample_scales, flag, s = self.local_Tracker.track_updater(image)
        update_flag = flag not in ['not_found', 'uncertain']
        update = update_flag
        max_score = max(score_map.flatten())
        self.all_map.append(score_map)
        local_state = np.array(state).reshape((1, 4))
        ap_dis = self.metric_eval(image, local_state, self.anchor_feature)
        self.dis_record.append(ap_dis.data.cpu().numpy()[0])
        h = image.shape[0]
        w = image.shape[1]
        self.state_record.append([local_state[0][0] / w, local_state[0][1] / h,
                                  (local_state[0][0] + local_state[0][2]) / w,
                                  (local_state[0][1] + local_state[0][3]) / h])
        self.rv_record.append(max_score)
        if len(self.state_record) >= tcopts['time_steps']:
            dis = np.array(self.dis_record[-tcopts["time_steps"]:]).reshape((tcopts["time_steps"], 1))
            rv = np.array(self.rv_record[-tcopts["time_steps"]:]).reshape((tcopts["time_steps"], 1))
            state_tc = np.array(self.state_record[-tcopts["time_steps"]:])
            map_input = np.array(self.all_map[-tcopts["time_steps"]:])
            map_input = np.reshape(map_input, [tcopts['time_steps'], 1, 19, 19])
            map_input = map_input.transpose((0, 2, 3, 1))
            X_input = np.concatenate((state_tc, rv, dis), axis=1)
            logits = self.sess.run(self.logits,
                                               feed_dict={self.X_input: np.expand_dims(X_input, axis=0),
                                                          self.maps: map_input})
            update = logits[0][0] < logits[0][1]


        hard_negative = (flag == 'hard_negative')
        learning_rate = getattr(self.local_Tracker.params, 'hard_negative_learning_rate', None) if hard_negative else None

        if update:
            # Get train sample
            train_x = test_x[scale_ind:scale_ind+1, ...]

            # Create target_box and label for spatial sample
            target_box = self.local_Tracker.get_iounet_box(self.local_Tracker.pos, self.local_Tracker.target_sz, sample_pos[scale_ind,:], sample_scales[scale_ind])

            # Update the classifier model
            self.local_Tracker.update_classifier(train_x, target_box, learning_rate, s[scale_ind,...])
        self.last_gt = [state[1], state[0], state[1]+state[3], state[0]+state[2]]
        # if self.p.visualization:
        #     show_res(cv.cvtColor(image, cv.COLOR_RGB2BGR), np.array(self.last_gt, dtype=np.int32), '2', groundtruth=self.groundtruth, frame_id=self.i)
        # print("%s: %d / %d" % (video, id, len(img_list)))
        return state, score_map, update, max_score, ap_dis.data.cpu().numpy()[0]
    def locate(self, image):

        # Convert image
        im = numpy_to_torch(image)
        self.local_Tracker.im = im  # For debugging only

        # ------- LOCALIZATION ------- #

        # Get sample
        sample_pos = self.local_Tracker.pos.round()
        sample_scales = self.local_Tracker.target_scale * self.local_Tracker.params.scale_factors
        test_x = self.local_Tracker.extract_processed_sample(im, self.local_Tracker.pos, sample_scales, self.local_Tracker.img_sample_sz)

        # Compute scores
        scores_raw = self.local_Tracker.apply_filter(test_x)
        translation_vec, scale_ind, s, flag = self.local_Tracker.localize_target(scores_raw)
        return translation_vec, scale_ind, s, flag, sample_pos, sample_scales, test_x
    def local_update(self, sample_pos, translation_vec, scale_ind, sample_scales, s, test_x, update_flag=None):

        # Check flags and set learning rate if hard negative
        if update_flag is None:
            update_flag = self.flag not in ['not_found', 'uncertain']
        hard_negative = (self.flag == 'hard_negative')
        learning_rate = self.local_Tracker.params.hard_negative_learning_rate if hard_negative else None

        if update_flag:
            # Get train sample
            train_x = TensorList([x[scale_ind:scale_ind + 1, ...] for x in test_x])

            # Create label for sample
            train_y = self.local_Tracker.get_label_function(sample_pos, sample_scales[scale_ind])

            # Update memory
            self.local_Tracker.update_memory(train_x, train_y, learning_rate)

        # Train filter
        if hard_negative:
            self.local_Tracker.filter_optimizer.run(self.local_Tracker.params.hard_negative_CG_iter)
        elif (self.local_Tracker.frame_num - 1) % self.local_Tracker.params.train_skipping == 0:
            self.local_Tracker.filter_optimizer.run(self.local_Tracker.params.CG_iter)




    def tracking2(self, image):
        self.i += 1
        local_state1, self.score_map, update, score_max, dis = self.local_track(image)
        gt_err = self.groundtruth[self.i, 2] < 3 or self.groundtruth[self.i, 3] < 3
        gt_nan = any(np.isnan(self.groundtruth[self.i]))
        if gt_err:
            iou = -1
        elif gt_nan:
            iou = 0
        else:
            iou = _compute_iou(self.groundtruth[self.i], local_state1)
        ##------------------------------------------------------##

        # self.local_update(sample_pos, translation_vec, scale_ind, sample_scales, s, test_x)

        width = self.last_gt[3] - self.last_gt[1]
        height = self.last_gt[2] - self.last_gt[0]

        self.g_conf.writelines(["%f" % score_max + '\n'])
        self.g_region.writelines(["%.4f" % float(self.last_gt[1]) + ',' + "%.4f" % float(
            self.last_gt[0]) + ',' + "%.4f" % float(width) + ',' + "%.4f" % float(height) + '\n'])
        self.g_time.writelines(['0.021305\n'])
        if self.p.visualization:
            show_res(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), np.array(self.last_gt, dtype=np.int32), '2',
                     groundtruth=self.groundtruth,score_max=0,confidence=update,
                     frame_id=self.i, score=max(self.score_map.flatten()))

        # print("frame: " + "%d  " % self.i + "Region: " + "%.2f" % float(self.last_gt[1]) + ",%.2f" % float(
        #     self.last_gt[0]) + ",%.2f" % float(width) + ",%.2f" % float(height))
        # return vot.Rectangle(float(outputs[1]), float(outputs[0]), float(outputs[3]-outputs[1]), float(outputs[2]-outputs[0])),confidence_score#scores[0,max_idx]
        return [float(self.last_gt[1]), float(self.last_gt[0]), float(width), float(height)], self.score_map, iou, score_max, dis

def eval_tracking(Dataset, video_spe=None, save=False, p=None, classes=None, mode=None):
    if Dataset == "votlt":
        data_dir = '/home/daikenan/dataset/VOT18_long'
    elif Dataset == 'otb':
        data_dir = '/home/dkn/daikenan/dataset/OTB'
    elif Dataset == "votlt19":
        data_dir = '/home/dkn/data/lt2019'
    elif Dataset == "tlp":
        data_dir = '/media/dkn/Data2/TLP'
    elif Dataset == "lasot":
        data_dir = os.path.join('/media/dkn/Data/LaSOTBenchmark/', classes)
    sequence_list = os.listdir(data_dir)
    sequence_list.sort()
    sequence_list = [title for title in sequence_list if not title.endswith("txt")]
    testing_set_dir = './tracker_controller/testing_set.txt'
    testing_set = list(np.loadtxt(testing_set_dir, dtype=str))
    if mode == 'test':
        print('test data')
        sequence_list = [vid for vid in sequence_list if vid in testing_set]
    elif mode == 'train':
        print('train data')
        sequence_list = [vid for vid in sequence_list if vid not in testing_set]
    else:
        print("all data")
    if video_spe is not None:
        sequence_list = [video_spe]

    m_shape = 19
    base_save_path = os.path.join('/media/dkn/Data2/dimp/',p.name, Dataset)
    if not os.path.exists(base_save_path):
        os.makedirs(base_save_path)

    for seq_id, video in enumerate(sequence_list):
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
        if Dataset == 'tlp':
            groundtruth = groundtruth[:, 1:5]
        region = Region(groundtruth[0, 0], groundtruth[0,1],groundtruth[0,2],groundtruth[0,3])
        image_dir = sequence_dir + image_list[0]
        image = cv2.cvtColor(cv2.imread(image_dir), cv2.COLOR_BGR2RGB)
        h = image.shape[0]
        w = image.shape[1]
        region1 = groundtruth[0]
        box = np.array([region1[0]/w, region1[1]/h, (region1[0]+region1[2])/w, (region1[1]+region1[3])/h])
        # if groundtruth[0, 2] <= 9 or groundtruth[0, 3] <= 9:
        #     continue
        tracker = MobileTracker(image, region, video=video, p=p,groundtruth=groundtruth)
        score_map, score_max = tracker.get_first_state()

        num_frames = len(image_list)
        all_map = np.zeros((num_frames, m_shape, m_shape))
        all_map[0] = cv2.resize(score_map, (m_shape, m_shape))
        bBoxes = np.zeros((num_frames, 8))
        bBoxes[0, :] = [box[0], box[1], box[2], box[3], 0, 1, score_max, 0]
        # bBoxes[0, :] = [box[0], box[1], box[2], box[3]]



        for im_id in range(1, len(image_list)):
            imagefile = sequence_dir + image_list[im_id]
            image = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)
            # image = cv2.imread(imagefile)
            # image = image[:,:,::-1]
            print("%d: " % seq_id + video + ": %d /" % im_id + "%d" % len(image_list))
            region, score_map, iou, score_max, dis = tracker.tracking2(image)
            all_map[im_id] = cv2.resize(score_map, (m_shape, m_shape))
            # region, _ = tracker.local_track(image)
            # region = tracker.pymdnet_track(image)
            # region, region_bb = tracker.rtmdnet_track(image)
            bBoxes[im_id, :] = region
            # box = np.array(
            #     [region[0] / w, region[1] / h, (region[0] + region[2]) / w, (region[1] + region[3]) / h])
            # bBoxes[im_id, :] = [box[0], box[1], box[2], box[3], im_id, iou, score_max, dis]
            # bBoxes2[im_id, :] = region
        if save:
            # np.savetxt(result_save_path, bBoxes, fmt="%.8f %.8f %.8f %.8f")
            np.savetxt(result_save_path, bBoxes, fmt="%.8f,%.8f,%.8f,%.8f,%d,%.8f,%.8f,%.8f")
            np.save(os.path.join(base_save_path, video+'_map'), all_map)
        # np.savetxt(os.path.join('/home/daikenan/Desktop/MDNet_tf/', video+'.txt'), bBoxes2, fmt="%.6f,%.6f,%.6f,%.6f")
        tracker.sess.close()
        tf.reset_default_graph()



if __name__ == '__main__':
    lasot_dir = '/media/dkn/Data/LaSOTBenchmark/'
    classes = os.listdir(lasot_dir)
    classes.sort()
    # # for i in range(len(videos)):
    # #     eval_tracking('votlt19', p=p, video_spe=videos[i])
    p = p_config()
    p.name = 'test3'
    p.start_frame = 20
    # eval_tracking('tlp', p=p, save=True)
    for c in classes:
        eval_tracking('lasot', p=p, save=True, classes=c, mode='all')

    # p = p_config()
    # p.name = 'test1_100'
    # p.start_frame = 100
    # for c in classes:
    #     eval_tracking('lasot', p=p, save=True, classes=c, mode='test')


