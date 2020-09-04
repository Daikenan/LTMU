from vot_path import base_path
import cv2
import os
import torch
import numpy as np
import math
import tensorflow as tf
import sys
sys.path.append(os.path.join(base_path, 'DiMP_LTMU/meta_updater'))
sys.path.append(os.path.join(base_path, 'utils/metric_net'))
from tcNet import tclstm
from tcopt import tcopts
from metric_model import ft_net
from torch.autograd import Variable
from me_sample_generator import *
import vot
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Dimp
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
from tracking_utils import compute_iou, show_res, process_regions

# Global Tracker
import Global_Track._init_paths
import neuron.data as data
from Global_tracker import *

sys.path.append(os.path.join(base_path, 'DiMP_LTMU/pyMDNet/modules'))
sys.path.append(os.path.join(base_path, 'DiMP_LTMU/pyMDNet/tracking'))
# pymdnet
from pyMDNet.modules.model import *
sys.path.insert(0, './pyMDNet')
from pyMDNet.modules.model import MDNet, BCELoss, set_optimizer
from pyMDNet.modules.sample_generator import SampleGenerator
from pyMDNet.modules.utils import overlap_ratio
from pyMDNet.tracking.data_prov import RegionExtractor
from pyMDNet.tracking.run_tracker import *
from bbreg import BBRegressor
from gen_config import gen_config
opts = yaml.safe_load(open(os.path.join(base_path, 'DiMP_LTMU/pyMDNet/tracking/options.yaml'), 'r'))

# siammask
sys.path.append(os.path.join(base_path, 'DiMP_LTMU/SiamMask'))
sys.path.append(os.path.join(base_path, 'DiMP_LTMU/SiamMask/experiments/siammask'))
from custom import Custom
from tools.test import *


class p_config(object):
    Verification = "rtmdnet"
    name = 'Dimp_MU'
    model_dir = 'dimp_mu_votlt'
    checkpoint = 220000
    start_frame = 200
    R_candidates = 20
    save_results = False
    use_mask = True
    save_training_data = False
    visualization = True


class Dimp_LTMU_Tracker(object):
    def __init__(self, image, region, p=None, groundtruth=None):
        self.p = p
        self.i = 0
        self.t_id = 0
        if groundtruth is not None:
            self.groundtruth = groundtruth

        tfconfig = tf.ConfigProto()
        tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.3
        self.sess = tf.Session(config=tfconfig)
        init_gt1 = [region.x, region.y, region.width, region.height]
        init_gt = [init_gt1[1], init_gt1[0], init_gt1[1]+init_gt1[3], init_gt1[0]+init_gt1[2]]  # ymin xmin ymax xmax

        self.last_gt = init_gt
        self.init_pymdnet(image, init_gt1)
        self.local_init(image, init_gt1)
        self.Golbal_Track_init(image, init_gt1)
        if self.p.use_mask:
            self.siammask_init(image, init_gt1)
        self.tc_init(self.p.model_dir)
        self.metric_init(image, np.array(init_gt1))
        self.dis_record = []
        self.state_record = []
        self.rv_record = []
        self.all_map = []
        self.count = 0
        local_state1, self.score_map, update, self.score_max, dis, flag = self.local_track(image)
        self.local_Tracker.pos = torch.FloatTensor(
            [(self.last_gt[0] + self.last_gt[2] - 1) / 2, (self.last_gt[1] + self.last_gt[3] - 1) / 2])
        self.local_Tracker.target_sz = torch.FloatTensor(
            [(self.last_gt[2] - self.last_gt[0]), (self.last_gt[3] - self.last_gt[1])])

    def get_first_state(self):
        return self.score_map, self.score_max

    def siammask_init(self, im, init_gt):
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        parser = argparse.ArgumentParser(description='PyTorch Tracking Demo')

        parser.add_argument('--resume', default=os.path.join(base_path, 'DiMP_LTMU/SiamMask/experiments/siammask/SiamMask_VOT_LD.pth'),
                            type=str,
                            metavar='PATH', help='path to latest checkpoint (default: none)')
        parser.add_argument('--config', dest='config',
                            default=os.path.join(base_path, 'DiMP_LTMU/SiamMask/experiments/siammask/config_vot19lt.json'),
                            help='hyper-parameter of SiamMask in json format')
        args = parser.parse_args()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.backends.cudnn.benchmark = True

        # Setup Model
        cfg = load_config(args)
        self.siammask = Custom(anchors=cfg['anchors'])
        if args.resume:
            assert isfile(args.resume), '{} is not a valid file'.format(args.resume)
            self.siammask = load_pretrain(self.siammask, args.resume)

        self.siammask.eval().to(device)
        x = init_gt[0]
        y = init_gt[1]
        w = init_gt[2]
        h = init_gt[3]
        target_pos = np.array([x + w / 2, y + h / 2])
        target_sz = np.array([w, h])
        self.siamstate = siamese_init(im, target_pos, target_sz, self.siammask, cfg['hp'])

    def siammask_track(self, im):
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        self.siamstate = siamese_track(self.siamstate, im, mask_enable=True, refine_enable=True)  # track
        # pdb.set_trace()
        score = np.max(self.siamstate['score'])
        location = self.siamstate['ploygon'].flatten()
        mask = self.siamstate['mask'] > self.siamstate['p'].seg_thr

        # im[:, :, 2] = (mask > 0) * 255 + (mask == 0) * im[:, :, 2]
        #
        # cv2.namedWindow("SiamMask", cv2.WINDOW_NORMAL)
        # cv2.rectangle(im, (int(self.siamstate['target_pos'][0] - self.siamstate['target_sz'][0] / 2.0),
        #                    int(self.siamstate['target_pos'][1] - self.siamstate['target_sz'][1] / 2.0)),
        #               (int(self.siamstate['target_pos'][0] + self.siamstate['target_sz'][0] / 2.0),
        #                int(self.siamstate['target_pos'][1] + self.siamstate['target_sz'][1] / 2.0)), [0, 255, 0], 2)
        # # cv2.imwrite("/home/xiaobai/Desktop/MBMD_vot_code/figure/%05d.jpg"%frame_id, im[:, :, -1::-1])
        # cv2.imshow("SiamMask", im)
        # cv2.waitKey(1)
        return score, mask

    def Golbal_Track_init(self, image, init_box):
        cfg_file = os.path.join(base_path, 'DiMP_LTMU/Global_Track/configs/qg_rcnn_r50_fpn.py')
        ckp_file = os.path.join(base_path, 'DiMP_LTMU/Global_Track/checkpoints/qg_rcnn_r50_fpn_coco_got10k_lasot.pth')
        transforms = data.BasicPairTransforms(train=False)
        self.Global_Tracker = GlobalTrack(
            cfg_file, ckp_file, transforms,
            name_suffix='qg_rcnn_r50_fpn')
        self.Global_Tracker.init(image, init_box)

    def Global_Track_eval(self, image, num):
        # xywh
        results = self.Global_Tracker.update(image)
        index = np.argsort(results[:, -1])[::-1]
        max_index = index[:num]
        can_boxes = results[max_index][:, :4]
        can_boxes = np.array([can_boxes[:, 0], can_boxes[:, 1], can_boxes[:, 2]-can_boxes[:, 0], can_boxes[:, 3]-can_boxes[:, 1]]).transpose()
        return can_boxes

    def init_pymdnet(self, image, init_bbox):
        target_bbox = np.array(init_bbox)
        self.last_result = target_bbox
        self.pymodel = MDNet(os.path.join(base_path, 'DiMP_LTMU/pyMDNet/models/mdnet_imagenet_vid.pth'))
        if opts['use_gpu']:
            self.pymodel = self.pymodel.cuda()
        self.pymodel.set_learnable_params(opts['ft_layers'])

        # Init criterion and optimizer
        self.criterion = BCELoss()
        init_optimizer = set_optimizer(self.pymodel, opts['lr_init'], opts['lr_mult'])
        self.update_optimizer = set_optimizer(self.pymodel, opts['lr_update'], opts['lr_mult'])

        tic = time.time()

        # Draw pos/neg samples
        pos_examples = SampleGenerator('gaussian', image.size, opts['trans_pos'], opts['scale_pos'])(
            target_bbox, opts['n_pos_init'], opts['overlap_pos_init'])

        neg_examples = np.concatenate([
            SampleGenerator('uniform', image.size, opts['trans_neg_init'], opts['scale_neg_init'])(
                target_bbox, int(opts['n_neg_init'] * 0.5), opts['overlap_neg_init']),
            SampleGenerator('whole', image.size)(
                target_bbox, int(opts['n_neg_init'] * 0.5), opts['overlap_neg_init'])])
        neg_examples = np.random.permutation(neg_examples)

        # Extract pos/neg features
        pos_feats = forward_samples(self.pymodel, image, pos_examples, opts)
        neg_feats = forward_samples(self.pymodel, image, neg_examples, opts)
        self.feat_dim = pos_feats.size(-1)

        # Initial training
        train(self.pymodel, self.criterion, init_optimizer, pos_feats, neg_feats, opts['maxiter_init'], opts=opts)
        del init_optimizer, neg_feats
        torch.cuda.empty_cache()

        # Train bbox regressor
        bbreg_examples = SampleGenerator('uniform', image.size, opts['trans_bbreg'], opts['scale_bbreg'],
                                         opts['aspect_bbreg'])(
            target_bbox, opts['n_bbreg'], opts['overlap_bbreg'])
        bbreg_feats = forward_samples(self.pymodel, image, bbreg_examples, opts)
        self.bbreg = BBRegressor(image.size)
        self.bbreg.train(bbreg_feats, bbreg_examples, target_bbox)
        del bbreg_feats
        torch.cuda.empty_cache()
        # Init sample generators
        self.sample_generator = SampleGenerator('gaussian', image.size, opts['trans'], opts['scale'])
        self.pos_generator = SampleGenerator('gaussian', image.size, opts['trans_pos'], opts['scale_pos'])
        self.neg_generator = SampleGenerator('uniform', image.size, opts['trans_neg'], opts['scale_neg'])

        # Init pos/neg features for update
        neg_examples = self.neg_generator(target_bbox, opts['n_neg_update'], opts['overlap_neg_init'])
        neg_feats = forward_samples(self.pymodel, image, neg_examples, opts)
        self.pos_feats_all = [pos_feats]
        self.neg_feats_all = [neg_feats]

        spf_total = time.time() - tic

    def pymdnet_eval(self, image, samples):
        sample_scores = forward_samples(self.pymodel, image, samples, out_layer='fc6', opts=opts)
        return sample_scores[:, 1][:].cpu().numpy()
    # def pymdnet_track(self, image):
    #     self.image = image
    #     target_bbox = self.last_result
    #     samples = self.sample_generator(target_bbox, opts['n_samples'])
    #     sample_scores = forward_samples(self.pymodel, image, samples, out_layer='fc6', opts=opts)
    #
    #     top_scores, top_idx = sample_scores[:, 1].topk(5)
    #     top_idx = top_idx.cpu().numpy()
    #     target_score = top_scores.mean()
    #     target_bbox = samples[top_idx].mean(axis=0)
    #
    #     success = target_score > 0
    #
    #     # Expand search area at failure
    #     if success:
    #         self.sample_generator.set_trans(opts['trans'])
    #     else:
    #         self.sample_generator.expand_trans(opts['trans_limit'])
    #
    #     self.last_result = target_bbox
    #     # Bbox regression
    #     bbreg_bbox = self.pymdnet_bbox_reg(success, samples, top_idx)
    #
    #     # Save result
    #     region = bbreg_bbox
    #
    #     # Data collect
    #     if success:
    #         self.collect_samples_pymdnet()
    #
    #     # Short term update
    #     if not success:
    #         self.pymdnet_short_term_update()
    #
    #     # Long term update
    #     elif self.i % opts['long_interval'] == 0:
    #         self.pymdnet_long_term_update()
    #
    #     return region, target_score

    def collect_samples_pymdnet(self, image):
        self.t_id += 1
        target_bbox = np.array([self.last_gt[1], self.last_gt[0], self.last_gt[3]-self.last_gt[1], self.last_gt[2]-self.last_gt[0]])
        pos_examples = self.pos_generator(target_bbox, opts['n_pos_update'], opts['overlap_pos_update'])
        if len(pos_examples) > 0:
            pos_feats = forward_samples(self.pymodel, image, pos_examples, opts)
            self.pos_feats_all.append(pos_feats)
        if len(self.pos_feats_all) > opts['n_frames_long']:
            del self.pos_feats_all[0]

        neg_examples = self.neg_generator(target_bbox, opts['n_neg_update'], opts['overlap_neg_update'])
        if len(neg_examples) > 0:
            neg_feats = forward_samples(self.pymodel, image, neg_examples, opts)
            self.neg_feats_all.append(neg_feats)
        if len(self.neg_feats_all) > opts['n_frames_short']:
            del self.neg_feats_all[0]

    def pymdnet_short_term_update(self):
        # Short term update
        nframes = min(opts['n_frames_short'], len(self.pos_feats_all))
        pos_data = torch.cat(self.pos_feats_all[-nframes:], 0)
        neg_data = torch.cat(self.neg_feats_all, 0)
        train(self.pymodel, self.criterion, self.update_optimizer, pos_data, neg_data, opts['maxiter_update'],
              opts=opts)

    def pymdnet_long_term_update(self):
        if self.t_id % opts['long_interval'] == 0:
            # Long term update
            pos_data = torch.cat(self.pos_feats_all, 0)
            neg_data = torch.cat(self.neg_feats_all, 0)
            train(self.pymodel, self.criterion, self.update_optimizer, pos_data, neg_data, opts['maxiter_update'],
                  opts=opts)
    #
    # def pymdnet_bbox_reg(self, success, samples, top_idx):
    #     target_bbox = self.last_result
    #     if success:
    #         bbreg_samples = samples[top_idx]
    #         if top_idx.shape[0] == 1:
    #             bbreg_samples = bbreg_samples[None, :]
    #         bbreg_feats = forward_samples(self.pymodel, self.image, bbreg_samples, opts)
    #         bbreg_samples = self.bbreg.predict(bbreg_feats, bbreg_samples)
    #         bbreg_bbox = bbreg_samples.mean(axis=0)
    #     else:
    #         bbreg_bbox = target_bbox
    #     return bbreg_bbox

    def metric_init(self, im, init_box):
        self.metric_model = ft_net(class_num=1120)
        path = os.path.join(base_path, 'utils/metric_net/metric_model/metric_model.pt')
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
        if self.p.checkpoint is None:
            checkpoint = tf.train.latest_checkpoint(os.path.join(base_path, 'DiMP_LTMU/meta_updater', model_dir))
        else:
            checkpoint = os.path.join(base_path, 'DiMP_LTMU/meta_updater/' + self.p.model_dir + '/lstm_model.ckpt-' + str(self.p.checkpoint))
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
        init_box['init_bbox'] = init_bbox
        self.local_Tracker.initialize(image, init_box)

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
        if len(self.state_record) >= self.p.start_frame:
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
            target_box = self.local_Tracker.get_iounet_box(self.local_Tracker.pos, self.local_Tracker.target_sz,
                                                           sample_pos[scale_ind, :], sample_scales[scale_ind])

            # Update the classifier model
            self.local_Tracker.update_classifier(train_x, target_box, learning_rate, s[scale_ind,...])
        self.last_gt = [state[1], state[0], state[1]+state[3], state[0]+state[2]]
        return state, score_map, update, max_score, ap_dis.data.cpu().numpy()[0], flag

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


    def tracking(self, image):
        self.i += 1
        mask = None
        candidate_bboxes = None
        # state, pyscore = self.pymdnet_track(image)
        # self.last_gt = [state[1], state[0], state[1] + state[3], state[0] + state[2]]
        self.local_Tracker.pos = torch.FloatTensor(
            [(self.last_gt[0] + self.last_gt[2] - 1) / 2, (self.last_gt[1] + self.last_gt[3] - 1) / 2])
        self.local_Tracker.target_sz = torch.FloatTensor(
            [(self.last_gt[2] - self.last_gt[0]), (self.last_gt[3] - self.last_gt[1])])
        tic = time.time()
        local_state, self.score_map, update, local_score, dis, flag = self.local_track(image)

        md_score = self.pymdnet_eval(image, np.array(local_state).reshape([-1, 4]))[0]
        self.score_max = md_score

        if md_score > 0 and flag == 'normal':
            self.flag = 'found'
            if self.p.use_mask:
                self.siamstate['target_pos'] = self.local_Tracker.pos.numpy()[::-1]
                self.siamstate['target_sz'] = self.local_Tracker.target_sz.numpy()[::-1]
                siamscore, mask = self.siammask_track(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                self.local_Tracker.pos = torch.FloatTensor(self.siamstate['target_pos'][::-1].copy())
                self.local_Tracker.target_sz = torch.FloatTensor(self.siamstate['target_sz'][::-1].copy())
                local_state = torch.cat(
                    (self.local_Tracker.pos[[1, 0]] - (self.local_Tracker.target_sz[[1, 0]] - 1) / 2,
                     self.local_Tracker.target_sz[[1, 0]])).data.cpu().numpy()
            self.last_gt = np.array(
                [local_state[1], local_state[0], local_state[1] + local_state[3], local_state[0] + local_state[2]])
        elif md_score < 0 or flag == 'not_found':
            self.count += 1
            self.flag = 'not_found'
            candidate_bboxes = self.Global_Track_eval(image, 10)
            candidate_scores = self.pymdnet_eval(image, candidate_bboxes)
            max_id = np.argmax(candidate_scores)
            if candidate_scores[max_id] > 0:
                redet_bboxes = candidate_bboxes[max_id]
                if self.count >= 5:
                    self.last_gt = np.array([redet_bboxes[1], redet_bboxes[0], redet_bboxes[1] + redet_bboxes[3],
                                             redet_bboxes[2] + redet_bboxes[0]])
                    self.local_Tracker.pos = torch.FloatTensor(
                        [(self.last_gt[0] + self.last_gt[2] - 1) / 2, (self.last_gt[1] + self.last_gt[3] - 1) / 2])
                    self.local_Tracker.target_sz = torch.FloatTensor(
                        [(self.last_gt[2] - self.last_gt[0]), (self.last_gt[3] - self.last_gt[1])])
                    self.score_max = candidate_scores[max_id]
                    self.count = 0
        if update:
            self.collect_samples_pymdnet(image)

        self.pymdnet_long_term_update()

        width = self.last_gt[3] - self.last_gt[1]
        height = self.last_gt[2] - self.last_gt[0]
        toc = time.time() - tic
        print(toc)
        # if self.flag == 'found' and self.score_max > 0:
        #     confidence_score = 0.99
        # elif self.flag == 'not_found':
        #     confidence_score = 0.0
        # else:
        #     confidence_score = np.clip((local_score+np.arctan(0.2*self.score_max)/math.pi+0.5)/2, 0, 1)
        confidence_score = np.clip((local_score + np.arctan(0.2 * self.score_max) / math.pi + 0.5) / 2, 0, 1)
        if self.p.visualization:
            show_res(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), np.array(self.last_gt, dtype=np.int32), '2',
                     update=update, can_bboxes=candidate_bboxes,
                     frame_id=self.i, tracker_score=md_score, mask=mask)

        return vot.Rectangle(float(self.last_gt[1]), float(self.last_gt[0]), float(width),
                float(height)), confidence_score


handle = vot.VOT("rectangle")

selection = handle.region()
imagefile = handle.frame()
p = p_config()
# if not imagefile:
#     sys.exit(0)
image = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)
tracker = Dimp_LTMU_Tracker(image, selection, p=p)

while True:
    imagefile = handle.frame()
    #print(imagefile)
    if not imagefile:
        break
    image = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)
    region, confidence = tracker.tracking(image)
    handle.report(region, confidence)

