#coding=utf-8
import cv2 as cv
import os
import time
import tensorflow as tf
import numpy as np
import sys

sys.path.append('./RT_MDNet')
sys.path.insert(0,'./RT_MDNet/modules')
sys.path.append('./meta_updater')
sys.path.append(os.path.join('../utils/metric_net'))

from tcNet import tclstm
from tcopt import tcopts

from metric_model import ft_net
from me_sample_generator import *
# rtmdnet
from rtmdnet_utils import *
from rt_sample_generator import *
from data_prov import *

from rtmdnet_model import *
from rtmdnet_options import *
from img_cropper import *

from roi_align.modules.roi_align import RoIAlignAvg,RoIAlignMax,RoIAlignAdaMax,RoIAlignDenseAdaMax

from bbreg import *

from RT_MDNet.tracker import set_optimizer, rt_train


from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000
from rt_sample_generator import *
from tracking_utils import *




class p_config(object):
    name = 'a'
    lose_count = 5
    R_loss_thr = 0.3
    R_center_redet = 0.3
    R_global_redet = 0.5
    Verification = "rtmdnet"
    Regressor = "mrpn"
    visualization = False

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
    regions[:,:, :, 0] = (regions[:,:, :, 0] - 0.485) / 0.229
    regions[:,:, :, 1] = (regions[:,:, :, 1] - 0.456) / 0.224
    regions[:,:, :, 2] = (regions[:,:, :, 2] - 0.406) / 0.225
    regions = np.transpose(regions, (0,3, 1, 2))
    #regions = np.expand_dims(regions, axis=0)
    #regions = np.tile(regions, (2,1,1,1))
    return regions

def get_mmresult(img, result, dataset='coco', score_thr=0.3):
    # class_names = get_classes(dataset)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(result)
    ]
    labels = np.concatenate(labels)
    bboxes = np.vstack(result)
    # img = mmcv.imread(img)
    # mmcv.imshow_det_bboxes(
    #     img.copy(),
    #     bboxes,
    #     labels,
    #     class_names=class_names,
    #     score_thr=score_thr)
    if score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]
    return bboxes, labels
class Region:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

class RTMD_MU_Tracker(object):
    def __init__(self, image, region, imagefile=None, video=None, p=None, groundtruth=None):
        # seed(1)
        # set_random_seed(2)
        # torch.manual_seed(456)
        # torch.cuda.manual_seed(789)
        self.p = p
        self.i = 0
        self.globalmode = True
        if groundtruth is not None:
            self.groundtruth = groundtruth

        tfconfig = tf.ConfigProto()
        tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.1
        self.sess = tf.Session(config=tfconfig)
        init_img = Image.fromarray(image)
        init_gt1 = [region.x,region.y,region.width,region.height]
        # init_gt1 = [region[0], region[1], region[2], region[3]]
        init_gt = [init_gt1[1], init_gt1[0], init_gt1[1]+init_gt1[3], init_gt1[0]+init_gt1[2]] # ymin xmin ymax xmax
        # tc init
        self.tc_init(self.p.model_dir)
        self.metric_init(image, np.array(init_gt1))
        self.last_gt = init_gt



        self.init_rtmdnet(image, init_gt1)


        target_bbox = self.last_result
        ishape = image.shape
        samples = gen_samples(
            SampleGenerator('gaussian', (ishape[1], ishape[0]), self.trans_f, rt_opts['scale_f'], valid=True),
            target_bbox, rt_opts['n_samples'])
        sample_scores, sample_feats = self.rtmdnet_eval(samples, image, target_bbox)

        first_score, top_idx = sample_scores[:, 1].topk(5)
        self.first_score = first_score.data.cpu().numpy()
        self.dis_record = []
        self.state_record = []
        self.rv_record = []

    def init_rtmdnet(self, image, init_bbox):
        target_bbox = np.array(init_bbox)
        self.last_result = target_bbox
        self.rtmodel = RTMDNet(rt_opts['model_path'])
        if rt_opts['adaptive_align']:
            align_h = self.rtmodel.roi_align_model.aligned_height
            align_w = self.rtmodel.roi_align_model.aligned_width
            spatial_s = self.rtmodel.roi_align_model.spatial_scale
            self.rtmodel.roi_align_model = RoIAlignAdaMax(align_h, align_w, spatial_s)
        if rt_opts['use_gpu']:
            self.rtmodel = self.rtmodel.cuda()

            self.rtmodel.set_learnable_params(rt_opts['ft_layers'])

        # Init image crop model
        self.img_crop_model = imgCropper(1.)
        if rt_opts['use_gpu']:
            self.img_crop_model.gpuEnable()

        # Init criterion and optimizer
        self.criterion = BinaryLoss()
        init_optimizer = set_optimizer(self.rtmodel, rt_opts['lr_init'])
        self.rtupdate_optimizer = set_optimizer(self.rtmodel, rt_opts['lr_update'])

        tic = time.time()
        # Load first image
        cur_image = Image.fromarray(image).convert('RGB')
        cur_image = np.asarray(cur_image)

        # Draw pos/neg samples
        ishape = cur_image.shape
        pos_examples = gen_samples(RT_SampleGenerator('gaussian', (ishape[1], ishape[0]), 0.1, 1.2),
                                   target_bbox, rt_opts['n_pos_init'], rt_opts['overlap_pos_init'])
        neg_examples = gen_samples(RT_SampleGenerator('uniform', (ishape[1], ishape[0]), 1, 2, 1.1),
                                   target_bbox, rt_opts['n_neg_init'], rt_opts['overlap_neg_init'])
        neg_examples = np.random.permutation(neg_examples)

        cur_bbreg_examples = gen_samples(SampleGenerator('uniform', (ishape[1], ishape[0]), 0.3, 1.5, 1.1),
                                         target_bbox, rt_opts['n_bbreg'], rt_opts['overlap_bbreg'], rt_opts['scale_bbreg'])

        # compute padded sample
        padded_x1 = (neg_examples[:, 0] - neg_examples[:, 2] * (rt_opts['padding'] - 1.) / 2.).min()
        padded_y1 = (neg_examples[:, 1] - neg_examples[:, 3] * (rt_opts['padding'] - 1.) / 2.).min()
        padded_x2 = (neg_examples[:, 0] + neg_examples[:, 2] * (rt_opts['padding'] + 1.) / 2.).max()
        padded_y2 = (neg_examples[:, 1] + neg_examples[:, 3] * (rt_opts['padding'] + 1.) / 2.).max()
        padded_scene_box = np.reshape(np.asarray((padded_x1, padded_y1, padded_x2 - padded_x1, padded_y2 - padded_y1)),
                                      (1, 4))

        scene_boxes = np.reshape(np.copy(padded_scene_box), (1, 4))
        if rt_opts['jitter']:
            ## horizontal shift
            jittered_scene_box_horizon = np.copy(padded_scene_box)
            jittered_scene_box_horizon[0, 0] -= 4.
            jitter_scale_horizon = 1.

            ## vertical shift
            jittered_scene_box_vertical = np.copy(padded_scene_box)
            jittered_scene_box_vertical[0, 1] -= 4.
            jitter_scale_vertical = 1.

            jittered_scene_box_reduce1 = np.copy(padded_scene_box)
            jitter_scale_reduce1 = 1.1 ** (-1)

            ## vertical shift
            jittered_scene_box_enlarge1 = np.copy(padded_scene_box)
            jitter_scale_enlarge1 = 1.1 ** (1)

            ## scale reduction
            jittered_scene_box_reduce2 = np.copy(padded_scene_box)
            jitter_scale_reduce2 = 1.1 ** (-2)
            ## scale enlarge
            jittered_scene_box_enlarge2 = np.copy(padded_scene_box)
            jitter_scale_enlarge2 = 1.1 ** (2)

            scene_boxes = np.concatenate(
                [scene_boxes, jittered_scene_box_horizon, jittered_scene_box_vertical, jittered_scene_box_reduce1,
                 jittered_scene_box_enlarge1, jittered_scene_box_reduce2, jittered_scene_box_enlarge2], axis=0)
            jitter_scale = [1., jitter_scale_horizon, jitter_scale_vertical, jitter_scale_reduce1,
                            jitter_scale_enlarge1, jitter_scale_reduce2, jitter_scale_enlarge2]
        else:
            jitter_scale = [1.]

            self.rtmodel.eval()
        for bidx in range(0, scene_boxes.shape[0]):
            crop_img_size = (scene_boxes[bidx, 2:4] * ((rt_opts['img_size'], rt_opts['img_size']) / target_bbox[2:4])).astype(
                'int64') * jitter_scale[bidx]
            cropped_image, cur_image_var = self.img_crop_model.crop_image(cur_image, np.reshape(scene_boxes[bidx], (1, 4)),
                                                                     crop_img_size)
            cropped_image = cropped_image - 128.

            feat_map = self.rtmodel(cropped_image, out_layer='conv3')

            rel_target_bbox = np.copy(target_bbox)
            rel_target_bbox[0:2] -= scene_boxes[bidx, 0:2]

            batch_num = np.zeros((pos_examples.shape[0], 1))
            cur_pos_rois = np.copy(pos_examples)
            cur_pos_rois[:, 0:2] -= np.repeat(np.reshape(scene_boxes[bidx, 0:2], (1, 2)), cur_pos_rois.shape[0], axis=0)
            scaled_obj_size = float(rt_opts['img_size']) * jitter_scale[bidx]
            cur_pos_rois = samples2maskroi(cur_pos_rois, self.rtmodel.receptive_field, (scaled_obj_size, scaled_obj_size),
                                           target_bbox[2:4], rt_opts['padding'])
            cur_pos_rois = np.concatenate((batch_num, cur_pos_rois), axis=1)
            cur_pos_rois = Variable(torch.from_numpy(cur_pos_rois.astype('float32'))).cuda()
            cur_pos_feats = self.rtmodel.roi_align_model(feat_map, cur_pos_rois)
            cur_pos_feats = cur_pos_feats.view(cur_pos_feats.size(0), -1).data.clone()

            batch_num = np.zeros((neg_examples.shape[0], 1))
            cur_neg_rois = np.copy(neg_examples)
            cur_neg_rois[:, 0:2] -= np.repeat(np.reshape(scene_boxes[bidx, 0:2], (1, 2)), cur_neg_rois.shape[0], axis=0)
            cur_neg_rois = samples2maskroi(cur_neg_rois, self.rtmodel.receptive_field, (scaled_obj_size, scaled_obj_size),
                                           target_bbox[2:4], rt_opts['padding'])
            cur_neg_rois = np.concatenate((batch_num, cur_neg_rois), axis=1)
            cur_neg_rois = Variable(torch.from_numpy(cur_neg_rois.astype('float32'))).cuda()
            cur_neg_feats = self.rtmodel.roi_align_model(feat_map, cur_neg_rois)
            cur_neg_feats = cur_neg_feats.view(cur_neg_feats.size(0), -1).data.clone()

            ## bbreg rois
            batch_num = np.zeros((cur_bbreg_examples.shape[0], 1))
            cur_bbreg_rois = np.copy(cur_bbreg_examples)
            cur_bbreg_rois[:, 0:2] -= np.repeat(np.reshape(scene_boxes[bidx, 0:2], (1, 2)), cur_bbreg_rois.shape[0],
                                                axis=0)
            scaled_obj_size = float(rt_opts['img_size']) * jitter_scale[bidx]
            cur_bbreg_rois = samples2maskroi(cur_bbreg_rois, self.rtmodel.receptive_field, (scaled_obj_size, scaled_obj_size),
                                             target_bbox[2:4], rt_opts['padding'])
            cur_bbreg_rois = np.concatenate((batch_num, cur_bbreg_rois), axis=1)
            cur_bbreg_rois = Variable(torch.from_numpy(cur_bbreg_rois.astype('float32'))).cuda()
            cur_bbreg_feats = self.rtmodel.roi_align_model(feat_map, cur_bbreg_rois)
            cur_bbreg_feats = cur_bbreg_feats.view(cur_bbreg_feats.size(0), -1).data.clone()

            self.rtfeat_dim = cur_pos_feats.size(-1)

            if bidx == 0:
                pos_feats = cur_pos_feats
                neg_feats = cur_neg_feats
                ##bbreg feature
                bbreg_feats = cur_bbreg_feats
                bbreg_examples = cur_bbreg_examples
            else:
                pos_feats = torch.cat((pos_feats, cur_pos_feats), dim=0)
                neg_feats = torch.cat((neg_feats, cur_neg_feats), dim=0)
                ##bbreg feature
                bbreg_feats = torch.cat((bbreg_feats, cur_bbreg_feats), dim=0)
                bbreg_examples = np.concatenate((bbreg_examples, cur_bbreg_examples), axis=0)

        if pos_feats.size(0) > rt_opts['n_pos_init']:
            pos_idx = np.asarray(range(pos_feats.size(0)))
            np.random.shuffle(pos_idx)
            pos_feats = pos_feats[pos_idx[0:rt_opts['n_pos_init']], :]
        if neg_feats.size(0) > rt_opts['n_neg_init']:
            neg_idx = np.asarray(range(neg_feats.size(0)))
            np.random.shuffle(neg_idx)
            neg_feats = neg_feats[neg_idx[0:rt_opts['n_neg_init']], :]

        ##bbreg
        if bbreg_feats.size(0) > rt_opts['n_bbreg']:
            bbreg_idx = np.asarray(range(bbreg_feats.size(0)))
            np.random.shuffle(bbreg_idx)
            bbreg_feats = bbreg_feats[bbreg_idx[0:rt_opts['n_bbreg']], :]
            bbreg_examples = bbreg_examples[bbreg_idx[0:rt_opts['n_bbreg']], :]
            # print bbreg_examples.shape

        ## open images and crop patch from obj
        extra_obj_size = np.array((rt_opts['img_size'], rt_opts['img_size']))
        extra_crop_img_size = extra_obj_size * (rt_opts['padding'] + 0.6)
        replicateNum = 100
        for iidx in range(replicateNum):
            extra_target_bbox = np.copy(target_bbox)

            extra_scene_box = np.copy(extra_target_bbox)
            extra_scene_box_center = extra_scene_box[0:2] + extra_scene_box[2:4] / 2.
            extra_scene_box_size = extra_scene_box[2:4] * (rt_opts['padding'] + 0.6)
            extra_scene_box[0:2] = extra_scene_box_center - extra_scene_box_size / 2.
            extra_scene_box[2:4] = extra_scene_box_size

            extra_shift_offset = np.clip(2. * np.random.randn(2), -4, 4)
            cur_extra_scale = 1.1 ** np.clip(np.random.randn(1), -2, 2)

            extra_scene_box[0] += extra_shift_offset[0]
            extra_scene_box[1] += extra_shift_offset[1]
            extra_scene_box[2:4] *= cur_extra_scale[0]

            scaled_obj_size = float(rt_opts['img_size']) / cur_extra_scale[0]

            cur_extra_cropped_image, _ = self.img_crop_model.crop_image(cur_image, np.reshape(extra_scene_box, (1, 4)),
                                                                   extra_crop_img_size)
            cur_extra_cropped_image = cur_extra_cropped_image.detach()

            cur_extra_pos_examples = gen_samples(SampleGenerator('gaussian', (ishape[1], ishape[0]), 0.1, 1.2),
                                                 extra_target_bbox, rt_opts['n_pos_init'] // replicateNum,
                                                 rt_opts['overlap_pos_init'])
            cur_extra_neg_examples = gen_samples(SampleGenerator('uniform', (ishape[1], ishape[0]), 0.3, 2, 1.1),
                                                 extra_target_bbox, rt_opts['n_neg_init'] // replicateNum // 4,
                                                 rt_opts['overlap_neg_init'])

            ##bbreg sample
            cur_extra_bbreg_examples = gen_samples(SampleGenerator('uniform', (ishape[1], ishape[0]), 0.3, 1.5, 1.1),
                                                   extra_target_bbox, rt_opts['n_bbreg'] // replicateNum // 4,
                                                   rt_opts['overlap_bbreg'], rt_opts['scale_bbreg'])

            batch_num = iidx * np.ones((cur_extra_pos_examples.shape[0], 1))
            cur_extra_pos_rois = np.copy(cur_extra_pos_examples)
            cur_extra_pos_rois[:, 0:2] -= np.repeat(np.reshape(extra_scene_box[0:2], (1, 2)),
                                                    cur_extra_pos_rois.shape[0], axis=0)
            cur_extra_pos_rois = samples2maskroi(cur_extra_pos_rois, self.rtmodel.receptive_field,
                                                 (scaled_obj_size, scaled_obj_size), extra_target_bbox[2:4],
                                                 rt_opts['padding'])
            cur_extra_pos_rois = np.concatenate((batch_num, cur_extra_pos_rois), axis=1)

            batch_num = iidx * np.ones((cur_extra_neg_examples.shape[0], 1))
            cur_extra_neg_rois = np.copy(cur_extra_neg_examples)
            cur_extra_neg_rois[:, 0:2] -= np.repeat(np.reshape(extra_scene_box[0:2], (1, 2)),
                                                    cur_extra_neg_rois.shape[0], axis=0)
            cur_extra_neg_rois = samples2maskroi(cur_extra_neg_rois, self.rtmodel.receptive_field,
                                                 (scaled_obj_size, scaled_obj_size), extra_target_bbox[2:4],
                                                 rt_opts['padding'])
            cur_extra_neg_rois = np.concatenate((batch_num, cur_extra_neg_rois), axis=1)

            ## bbreg rois
            batch_num = iidx * np.ones((cur_extra_bbreg_examples.shape[0], 1))
            cur_extra_bbreg_rois = np.copy(cur_extra_bbreg_examples)
            cur_extra_bbreg_rois[:, 0:2] -= np.repeat(np.reshape(extra_scene_box[0:2], (1, 2)),
                                                      cur_extra_bbreg_rois.shape[0], axis=0)
            cur_extra_bbreg_rois = samples2maskroi(cur_extra_bbreg_rois, self.rtmodel.receptive_field,
                                                   (scaled_obj_size, scaled_obj_size), extra_target_bbox[2:4],
                                                   rt_opts['padding'])
            cur_extra_bbreg_rois = np.concatenate((batch_num, cur_extra_bbreg_rois), axis=1)

            if iidx == 0:
                extra_cropped_image = cur_extra_cropped_image

                extra_pos_rois = np.copy(cur_extra_pos_rois)
                extra_neg_rois = np.copy(cur_extra_neg_rois)
                ##bbreg rois
                extra_bbreg_rois = np.copy(cur_extra_bbreg_rois)
                extra_bbreg_examples = np.copy(cur_extra_bbreg_examples)
            else:
                extra_cropped_image = torch.cat((extra_cropped_image, cur_extra_cropped_image), dim=0)

                extra_pos_rois = np.concatenate((extra_pos_rois, np.copy(cur_extra_pos_rois)), axis=0)
                extra_neg_rois = np.concatenate((extra_neg_rois, np.copy(cur_extra_neg_rois)), axis=0)
                ##bbreg rois
                extra_bbreg_rois = np.concatenate((extra_bbreg_rois, np.copy(cur_extra_bbreg_rois)), axis=0)
                extra_bbreg_examples = np.concatenate((extra_bbreg_examples, np.copy(cur_extra_bbreg_examples)), axis=0)

        extra_pos_rois = Variable(torch.from_numpy(extra_pos_rois.astype('float32'))).cuda()
        extra_neg_rois = Variable(torch.from_numpy(extra_neg_rois.astype('float32'))).cuda()
        ##bbreg rois
        extra_bbreg_rois = Variable(torch.from_numpy(extra_bbreg_rois.astype('float32'))).cuda()

        extra_cropped_image -= 128.

        extra_feat_maps = self.rtmodel(extra_cropped_image, out_layer='conv3')
        # Draw pos/neg samples
        ishape = cur_image.shape

        extra_pos_feats = self.rtmodel.roi_align_model(extra_feat_maps, extra_pos_rois)
        extra_pos_feats = extra_pos_feats.view(extra_pos_feats.size(0), -1).data.clone()

        extra_neg_feats = self.rtmodel.roi_align_model(extra_feat_maps, extra_neg_rois)
        extra_neg_feats = extra_neg_feats.view(extra_neg_feats.size(0), -1).data.clone()
        ##bbreg feat
        extra_bbreg_feats = self.rtmodel.roi_align_model(extra_feat_maps, extra_bbreg_rois)
        extra_bbreg_feats = extra_bbreg_feats.view(extra_bbreg_feats.size(0), -1).data.clone()

        ## concatenate extra features to original_features
        pos_feats = torch.cat((pos_feats, extra_pos_feats), dim=0)
        neg_feats = torch.cat((neg_feats, extra_neg_feats), dim=0)
        ## concatenate extra bbreg feats to original_bbreg_feats
        bbreg_feats = torch.cat((bbreg_feats, extra_bbreg_feats), dim=0)
        bbreg_examples = np.concatenate((bbreg_examples, extra_bbreg_examples), axis=0)

        torch.cuda.empty_cache()
        self.rtmodel.zero_grad()

        # Initial training
        rt_train(self.rtmodel, self.criterion, init_optimizer, pos_feats, neg_feats, rt_opts['maxiter_init'])
        ##bbreg train
        if bbreg_feats.size(0) > rt_opts['n_bbreg']:
            bbreg_idx = np.asarray(range(bbreg_feats.size(0)))
            np.random.shuffle(bbreg_idx)
            bbreg_feats = bbreg_feats[bbreg_idx[0:rt_opts['n_bbreg']], :]
            bbreg_examples = bbreg_examples[bbreg_idx[0:rt_opts['n_bbreg']], :]
        self.bbreg = BBRegressor((ishape[1], ishape[0]))
        self.bbreg.train(bbreg_feats, bbreg_examples, target_bbox)

        if pos_feats.size(0) > rt_opts['n_pos_update']:
            pos_idx = np.asarray(range(pos_feats.size(0)))
            np.random.shuffle(pos_idx)
            self.rtpos_feats_all = [pos_feats.index_select(0, torch.from_numpy(pos_idx[0:rt_opts['n_pos_update']]).cuda())]
        if neg_feats.size(0) > rt_opts['n_neg_update']:
            neg_idx = np.asarray(range(neg_feats.size(0)))
            np.random.shuffle(neg_idx)
            self.rtneg_feats_all = [neg_feats.index_select(0, torch.from_numpy(neg_idx[0:rt_opts['n_neg_update']]).cuda())]

        spf_total = time.time() - tic
        self.trans_f = rt_opts['trans_f']
        return

    def get_first_state(self):
        return self.first_score

    def rtmdnet_track(self, image):
        # self.i += 1
        cur_image = Image.fromarray(image).convert('RGB')
        cur_image = np.asarray(cur_image)

        target_bbox = self.last_result
        # Estimate target bbox
        ishape = cur_image.shape
        samples = gen_samples(SampleGenerator('gaussian', (ishape[1], ishape[0]), self.trans_f, rt_opts['scale_f'], valid=True),
                              target_bbox, rt_opts['n_samples'])
        sample_scores, sample_feats = self.rtmdnet_eval(samples, cur_image, target_bbox)
        top_scores, top_idx = sample_scores[:, 1].topk(5)
        top_idx = top_idx.data.cpu().numpy()
        target_score = top_scores.data.mean()
        target_bbox = samples[top_idx].mean(axis=0)
        self.last_result = target_bbox
        success = target_score > rt_opts['success_thr']

        # # Expand search area at failure
        if success:
            self.trans_f = rt_opts['trans_f']
        else:
            self.trans_f = rt_opts['trans_f_expand']

        ## Bbox regression
        if success:
            bbreg_feats = sample_feats[top_idx, :]
            bbreg_samples = samples[top_idx]
            bbreg_samples = self.bbreg.predict(bbreg_feats.data, bbreg_samples)
            bbreg_bbox = bbreg_samples.mean(axis=0)
        else:
            bbreg_bbox = target_bbox


        return target_bbox, bbreg_bbox, top_scores.data.cpu().numpy(),success

    def rtmdnet_eval(self, samples, cur_image, target_bbox):
        cur_image = np.asarray(cur_image)

        padded_x1 = (samples[:, 0] - (3*samples[:, 2]+1*samples[:, 3])/2.0 * (rt_opts['padding'] - 1.) / 2.).min()
        padded_y1 = (samples[:, 1] - (3*samples[:, 3]+1*samples[:, 2])/2.0 * (rt_opts['padding'] - 1.) / 2.).min()
        padded_x2 = (samples[:, 0] + (3*samples[:, 2]+1*samples[:, 3])/2.0 * (rt_opts['padding'] + 1.) / 2.).max()
        padded_y2 = (samples[:, 1] + (3*samples[:, 3]+1*samples[:, 2])/2.0 * (rt_opts['padding'] + 1.) / 2.).max()
        padded_scene_box = np.asarray((padded_x1, padded_y1, padded_x2 - padded_x1, padded_y2 - padded_y1))

        if padded_scene_box[0] > cur_image.shape[1]:
            padded_scene_box[0] = cur_image.shape[1] - 1
        if padded_scene_box[1] > cur_image.shape[0]:
            padded_scene_box[1] = cur_image.shape[0] - 1
        if padded_scene_box[0] + padded_scene_box[2] < 0:
            padded_scene_box[2] = -padded_scene_box[0] + 1
        if padded_scene_box[1] + padded_scene_box[3] < 0:
            padded_scene_box[3] = -padded_scene_box[1] + 1

        crop_img_size = (
                    padded_scene_box[2:4] * ((rt_opts['img_size'], rt_opts['img_size']) / target_bbox[2:4])).astype(
            'int64')
        crop_img_size[0] = np.clip(crop_img_size[0], 84, 2000)
        crop_img_size[1] = np.clip(crop_img_size[1], 84, 2000)
        cropped_image, cur_image_var = self.img_crop_model.crop_image(cur_image, np.reshape(padded_scene_box, (1, 4)),
                                                                      crop_img_size)
        cropped_image = cropped_image - 128.

        self.rtmodel.eval()
        feat_map = self.rtmodel(cropped_image, out_layer='conv3')

        # relative target bbox with padded_scene_box
        rel_target_bbox = np.copy(target_bbox)
        rel_target_bbox[0:2] -= padded_scene_box[0:2]

        # Extract sample features and get target location
        batch_num = np.zeros((samples.shape[0], 1))
        sample_rois = np.copy(samples)
        sample_rois[:, 0:2] -= np.repeat(np.reshape(padded_scene_box[0:2], (1, 2)), sample_rois.shape[0], axis=0)
        sample_rois = samples2maskroi(sample_rois, self.rtmodel.receptive_field,
                                      (rt_opts['img_size'], rt_opts['img_size']),
                                      target_bbox[2:4], rt_opts['padding'])
        sample_rois = np.concatenate((batch_num, sample_rois), axis=1)
        sample_rois = Variable(torch.from_numpy(sample_rois.astype('float32'))).cuda()
        sample_feats = self.rtmodel.roi_align_model(feat_map, sample_rois)
        sample_feats = sample_feats.view(sample_feats.size(0), -1).clone()
        sample_scores = self.rtmodel(sample_feats, in_layer='fc4')
        return sample_scores, sample_feats

    def collect_samples_rtmdnet(self, cur_image, target_bbox):
        cur_image = np.asarray(cur_image)

        # Draw pos/neg samples
        ishape = cur_image.shape
        pos_examples = gen_samples(
            SampleGenerator('gaussian', (ishape[1], ishape[0]), 0.1, 1.2), target_bbox,
            rt_opts['n_pos_update'],
            rt_opts['overlap_pos_update'])
        neg_examples = gen_samples(
            SampleGenerator('uniform', (ishape[1], ishape[0]), 1.5, 1.2), target_bbox,
            rt_opts['n_neg_update'],
            rt_opts['overlap_neg_update'])
        if pos_examples.shape[0] == rt_opts['n_pos_update'] and neg_examples.shape[0] == rt_opts['n_neg_update']:
            padded_x1 = (neg_examples[:, 0] - neg_examples[:, 2] * (rt_opts['padding'] - 1.) / 2.).min()
            padded_y1 = (neg_examples[:, 1] - neg_examples[:, 3] * (rt_opts['padding'] - 1.) / 2.).min()
            padded_x2 = (neg_examples[:, 0] + neg_examples[:, 2] * (rt_opts['padding'] + 1.) / 2.).max()
            padded_y2 = (neg_examples[:, 1] + neg_examples[:, 3] * (rt_opts['padding'] + 1.) / 2.).max()
            padded_scene_box = np.reshape(
                np.asarray((padded_x1, padded_y1, padded_x2 - padded_x1, padded_y2 - padded_y1)), (1, 4))

            scene_boxes = np.reshape(np.copy(padded_scene_box), (1, 4))
            jitter_scale = [1.]

            for bidx in range(0, scene_boxes.shape[0]):
                crop_img_size = (scene_boxes[bidx, 2:4] * (
                        (rt_opts['img_size'], rt_opts['img_size']) / target_bbox[2:4])).astype('int64') * jitter_scale[
                                    bidx]
                cropped_image, cur_image_var = self.img_crop_model.crop_image(cur_image,
                                                                              np.reshape(scene_boxes[bidx], (1, 4)),
                                                                              crop_img_size)
                cropped_image = cropped_image - 128.

                feat_map = self.rtmodel(cropped_image, out_layer='conv3')

                rel_target_bbox = np.copy(target_bbox)
                rel_target_bbox[0:2] -= scene_boxes[bidx, 0:2]

                batch_num = np.zeros((pos_examples.shape[0], 1))
                cur_pos_rois = np.copy(pos_examples)
                cur_pos_rois[:, 0:2] -= np.repeat(np.reshape(scene_boxes[bidx, 0:2], (1, 2)), cur_pos_rois.shape[0],
                                                  axis=0)
                scaled_obj_size = float(rt_opts['img_size']) * jitter_scale[bidx]
                cur_pos_rois = samples2maskroi(cur_pos_rois, self.rtmodel.receptive_field,
                                               (scaled_obj_size, scaled_obj_size),
                                               target_bbox[2:4], rt_opts['padding'])
                cur_pos_rois = np.concatenate((batch_num, cur_pos_rois), axis=1)
                cur_pos_rois = Variable(torch.from_numpy(cur_pos_rois.astype('float32'))).cuda()
                cur_pos_feats = self.rtmodel.roi_align_model(feat_map, cur_pos_rois)
                cur_pos_feats = cur_pos_feats.view(cur_pos_feats.size(0), -1).data.clone()

                batch_num = np.zeros((neg_examples.shape[0], 1))
                cur_neg_rois = np.copy(neg_examples)
                cur_neg_rois[:, 0:2] -= np.repeat(np.reshape(scene_boxes[bidx, 0:2], (1, 2)), cur_neg_rois.shape[0],
                                                  axis=0)
                cur_neg_rois = samples2maskroi(cur_neg_rois, self.rtmodel.receptive_field,
                                               (scaled_obj_size, scaled_obj_size),
                                               target_bbox[2:4], rt_opts['padding'])
                cur_neg_rois = np.concatenate((batch_num, cur_neg_rois), axis=1)
                cur_neg_rois = Variable(torch.from_numpy(cur_neg_rois.astype('float32'))).cuda()
                cur_neg_feats = self.rtmodel.roi_align_model(feat_map, cur_neg_rois)
                cur_neg_feats = cur_neg_feats.view(cur_neg_feats.size(0), -1).data.clone()

                feat_dim = cur_pos_feats.size(-1)

                if bidx == 0:
                    pos_feats = cur_pos_feats  ##index select
                    neg_feats = cur_neg_feats
                else:
                    pos_feats = torch.cat((pos_feats, cur_pos_feats), dim=0)
                    neg_feats = torch.cat((neg_feats, cur_neg_feats), dim=0)

            if pos_feats.size(0) > rt_opts['n_pos_update']:
                pos_idx = np.asarray(range(pos_feats.size(0)))
                np.random.shuffle(pos_idx)
                pos_feats = pos_feats.index_select(0, torch.from_numpy(pos_idx[0:rt_opts['n_pos_update']]).cuda())
            if neg_feats.size(0) > rt_opts['n_neg_update']:
                neg_idx = np.asarray(range(neg_feats.size(0)))
                np.random.shuffle(neg_idx)
                neg_feats = neg_feats.index_select(0, torch.from_numpy(neg_idx[0:rt_opts['n_neg_update']]).cuda())

            self.rtpos_feats_all.append(pos_feats)
            self.rtneg_feats_all.append(neg_feats)

            if len(self.rtpos_feats_all) > rt_opts['n_frames_long']:
                del self.rtpos_feats_all[0]
            if len(self.rtneg_feats_all) > rt_opts['n_frames_short']:
                del self.rtneg_feats_all[0]


    def rtmdnet_update(self, use_short_update=False):
        # Short term update
        if use_short_update:
            nframes = min(rt_opts['n_frames_short'], len(self.rtpos_feats_all))
            pos_data = torch.stack(self.rtpos_feats_all[-nframes:], 0).view(-1, self.rtfeat_dim)
            neg_data = torch.stack(self.rtneg_feats_all, 0).view(-1, self.rtfeat_dim)
            rt_train(self.rtmodel, self.criterion, self.rtupdate_optimizer, pos_data, neg_data, rt_opts['maxiter_update'])

        # Long term update
        elif self.i % rt_opts['long_interval'] == 0:
            pos_data = torch.stack(self.rtpos_feats_all, 0).view(-1, self.rtfeat_dim)
            neg_data = torch.stack(self.rtneg_feats_all, 0).view(-1, self.rtfeat_dim)
            rt_train(self.rtmodel, self.criterion, self.rtupdate_optimizer, pos_data, neg_data, rt_opts['maxiter_update'])


    def local_init(self, image, init_bbox):
        local_tracker = Tracker('atom', 'default', None)
        self.local_Tracker = local_tracker.tracker_class(local_tracker.parameters)
        self.local_Tracker.initialize(image, init_bbox)
        # if self.p.visualization:
        #     show_res(cv.cvtColor(image, cv.COLOR_RGB2BGR), np.array(self.last_gt, dtype=np.int32), '2', groundtruth=self.groundtruth,frame_id=self.i)
    def local_track(self, image):
        state, flag, score = self.local_Tracker.track(image)
        self.last_gt = [state[1], state[0], state[1]+state[3], state[0]+state[2]]
        # if self.p.visualization:
        #     show_res(cv.cvtColor(image, cv.COLOR_RGB2BGR), np.array(self.last_gt, dtype=np.int32), '2', groundtruth=self.groundtruth, frame_id=self.i)
        # print("%s: %d / %d" % (video, id, len(img_list)))
        if flag is not None:
            print(flag)
        return state, flag, score
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

    def metric_init(self, im, init_box):
        self.metric_model = ft_net(class_num=1120)
        path = '../utils/metric_net/metric_model/metric_model.pt'
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
        self.tc_input = tf.placeholder("float", [None, tcopts['time_steps'], tcopts['lstm_num_input']])
        # self.tc_label = tf.placeholder("float", [None, tcopts['lstm_num_classes']])

        self.tclogits = self.tc_model.net(self.tc_input)

        variables_to_restore = [var for var in tf.global_variables() if (var.name.startswith('tclstm'))]
        saver = tf.train.Saver(var_list=variables_to_restore)
        checkpoint = tf.train.latest_checkpoint(os.path.join('./meta_updater', model_dir))
        # if checkpoint is not None:
        saver.restore(self.sess, checkpoint)




    def tracking(self, image):
        self.i += 1
        update = True
        update_ouput = [0, 0]
        cur_ori_img = Image.fromarray(image).convert('RGB')
        cur_image = np.asarray(cur_ori_img)

        target_bbox, reg_box, top_scores, success = self.rtmdnet_track(image)
        # updater
        local_state = reg_box.reshape((1, 4))
        ap_dis = self.metric_eval(image, local_state, self.anchor_feature)
        self.dis_record.append(ap_dis.data.cpu().numpy()[0])
        self.state_record.append([local_state[0][0] / cur_ori_img.width, local_state[0][1] / cur_ori_img.height,
                                  (local_state[0][0] + local_state[0][2]) / cur_ori_img.width,
                                  (local_state[0][1] + local_state[0][3]) / cur_ori_img.height])
        self.rv_record.append(top_scores)
        if len(self.state_record) >= self.p.start_frame:
            dis = np.array(self.dis_record[-tcopts["time_steps"]:]).reshape((tcopts["time_steps"], 1))
            rv = np.array(self.rv_record[-tcopts["time_steps"]:])
            state_tc = np.array(self.state_record[-tcopts["time_steps"]:])
            tc_input = np.concatenate((state_tc, rv, dis), axis=1)
            logits = self.sess.run(self.tclogits, feed_dict={self.tc_input: np.expand_dims(tc_input, axis=0)})
            update = logits[0][0] < logits[0][1]
            update_ouput = logits[0]
            print(logits)

        # Data collect
        if update:
            self.collect_samples_rtmdnet(cur_image, target_bbox)

        self.rtmdnet_update(use_short_update=not success)

        self.last_gt = [reg_box[1], reg_box[0], reg_box[1]+reg_box[3], reg_box[0]+reg_box[2]]
        # gt_err = self.groundtruth[self.i, 2] < 3 or self.groundtruth[self.i, 3] < 3
        # gt_nan = any(np.isnan(self.groundtruth[self.i]))
        try:
            iou = _compute_iou(self.groundtruth[self.i], reg_box)
        except:
            iou = 0
        ##------------------------------------------------------##

        # self.local_update(sample_pos, translation_vec, scale_ind, sample_scales, s, test_x)

        width = self.last_gt[3] - self.last_gt[1]
        height = self.last_gt[2] - self.last_gt[0]

        if self.p.visualization:
            show_res(cv.cvtColor(image, cv.COLOR_RGB2BGR), np.array(self.last_gt, dtype=np.int32), '2',
                     groundtruth=self.groundtruth,score_max=update_ouput,update=update,
                     frame_id=self.i)

        return [float(self.last_gt[1]), float(self.last_gt[0]), float(width), float(height)], top_scores, iou, ap_dis.data.cpu().numpy()[0]

def eval_tracking(Dataset, video_spe=None, save=False, p=None, classes=None, mode=None):
    if Dataset == "votlt":
        data_dir = '/home/daikenan/dataset/VOT18_long'
    elif Dataset == 'otb':
        data_dir = '/home/dkn/daikenan/dataset/OTB'
    elif Dataset == "votlt19":
        data_dir = '/home/dkn/data/lt2019'
    elif Dataset == "lasot":
        data_dir = os.path.join('/home/dkn/daikenan/dataset/LASOT/LaSOTBenchmark/', classes)
    sequence_list = os.listdir(data_dir)
    testing_set_dir = './testing_set.txt'
    testing_set = list(np.loadtxt(testing_set_dir, dtype=str))
    if mode == 'test':
        print('test data')
        sequence_list = [vid for vid in sequence_list if vid in testing_set]
    elif mode == 'train':
        print('train data')
        sequence_list = [vid for vid in sequence_list if vid not in testing_set]
    else:
        print("all data")
    sequence_list.sort()
    sequence_list = [title for title in sequence_list if not title.endswith("txt")]
    if video_spe is not None:
        sequence_list = [video_spe]
    precisions = np.zeros(len(sequence_list))
    precisions_auc = np.zeros(len(sequence_list))
    ious = np.zeros(len(sequence_list))
    lengths = np.zeros(len(sequence_list))
    speed = np.zeros(len(sequence_list))
    base_save_path = os.path.join('/home/dkn/Desktop/rtmd_new/', p.name, Dataset)
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
        result_save_path = os.path.join(base_save_path, video+'.txt')
        if os.path.exists(result_save_path) and save:
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
        image = cv.cvtColor(cv.imread(image_dir), cv.COLOR_BGR2RGB)
        h = image.shape[0]
        w = image.shape[1]
        region1 = groundtruth[0]
        box = np.array([region1[0]/w, region1[1]/h, (region1[0]+region1[2])/w, (region1[1]+region1[3])/h])
        if groundtruth[0, 2] <= 9 or groundtruth[0, 3] <= 9:
            continue
        tracker = RTMD_MU_Tracker(image, region, video=video, p=p,groundtruth=groundtruth)
        v_score = tracker.get_first_state()

        num_frames = len(image_list)
        bBoxes = np.zeros((num_frames, 12))
        bBoxes[0, :] = [box[0], box[1], box[2], box[3], 0, 1, v_score[0], v_score[1], v_score[2], v_score[3], v_score[4], 0]


        tic = time.time()
        for im_id in range(1, len(image_list)):
            imagefile = sequence_dir + image_list[im_id]
            image = cv.cvtColor(cv.imread(imagefile), cv.COLOR_BGR2RGB)
            # image = cv2.imread(imagefile)
            # image = image[:,:,::-1]
            print("%d: " % seq_id + video + ": %d /" % im_id + "%d" % len(image_list))
            region, v_score, iou, dis = tracker.tracking2(image)
            # region, _ = tracker.local_track(image)
            # region = tracker.pymdnet_track(image)
            # region, region_bb = tracker.rtmdnet_track(image)
            box = np.array(
                [region[0] / w, region[1] / h, (region[0] + region[2]) / w, (region[1] + region[3]) / h])
            bBoxes[im_id, :] = [box[0], box[1], box[2], box[3], im_id, iou, v_score[0], v_score[1], v_score[2], v_score[3], v_score[4], dis]
            # bBoxes2[im_id, :] = region
        if save:
            np.savetxt(result_save_path, bBoxes, fmt="%.8f,%.8f,%.8f,%.8f,%d,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f")
        # np.savetxt(os.path.join('/home/daikenan/Desktop/MDNet_tf/', video+'.txt'), bBoxes2, fmt="%.6f,%.6f,%.6f,%.6f")
        tracker.sess.close()
        tf.reset_default_graph()
        toc = time.time()-tic
        fps = len(image_list)/toc
        print(fps)



