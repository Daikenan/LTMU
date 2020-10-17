import tensorflow as tf
import numpy as np
from google.protobuf import text_format
# from object_detection.protos import pipeline_pb2
# from core.model_builder import build_man_model
# from object_detection.core import box_list
# from object_detection.core import box_list_ops
from region_to_bbox import region_to_bbox
from PIL import Image
import cv2
import os
import time
import random
import math
import matplotlib.pyplot as plt

def compute_overall_results(dataset, result_path, save_path, eval_mode=None):
    g = open(save_path, 'w')
    if dataset == 'otb':
        results = np.zeros((100, 4))
        base_dir = '/home/daikenan/dataset/OTB/'
        videos = os.listdir(base_dir)
        videos.sort()
        for id, video in enumerate(videos):
            try:
                gt = np.loadtxt(base_dir+video+'/groundtruth_rect.txt')
            except:
                gt = np.loadtxt(base_dir + video + '/groundtruth_rect.txt', delimiter=',')

            try:
                bbox = np.loadtxt(os.path.join(result_path, video+'.txt'), delimiter=',')
            except IOError:
                try:
                    bbox = np.loadtxt(os.path.join(result_path, video[0].lower()+video[1:] + '.txt'), delimiter=',')
                except:
                    bbox = np.loadtxt(os.path.join(result_path, str.lower(video) + '.txt'), delimiter=',')
            results[id] = _compile_results(gt, bbox, 20)
            g.writelines([str(id) + ' -- ' + video +
                          ' -- Precision: ' + "%.2f" % results[id, 1] +
                          ' -- Precisions AUC: ' + "%.2f" % results[id, 2] +
                          ' -- IOU: ' + "%.2f" % results[id, 3] + ' --\n'])
    elif dataset == 'votlt':
        results = np.zeros((35, 4))
        base_dir = '/home/daikenan/dataset/VOT18_long/'
        videos = os.listdir(base_dir)
        videos.sort()
        for id, video in enumerate(videos):
            gt = np.loadtxt(base_dir + video + '/groundtruth.txt', delimiter=',')
            try:
                bbox = np.loadtxt(os.path.join(result_path, video + '.txt'), delimiter=',')
            except IOError:
                try:
                    bbox = np.loadtxt(os.path.join(result_path, video[0].lower() + video[1:] + '.txt'), delimiter=',')
                except:
                    bbox = np.loadtxt(os.path.join(result_path, str.lower(video) + '.txt'), delimiter=',')
            results[id] = _compile_results(gt, bbox, 20)
            g.writelines([str(id) + ' -- ' + video +
                          ' -- Precision: ' + "%.2f" % results[id, 1] +
                          ' -- Precisions AUC: ' + "%.2f" % results[id, 2] +
                          ' -- IOU: ' + "%.2f" % results[id, 3] + ' --\n'])

    overall_pre = np.mean(results[:, 1])
    overall_auc = np.mean(results[:, 2])
    overall_iou = np.mean(results[:, 3])
    g.writelines(['-- Overall stats (averaged per frame) on ' + str(
        len(videos)) + ' -- Precision ' + "(20 px)" + ': ' + "%.2f" % overall_pre +
                  ' -- Precisions AUC: ' + "%.2f" % overall_auc +
                  ' -- IOU: ' + "%.2f" % overall_iou + ' --\n'])
    g.close()





def show_boxes(im, boxes, step=1):
    import matplotlib.patches as patches
    fig, ax = plt.subplots(1)
    ax.imshow(im)
    for i in range(0, boxes.shape[0]-1, step):
        rect = patches.Rectangle((boxes[i][0], boxes[i][1]), boxes[i][2], boxes[i][3], linewidth=0.5, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.show()

def _compile_results(gt, bboxes, dist_threshold):
    l = np.size(bboxes, 0)
    gt4 = np.zeros((l, 4))
    new_distances = np.zeros(l)
    new_ious = np.zeros(l)
    n_thresholds = 50
    precisions_ths = np.zeros(n_thresholds)

    for i in range(l):
        gt4[i, :] = region_to_bbox(gt[i, :], center=False)
        new_distances[i] = _compute_distance(bboxes[i, :], gt4[i, :])
        new_ious[i] = _compute_iou(bboxes[i, :], gt4[i, :])

    # what's the percentage of frame in which center displacement is inferior to given threshold? (OTB metric)
    precision = sum(new_distances < dist_threshold)/float(np.size(new_distances)) * 100.0

    # find above result for many thresholds, then report the AUC
    thresholds = np.linspace(0, 25, n_thresholds+1)
    thresholds = thresholds[-n_thresholds:]
    # reverse it so that higher values of precision goes at the beginning
    thresholds = thresholds[::-1]
    for i in range(n_thresholds):
        precisions_ths[i] = sum(new_distances < thresholds[i])/float(np.size(new_distances))

    # integrate over the thresholds
    precision_auc = np.trapz(precisions_ths)

    # per frame averaged intersection over union (OTB metric)
    iou = np.mean(new_ious) * 100

    return l, precision, precision_auc, iou

def _compute_distance(boxA, boxB):
    a = np.array((boxA[0]+boxA[2]/2, boxA[1]+boxA[3]/2))
    b = np.array((boxB[0]+boxB[2]/2, boxB[1]+boxB[3]/2))
    dist = np.linalg.norm(a - b)

    assert dist >= 0
    assert dist != float('Inf')

    return dist


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


def get_configs_from_pipeline_file(config_file):
  """Reads training configuration from a pipeline_pb2.TrainEvalPipelineConfig.

  Reads training config from file specified by pipeline_config_path flag.

  Returns:
    model_config: model_pb2.DetectionModel
    train_config: train_pb2.TrainConfig
    input_config: input_reader_pb2.InputReader
  """
  pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
  with tf.gfile.GFile(config_file, 'r') as f:
    text_format.Merge(f.read(), pipeline_config)
  model_config = pipeline_config.model.ssd
  train_config = pipeline_config.train_config
  input_config = pipeline_config.train_input_reader
  eval_config = pipeline_config.eval_config

  return model_config, train_config, input_config, eval_config

def show_res(im, box, win_name,score=None,save_path=None,frame_id=None,all_frame=None,score_max=None):
    cv2.namedWindow(win_name,cv2.WINDOW_NORMAL)
    cv2.rectangle(im, (box[1], box[0]),
                  (box[3], box[2]), [0, 255, 0], 2)
    if score is not None:
        cv2.putText(im,str(score),(20,40), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1)
    if score_max is not None:
        cv2.putText(im,str(score_max),(20,60), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1)
    if frame_id is not None:
        cv2.putText(im,str(frame_id),(20,20), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1)
    #cv2.imwrite("/home/xiaobai/Desktop/MBMD_vot_code/figure/%05d.jpg"%frame_id, im[:, :, -1::-1])
    cv2.imshow(win_name, im)
    cv2.waitKey(1)

def restore_model(sess, model_scope, checkpoint_path, variables_to_restore):
    # variables_to_restore = tf.global_variables()
    name_to_var_dict = dict([(var.op.name.lstrip(model_scope+'/'), var) for var in variables_to_restore
                             if (var.op.name.startswith("model") and not var.op.name.endswith('Momentum'))])
    saver = tf.train.Saver(name_to_var_dict)
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_path)
    saver.restore(sess, latest_checkpoint)

def crop_search_region(img, gt, win_size, scale=4, mean_rgb=128, offset=None):
    # gt: [ymin, xmin, ymax, xmax]
    bnd_ymin, bnd_xmin, bnd_ymax, bnd_xmax = gt
    bnd_w = bnd_xmax - bnd_xmin
    bnd_h = bnd_ymax - bnd_ymin
    # cx, cy = gt[:2] + gt[2:] / 2
    cy, cx = (bnd_ymin + bnd_ymax)/2, (bnd_xmin+bnd_xmax)/2
    diag = np.sum( bnd_h** 2 + bnd_w**2) ** 0.5
    origin_win_size = diag * scale
    origin_win_size_h, origin_win_size_w = bnd_h * scale, bnd_w * scale
    # origin_win_size_h = origin_win_size
    # origin_win_size_w = origin_win_size
    im_size = img.size[1::-1]
    min_x = np.round(cx - origin_win_size_w / 2).astype(np.int32)
    max_x = np.round(cx + origin_win_size_w / 2).astype(np.int32)
    min_y = np.round(cy - origin_win_size_h / 2).astype(np.int32)
    max_y = np.round(cy + origin_win_size_h / 2).astype(np.int32)
    if offset is not None:
        min_offset_y, max_offset_y = (bnd_ymax - max_y, bnd_ymin - min_y)
        min_offset_x, max_offset_x = (bnd_xmax - max_x, bnd_xmin - min_x)
        offset[0] = np.clip(offset[0] * origin_win_size_h, min_offset_y, max_offset_y)
        offset[1] = np.clip(offset[1] * origin_win_size_w, min_offset_x, max_offset_x)
        offset = np.int32(offset)
        min_y += offset[0]
        max_y += offset[0]
        min_x += offset[1]
        max_x += offset[1]

    win_loc = np.array([min_y, min_x])
    gt_x_min, gt_y_min = ((bnd_xmin-min_x)/origin_win_size_w, (bnd_ymin - min_y)/origin_win_size_h) #coordinates on window
    gt_x_max, gt_y_max = [(bnd_xmax-min_x)/origin_win_size_w, (bnd_ymax - min_y)/origin_win_size_h] #relative coordinates of gt bbox to the search region

    unscaled_w, unscaled_h = [max_x - min_x + 1, max_y - min_y + 1]
    min_x_win, min_y_win, max_x_win, max_y_win = (0, 0, unscaled_w, unscaled_h)
    min_x_im, min_y_im, max_x_im, max_y_im = (min_x, min_y, max_x+1, max_y+1)

    img = img.crop([min_x_im, min_y_im, max_x_im, max_y_im])
    img_array = np.array(img)

    if min_x < 0:
        min_x_im = 0
        min_x_win = 0 - min_x
    if min_y < 0:
        min_y_im = 0
        min_y_win = 0 - min_y
    if max_x+1 > im_size[1]:
        max_x_im = im_size[1]
        max_x_win = unscaled_w - (max_x + 1 - im_size[1])
    if max_y+1 > im_size[0]:
        max_y_im = im_size[0]
        max_y_win = unscaled_h- (max_y +1 - im_size[0])

    unscaled_win = np.ones([unscaled_h, unscaled_w, 3], dtype=np.uint8) * np.uint8(mean_rgb)
    unscaled_win[min_y_win:max_y_win, min_x_win:max_x_win] = img_array[min_y_win:max_y_win, min_x_win:max_x_win]

    unscaled_win = Image.fromarray(unscaled_win)
    height_scale, width_scale = np.float32(unscaled_h)/win_size, np.float32(unscaled_w)/win_size
    win = unscaled_win.resize([win_size, win_size], resample=Image.BILINEAR)
    # win = sp.misc.imresize(unscaled_win, [win_size, win_size])
    return win, np.array([gt_y_min, gt_x_min, gt_y_max, gt_x_max]), win_loc, [height_scale, width_scale]
    # return win, np.array([gt_x_min, gt_y_min, gt_x_max, gt_y_max]), diag, np.array(win_loc)

def generate_init_training_samples(img, box, win_size, src_scales=None, tar_scales=None, batch_size=20, mean_rgb=128):
    if src_scales is None:
        src_scales = [1.2, 3]
    if tar_scales is None:
        tar_scales = [3.7, 4.5]
    out_images = np.zeros([batch_size, 1, win_size, win_size, 3], dtype=np.uint8)
    out_gt_box = np.zeros([batch_size, 1, 4], dtype=np.float32)
    init_img = img.crop(np.int32([box[1], box[0], box[3], box[2]]))
    init_img = init_img.resize([128,128], resample=Image.BILINEAR)
    init_img = np.array(init_img)
    init_img = np.expand_dims(np.expand_dims(init_img,axis=0),axis=0)
    init_img = np.tile(init_img,(batch_size,1,1,1,1))
    for ind in range(batch_size):
        src_scale = np.random.rand(1)[0]*(src_scales[1]-src_scales[0]) + src_scales[0]
        tar_scale = np.random.rand(1)[0]*(tar_scales[1]-tar_scales[0]) + tar_scales[0]
        src_offset = np.random.laplace(0, 0.2, [2])
        tar_offset = np.random.laplace(0, 0.2, [2])
        # src_win, src_gt, _, _ = crop_search_region(img, box, win_size, src_scale, offset=src_offset)
        tar_win, tar_gt, _, _ = crop_search_region(img, box, win_size, tar_scale, offset=tar_offset)
        #out_images[ind, 0] = init_img
        out_images[ind, 0] = tar_win
        out_gt_box[ind, 0] = tar_gt
    return out_images, init_img,out_gt_box

def build_init_graph(model, model_scope, reuse=None):
    input_init_image = tf.placeholder(dtype=tf.uint8, shape=[128,128,3])
    float_init_image = tf.to_float(input_init_image)
    float_init_image = tf.expand_dims(tf.expand_dims(float_init_image, axis=0), axis=0)
    preprocessed_init_image = model.preprocess(float_init_image, [128,128])
    with tf.variable_scope(model_scope, reuse=reuse):
        init_feature_maps = model.extract_init_feature(preprocessed_init_image)
    return init_feature_maps,input_init_image

def build_box_predictor(model, model_scope,init_feature_maps,reuse=None):
    input_cur_image = tf.placeholder(dtype=tf.uint8, shape=[300, 300, 3])
    images = tf.expand_dims(input_cur_image, axis=0)
    float_images = tf.to_float(images)
    preprocessed_images = model.preprocess(float_images)
    preprocessed_images = tf.expand_dims(preprocessed_images, axis=0)
    input_init_gt_box = tf.constant(np.zeros((1, 4)), dtype=tf.float32)
    init_gt_box = tf.reshape(input_init_gt_box, shape=[1,1,4])
    groundtruth_classes = tf.ones(dtype=tf.float32, shape=[1, 1, 1])
    model.provide_groundtruth(init_gt_box,
                              groundtruth_classes,
                              None)
    with tf.variable_scope(model_scope, reuse=reuse):
        prediction_dict = model.predict_box_with_init(init_feature_maps, preprocessed_images, istraining=False)

    detections, box_ori, score_ori = model.postprocess(prediction_dict)
    original_image_shape = tf.shape(preprocessed_images)
    absolute_detection_boxlist = box_list_ops.to_absolute_coordinates(
        box_list.BoxList(tf.squeeze(detections['detection_boxes'], axis=0)),
        original_image_shape[2], original_image_shape[3])
    return absolute_detection_boxlist.get(), detections['detection_scores'], input_cur_image

def build_test_graph(model, model_scope, reuse=None,weights_dict=None):
    input_init_gt_box = tf.constant(np.zeros((1,4)), dtype=tf.float32)
    # input_init_image = tf.constant(init_img_array, dtype=tf.uint8)
    input_init_image = tf.placeholder(dtype=tf.uint8, shape=[128,128,3])
    input_cur_image = tf.placeholder(dtype=tf.uint8, shape=[300,300,3])

    init_gt_box = tf.reshape(input_init_gt_box, shape=[1,1,4])
    groundtruth_classes = tf.ones(dtype=tf.float32, shape=[1,1,1])
    float_init_image = tf.to_float(input_init_image)
    float_init_image = tf.expand_dims(tf.expand_dims(float_init_image, axis=0), axis=0)
    preprocessed_init_image = model.preprocess(float_init_image, [128,128])
    images = tf.expand_dims(input_cur_image, axis=0)
    float_images = tf.to_float(images)
    preprocessed_images = model.preprocess(float_images)
    preprocessed_images = tf.expand_dims(preprocessed_images, axis=0)
    model.provide_groundtruth(init_gt_box,
                              groundtruth_classes,
                              None)
    with tf.variable_scope(model_scope, reuse=reuse):
        prediction_dict = model.predict(preprocessed_init_image, preprocessed_images,istraining=False,reuse=reuse)
    detections = model.postprocess(prediction_dict)
    original_image_shape = tf.shape(preprocessed_images)
    absolute_detection_boxlist = box_list_ops.to_absolute_coordinates(
        box_list.BoxList(tf.squeeze(detections['detection_boxes'], axis=0)),
        original_image_shape[2], original_image_shape[3])
    return absolute_detection_boxlist.get(), detections['detection_scores'], input_cur_image, input_init_image


def build_extract_feature_graph(model, model_scope,reuse=None):
    batch_size = 20
    seq_len = 1
    image = tf.placeholder(dtype=tf.uint8, shape=[batch_size, seq_len, 300,300,3])
    float_image = tf.to_float(image)
    float_image = tf.reshape(float_image,[-1,300,300,3])
    preprocessed_images = model.preprocess(float_image)
    preprocessed_images = tf.reshape(preprocessed_images,[batch_size,seq_len,300,300,3])

    random_noise = tf.random_normal([batch_size, seq_len, 300, 300, 3], mean=0, stddev=0.1)
    preprocessed_images = preprocessed_images + random_noise
    with tf.variable_scope(model_scope, reuse=reuse):
        output_dict = model.extract_feature(preprocessed_images)

    init_image = tf.placeholder(dtype=tf.uint8, shape=[1,seq_len, 128,128,3])
    float_init_image = tf.to_float(init_image)
    float_init_image = tf.reshape(float_init_image,[-1,128,128,3])
    preprocessed_init_images = model.preprocess(float_init_image,[128,128])
    preprocessed_init_images = tf.reshape(preprocessed_init_images,[1,seq_len,128,128,3])
    with tf.variable_scope(model_scope, reuse=reuse):
        init_feature_maps = model.extract_init_feature(preprocessed_init_images)

    return image, init_image, output_dict, init_feature_maps

def build_extract_feature_graph1(model, model_scope,reuse=None):
    batch_size = 5
    seq_len = 1
    image = tf.placeholder(dtype=tf.uint8, shape=[batch_size, seq_len, 300,300,3])
    float_image = tf.to_float(image)
    float_image = tf.reshape(float_image,[-1,300,300,3])
    preprocessed_images = model.preprocess(float_image)
    preprocessed_images = tf.reshape(preprocessed_images,[batch_size,seq_len,300,300,3])

    random_noise = tf.random_normal([batch_size, seq_len, 300, 300, 3], mean=0, stddev=0.1)
    preprocessed_images = preprocessed_images + random_noise
    with tf.variable_scope(model_scope, reuse=reuse):
        output_dict = model.extract_feature(preprocessed_images)

    init_image = tf.placeholder(dtype=tf.uint8, shape=[1,seq_len, 128,128,3])
    float_init_image = tf.to_float(init_image)
    float_init_image = tf.reshape(float_init_image,[-1,128,128,3])
    preprocessed_init_images = model.preprocess(float_init_image,[128,128])
    preprocessed_init_images = tf.reshape(preprocessed_init_images,[1,seq_len,128,128,3])
    with tf.variable_scope(model_scope, reuse=reuse):
        init_feature_maps = model.extract_init_feature(preprocessed_init_images)

    return image, init_image, output_dict, init_feature_maps
# def build_train_boxpredictor_graph(model, model_scope,reuse=None):
#     batch_size = 20
#     seq_len = 1
#     init_features = tf.placeholder(dtype=tf.float32, shape=[batch_size,seq_len,1,1,])

def build_train_graph(model,model_scope, lr=1e-5, reuse=None):
    batch_size = 20
    seq_len = 1
    featureOp0 = tf.placeholder(dtype=tf.float32, shape=[batch_size,19,19,512])
    featureOp1 = tf.placeholder(dtype=tf.float32, shape=[batch_size,10,10,512])
    # featureOp2 = tf.placeholder(dtype=tf.float32, shape=[batch_size,5,5,256])
    # featureOp3 = tf.placeholder(dtype=tf.float32, shape=[batch_size,3,3,256])
    # featureOp4 = tf.placeholder(dtype=tf.float32, shape=[batch_size,2,2,256])
    # featureOp5 = tf.placeholder(dtype=tf.float32, shape=[batch_size,1,1,256])
    initFeatureOp = tf.placeholder(dtype=tf.float32, shape=[batch_size,1,1,512])
    feature_maps = [featureOp0,featureOp1]

    train_gt_box = tf.placeholder(dtype=tf.float32, shape=[batch_size,seq_len,4])
    train_gt_class = tf.ones(dtype=tf.uint8, shape=[batch_size,seq_len,1])
    model.provide_groundtruth(train_gt_box,train_gt_class,None)

    with tf.variable_scope(model_scope,reuse=reuse):
        train_prediction_dict = model.predict_box(initFeatureOp,feature_maps,istraining=True)

    losses_dict = model.loss(train_prediction_dict)
    total_loss = 0
    # total_loss = losses_dict['classification_loss']
    for loss in losses_dict.values():
        total_loss += loss
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9)
    # optimizer = tf.train.AdamOptimizer()
    variables_to_restore = tf.global_variables()
    all_trainable_variables = tf.trainable_variables()
    trainable_variables = [var for var in all_trainable_variables if (var.op.name.startswith(model_scope + '/BoxPredictor') )]
    grad_vars = optimizer.compute_gradients(total_loss, trainable_variables)
    for grad, var in grad_vars:
        if grad is not None:
            if var.name.endswith("Conv3x3_OutPut_40/weights:0") or var.name.endswith("Conv3x3_OutPut_40/biases:0") or var.name.endswith("Conv3x3_OutPut_20/weights:0") \
                or var.name.endswith("Conv3x3_OutPut_20/biases:0") or var.name.endswith("Conv1x1_OutPut_20/weights:0") or var.name.endswith("Conv1x1_OutPut_20/biases:0") \
                    or var.name.endswith("Conv1x1_OutPut_10/weights:0") or var.name.endswith(
                "Conv1x1_OutPut_10/biases:0"):
                grad *= 10.0
    grad_updates = optimizer.apply_gradients(grad_vars)
    update_ops.append(grad_updates)
    update_op = tf.group(*update_ops)
    with tf.control_dependencies([update_op]):
        train_tensor = tf.identity(total_loss, name='train_op')

    return train_tensor, variables_to_restore,featureOp0, featureOp1, initFeatureOp, train_gt_box


def crop_init_array(init_img,gt_boxes):
    img1_xiaobai = np.array(init_img)
    pad_x = 36.0 / 264.0 * (gt_boxes[0, 3] - gt_boxes[0, 1]) * init_img.width
    pad_y = 36.0 / 264.0 * (gt_boxes[0, 2] - gt_boxes[0, 0]) * init_img.height
    cx = (gt_boxes[0, 3] + gt_boxes[0, 1]) / 2.0 * init_img.width
    cy = (gt_boxes[0, 2] + gt_boxes[0, 0]) / 2.0 * init_img.height
    startx = gt_boxes[0, 1] * init_img.width - pad_x
    starty = gt_boxes[0, 0] * init_img.height - pad_y
    endx = gt_boxes[0, 3] * init_img.width + pad_x
    endy = gt_boxes[0, 2] * init_img.height + pad_y
    left_pad = max(0, int(-startx))
    top_pad = max(0, int(-starty))
    right_pad = max(0, int(endx - init_img.width + 1))
    bottom_pad = max(0, int(endy - init_img.height + 1))

    startx = int(startx + left_pad)
    starty = int(starty + top_pad)
    endx = int(endx + left_pad)
    endy = int(endy + top_pad)

    if top_pad or left_pad or bottom_pad or right_pad:
        r = np.pad(img1_xiaobai[:, :, 0], ((top_pad, bottom_pad), (left_pad, right_pad)), mode='constant',
                   constant_values=128)
        g = np.pad(img1_xiaobai[:, :, 1], ((top_pad, bottom_pad), (left_pad, right_pad)), mode='constant',
                   constant_values=128)
        b = np.pad(img1_xiaobai[:, :, 2], ((top_pad, bottom_pad), (left_pad, right_pad)), mode='constant',
                   constant_values=128)
        r = np.expand_dims(r, 2)
        g = np.expand_dims(g, 2)
        b = np.expand_dims(b, 2)

        img1_xiaobai = np.concatenate((r, g, b), axis=2)
    img1_xiaobai = Image.fromarray(img1_xiaobai)

    # gt_boxes resize
    init_img_crop = img1_xiaobai.crop(np.int32([startx, starty, endx, endy]))
    init_img_crop = init_img_crop.resize([128, 128], resample=Image.BILINEAR)
    init_img_array = np.array(init_img_crop)

    return init_img_array


def judgeSearch(cur_ori_img, search_gt, tracker, threshold=0.8):
    # search_gt[0] = cur_ori_img.height / 2.0 - self.first_h / 2.0 / 2.0
    # search_gt[2] = cur_ori_img.height / 2.0 + self.first_h / 2.0 / 2.0
    # search_gt[1] = cur_ori_img.width / 2.0 - self.first_w / 2.0 / 2.0
    # search_gt[3] = cur_ori_img.width / 2.0 + self.first_w / 2.0 / 2.0

    cropped_img1, last_gt_norm1, win_loc1, scale1 = crop_search_region(cur_ori_img, search_gt, 300,
                                                                       mean_rgb=128)
    cur_ori_img_array = np.array(cur_ori_img)
    cur_img_array = np.array(cropped_img1)
    detection_box_ori1, scores1 = tracker.sess.run([tracker.pre_box_tensor, tracker.scores_tensor],
                                                feed_dict={tracker.input_cur_image: cur_img_array,
                                                           tracker.initConstantOp: tracker.init_feature_maps})
    if scores1[0, 0] > threshold:
        detection_box_ori1[:, 0] = detection_box_ori1[:, 0] * scale1[0] + win_loc1[0]
        detection_box_ori1[:, 1] = detection_box_ori1[:, 1] * scale1[1] + win_loc1[1]
        detection_box_ori1[:, 2] = detection_box_ori1[:, 2] * scale1[0] + win_loc1[0]
        detection_box_ori1[:, 3] = detection_box_ori1[:, 3] * scale1[1] + win_loc1[1]
        detection_box_ori = detection_box_ori1.copy()
        # max_idx = 0
        search_box1 = detection_box_ori[0]

        search_box1[0] = np.clip(search_box1[0], 0, cur_ori_img.height - 1)
        search_box1[2] = np.clip(search_box1[2], 0, cur_ori_img.height - 1)
        search_box1[1] = np.clip(search_box1[1], 0, cur_ori_img.width - 1)
        search_box1[3] = np.clip(search_box1[3], 0, cur_ori_img.width - 1)
        if (search_box1[0] == search_box1[2]) or (search_box1[1] == search_box1[3]):
            score_max = -1
        else:
            search_box1 = [search_box1[1], search_box1[0], search_box1[3] - search_box1[1],
                           search_box1[2] - search_box1[0]]
            search_box1 = np.reshape(search_box1, (1, 4))

            search_regions = extract_regions(cur_ori_img_array, search_box1)
            search_regions = search_regions[:, :, :, ::-1]

            score_max = tracker.sess.run(tracker.outputsSingleOp, feed_dict={tracker.imageSingleOp: search_regions})
            score_max = score_max[0, 1]

        # search_box1 = [search_box1[1],search_box1[0],search_box1[3]-search_box1[1],search_box1[2]-search_box1[0]]
        # search_box1 = np.reshape(search_box1, (1, 4))
        # search_regions = extract_regions(cur_ori_img_array, search_box1)
        # score_max = sess.run(outputsSingleOp, feed_dict={imageSingleOp: search_regions})
        if score_max > 0:
            scores = scores1.copy()

            max_idx = 0
            detection_box = detection_box_ori[max_idx]

        if score_max < 0:
            search_box1 = detection_box_ori[:20]
            search_box = np.zeros_like(search_box1)
            search_box[:, 1] = search_box1[:, 0]
            search_box[:, 0] = search_box1[:, 1]
            search_box[:, 2] = search_box1[:, 3]
            search_box[:, 3] = search_box1[:, 2]
            haha = np.ones_like(search_box[:, 2]) * 3
            search_box[:, 2] = search_box[:, 2] - search_box[:, 0]
            search_box[:, 3] = search_box[:, 3] - search_box[:, 1]
            search_box[:, 2] = np.maximum(search_box[:, 2], haha)
            search_box[:, 3] = np.maximum(search_box[:, 3], haha)
            haha2 = np.zeros_like(search_box[:, 0])
            search_box[:, 0] = np.maximum(search_box[:, 0], haha2)
            search_box[:, 1] = np.maximum(search_box[:, 1], haha2)
            haha = np.ones_like(search_box[:, 2]) * cur_ori_img.width - 1 - search_box[:, 2]
            search_box[:, 0] = np.minimum(search_box[:, 0], haha)
            haha2 = np.ones_like(search_box[:, 3]) * cur_ori_img.height - 1 - search_box[:, 3]
            search_box[:, 1] = np.minimum(search_box[:, 1], haha2)

            search_regions = extract_regions(cur_ori_img_array, search_box)
            search_regions = search_regions[:, :, :, ::-1]
            mdnet_scores = tracker.sess.run(tracker.outputsOp, feed_dict={tracker.imageOp: search_regions})
            mdnet_scores = mdnet_scores[:, 1]
            for max_idx1 in range(1, 20):
                if mdnet_scores[max_idx1] > 0 and scores1[0, max_idx1] > 0.3:
                    score_max = mdnet_scores[max_idx1]
                    max_idx = max_idx1
                    scores = scores1.copy()

                    detection_box = detection_box_ori[max_idx]
                    return scores, detection_box, detection_box_ori, max_idx, score_max
                elif scores1[0, max_idx1] < 0.3:
                    return None

            return None

def calc_ent(x):
    """
        calculate shanno ent of x
    """
    result = 0
    for i in range(x.shape[0]):
        result+=(-x[i])*math.log(x[i]+1e-7,2)
    return result