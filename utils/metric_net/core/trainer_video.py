import functools
import tensorflow as tf
from object_detection.builders import preprocessor_builder, optimizer_builder
from object_detection.utils import variables_helper
from object_detection.core import standard_fields as fields
from object_detection.core import preprocessor, batcher
#import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import matplotlib.patches as patches
import cv2
import random
import math
import numpy as np
import os
from core.preprocessor import  preprocess
slim=tf.contrib.slim


def _augment_negative_samples(image_batch, video_batch):
    image = image_batch[fields.InputDataFields.image]
    image_boxes = image_batch[fields.InputDataFields.groundtruth_boxes]
    image_label = image_batch[fields.InputDataFields.groundtruth_classes]
    video_frame = video_batch[fields.InputDataFields.image]
    video_boxes = video_batch[fields.InputDataFields.groundtruth_boxes]
    video_label = video_batch[fields.InputDataFields.groundtruth_classes]
    batch_size, seq_length,_ = image_label.get_shape().as_list()
    image_batch[fields.InputDataFields.image] = tf.concat([image, video_frame], axis=1)
    image_batch[fields.InputDataFields.groundtruth_boxes] = tf.concat([image_boxes, video_boxes], axis=1)
    image_batch[fields.InputDataFields.groundtruth_classes] = tf.concat([image_label, tf.zeros(shape=[batch_size,seq_length,1], dtype=tf.int32)],
                                                                        axis=1)
    video_batch[fields.InputDataFields.image] = tf.concat([video_frame, image], axis=1)
    video_batch[fields.InputDataFields.groundtruth_boxes] = tf.concat([video_boxes, image_boxes], axis=1)
    video_batch[fields.InputDataFields.groundtruth_classes] = tf.concat([video_label, tf.zeros(shape=[batch_size,seq_length,1], dtype=tf.int32)],
                                                                        axis=1)


def crop_search_region(img, gt, win_size, scales, mean_rgb=128, offsets=None):
    # gt: [ymin, xmin, ymax, xmax]
    bnd_ymin, bnd_xmin, bnd_ymax, bnd_xmax = [gt[0,0]*win_size[0], gt[0,1]*win_size[0],
                                              gt[0,2]*win_size[0], gt[0,3]*win_size[0]]
    bnd_w = bnd_xmax - bnd_xmin
    bnd_h = bnd_ymax - bnd_ymin
    cy, cx = (bnd_ymin + bnd_ymax)/2, (bnd_xmin+bnd_xmax)/2
    origin_win_size_h, origin_win_size_w = bnd_h * scales[0], bnd_w * scales[1]
    im_size = img.size[1::-1]
    min_x = np.round(cx - origin_win_size_w / 2).astype(np.int32)
    max_x = np.round(cx + origin_win_size_w / 2).astype(np.int32)
    min_y = np.round(cy - origin_win_size_h / 2).astype(np.int32)
    max_y = np.round(cy + origin_win_size_h / 2).astype(np.int32)
    if offsets is not None:
        min_offset_y, max_offset_y = (bnd_ymax - max_y, bnd_ymin - min_y)
        min_offset_x, max_offset_x = (bnd_xmax - max_x, bnd_xmin - min_x)
        offsets[0] = np.clip(offsets[0] * origin_win_size_h, min_offset_y, max_offset_y)
        offsets[1] = np.clip(offsets[1] * origin_win_size_w, min_offset_x, max_offset_x)
        offset = np.int32(offsets)
        min_y += offset[0]
        max_y += offset[0]
        min_x += offset[1]
        max_x += offset[1]

    gt_x_min, gt_y_min = ((bnd_xmin-min_x)/origin_win_size_w, (bnd_ymin - min_y)/origin_win_size_h) #coordinates on window
    gt_x_max, gt_y_max = [(bnd_xmax-min_x)/origin_win_size_w, (bnd_ymax - min_y)/origin_win_size_h] #relative coordinates of gt bbox to the search region

    unscaled_w, unscaled_h = [max_x - min_x + 1, max_y - min_y + 1]
    min_x_win, min_y_win, max_x_win, max_y_win = (0, 0, unscaled_w, unscaled_h)
    min_x_im, min_y_im, max_x_im, max_y_im = (min_x, min_y, max_x+1, max_y+1)

    img = img.crop([min_x_im, min_y_im, max_x_im, max_y_im])
    img_array = np.array(img)

    if min_x < 0:
        min_x_win = 0 - min_x
    if min_y < 0:
        min_y_win = 0 - min_y
    if max_x+1 > im_size[1]:
        max_x_win = unscaled_w - (max_x + 1 - im_size[1])
    if max_y+1 > im_size[0]:
        max_y_win = unscaled_h- (max_y +1 - im_size[0])

    unscaled_win = np.ones([unscaled_h, unscaled_w, 3], dtype=np.uint8) * np.uint8(mean_rgb)
    unscaled_win[min_y_win:max_y_win, min_x_win:max_x_win] = img_array[min_y_win:max_y_win, min_x_win:max_x_win]
    unscaled_win = Image.fromarray(unscaled_win)
    win = unscaled_win.resize(win_size, resample=Image.BILINEAR)
    return win, np.array([gt_y_min, gt_x_min, gt_y_max, gt_x_max])



def _concat_tensor_dicts(tensor_dicts):
    concat_tensor_dict = dict()
    for k in tensor_dicts[0].keys():
        tensor_list = [t[k] for t in tensor_dicts]
        concat_tensor_dict[k] = tf.concat(tensor_list, axis=0)
    return concat_tensor_dict
def _stack_tensor_dicts(tensor_dicts):
    concat_tensor_dict = dict()
    for k in tensor_dicts[0].keys():
        tensor_list = [t[k] for t in tensor_dicts]
        concat_tensor_dict[k] = tf.stack(tensor_list, axis=0)
    return concat_tensor_dict

def _perturbate_box(box):
    ymin, xmin, ymax, xmax = np.array(box).flatten()
    squeeze_max = 0.5
    height, width = ymax-ymin, xmax-xmin
    height_squeeze = np.abs(np.random.randn() * height / 3)
    height_squeeze = np.clip(height_squeeze, 0, squeeze_max * height)
    width_squeeze = np.abs(np.random.randn() * width / 3)
    width_squeeze = np.clip(width_squeeze, 0, squeeze_max * width)
    new_height, new_width = height - height_squeeze, width - width_squeeze
    y_range, x_range = height - new_height, width - new_width
    y_inc, x_inc = np.random.randn(2)* [y_range/4, x_range/4] + [y_range/2, x_range/2]
    ymin += y_inc
    xmin += x_inc
    ymax = ymin + new_height
    xmax = xmin + new_width
    return np.array([[ymin, xmin, ymax, xmax]])


def _random_exp_norm(alpha, beta, t=1.0):
    '''
    out = alpha x beta^r
    r subjects to a truncated normal distribution at [-t,t] 
    '''
    r = np.random.randn() * (t / 3.0)
    r_is_within_range = r >= -t and r <= t
    r = r_is_within_range * r
    out = alpha * beta ** r
    return out

def _random_exp_uniform(alpha, beta, t=1.0):
    '''
    out = alpha x beta^r
    r subjects to a uniform distribution at [-t,t] 
    '''
    r = np.random.uniform(-t, t)
    out = alpha * beta ** r
    return out


def _create_input_queue(batch_size_per_clone, create_tensor_dict_fn, detection_model,
                        batch_queue_capacity, num_batch_queue_threads,
                        prefetch_queue_capacity, data_augmentation_options, image_path, video_path):
    """Sets up reader, prefetcher and returns input queue.
  
    Args:
      batch_size_per_clone: batch size to use per clone.
      create_tensor_dict_fn: function to create tensor dictionary.
      batch_queue_capacity: maximum number of elements to store within a queue.
      num_batch_queue_threads: number of threads to use for batching.
      prefetch_queue_capacity: maximum capacity of the queue used to prefetch
                               assembled batches.
      data_augmentation_options: a list of tuples, where each tuple contains a
        data augmentation function and a dictionary containing arguments and their
        values (see preprocessor.py).
  
    Returns:
      input queue: a batcher.BatchQueue object holding enqueued tensor_dicts
        (which hold images, boxes and targets).  To get a batch of tensor_dicts,
        call input_queue.Dequeue().
    """
    pre_video_tensor_dict, pre_image_tensor_dict = create_tensor_dict_fn()

    def _read_image(im_names, gt_boxes, win_size=None, box_size=None, seq_length=1):
        if win_size == None:
            win_size = [300,300]
        if box_size == None:
            box_size = [128,128]
        img = Image.open(os.path.join(image_path, im_names[0,0]+'.JPEG'))
        im_size = np.random.randint(50,100)
        img = img.resize([im_size,im_size], resample=Image.BILINEAR)
        img = img.resize(win_size, resample=Image.BILINEAR)
        img_array = np.array(img)
        if img_array.ndim < 3:
            img_array = np.expand_dims(img_array, axis=2)
            img_array = img_array.repeat(3, axis=2)
            img = Image.fromarray(img_array)
        # gt_boxes resize
        # gt_boxes = _perturbate_box(gt_boxes)
        #init_win,_ = crop_search_region(img, gt_boxes, win_size=[300,300], scales=[1.2, 1.2])
        img1_xiaobai = np.array(img)
        pad_x = 36.0/264.0*(gt_boxes[0,3] - gt_boxes[0,1])*img.width
        pad_y = 36.0/264.0*(gt_boxes[0,2] - gt_boxes[0,0])*img.height
        cx = (gt_boxes[0,3] + gt_boxes[0,1]) / 2.0 * img.width
        cy = (gt_boxes[0,2] + gt_boxes[0,0]) / 2.0 *img.height
        startx = gt_boxes[0,1] * img.width - pad_x
        starty = gt_boxes[0,0] * img.height - pad_y
        endx = gt_boxes[0,3] * img.width + pad_x
        endy = gt_boxes[0,2] * img.height + pad_y
        left_pad = max(0, int(-startx))
        top_pad = max(0, int(-starty))
        right_pad = max(0, int(endx - img.width + 1))
        bottom_pad = max(0, int(endy - img.height + 1))

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
        init_win = img1_xiaobai.crop(np.int32([startx, starty,endx, endy]))


        flip_image = np.random.rand() > 0.5
        if flip_image:
            init_win = ImageOps.mirror(init_win)
            img = ImageOps.mirror(img)
            xmin = gt_boxes[0,1]
            xmax = gt_boxes[0,3]
            gt_boxes[0,1] = 1 - xmax
            gt_boxes[0,3] = 1 - xmin

        init_win = init_win.resize(box_size, resample=Image.BILINEAR)
        init_win_array = np.expand_dims(np.array(init_win), axis=0)

        area = 128*128
        target_area = random.uniform(0.02,0.4) * area
        aspect_ratio = random.uniform(0.3,1.0/0.3)
        h = int(round(math.sqrt(target_area * aspect_ratio)))
        w = int(round(math.sqrt(target_area / aspect_ratio)))
        if h < 128 and w < 128:
            x1 = random.randint(0,128-h)
            y1 = random.randint(0,128-w)
            init_win_array[0,x1:x1+h,y1:y1+w,0] = 123.68
            init_win_array[0,x1:x1+h,y1:y1+w,1] = 116.779
            init_win_array[0,x1:x1+h,y1:y1+w,2] = 103.939

        out_img = np.zeros([seq_length, win_size[0], win_size[1], 3], dtype=np.uint8)
        groundtruth_boxes = np.zeros([seq_length, 4], dtype=np.float32)
        groundtruth_classes = np.ones([seq_length, 1], dtype=np.int32)

        scale_max = 0.5
        scale_alpha = 4
        scale_beta = 0.5
        for ind in range(0,seq_length):
            scaleh = _random_exp_norm(scale_alpha, scale_beta, scale_max)
            scalew = _random_exp_norm(scale_alpha, scale_beta, scale_max)
            scales = [scaleh, scalew]
            offsets = np.random.laplace(0, 0.2, [2])
            win2, gt_box2 = crop_search_region(img, gt_boxes, win_size, scales=scales, offsets=offsets)
            out_img[ind] = np.array(win2)
            #out_img[ind] = cv2.resize(np.array(img),(300,300))
            groundtruth_boxes[ind] = gt_box2
        return init_win_array, out_img, groundtruth_boxes, groundtruth_classes

    def _read_video(folder, im_names, gt_boxes, win_size=None, box_size=None, seq_length=1):
        if win_size == None:
            win_size = [300,300]
        if box_size == None:
            box_size = [128,128]
        num_images = len(im_names)
        video_path = '/home/xiaobai/Documents/ILSVRC2015/Data/VID/train'

        selected_id = np.random.randint(0,num_images,[2])
        img1 = Image.open(os.path.join(video_path, im_names[selected_id[0]]))
        img2 = Image.open(os.path.join(video_path, im_names[selected_id[1]]))

        img1 = img1.resize(win_size, resample=Image.BILINEAR)
        img2 = img2.resize(win_size, resample=Image.BILINEAR)
        img1_array = np.array(img1)
        if img1_array.ndim < 3:
            img1_array = np.expand_dims(img1_array, axis=2)
            img1_array = img1_array.repeat(3, axis=2)
            img1 = Image.fromarray(img1_array)
            img2_array = np.array(img2)
            img2_array = np.expand_dims(img2_array, axis=2)
            img2_array = img2_array.repeat(3, axis=2)
            img2 = Image.fromarray(img2_array)

        # gt_boxes resize
        img1_xiaobai = np.array(img1)
        pad_x = 36.0/264.0*(gt_boxes[selected_id[0],3] - gt_boxes[selected_id[0],1])*img1.width
        pad_y = 36.0/264.0*(gt_boxes[selected_id[0],2] - gt_boxes[selected_id[0],0])*img1.height
        cx = (gt_boxes[selected_id[0],3] + gt_boxes[selected_id[0],1]) / 2.0 * img1.width
        cy = (gt_boxes[selected_id[0],2] + gt_boxes[selected_id[0],0]) / 2.0 *img1.height
        startx = gt_boxes[selected_id[0],1] * img1.width - pad_x
        starty = gt_boxes[selected_id[0],0] * img1.height - pad_y
        endx = gt_boxes[selected_id[0],3] * img1.width + pad_x
        endy = gt_boxes[selected_id[0],2] * img1.height + pad_y
        left_pad = max(0, int(-startx))
        top_pad = max(0, int(-starty))
        right_pad = max(0, int(endx - img1.width + 1))
        bottom_pad = max(0, int(endy - img1.height + 1))

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

            # h, w = r.shape
            # r1 = np.zeros([h, w, 1], dtype=np.float32)
            # r1[:, :, 0] = r
            # g1 = np.zeros([h, w, 1], dtype=np.float32)
            # g1[:, :, 0] = g
            # b1 = np.zeros([h, w, 1], dtype=np.float32)
            # b1[:, :, 0] = b

            img1_xiaobai = np.concatenate((r, g, b), axis=2)
        img1_xiaobai = Image.fromarray(img1_xiaobai)

        # gt_boxes resize
        init_win = img1_xiaobai.crop(np.int32([startx, starty,endx, endy]))
        #init_win, _ = crop_search_region(img1, gt_boxes[[selected_id[0]]], win_size=[300,300], scales=[1.2, 1.2])
        flip_image = np.random.rand() > 0.5
        if flip_image:
            init_win = ImageOps.mirror(init_win)
            img2 = ImageOps.mirror(img2)
            xmin = gt_boxes[selected_id[1]][1]
            xmax = gt_boxes[selected_id[1]][3]
            gt_boxes[selected_id[1]][1] = 1 - xmax
            gt_boxes[selected_id[1]][3] = 1 - xmin

        init_win = init_win.resize(box_size, resample=Image.BILINEAR)
        init_win_array = np.expand_dims(np.array(init_win), axis=0)

        # area = 128*128
        # target_area = random.uniform(0.02,0.4) * area
        # aspect_ratio = random.uniform(0.3,1.0/0.3)
        # h = int(round(math.sqrt(target_area * aspect_ratio)))
        # w = int(round(math.sqrt(target_area / aspect_ratio)))
        # if h < 128 and w < 128:
        #     x1 = random.randint(0,128-h)
        #     y1 = random.randint(0,128-w)
        #     init_win_array[0,x1:x1+h,y1:y1+w,0] = 0#123.68
        #     init_win_array[0,x1:x1+h,y1:y1+w,1] = 0#116.779
        #     init_win_array[0,x1:x1+h,y1:y1+w,2] = 0#103.939

        out_img = np.zeros([seq_length, win_size[0], win_size[1], 3], dtype=np.uint8)
        groundtruth_boxes = np.zeros([seq_length, 4], dtype=np.float32)
        groundtruth_classes = np.ones([seq_length, 1], dtype=np.int32)
        scale_max = 0.5
        scale_alpha = 4
        scale_beta = 0.5
        for ind in range(0,seq_length):
            scaleh = _random_exp_norm(scale_alpha, scale_beta, scale_max)
            scalew = _random_exp_norm(scale_alpha, scale_beta, scale_max)
            scales = [scaleh, scalew]
            offsets = np.random.laplace(0, 0.2, [2])
            win2, gt_box2 = crop_search_region(img2, gt_boxes[selected_id[[1]]], win_size, scales=scales, offsets=offsets)
            out_img[ind] = np.array(win2)
            #out_img[ind] = cv2.resize(np.array(img2),(300,300))
            groundtruth_boxes[ind] = gt_box2
        return init_win_array, out_img, groundtruth_boxes, groundtruth_classes

    #
    # sess = tf.Session()
    # coord = tf.train.Coordinator()
    # threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # for i in range(1000):
    #    a = sess.run(pre_video_tensor_dict)
    #    init_box, train_images, gt_boxes, gt_classes =_read_video(a['folder'], a['filename'], a['groundtruth_boxes'])
    #    cv2.startWindowThread()
    #    cv2.namedWindow('im1')
    #    cv2.namedWindow('im2')
    #    im1 = init_box[0]
    #    im2 = train_images[0]
    #    cv2.imshow('im1', im1[:,:,-1::-1])
    #    cv2.rectangle(im2, (np.int32(gt_boxes[0,1]*300), np.int32(gt_boxes[0,0]*300)),
    #                  (np.int32(gt_boxes[0,3]*300), np.int32(gt_boxes[0,2]*300)), [0,0,255], 2)
    #    cv2.imshow('im2', im2[:, :, -1::-1])
    #
    # for i in range(1000):
    #     a = sess.run(pre_image_tensor_dict)
    #     init_box, train_images, gt_boxes, gt_classes = _read_image(a['filename'], a['groundtruth_boxes'])
    #
    #     cv2.startWindowThread()
    #     cv2.namedWindow('im1')
    #     cv2.namedWindow('im2')
    #     im1 = init_box[0]
    #     im2 = train_images[0]
    #     cv2.imshow('im1', im1[:,:,-1::-1])
    #     cv2.rectangle(im2, (np.int32(gt_boxes[0,1]*300), np.int32(gt_boxes[0,0]*300)),
    #                   (np.int32(gt_boxes[0,3]*300), np.int32(gt_boxes[0,2]*300)), [0,0,255], 2)
    #     cv2.imshow('im2', im2[:, :, -1::-1])

    def _batch_image_input(tensor_dict):
        init_box, images, groundtruth_boxes, groundtruth_classes = tf.py_func(_read_image,
                                                                    [tensor_dict[
                                                                         fields.InputDataFields.filename],
                                                                     tensor_dict[
                                                                         fields.InputDataFields.groundtruth_boxes]],
                                                                    [tf.uint8, tf.uint8, tf.float32, tf.int32])

        seq_length = 1
        box_size = 128
        win_size = 300
        init_box.set_shape([1, box_size, box_size, 3])

        images.set_shape([seq_length, win_size, win_size, 3])
        groundtruth_boxes.set_shape([seq_length, 4])
        groundtruth_classes.set_shape([seq_length,1])
        image_tensor_dict = dict()
        float_init_box = tf.to_float(init_box)
        float_images = tf.to_float(images)
        image_tensor_dict['InitBox'] = detection_model.preprocess(float_init_box, win_size=[box_size, box_size])
        image_tensor_dict[fields.InputDataFields.image] = detection_model.preprocess(float_images)
        image_tensor_dict[fields.InputDataFields.groundtruth_classes] = groundtruth_classes
        image_tensor_dict[fields.InputDataFields.groundtruth_boxes] = groundtruth_boxes
        image_batched_tensor = tf.train.batch(image_tensor_dict,
                                        capacity=batch_queue_capacity,
                                        batch_size=batch_size_per_clone,
                                        num_threads=num_batch_queue_threads,
                                        dynamic_pad=True
                                        )
        return image_batched_tensor
    def _batch_video_input(tensor_dict):
        init_box, images, groundtruth_boxes, groundtruth_classes = tf.py_func(_read_video,
                                                                    [tensor_dict['folder'],
                                                                    tensor_dict[fields.InputDataFields.filename],
                                                                    tensor_dict[fields.InputDataFields.groundtruth_boxes]],
                                                                    [tf.uint8, tf.uint8, tf.float32, tf.int32])
        seq_length = 1
        box_size = 128
        win_size=300
        init_box.set_shape([1, box_size, box_size, 3])
        images.set_shape([seq_length, win_size, win_size, 3])
        groundtruth_boxes.set_shape([seq_length, 4])
        groundtruth_classes.set_shape([seq_length,1])
        video_tensor_dict = dict()
        float_init_box = tf.to_float(init_box)
        float_images = tf.to_float(images)
        video_tensor_dict['InitBox'] = detection_model.preprocess(float_init_box, [box_size,  box_size])
        video_tensor_dict[fields.InputDataFields.image] = detection_model.preprocess(float_images)
        video_tensor_dict[fields.InputDataFields.groundtruth_classes] = groundtruth_classes
        video_tensor_dict[fields.InputDataFields.groundtruth_boxes] = groundtruth_boxes
        video_batched_tensor = tf.train.batch(video_tensor_dict,
                                        capacity=batch_queue_capacity,
                                        batch_size=batch_size_per_clone,
                                        num_threads=num_batch_queue_threads,
                                        dynamic_pad=True
                                        )
        return video_batched_tensor

    image_batch = _batch_image_input(pre_image_tensor_dict)
    video_batch = _batch_video_input(pre_video_tensor_dict)
    _augment_negative_samples(image_batch, video_batch)

    dtypes = [t.dtype for t in video_batch.values()]
    shapes = [t.get_shape() for t in video_batch.values()]
    names = list(video_batch.keys())

    prefetch_queue = tf.FIFOQueue(capacity=prefetch_queue_capacity, dtypes=dtypes, shapes=shapes, names=names)
    #init_image_prefetch = prefetch_queue.enqueue(image_batch)
    #init_video_prefetch = prefetch_queue.enqueue(video_batch)
    mixed_batch = _stack_tensor_dicts([image_batch, video_batch])
    init_prefetch_queue = prefetch_queue.enqueue_many(mixed_batch)
    tf.train.add_queue_runner(
        tf.train.QueueRunner(prefetch_queue, [init_prefetch_queue] * (num_batch_queue_threads)))
    #tf.train.add_queue_runner(tf.train.QueueRunner(prefetch_queue, [init_image_prefetch] * (num_batch_queue_threads/2)))
    #tf.train.add_queue_runner(tf.train.QueueRunner(prefetch_queue, [init_video_prefetch] * (num_batch_queue_threads/2)))
    #x = prefetch_queue.dequeue()
    #sess = tf.Session()
    #coord = tf.train.Coordinator()
    #threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    #for i in range(1000):
    #    a = sess.run(x)
    #   # cv2.startWindowThread()
    #    cv2.namedWindow('im1')
    #   cv2.namedWindow('im2',cv2.WINDOW_NORMAL)
    #    cv2.namedWindow('im3')
    #    im1 = (a['InitBox'][0,0]+1)/2
    #    im2 = (a['image'][0,0]+1)/2
    #    im2 = cv2.resize(im2,(10,10))
    #    im3 = (a['image'][0,1]+1) / 2
    #    im3 = cv2.resize(im3,(19,19))
    #    cv2.imshow('im1', im1[:,:,-1::-1])
        #cv2.rectangle(im2, (np.int32(a['groundtruth_boxes'][0,0,1]*300), np.int32(a['groundtruth_boxes'][0,0,0]*300)),
        #              (np.int32(a['groundtruth_boxes'][0,0,3]*300), np.int32(a['groundtruth_boxes'][0,0,2]*300)), [0,0,255], 2)
    #    cv2.imshow('im2', im2[:, :, -1::-1])
    #    cv2.rectangle(im3, (np.int32(a['groundtruth_boxes'][0,1,1]*19), np.int32(a['groundtruth_boxes'][0,1,0]*19)),
    #                  (np.int32(a['groundtruth_boxes'][0,1,3]*19), np.int32(a['groundtruth_boxes'][0,1,2]*19)), [0,1,255], 2)
    #    cv2.imshow('im3', im3[:, :, -1::-1])
    #    cv2.waitKey(0)

    return prefetch_queue

def _get_inputs(input_queue):
    tensor_dict = input_queue.dequeue()
    init_box = tensor_dict['InitBox']
    images = tensor_dict[fields.InputDataFields.image]
    groundtruth_box = tensor_dict[fields.InputDataFields.groundtruth_boxes]
    groundtruth_class = tensor_dict[fields.InputDataFields.groundtruth_classes]
    return init_box, images, groundtruth_box, groundtruth_class, None

def _create_losses(groundtruth_boxes,InitBox,groundtruth_classes,image, detection_model,reuse=False):
  """Creates loss function for a DetectionModel.

  Args:
    input_queue: BatchQueue object holding enqueued tensor_dicts.
    create_model_fn: A function to create the DetectionModel.
  """

  detection_model.provide_groundtruth(groundtruth_boxes,
                                      groundtruth_classes,
                                      None)

  prediction_dict = detection_model.predict(InitBox, image,istraining=True,reuse=reuse)

  losses_dict = detection_model.loss(prediction_dict)
  loss = 0
  for loss_tensor in losses_dict.values():
    loss += loss_tensor
    tf.losses.add_loss(loss_tensor)

  return loss

def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

PS_OPS = ['Variable', 'VariableV2', 'AutoReloadVariable']

def assign_to_device(device, ps_device='/cpu:0'):
    def _assign(op):
        node_def = op if isinstance(op, tf.NodeDef) else op.node_def
        if node_def.op in PS_OPS:
            return "/" + ps_device
        else:
            return device

    return _assign

def train(create_model_fn, create_tensor_dict_fn, train_config, train_dir, img_root, video_root):
    detection_model = create_model_fn()
    data_augmentation_options = [
              preprocessor_builder.build(step)
              for step in train_config.data_augmentation_options]
    gpu_num = 2
    with tf.device('cpu:0'):
        global_step = slim.create_global_step()

        input_queue = _create_input_queue(train_config.batch_size*gpu_num,
                                          create_tensor_dict_fn,
                                          detection_model,
                                          train_config.batch_queue_capacity,
                                          train_config.num_batch_queue_threads,
                                          train_config.prefetch_queue_capacity,
                                          data_augmentation_options,
                                          img_root, video_root)

       # inputOp = input_queue.dequeue()
        (InitBox1, image1, groundtruth_boxes1, groundtruth_classes1,
         groundtruth_masks
         ) = _get_inputs(input_queue)

        reuse_vars = False
        #task_lossb = []
        tower_grads = []
        for gpu_id in range(gpu_num):
            with tf.device(assign_to_device('/gpu:{}'.format(gpu_id), ps_device='/cpu:0')):
                _groundtruth_boxes1 = groundtruth_boxes1[gpu_id*train_config.batch_size:(gpu_id+1)*train_config.batch_size]
                _InitBox1 = InitBox1[gpu_id*train_config.batch_size:(gpu_id+1)*train_config.batch_size]
                _groundtruth_classes1 = groundtruth_classes1[gpu_id*train_config.batch_size:(gpu_id+1)*train_config.batch_size]
                _image1 = image1[gpu_id*train_config.batch_size:(gpu_id+1)*train_config.batch_size]

                detection_model = create_model_fn()
                task_lossa = _create_losses(_groundtruth_boxes1,_InitBox1,_groundtruth_classes1,_image1, detection_model,reuse=reuse_vars)
                optimizer = optimizer_builder.build(train_config.optimizer,
                                                    set())
                grads = optimizer.compute_gradients(task_lossa)
                pretrained_regex_list = ['^FeatureExtractor/MobilenetV1/Conv2d']
                grads_and_vars = variables_helper.multiply_gradients_matching_regex(
                    grads,
                    pretrained_regex_list,
                    multiplier=0.0)
                pretrained_regex_list = ['^InitFeatureExtractor/MobilenetV1/Conv2d_[0-13]']
                grads_and_vars = variables_helper.multiply_gradients_matching_regex(
                    grads_and_vars,
                    pretrained_regex_list,
                    multiplier=0.0)
                reuse_vars = True
                tower_grads.append(grads_and_vars)

        tower_grads = average_gradients(tower_grads)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = optimizer.apply_gradients(tower_grads,global_step=global_step)
        update_ops.append(train_op)
        update_op = tf.group(*update_ops)
        total_loss = tf.losses.get_total_loss()
        with tf.control_dependencies([update_op]):
            train_tensor = tf.identity(total_loss, name='train_op')


    # create initial restore op
        init_fn = None
        if train_config.fine_tune_checkpoint:
          var_map = detection_model.restore_map(
              from_detection_checkpoint=train_config.from_detection_checkpoint)
          init_var_map = detection_model.restore_init_map(
              from_detection_checkpoint=train_config.from_detection_checkpoint)
          available_var_map = (variables_helper.
              get_variables_available_in_checkpoint(
              var_map, train_config.fine_tune_checkpoint))
          init_available_var_map = (variables_helper.
              get_variables_available_in_checkpoint(
              init_var_map, train_config.fine_tune_checkpoint))
          saver = tf.train.Saver(available_var_map)
          init_saver = tf.train.Saver(init_available_var_map)

          def initializer_fn(sess):
            saver.restore(sess, train_config.fine_tune_checkpoint)
            init_saver.restore(sess, train_config.fine_tune_checkpoint)
            init_saver.restore(sess, train_config.fine_tune_checkpoint)
          init_fn = initializer_fn

        # session_config.gpu_options.allow_growth = True
        keep_checkpoint_every_n_hours = train_config.keep_checkpoint_every_n_hours
        training_saver = tf.train.Saver(
            keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours)
        session_config = tf.ConfigProto(allow_soft_placement=True,
                                        log_device_placement=False)
        slim.learning.train(train_tensor,logdir=train_dir,session_config=session_config,
                            init_fn=init_fn, number_of_steps=(
            train_config.num_steps if train_config.num_steps else None),
        save_summaries_secs=120,
        saver=training_saver,
                           global_step=global_step)

    # coord = tf.train.Coordinator()
    # threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # training_optimizer = optimizer_builder.build(train_config.optimizer,
    #                                              global_summaries)
    # loss = tf.add_n(tf.get_collection(tf.GraphKeys.LOSSES))
    # optimize_op = training_optimizer.minimize(loss, global_step=slim.get_global_step())
    # sess.run(tf.global_variables_initializer())
    # init_fn(sess)
    # for i in range(train_config.num_steps):
    #     cur_loss, _ = sess.run([loss, optimize_op])
    #     print("Iteration %d, Loss: %f"%(i, cur_loss))
