# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Detection model evaluator.

This file provides a generic evaluation method that can be used to evaluate a
DetectionModel.
"""
import logging
import tensorflow as tf
from PIL import Image
from object_detection import eval_util
from object_detection.core import box_list
from object_detection.core import box_list_ops
from object_detection.core import prefetcher
from object_detection.core import standard_fields as fields
from core.trainer_seq import _concat_tensor_dicts, _split_tensor_dict
from object_detection.utils import ops
import numpy as np
import os
from core.preprocessor import preprocess
import cv2

slim = tf.contrib.slim

EVAL_METRICS_FN_DICT = {
    'pascal_voc_metrics': eval_util.evaluate_detection_results_pascal_voc
}



def _create_input_queue(batch_size_per_clone, create_tensor_dict_fn, detection_model,
                        batch_queue_capacity, num_batch_queue_threads,
                        prefetch_queue_capacity, image_path):
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
    tensor_dict = create_tensor_dict_fn()

    def _read_image(folder, im_names, groundtruth_boxes):
        num_frames = len(im_names)
        size = 300
        seq_length = 2
        frame_ids = np.random.randint(0, num_frames, seq_length)
        imgs = np.zeros([seq_length, size, size, 3], dtype=np.uint8)
        for ind, frame_id in enumerate(frame_ids):
            img = Image.open(os.path.join(image_path+folder, im_names[frame_id] + '.JPEG'))
            img = img.resize(np.int32([size, size]))
            img = np.array(img).astype(np.uint8)
            if img.ndim < 3:
                img = np.repeat(np.expand_dims(img, axis=2), repeats=3, axis=2)
            imgs[ind] = img
            # imgs.append(img)
        groundtruth_boxes = groundtruth_boxes[frame_ids,:]
        groundtruth_classes = np.ones([seq_length, 1], dtype=np.float32)
        return imgs, groundtruth_boxes, groundtruth_classes

    images, groundtruth_boxes, groundtruth_classes = tf.py_func(_read_image, [tensor_dict['folder'], tensor_dict['filename'],
                                     tensor_dict['groundtruth_boxes']], [tf.uint8, tf.float32, tf.float32])
    seq_length = 2
    images.set_shape([seq_length, 300, 300, 3])
    float_images = tf.to_float(images)

    groundtruth_boxes.set_shape([seq_length, 4])
    groundtruth_classes.set_shape([seq_length, 1])
    tensor_dict = dict()
    tensor_dict[fields.InputDataFields.image] = float_images
    tensor_dict[fields.InputDataFields.groundtruth_boxes] = groundtruth_boxes
    tensor_dict[fields.InputDataFields.groundtruth_classes] = groundtruth_classes

    tensor_dicts = _split_tensor_dict(tensor_dict, seq_length)
    tensor_dicts = [preprocess(tensor_dict.copy()) for tensor_dict in tensor_dicts]

    for i in range(seq_length):
        tensor_dicts[i][fields.InputDataFields.image] = \
            detection_model.preprocess(tensor_dicts[i][fields.InputDataFields.image])
        tensor_dicts[i]['original_image'] = tensor_dicts[i][fields.InputDataFields.image]
        tensor_dicts[i][fields.InputDataFields.groundtruth_classes].set_shape([1, 1])
        # tensor_dicts[i][fields.InputDataFields.filename].set_shape([1, 1])
        tensor_dicts[i][fields.InputDataFields.groundtruth_boxes].set_shape([1, 4])

    concat_tensor_dict = _concat_tensor_dicts(tensor_dicts)

    batched_tensor = tf.train.batch(concat_tensor_dict,
                                    capacity=batch_queue_capacity,
                                    batch_size=batch_size_per_clone,
                                    num_threads=num_batch_queue_threads,
                                    dynamic_pad=True
                                    )

    dtypes = [t.dtype for t in batched_tensor.values()]
    shapes = [t.get_shape() for t in batched_tensor.values()]
    names = list(batched_tensor.keys())


    prefetch_queue = tf.FIFOQueue(capacity=prefetch_queue_capacity, dtypes=dtypes, shapes=shapes, names=names)
    init_prefetch = prefetch_queue.enqueue(batched_tensor)
    tf.train.add_queue_runner(tf.train.QueueRunner(prefetch_queue, [init_prefetch] * num_batch_queue_threads))
    return prefetch_queue


def _get_inputs(input_queue):
    tensor_dict = input_queue.dequeue()
    images = tensor_dict[fields.InputDataFields.image]
    groundtruth_box = tensor_dict[fields.InputDataFields.groundtruth_boxes]
    groundtruth_class = tensor_dict[fields.InputDataFields.groundtruth_classes]
    original_image = tensor_dict['original_image']
    return images, groundtruth_box, groundtruth_class, original_image




def _extract_prediction_tensors(model,
                                create_input_dict_fn,
                                image_root,
                                ignore_groundtruth=False):
    """Restores the model in a tensorflow session.
    
    Args:
    model: model to perform predictions with.
    create_input_dict_fn: function to create input tensor dictionaries.
    ignore_groundtruth: whether groundtruth should be ignored.
    
    Returns:
    tensor_dict: A tensor dictionary with evaluations.
    """
    # input_dict = create_input_dict_fn()
    # prefetch_queue = prefetcher.prefetch(input_dict, capacity=500)
    # input_dict = prefetch_queue.dequeue()
    ## ##########################################3
    input_queue = _create_input_queue(batch_size_per_clone = 1,
                                      create_tensor_dict_fn = create_input_dict_fn,
                                      detection_model = model,
                                      batch_queue_capacity = 10,
                                      num_batch_queue_threads = 8,
                                      prefetch_queue_capacity = 10,
                                      image_path = image_root)

    (images, groundtruth_boxes, groundtruth_classes,
     original_image) = _get_inputs(input_queue)
    model.provide_groundtruth(groundtruth_boxes,
                                        groundtruth_classes,
                                        None)
    prediction_dict = model.predict(images)
    detections = model.postprocess(prediction_dict)


    # original_image = tf.expand_dims(input_dict[fields.InputDataFields.image], 0)
    # preprocessed_image = model.preprocess(tf.to_float(original_image))
    # prediction_dict = model.predict(preprocessed_image)
    # detections = model.postprocess(prediction_dict)

    original_image_shape = tf.shape(original_image)
    absolute_detection_boxlist = box_list_ops.to_absolute_coordinates(
        box_list.BoxList(tf.squeeze(detections['detection_boxes'], axis=0)),
        original_image_shape[2], original_image_shape[3])
    absolute_groundtruth_boxlist = box_list_ops.to_absolute_coordinates(
        box_list.BoxList(tf.squeeze(groundtruth_boxes, axis=0)),
    original_image_shape[2], original_image_shape[3])

    label_id_offset = 1
    tensor_dict = {
        'original_image': original_image,
        'detection_boxes': absolute_detection_boxlist.get(),
        'groundtruth_boxes': absolute_groundtruth_boxlist.get(),
        'detection_scores': tf.squeeze(detections['detection_scores'], axis=0),
        'detection_classes': (
            tf.squeeze(detections['detection_classes'], axis=0) +
            label_id_offset),
    }

    return tensor_dict


def evaluate(create_input_dict_fn, create_model_fn, eval_config, categories,
             checkpoint_dir, eval_dir, image_root):
    """Evaluation function for detection models.
    
    Args:
    create_input_dict_fn: a function to create a tensor input dictionary.
    create_model_fn: a function that creates a DetectionModel.
    eval_config: a eval_pb2.EvalConfig protobuf.
    categories: a list of category dictionaries. Each dict in the list should
                have an integer 'id' field and string 'name' field.
    checkpoint_dir: directory to load the checkpoints to evaluate from.
    eval_dir: directory to write evaluation metrics summary to.
    """

    model = create_model_fn()
    tensor_dict = _extract_prediction_tensors(
        model=model,
        create_input_dict_fn=create_input_dict_fn,
        image_root=image_root,
        ignore_groundtruth=eval_config.ignore_groundtruth)

    def _display_batch(tensor_dict, sess):
        res_tensor = sess.run(tensor_dict)
        original_images = ((res_tensor['original_image'] + 1)/2*255).astype(np.uint8)
        gt_boxes = res_tensor['groundtruth_boxes']
        detection_box = res_tensor['detection_boxes'][0]
        cv2.namedWindow('1')
        cv2.rectangle(original_images[0,0], (gt_boxes[0][1], gt_boxes[0][0]),
                      (gt_boxes[0][3], gt_boxes[0][2]), [255,0,0], 2)
        cv2.imshow('1', original_images[0,0,:,:,-1::-1])

        cv2.namedWindow('2')
        cv2.rectangle(original_images[0, 1], (gt_boxes[1][1], gt_boxes[1][0]),
                      (gt_boxes[1][3], gt_boxes[1][2]), [255, 0, 0], 2)
        cv2.rectangle(original_images[0, 1], (detection_box[1], detection_box[0]),
                      (detection_box[3], detection_box[2]), [0, 255, 0], 2)
        cv2.imshow('2', original_images[0, 1, :, :, -1::-1])
        print("Detection Score %f"%(res_tensor['detection_scores'][0]))



    variables_to_restore = tf.global_variables()
    global_step = slim.get_or_create_global_step()
    variables_to_restore.append(global_step)
    if eval_config.use_moving_averages:
        variable_averages = tf.train.ExponentialMovingAverage(0.0)
        variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)
    def _restore_latest_checkpoint(sess):
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        saver.restore(sess, latest_checkpoint)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    _restore_latest_checkpoint(sess)

    cv2.startWindowThread()
    for i in range(5000):
        _display_batch(tensor_dict, sess)